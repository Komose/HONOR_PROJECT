import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# -------- config --------
epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
batch_size = 1
train_epochs_if_no_pretrained = 3   # 若无预训练模型，简单训练几个 epoch（可改）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(pretrained_model), exist_ok=True)

# -------- Model definition (LeNet-like) --------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# -------- Data loaders --------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# -------- helper: denorm (to [0,1] range) --------
def denorm(batch, mean=[0.1307], std=[0.3081], device=device):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    # batch shape: (N, C, H, W)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# -------- FGSM attack function --------
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# -------- training function (only used if no pretrained model) --------
def train_simple(model, device, train_loader, epochs=1, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            total += data.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Train Epoch {epoch+1}/{epochs}, acc = {correct}/{total} = {correct/total:.4f}")
    return model

# -------- test under attack --------
def test(model, device, test_loader, epsilon):
    model.eval()
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # restore to original scale, run FGSM, then renormalize
        data_denorm = denorm(data)
        perturbed = fgsm_attack(data_denorm, epsilon, data_grad)
        # re-normalize with same transform
        perturbed_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed.squeeze(0)).unsqueeze(0)
        perturbed_normalized = perturbed_normalized.to(device)

        output2 = model(perturbed_normalized)
        final_pred = output2.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if epsilon == 0 and len(adv_examples) < 5:
                adv_examples.append((init_pred.item(), final_pred.item(), perturbed.squeeze().detach().cpu().numpy()))
        else:
            if len(adv_examples) < 5:
                adv_examples.append((init_pred.item(), final_pred.item(), perturbed.squeeze().detach().cpu().numpy()))

    # NOTE: len(test_loader) corresponds to number of batches (batch_size=1) -> equals number of samples
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples

# -------- main flow --------
def main():
    model = Net().to(device)

    # load or train
    if os.path.exists(pretrained_model):
        print("Loading pretrained model:", pretrained_model)
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
    else:
        print("No pretrained model found. Training a simple model for", train_epochs_if_no_pretrained, "epochs")
        model = train_simple(model, device, train_loader, epochs=train_epochs_if_no_pretrained)
        torch.save(model.state_dict(), pretrained_model)
        print("Saved pretrained model to", pretrained_model)

    model.eval()
    accuracies = []
    examples = []

    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    # plot accuracy vs epsilon
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(0, .35, 0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("accuracy_vs_epsilon.png")
    plt.show()

    # show some adversarial examples
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=10)
            orig, adv, ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.savefig("examples_grid.png")
    plt.show()

if __name__ == "__main__":
    main()

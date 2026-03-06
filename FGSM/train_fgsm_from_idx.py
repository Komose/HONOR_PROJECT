# train_fgsm_from_idx.py
import os
import time
import csv
import math
import gzip
import struct
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ========= 配置区域 =========
RAW_DIR = r"../data/MNIST/raw"   # 放置 idx 文件的目录（包含 train-images-idx3-ubyte 等）
SAVE_DIR = r"./trained_model"    # 训练好的模型与结果输出目录
EPOCHS = 5                       # 对抗训练轮数
LR = 1e-3                        # 学习率
BATCH_SIZE = 64
EPSILON_TRAIN = 0.15             # 训练时使用的 FGSM epsilon
MIX_RATIO = 0.5                  # 0~1；1 表示全对抗样本，0.5 表示对抗:干净=1:1
USE_PRED_LABEL = True            # 生成对抗样本时用模型预测标签，缓解 label leaking
RANDOM_START = True              # FGSM 随机起点（FGSM-RS）
EPSILONS_TEST = [0, .05, .1, .15, .2, .25, .3]  # 测试量化的 epsilons
SEED = 42
# ==========================

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 读取 IDX (ubyte/.gz) =========
def read_idx_images(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def read_idx_labels(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

# ========= 自定义 Dataset（仅用原始 IDX）=========
class MNISTIdxDataset(Dataset):
    def __init__(self, images_path, labels_path, normalize=True):
        self.images = read_idx_images(images_path)  # (N,28,28), uint8 [0,255]
        self.labels = read_idx_labels(labels_path)  # (N,), uint8 0..9
        assert len(self.images) == len(self.labels)
        self.normalize = normalize
        # 常用的 MNIST 归一化参数
        self.mean = 0.1307
        self.std = 0.3081

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # -> [0,1]
        img = torch.from_numpy(img).unsqueeze(0)           # (1,28,28)
        if self.normalize:
            img = (img - self.mean) / self.std
        label = int(self.labels[idx])
        return img, label

# ========= LeNet-like 模型 =========
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
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ========= 辅助：标准化 / 反标准化 =========
def denorm(x_normed, mean=0.1307, std=0.3081):
    return x_normed * std + mean

@torch.no_grad()
def _normalize_(x, mean=0.1307, std=0.3081):
    return (x - mean) / std

# ========= FGSM 生成（训练用）=========
def make_adversarial(model, x_normed, y, epsilon=0.15,
                     use_pred_label=True, random_start=True):
    """
    输入/输出均是“标准化空间”的张量，内部会自动反标准化到[0,1]添加扰动，再标准化回去。
    """
    model.zero_grad(set_to_none=True)
    x_normed = x_normed.detach().clone().requires_grad_(True)
    logits = model(x_normed)

    if use_pred_label:
        y_for_attack = logits.argmax(dim=1).detach()
    else:
        y_for_attack = y

    loss = F.nll_loss(logits, y_for_attack)
    loss.backward()
    grad = x_normed.grad.detach()

    # 反标准化到 [0,1]
    x_pix = denorm(x_normed)

    if random_start:
        # 随机起点：[-eps, +eps] 里随机扰动一步
        x_pix = torch.clamp(x_pix + (2 * epsilon) * torch.rand_like(x_pix) - epsilon, 0, 1)

    x_adv_pix = torch.clamp(x_pix + epsilon * grad.sign(), 0, 1)

    # 重新标准化
    x_adv_normed = _normalize_(x_adv_pix)
    return x_adv_normed.detach()

# ========= 对抗训练一个 epoch =========
def train_adv_epoch(model, loader, optimizer, epsilon=0.15, mix_ratio=0.5,
                    use_pred_label=True, random_start=True):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_adv = make_adversarial(model, x, y, epsilon=epsilon,
                                 use_pred_label=use_pred_label,
                                 random_start=random_start)

        # 按比例混合
        if 0 < mix_ratio < 1:
            n = x.size(0)
            k = int(n * mix_ratio)
            x_mix = torch.cat([x_adv[:k], x[k:]], dim=0)
            y_mix = torch.cat([y[:k],    y[k:]], dim=0)
        elif mix_ratio >= 1:
            x_mix, y_mix = x_adv, y
        else:
            x_mix, y_mix = x, y

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_mix)
        loss = F.nll_loss(logits, y_mix)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y_mix.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y_mix).sum().item()
        total += y_mix.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    print(f"[ADV-TRAIN] eps={epsilon} mix={mix_ratio}  loss={avg_loss:.4f}  acc={acc:.4f}")
    return avg_loss, acc

# ========= 测试（含 FGSM 攻击）=========
def test_under_attack(model, loader, epsilon, collect_examples=False, max_examples=5):
    model.eval()
    correct = 0
    collected = []

    with torch.no_grad():
        pass  # we will enable grad per-sample below

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # 需要梯度来取 sign(∇x J)
        x.requires_grad_(True)
        logits = model(x)
        init_pred = logits.argmax(dim=1)
        if init_pred.item() != y.item():
            x.requires_grad_(False)
            continue

        loss = F.nll_loss(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad = x.grad.detach()

        # 反标准化 -> 加扰动 -> 再标准化
        x_pix = denorm(x)
        x_adv_pix = torch.clamp(x_pix + epsilon * grad.sign(), 0, 1)
        x_adv = _normalize_(x_adv_pix).to(device)

        logits2 = model(x_adv)
        final_pred = logits2.argmax(dim=1)
        if final_pred.item() == y.item():
            correct += 1
            if epsilon == 0 and collect_examples and len(collected) < max_examples:
                collected.append((init_pred.item(), final_pred.item(), x_adv_pix.squeeze().cpu().detach().numpy()))

        else:
            if collect_examples and len(collected) < max_examples:
                collected.append((init_pred.item(), final_pred.item(), x_adv_pix.squeeze().cpu().detach().numpy()))


        x.requires_grad_(False)

    total = len(loader)  # batch_size=1 时等于样本数
    acc = correct / total
    return acc, collected

def evaluate(model, test_loader, epsilons, save_prefix):
    accuracies = []
    examples_by_eps = []
    for eps in epsilons:
        acc, ex = test_under_attack(model, test_loader, eps, collect_examples=True, max_examples=5)
        print(f"[EVAL] Epsilon: {eps}\tTest Accuracy = {acc:.4f}")
        accuracies.append(acc)
        examples_by_eps.append(ex)

    # 保存 CSV
    csv_path = os.path.join(SAVE_DIR, f"{save_prefix}_acc_vs_eps.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "accuracy"])
        for eps, acc in zip(epsilons, accuracies):
            w.writerow([eps, acc])
    print("Saved:", csv_path)

    # 画曲线
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(0, .35, 0.05))
    plt.title("Accuracy vs Epsilon (Adversarial Evaluation)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    fig1_path = os.path.join(SAVE_DIR, f"{save_prefix}_accuracy_vs_epsilon.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", fig1_path)

    # 可选：可视化对抗样本网格
    try:
        cnt = 0
        cols = max(1, len(examples_by_eps[0]))
        rows = len(epsilons)
        plt.figure(figsize=(2*cols+2, 2*rows+2))
        for i, exs in enumerate(examples_by_eps):
            for j, ex in enumerate(exs):
                cnt += 1
                plt.subplot(rows, cols, (i*cols)+j+1)
                plt.xticks([]); plt.yticks([])
                orig, adv, img = ex
                if j == 0:
                    plt.ylabel(f"Eps {epsilons[i]}", fontsize=10)
                plt.title(f"{orig}→{adv}", fontsize=9)
                plt.imshow(img, cmap="gray")
        fig2_path = os.path.join(SAVE_DIR, f"{save_prefix}_examples_grid.png")
        plt.tight_layout()
        plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", fig2_path)
    except Exception as e:
        print("Skip examples grid due to:", e)

    return accuracies

# ========= 构建 DataLoader =========
def get_dataloaders(raw_dir, batch_size=64):
    # 训练集：仅使用指定文件（满足你的要求）
    train_img = os.path.join(raw_dir, "train-images-idx3-ubyte")
    train_lbl = os.path.join(raw_dir, "train-labels-idx1-ubyte")
    if not os.path.exists(train_img) and os.path.exists(train_img + ".gz"):
        train_img += ".gz"
    if not os.path.exists(train_lbl) and os.path.exists(train_lbl + ".gz"):
        train_lbl += ".gz"

    if not (os.path.exists(train_img) and os.path.exists(train_lbl)):
        raise FileNotFoundError("Train IDX files not found. Expected 'train-images-idx3-ubyte(.gz)' and 'train-labels-idx1-ubyte(.gz)'")

    train_ds = MNISTIdxDataset(train_img, train_lbl, normalize=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 测试集：建议使用 t10k 文件进行量化
    test_img = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
    test_lbl = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")
    if not os.path.exists(test_img) and os.path.exists(test_img + ".gz"):
        test_img += ".gz"
    if not os.path.exists(test_lbl) and os.path.exists(test_lbl + ".gz"):
        test_lbl += ".gz"

    if not (os.path.exists(test_img) and os.path.exists(test_lbl)):
        raise FileNotFoundError("Test IDX files not found. Expected 't10k-images-idx3-ubyte(.gz)' and 't10k-labels-idx1-ubyte(.gz)'")

    # 为了逐样本攻击，测试时我们用 batch_size=1
    test_ds = MNISTIdxDataset(test_img, test_lbl, normalize=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, pin_memory=True)

    print(f"[DATA] Train: {len(train_ds)}  Test: {len(test_ds)}")
    return train_loader, test_loader

# ========= 主流程 =========
def main():
    train_loader, test_loader = get_dataloaders(RAW_DIR, batch_size=BATCH_SIZE)

    model = Net().to(device)

    pretrained_path = r"D:\PycharmProjects\HONER_PROJECT\FGSM\data\lenet_mnist_model.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"[INFO] Loaded pretrained model from {pretrained_path}")
    else:
        print(f"[WARN] Pretrained model not found at {pretrained_path}, training from scratch!")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("[INFO] Start FGSM adversarial training...")
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_adv_epoch(model, train_loader, optimizer,
                                    epsilon=EPSILON_TRAIN,
                                    mix_ratio=MIX_RATIO,
                                    use_pred_label=USE_PRED_LABEL,
                                    random_start=RANDOM_START)
        print(f"[EPOCH {epoch}/{EPOCHS}] train_loss={loss:.4f}  train_acc={acc:.4f}")

    # 保存模型（不覆盖，使用时间戳）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"lenet_mnist_adv_eps{str(EPSILON_TRAIN).replace('.', '_')}_mix{str(MIX_RATIO).replace('.', '_')}_{ts}.pth"
    ckpt_path = os.path.join(SAVE_DIR, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print("Saved model:", ckpt_path)

    # 评估与量化
    print("[INFO] Evaluating robustness across epsilons:", EPSILONS_TEST)
    save_prefix = f"report_eps{str(EPSILON_TRAIN).replace('.', '_')}_mix{str(MIX_RATIO).replace('.', '_')}_{ts}"
    accuracies = evaluate(model, test_loader, EPSILONS_TEST, save_prefix=save_prefix)

    # 另存一个 JSON 汇总
    summary = {
        "device": str(device),
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epsilon_train": EPSILON_TRAIN,
        "mix_ratio": MIX_RATIO,
        "use_pred_label": USE_PRED_LABEL,
        "random_start": RANDOM_START,
        "epsilons_test": EPSILONS_TEST,
        "accuracies": accuracies,
        "model_path": ckpt_path,
    }
    json_path = os.path.join(SAVE_DIR, f"{save_prefix}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", json_path)

if __name__ == "__main__":
    # Windows OpenMP 冲突提示可抑制（可选）
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

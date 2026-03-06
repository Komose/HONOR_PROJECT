# read_mnist_idx.py
import idx2numpy
import matplotlib.pyplot as plt
import os

# 指定 raw 文件路径（你截图位置）
raw_dir = r"D:/PycharmProjects/HONER_PROJECT/data/MNIST/raw"  # 按需修改
train_images_path = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
train_labels_path = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")

# 若是 .gz 文件，idx2numpy 也能直接读，例如 "train-images-idx3-ubyte.gz"
# imgs = idx2numpy.convert_from_file(train_images_path + ".gz")

imgs = idx2numpy.convert_from_file(train_images_path)
labels = idx2numpy.convert_from_file(train_labels_path)

print("imgs shape:", imgs.shape)    # (60000, 28, 28)
print("labels shape:", labels.shape)# (60000,)

# 展示前 10 张
outdir = "mnist_idx_vis"
os.makedirs(outdir, exist_ok=True)
for i in range(10):
    plt.imshow(imgs[i], cmap="gray")
    plt.title(f"Index {i} Label {labels[i]}")
    plt.axis("off")
    outpath = os.path.join(outdir, f"idx_img_{i}_label_{labels[i]}.png")
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0)
    plt.show()
    print("Saved", outpath)

## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import random

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def evaluate_attack(name, attack, model, inputs, targets):
    """
    对任意一个 Carlini 攻击（L2 / L0 / Li）做：
      - 生成对抗样本
      - 计算成功率
      - 计算 L2 扰动的均值 / 最小值 / 最大值
    """
    timestart = time.time()
    adv = attack.attack(inputs, targets)
    timeend = time.time()

    success = 0
    dists = []

    for i in range(len(adv)):
        # 预测对抗样本
        pred = model.model.predict(adv[i:i+1])
        pred_label = np.argmax(pred)

        if attack.TARGETED:
            # targeted：预测 == 目标标签 才算成功
            target_label = np.argmax(targets[i])
            if pred_label == target_label:
                success += 1
        else:
            # untargeted：预测 != 原标签 才算成功
            orig_label = np.argmax(targets[i])
            if pred_label != orig_label:
                success += 1

        # 不管是 L0/L2/L∞，都可以用 L2 来量化“总体扰动大小”
        dist = np.linalg.norm((adv[i] - inputs[i]).ravel(), 2)
        dists.append(dist)

    success_rate = success / len(adv)

    print(f"\nTargeted attack: {name}")
    print(f"  samples={len(inputs)}, targeted={attack.TARGETED}, start=0, inception=False")
    print(f"  Success rate: {success_rate}")
    print(f"  Mean L2 distortion: {np.mean(dists)}")
    print(f"  Min  L2 distortion: {np.min(dists)}")
    print(f"  Max  L2 distortion: {np.max(dists)}")
    print(f"  Time taken: {timeend - timestart:.2f} seconds")

    save_attack_images(name, inputs, adv, idx=0)

    # 如果后面写报告要用，可以把结果返回成字典
    return {
        "name": name,
        "success_rate": success_rate,
        "mean_l2": float(np.mean(dists)),
        "min_l2": float(np.min(dists)),
        "max_l2": float(np.max(dists)),
        "time": float(timeend - timestart),
    }

def save_attack_images(name, inputs, adv, idx=0):
    """ 保存原图、对抗图、扰动图 """
    orig = inputs[idx]
    adv_img = adv[idx]
    diff = (adv_img - orig) * 20  # 放大扰动可视化

    plt.figure(figsize=(9,3))

    plt.subplot(131)
    plt.title("Original")
    plt.imshow(orig)
    plt.axis('off')

    plt.subplot(132)
    plt.title(f"Adversarial ({name})")
    plt.imshow(adv_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title("Perturbation x20")
    plt.imshow(diff)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"attack_{name}.png")
    plt.close()

if __name__ == "__main__":
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        attack_l2 = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=5)
        attack_l0 = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
                          largest_const=15)
        attack_li = CarliniLi(sess, model, max_iterations=1000,
                           initial_const=10)

        inputs, targets = generate_data(data, samples=10, targeted=True,
                                        start=0, inception=False)
        results_l2 = evaluate_attack("L2", attack_l2, model, inputs, targets)
        results_l0 = evaluate_attack("L0", attack_l0, model, inputs, targets)
        results_li = evaluate_attack("Li", attack_li, model, inputs, targets)
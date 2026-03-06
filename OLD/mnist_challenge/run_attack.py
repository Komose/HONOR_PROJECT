"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

# ====== 兼容 TF1/TF2 的写法（如果你是 TF2，放开下面两行）======
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# ❌ 旧的：from tensorflow.examples.tutorials.mnist import input_data
# ✅ 新的：
from tensorflow.keras.datasets import mnist as keras_mnist
import numpy as np

from model import Model

# ---- 小工具：做个简单的“命名空间对象”来模拟 mnist.test.* ----
class _NS:
    pass

def _load_mnist_like_input_data():
    # 读取并转换到你原代码期望的形状与范围
    (_x_train, _y_train), (_x_test, _y_test) = keras_mnist.load_data()
    x_test = _x_test.reshape(-1, 28*28).astype(np.float32) / 255.0  # (10000, 784), [0,1]
    y_test = _y_test.astype(np.int64)  # 非 one-hot，与你原代码一致

    mn = _NS()
    mn.test = _NS()
    mn.test.images = x_test
    mn.test.labels = y_test
    return mn
# ----------------------------------------------------------------

def run_attack(checkpoint, x_adv, epsilon):
  # ❌ 旧的：mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
  # ✅ 新的：
  mnist = _load_mnist_like_input_data()

  model = Model()
  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 64

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  x_nat = mnist.test.images
  l_inf = np.amax(np.abs(x_nat - x_adv))

  if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    return

  y_pred = [] # label accumulator

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      cur_corr, y_pred_batch = sess.run([model.num_correct, model.y_pred],
                                        feed_dict=dict_adv)

      total_corr += cur_corr
      y_pred.append(y_pred_batch)

  accuracy = total_corr / num_eval_examples

  print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
  y_pred = np.concatenate(y_pred, axis=0)
  np.save('pred.npy', y_pred)
  print('Output saved at pred.npy')

if __name__ == '__main__':
  with open('config.json') as config_file:
    config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv = np.load(config['store_adv_path'])

  if checkpoint is None:
    print('No checkpoint found')
  elif x_adv.shape != (10000, 784):
    print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
  elif np.amax(x_adv) > 1.0001 or \
       np.amin(x_adv) < -0.0001 or \
       np.isnan(np.amax(x_adv)):
    print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                                                              np.amin(x_adv),
                                                              np.amax(x_adv)))
  else:
    run_attack(checkpoint, x_adv, config['epsilon'])

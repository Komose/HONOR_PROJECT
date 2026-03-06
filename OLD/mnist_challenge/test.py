
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
print(tf.__version__.startswith('2'))  # True 表示 TF2

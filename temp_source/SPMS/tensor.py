#%%
import tensorflow as tf
import numpy as np

c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
c2 = tf.constant([1, 3, 5])
v1 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
v2 = tf.constant([[1, 2, 3], [7, 8, 9]])

print('-----------split------------')
print(tf.split(0, 2, c1)) # [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]]
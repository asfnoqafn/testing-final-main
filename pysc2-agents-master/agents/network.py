from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from warnings import filters

import tensorflow as tf
import tensorflow.keras.layers as layers

def build_net(minimap, screen, info, msize, ssize, num_action):
  #print(minimap)
  #print(screen)
  # Extract features
  mconv1 = layers.Conv2D(filters=16,
                         kernel_size=5,
                         strides=1
                         )(tf.transpose(minimap, [0, 2, 3, 1]))
  mconv2 = layers.Conv2D(filters=32,
                         kernel_size=3,
                         strides=1)(mconv1)
  sconv1 = layers.Conv2D(filters=16,
                         kernel_size=5,
                         strides=1)(tf.transpose(screen, [0, 2, 3, 1]))
  sconv2 = layers.Conv2D(filters=32,
                         kernel_size=3,
                         strides=1)(sconv1)
  info_fc = layers.Dense(units=256,activation=tf.tanh)(layers.Flatten()(info))

  # Compute spatial actions
  feat_conv = tf.concat([mconv2, sconv2], axis=3)
  spatial_action = layers.Conv2D(filters=1,
                                 kernel_size=1,
                                 strides=1,
                                 activation=None)(feat_conv)
  spatial_dense = layers.Dense(units=ssize**2)(layers.Flatten()(spatial_action))
  spatial_action = tf.nn.softmax(spatial_dense)

  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.Flatten()(mconv2), layers.Flatten()(sconv2), info_fc], axis=1)
  feat_fc = layers.Dense(units=256,
                                   activation=tf.nn.relu)(feat_fc)
  non_spatial_action = layers.Dense(units=num_action,
                                              activation=tf.nn.softmax)(feat_fc)
  value = tf.reshape(layers.Dense(units=1,activation=None)(feat_fc), [-1])

  return spatial_action, non_spatial_action, value

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from warnings import filters

import tensorflow as tf
import tensorflow.keras.layers as layers


def build_net(minimap, screen, info, msize, ssize, num_action, ntype):
  if ntype == 'atari':
    return build_atari(minimap, screen, info, msize, ssize, num_action)
  elif ntype == 'fcn':
    return build_fcn(minimap, screen, info, msize, ssize, num_action)
  else:
    raise 'FLAGS.net must be atari or fcn'

##### fcn ist default
def build_atari(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  mconv1 = layers.Conv2D(tf.transpose(minimap, [0, 2, 3, 1]),
                         filters=16,
                         kernel_size=8,
                         strides=4,
                         scope='mconv1')
  mconv2 = layers.Conv2D(mconv1,
                         filters=32,
                         kernel_size=4,
                         strides=2,
                         scope='mconv2')
  sconv1 = layers.Conv2D(tf.transpose(screen, [0, 2, 3, 1]),
                         filters=16,
                         kernel_size=8,
                         strides=4,
                         scope='sconv1')
  sconv2 = layers.Conv2D(sconv1,
                         filters=32,
                         kernel_size=4,
                         strides=2)
  info_fc = layers.fully_connected(layers.flatten(info),
                                   filters=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions, non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   filters=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')

  spatial_action_x = layers.fully_connected(feat_fc,
                                            filters=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_x')
  spatial_action_y = layers.fully_connected(feat_fc,
                                            filters=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_y')
  spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
  spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
  spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
  spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
  spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

  non_spatial_action = layers.fully_connected(feat_fc,
                                              filters=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            filters=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value


def build_fcn(minimap, screen, info, msize, ssize, num_action):
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
  spatial_action = tf.nn.softmax(layers.Flatten()(spatial_action))

  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.Flatten()(mconv2), layers.Flatten()(sconv2), info_fc], axis=1)
  feat_fc = layers.Dense(units=256,
                                   activation=tf.nn.relu)(feat_fc)
  non_spatial_action = layers.Dense(units=num_action,
                                              activation=tf.nn.softmax)(feat_fc)
  value = tf.reshape(layers.Dense(units=1,activation=None)(feat_fc), [-1])

  return spatial_action, non_spatial_action, value

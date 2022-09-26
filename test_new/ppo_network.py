from unicodedata import name
import tensorflow as tf
from pysc2.lib import actions
from tensorflow import keras
import utils
from tensorflow.keras import layers


class PPO_Network_actor(tf.keras.Model):
    def __init__(self, FLAGS):
        super(PPO_Network_actor, self).__init__()
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=5,
                                   strides=1,
                                   name="Conv1")
        self.conv2 = layers.Conv2D(filters=32,
                                   kernel_size=3,
                                   strides=1,
                                   name="Conv2")
        self.info_dense = layers.Dense(units=256, activation=tf.tanh,
                                       name="InfoDense")

        self.spatial = layers.Conv2D(filters=1,
                                     kernel_size=1,
                                     strides=1,
                                     activation=None,
                                     name="SpatialConv")
        self.spatial_dense = layers.Dense(units=FLAGS.screen_resolution**2,
                                          activation=tf.nn.softmax,
                                          name="SpatialDence")

        self.feat_fc = layers.Dense(units=256,
                                    activation=tf.nn.relu,
                                    name="FeatFc")
        self.non_spatial = layers.Dense(units=len(actions.FUNCTIONS),
                                        activation=tf.nn.softmax,
                                        name="NonSpatialDense")

    @tf.function
    def call(self, screen_input, available_actions):
        spatial_out = self.conv1(tf.transpose(screen_input, [0, 2, 3, 1]))
        spatial_out_conv2 = self.conv2(spatial_out)
        spatial_out = self.spatial(spatial_out_conv2)
        spatial_out = self.spatial_dense(layers.Flatten()(spatial_out))

        avail_out = self.info_dense(layers.Flatten()(available_actions))
        avail_out = tf.concat(
            [layers.Flatten()(spatial_out_conv2), avail_out], axis=1)
        avail_out = self.feat_fc(avail_out)
        avail_out = self.non_spatial(avail_out)

        return avail_out, spatial_out


class PPO_Network_critic(tf.keras.Model):
    def __init__(self, FLAGS):
        super(PPO_Network_critic, self).__init__()
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=5,
                                   strides=1,
                                   name="Conv1")
        self.conv2 = layers.Conv2D(filters=32,
                                   kernel_size=3,
                                   strides=1,
                                   name="Conv2")
        self.info_dense = layers.Dense(units=256, activation=tf.tanh,
                                       name="InfoDense")

        self.spatial = layers.Conv2D(filters=1,
                                     kernel_size=1,
                                     strides=1,
                                     activation=None,
                                     name="ConvSpatial")

        self.feat_fc = layers.Dense(units=256,
                                    activation=tf.nn.relu,
                                    name="FeatFc")

        self.critic_value = layers.Dense(units=1, activation=None,
                                         name="ValueDense")

    @tf.function
    def call(self, screen_input, available_actions):
        spatial_out = self.conv1(tf.transpose(screen_input, [0, 2, 3, 1]))
        spatial_out = self.conv2(spatial_out)

        avail_out = self.info_dense(layers.Flatten()(available_actions))
        avail_out = tf.concat(
            [layers.Flatten()(spatial_out), avail_out], axis=1)
        avail_out_fc = self.feat_fc(avail_out)

        return self.critic_value(avail_out_fc)

from absl import flags
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pysc2.env import sc2_env
import scipy.signal
import time
import utils
from pysc2.lib import actions
from buffer import Buffer
from ppo_network import PPO_Network_actor, PPO_Network_critic
import os

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "steps_per_epoch",
    300,
    "How many steps to do while one epoch iteration.")
flags.DEFINE_integer("epochs", 1000, "Number of epochs for training.")
flags.DEFINE_float("gamma", 0.99, "Discount value for future rewards.")
flags.DEFINE_float("clip_ratio", 0.2, "Discount value for future rewards.")
flags.DEFINE_float(
    "policy_learning_rate",
    7e-6,
    "Learning rate for the policy function.")
flags.DEFINE_float("value_function_learning_rate", 1e-5,
                   "Learning rate for the value funtion.")
flags.DEFINE_integer(
    "train_policy_iterations",
    75,
    "Max training iterations for policy funtion.")
flags.DEFINE_integer(
    "train_value_iterations",
    75,
    "Training iterations for value funtion.")
flags.DEFINE_float(
    "lam",
    0.9,
    "Smoothing parameter used for reducing variance.")
flags.DEFINE_float(
    "target_kl",
    0.01,
    "Target Kullback Leibler divergence to stop at if reached while policy update.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer(
    "screen_resolution",
    64,
    "Resolution for screen feature layers.")
flags.DEFINE_integer(
    "minimap_resolution",
    64,
    "Resolution for minimap feature layers.")
flags.DEFINE_integer("max_agent_steps", 120, "Total agent steps.")

FLAGS(sys.argv)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except BaseException:
    pass

env = sc2_env.SC2Env(
    map_name="DefeatZerglingsAndBanelings",
    players=[sc2_env.Agent(sc2_env.Race.protoss)],
    step_mul=FLAGS.step_mul,
    agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=FLAGS.screen_resolution,
            minimap=FLAGS.minimap_resolution),
        use_feature_units=True),
    visualize=True)

buffer = Buffer((utils.screen_channel(),
                 FLAGS.screen_resolution,
                 FLAGS.screen_resolution),
                len(actions.FUNCTIONS),
                FLAGS.steps_per_epoch,
                lam=FLAGS.lam)

actor = PPO_Network_actor(FLAGS)
critic = PPO_Network_critic(FLAGS)


policy_optimizer = keras.optimizers.Adam(
    learning_rate=FLAGS.policy_learning_rate)
value_optimizer = keras.optimizers.Adam(
    learning_rate=FLAGS.value_function_learning_rate)

observation = env.reset()
episode_return = 0
episode_length = 0
screen = np.array(
    observation[0].observation['feature_screen'],
    dtype=np.float32)
screen = np.expand_dims(utils.preprocess_screen(screen), axis=0)


@tf.function
def logprobabilities(logits, a):
    logprobabilities_all = tf.nn.log_softmax(logits, axis=-1)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, logits.shape[-1]) * logprobabilities_all, axis=-1
    )
    return logprobability


@tf.function
def train_policy(
        screen_buffer,
        avail_actions_buffer,
        non_spatial_action_buffer,
        spatial_action_buffer,
        logprobability_buffer,
        advantage_buffer):
    non_spatial_logprobability_t, spatial_logprobability_t = tf.split(
        logprobability_buffer, num_or_size_splits=2, axis=1)
    with tf.GradientTape() as non_spatial_tape, tf.GradientTape() as spatial_tape:
        non_spatial_logits, spatial_out_logits = actor(
            screen_buffer, avail_actions_buffer)
        non_spatial_ratio = tf.exp(
            logprobabilities(non_spatial_logits, non_spatial_action_buffer)
            - non_spatial_logprobability_t
        )
        spatial_ratio = tf.exp(
            logprobabilities(spatial_out_logits, spatial_action_buffer)
            - spatial_logprobability_t
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + FLAGS.clip_ratio) * advantage_buffer,
            (1 - FLAGS.clip_ratio) * advantage_buffer,
        )

        non_spatial_policy_loss = -tf.reduce_mean(
            tf.minimum(non_spatial_ratio * advantage_buffer, min_advantage)
        )
        spatial_policy_loss = -tf.reduce_mean(
            tf.minimum(spatial_ratio * advantage_buffer, min_advantage)
        )
    non_spatial_policy_grads = non_spatial_tape.gradient(
        non_spatial_policy_loss, actor.trainable_variables)
    spatial_policy_grads = spatial_tape.gradient(
        spatial_policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(
        zip(non_spatial_policy_grads, actor.trainable_variables))
    policy_optimizer.apply_gradients(
        zip(spatial_policy_grads, actor.trainable_variables))

    non_spatial_logits, spatial_out_logits = actor(
        screen_buffer, avail_actions_buffer)
    non_spatial_kl = tf.reduce_mean(
        non_spatial_logprobability_t
        - logprobabilities(non_spatial_logits, non_spatial_action_buffer)
    )
    spatial_kl = tf.reduce_mean(
        spatial_logprobability_t
        - logprobabilities(spatial_out_logits, spatial_action_buffer)
    )
    kl = tf.add(tf.reduce_sum(non_spatial_kl), tf.reduce_sum(spatial_kl))
    return kl


def step_agent(screen, available_actions, available_actions_one_hot):
    non_spatial_logits, spatial_out_logits = actor(
        screen, available_actions_one_hot)
    spatial_action = tf.squeeze(
        tf.random.categorical(
            spatial_out_logits, 1), axis=1)

    non_spatial_logits = non_spatial_logits.numpy().ravel()
    spatial_out_logits = spatial_out_logits.numpy().ravel()
    non_spatial_action = available_actions[tf.random.categorical(
        [non_spatial_logits[available_actions]], 1)[0][0]]

    target = spatial_action.numpy()[0]
    target = [int(target // FLAGS.screen_resolution),
              int(target % FLAGS.screen_resolution)]

    act_args = []
    for arg in actions.FUNCTIONS[non_spatial_action].args:
        if arg.name in ('screen', 'minimap', 'screen2'):
            act_args.append([target[1], target[0]])
        else:
            act_args.append([0])

    value_t = critic(screen, available_actions_one_hot)
    func_actions = actions.FunctionCall(non_spatial_action, act_args)

    return func_actions, non_spatial_logits, spatial_out_logits, non_spatial_action, spatial_action, value_t

@tf.function
def train_value_function(
        screen_buffer,
        available_actions_buffer,
        return_buffer):
    with tf.GradientTape() as tape:
        value_loss = tf.reduce_mean(
            (return_buffer -
             critic(
                 screen_buffer,
                 available_actions_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(
        zip(value_grads, critic.trainable_variables))


for epoch in range(FLAGS.epochs):
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    num_frames = 0

    for t in range(FLAGS.steps_per_epoch):
        available_actions = observation[0].observation['available_actions']
        available_actions_one_hot = np.zeros(
            [1, len(actions.FUNCTIONS)], dtype=np.float32)
        available_actions_one_hot[0, ] = 1

        func_actions, non_spatial_logits, spatial_out_logits, non_spatial_action, spatial_action, value_t = step_agent(
            screen, available_actions, available_actions_one_hot)
        observation_new = env.step([func_actions])
        episode_return += observation_new[0].reward
        episode_length += 1
        num_frames += 1

        spatial_logprobability_t = logprobabilities(
            spatial_out_logits, spatial_action)
        non_spatial_logprobability_t = logprobabilities(
            non_spatial_logits, non_spatial_action)

        buffer.store(
            screen,
            available_actions_one_hot,
            non_spatial_action,
            spatial_action,
            observation_new[0].reward,
            value_t,
            (non_spatial_logprobability_t,
             spatial_logprobability_t))

        observation = observation_new
        screen = np.array(
            observation[0].observation['feature_screen'],
            dtype=np.float32)
        screen = np.expand_dims(utils.preprocess_screen(screen), axis=0)

        terminal = (
            num_frames >= FLAGS.max_agent_steps) or observation[0].last()
        if terminal or (t == FLAGS.steps_per_epoch - 1):
            last_value = 0 if terminal else critic(
                screen, available_actions_one_hot)
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            num_frames = 0
            observation, episode_return, episode_length = env.reset(), 0, 0

    (
        screen_buffer,
        available_actions_buffer,
        non_spatial_action_buffer,
        spatial_action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    for test in range(FLAGS.train_policy_iterations):
        kl = train_policy(
            screen_buffer,
            available_actions_buffer,
            non_spatial_action_buffer,
            spatial_action_buffer,
            logprobability_buffer,
            advantage_buffer)
        if kl > 1.5 * FLAGS.target_kl:
            break

    for test2 in range(FLAGS.train_value_iterations):
        train_value_function(
            screen_buffer,
            available_actions_buffer,
            return_buffer)

    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )

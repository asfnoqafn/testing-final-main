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
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "1", "Device for training.")

flags.DEFINE_string("map", "DefeatZerglingsAndBanelings", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_integer("max_agent_steps", 120, "Total agent steps.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

FLAGS(sys.argv)
if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/cpu:0']
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']

steps_per_epoch = 200
epochs = 100
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 5e-5
value_function_learning_rate = 1e-4
train_policy_iterations = 75
train_value_iterations = 75
lam = 0.9
target_kl = 0.01

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

env = sc2_env.SC2Env(
    map_name="DefeatZerglingsAndBanelings",
    players= [sc2_env.Agent(sc2_env.Race.protoss)],
    step_mul=FLAGS.step_mul,
    agent_interface_format = sc2_env.AgentInterfaceFormat(
                            feature_dimensions=sc2_env.Dimensions(
                            screen=FLAGS.screen_resolution,
                            minimap=FLAGS.minimap_resolution),
                            use_feature_units=True),
    visualize=True)
    
buffer = Buffer((utils.screen_channel(), FLAGS.screen_resolution, FLAGS.screen_resolution), len(actions.FUNCTIONS), steps_per_epoch, lam=lam)

actor = PPO_Network_actor(FLAGS)
critic = PPO_Network_critic(FLAGS)


policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

observation = env.reset()
episode_return = 0
episode_length = 0
screen = np.array(observation[0].observation['feature_screen'], dtype=np.float32)
screen = np.expand_dims(utils.preprocess_screen(screen), axis=0)
available_actions = np.zeros([1, len(actions.FUNCTIONS)], dtype=np.float32)
available_actions[0, observation[0].observation['available_actions']] = 1

@tf.function
def logprobabilities(logits, a):
    logprobabilities_all = tf.nn.log_softmax(logits, axis=-1)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, logits.shape[-1]) * logprobabilities_all, axis=-1
    )
    return logprobability

@tf.function
def sample_action(screen, avail_actions):
    non_spatial_logits, spatial_out_logits = actor(screen, avail_actions)

    spatial_action = tf.squeeze(tf.random.categorical(spatial_out_logits, 1), axis=1)
    non_spatial_action = tf.squeeze(tf.random.categorical(non_spatial_logits, 1), axis=1)
    return non_spatial_logits, spatial_out_logits, non_spatial_action, spatial_action

@tf.function
def train_policy(screen_buffer, avail_actions_buffer, non_spatial_action_buffer, spatial_action_buffer, logprobability_buffer, advantage_buffer):
    non_spatial_logprobability_t, spatial_logprobability_t = tf.split(logprobability_buffer, num_or_size_splits=2, axis=1)
    with tf.GradientTape() as non_spatial_tape, tf.GradientTape() as spatial_tape:
        non_spatial_logits, spatial_out_logits = actor(screen_buffer, avail_actions_buffer)
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
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        non_spatial_policy_loss = -tf.reduce_mean(
            tf.minimum(non_spatial_ratio * advantage_buffer, min_advantage)
        )
        spatial_policy_loss = -tf.reduce_mean(
            tf.minimum(spatial_ratio * advantage_buffer, min_advantage)
        )
    non_spatial_policy_grads = non_spatial_tape.gradient(non_spatial_policy_loss, actor.trainable_variables)
    spatial_policy_grads = spatial_tape.gradient(spatial_policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(non_spatial_policy_grads, actor.trainable_variables))
    policy_optimizer.apply_gradients(zip(spatial_policy_grads, actor.trainable_variables))

    non_spatial_logits, spatial_out_logits = actor(screen_buffer, avail_actions_buffer)
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

def step(obs, screen, available_actions):
    non_spatial_logits, spatial_out_logits, non_spatial_action, spatial_action = sample_action(screen, available_actions)

    # Select an action and a spatial target
    non_spatial_logits = non_spatial_logits.numpy().ravel()
    spatial_out_logits = spatial_out_logits.numpy().ravel()
    valid_actions = obs.observation['available_actions']
    act_id = valid_actions[np.argmax(non_spatial_logits[valid_actions])]
    target = np.argmax(spatial_out_logits)
    target = [int(target // FLAGS.screen_resolution), int(target % FLAGS.screen_resolution)]

    dy = np.random.randint(-4, 5)
    target[0] = int(max(0, min(FLAGS.screen_resolution-1, target[0]+dy)))
    dx = np.random.randint(-4, 5)
    target[1] = int(max(0, min(FLAGS.screen_resolution-1, target[1]+dx)))

    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])
        
    value_t = critic(screen, available_actions)
    func_actions = actions.FunctionCall(act_id, act_args)

    return func_actions, non_spatial_logits, spatial_out_logits, non_spatial_action, spatial_action, value_t

# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(screen_buffer, available_actions_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(screen_buffer, available_actions_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


for epoch in range(epochs):
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    num_frames = 0

    for t in range(steps_per_epoch):
        func_actions, non_spatial_logits, spatial_out_logits, non_spatial_action, spatial_action, value_t = step(observation[0], screen, available_actions)
        observation_new = env.step([func_actions])
        episode_return += observation_new[0].reward
        episode_length += 1
        num_frames += 1

        # Get the value and log-probability of the action
        spatial_logprobability_t = logprobabilities(spatial_out_logits, spatial_action)
        non_spatial_logprobability_t = logprobabilities(non_spatial_logits, non_spatial_action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(screen, available_actions, non_spatial_action, spatial_action, observation_new[0].reward, value_t, (non_spatial_logprobability_t, spatial_logprobability_t))

        # Update the observation
        observation = observation_new
        screen = np.array(observation[0].observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(utils.preprocess_screen(screen), axis=0)
        available_actions = np.zeros([1, len(actions.FUNCTIONS)], dtype=np.float32)
        available_actions[0, observation[0].observation['available_actions']] = 1

        # Finish trajectory if reached to a terminal state
        terminal = (num_frames >= FLAGS.max_agent_steps) or observation[0].last()
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if terminal else critic(screen, available_actions)
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0
            break

    # Get values from the buffer
    (
      screen_buffer,
      available_actions_buffer,
      non_spatial_action_buffer,
      spatial_action_buffer,
      advantage_buffer,
      return_buffer,
      logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for test in range(train_policy_iterations):
        kl = train_policy(
            screen_buffer, available_actions_buffer, non_spatial_action_buffer, spatial_action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for test2 in range(train_value_iterations):
        train_value_function(screen_buffer, available_actions_buffer, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
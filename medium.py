import numpy as np
import pysc2
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions

from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)



class RandomAgent(base_agent.BaseAgent):
  """A random agent for Starcraft II pysc2."""  
  def step(self, obs):
    super(RandomAgent, self).step(obs)
    function_id = np.random.choice(obs.observation.available_actions)
    args = [[np.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)
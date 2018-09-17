"""OpenAI gym environment implementation for the NEL 
simulator."""

from __future__ import absolute_import, division, print_function

try:
  import gym
  from gym import spaces, logger
  modules_loaded = True
except:
  modules_loaded = False

import numpy as np

from .agent import Agent
from .direction import RelativeDirection
from .simulator import Simulator
from .visualizer import MapVisualizer

__all__ = ['NELEnv']


class NELEnv(gym.Env):
  def __init__(self, sim_config, reward_fn, render=False):
    """

    """
    self.sim_config = sim_config
    self._sim = None
    self._reward_fn = reward_fn
    self._render = render

    self.reset()

    if self._render:
      self._painter = MapVisualizer(
        self._sim, self.sim_config, 
        bottom_left=(-70, -70), top_right=(70, 70))

    # Computing shapes for the observation space.
    scent_shape = [len(self.sim_config.items[0].scent)]
    vision_dim = len(self.sim_config.items[0].color)
    vision_range = self.sim_config.vision_range
    vision_shape = [
      2 * vision_range + 1, 
      2 * vision_range + 1, 
      vision_dim]
    
    min_float = np.finfo(np.float32).min
    max_float = np.finfo(np.float32).max
    min_scent = min_float * np.ones(scent_shape)
    max_scent = max_float * np.ones(scent_shape)
    min_vision = min_float * np.ones(vision_shape)
    max_vision = max_float * np.ones(vision_shape)

    # Observations in this environment consist of a scent 
    # vector and a vision matrix.
    self.observation_space = spaces.Dict({
      'scent': spaces.Box(low=min_scent, high=max_scent), 
      'vision': spaces.Box(low=min_vision, high=max_vision)})

    # There are three possible actions:
    #   1. Move forward,
    #   2. Turn left,
    #   3. Turn right.
    self.action_space = spaces.Discrete(3)

  def step(self, action):
    prev_items = self._agent.collected_items()

    self._agent._next_action = action
    self._agent.do_next_action()

    items = self._agent.collected_items()
    reward = self._reward_fn(prev_items, items)
    done = False

    self.state = {
      'scent': self._agent.scent(), 
      'vision': self._agent.vision()}
    
    return self.state, reward, done, {}
    
  def reset(self):
    del self._sim
    self._sim = Simulator(sim_config=self.sim_config)
    self._agent = _NELEnvAgent(self._sim)
    if self._render:
      self._painter = MapVisualizer(
        self._sim, self.sim_config, 
        bottom_left=(-70, -70), top_right=(70, 70))
    self.state = {
      'scent': self._agent.scent(), 
      'vision': self._agent.vision()}
    return self.state

  def render(self, mode='matplotlib'):
    """Renders this environment in its current state.

    Note that, in order to support rendering, 
    `render=True` must be passed to the environment 
    constructor.
    
    Arguments:
      mode(str) Rendering mode. Currently, only 
                `"matplotlib"` is supported.
    """
    if mode == 'matplotlib' and self._render:
      self._painter.draw()
    elif not self._render:
      logger.warn(
        'Need to pass `render=True` to support '
        'rendering.')
    else:
      logger.warn(
        'Invalid rendering mode "%s". '
        'Only "matplotlib" is supported.')

  def close(self):
    """Deletes the underlying simulator and deallocates 
    all associated memory. This environment cannot be used
    again after it's been closed."""
    del self._sim
    return

  def seed(self, seed=None):
    logger.warn(
      'Could not seed environment %s after it\'s '
      'initialized.', self)
    return


class _NELEnvAgent(Agent):
  """Helper class for the NEL environment, that represents
  a NEL agent living in the simulator.
  """

  def __init__(self, simulator):
    """Creates a new NEL environment agent.
    
    Arguments:
      simulator(Simulator)  The simulator the agent lives in.
    """
    super(_NELEnvAgent, self).__init__(
      simulator, load_filepath=None)
    self._next_action = None

  def do_next_action(self):
    if self._next_action == 0:
      self.move(RelativeDirection.FORWARD)
    elif self._next_action == 1:
      self.turn(RelativeDirection.LEFT)
    elif self._next_action == 2:
      self.turn(RelativeDirection.RIGHT)
    else:
      logger.warn(
        'Ignoring invalid action %d.' 
        % self._next_action)

  # There is no need for saving and loading an agent's
  # state, as that can be done outside the gym environment.

  def save(self, filepath):
    pass

  def _load(self, filepath):
    pass

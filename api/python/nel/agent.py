from __future__ import absolute_import, division, print_function

import abc, six

from .position import Position

__all__ = ['AgentState', 'Agent']


class AgentState(object):
  def __init__(self, position, scent, vision):
    self.position = position
    self.scent = scent
    self.vision = vision


class Agent(six.with_metaclass(abc.ABCMeta, object)):
  def __init__(self, simulator):
    self._simulator = simulator
    self._id, self._current_state = simulator._add_agent(self)
  
  def move(self, direction, num_steps=1):
    self._simulator._move(self._id, direction, num_steps)

  def _update_state(self, state):
    self._current_state = state

  @abc.abstractmethod
  def on_step(self):
    pass

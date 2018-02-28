from __future__ import absolute_import, division, print_function

import abc

from .position import Position

__all__ = ['AgentState', 'Agent']


class AgentState(object):
  def __init__(self, position, scent, vision):
    self.position = position
    self.scent = scent
    self.vision = vision


class Agent(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, simulator):
    self._simulator = simulator
    self._id = simulator._add_agent(self)
  
  def move(self, direction, num_steps=1):
    return self._simulator._move(self._id, direction, num_steps)

  def position(self):
    return self._simulator._position(self._id)

  def scent(self):
    return self._simulator._scent(self._id)

  def vision(self):
    return self._simulator._vision(self._id)

  def collected_items(self):
    return self._simulator._collected_items(self._id)

  @abc.abstractmethod
  def on_step(self, saved):
    pass

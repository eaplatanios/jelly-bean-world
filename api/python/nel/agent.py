from __future__ import absolute_import, division, print_function

from . import simulator

__all__ = ['Agent']

class Agent(object):
  def __init__(self, simulator):
    self._simulator = simulator
    self._id = simulator._add_agent()
  
  def move(self, direction, num_steps=1):
    self._simulator._move(self._id, direction, num_steps)

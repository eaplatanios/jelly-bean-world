from __future__ import absolute_import, division, print_function

import abc

__all__ = ['AddAgentError', 'Agent']


class AddAgentError(Exception):
  pass

class Agent(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, simulator, load_filepath):
    self._simulator = simulator
    self._position = None
    self._direction = None
    self._scent = None
    self._vision = None
    self._items = None
    if load_filepath == None:
      self._id = simulator._add_agent(self)
    else:
      self._load(load_filepath)

  def position(self):
    return self._position

  def direction(self):
    return self._direction

  def scent(self):
    return self._scent

  def vision(self):
    return self._vision

  def collected_items(self):
    return self._items

  def move(self, direction, num_steps=1):
    return self._simulator.move(self, direction, num_steps)

  def turn(self, direction):
    return self._simulator.turn(self, direction)

  def set_active(self, active):
    self._simulator.set_active(self, active)

  def is_active(self):
    return self._simulator.is_active(self)

  @abc.abstractmethod
  def do_next_action(self):
    pass

  @abc.abstractmethod
  def save(self, filepath):
    pass

  @abc.abstractmethod
  def _load(self, filepath):
    pass

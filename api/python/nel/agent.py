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
    self._scent = None
    self._vision = None
    self._items = None
    if load_filepath == None:
      self._id = simulator._add_agent(self)
    else:
      self._load(load_filepath)

  def position(self):
    return self._position

  def scent(self):
    return self._scent

  def vision(self):
    return self._vision

  def collected_items(self):
    return self._items

  @abc.abstractmethod
  def next_move(self):
    pass

  @abc.abstractmethod
  def save(self, filepath):
    pass

  @abc.abstractmethod
  def _load(self, filepath):
    pass

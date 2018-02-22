from __future__ import absolute_import, division, print_function

from enum import Enum

__all__ = ['Item', 'IntensityFunction', 'InteractionFunction']

class Item(object):
  """Represents an item in the world (e.g., jelly beans)."""

  def __init__(self, name, scent, color, auto_collected):
    """Creates a new item.
  
    Arguments:
      name:           Name, represented as a string.
      scent:          Scent, represented as a list of floats.
      color:          Color, represented as a list of floats.
      auto_collected: Whether this item is automatically collected by agents.
    """
    self.name = name
    self.scent = scent
    self.color = color
    self.auto_collected = auto_collected


class IntensityFunction(Enum):
  """Item intensity function used in the Gibbs sampler for map generation."""

  ZERO = 0
  CONSTANT = 1


class InteractionFunction(Enum):
  """Item interaction function used in the Gibbs sampler for map generation."""

  ZERO = 0
  PIECEWISE_BOX = 1

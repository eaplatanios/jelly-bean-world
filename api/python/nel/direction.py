from __future__ import absolute_import, division, print_function

from enum import Enum

__all__ = ['Direction']

class Direction(Enum):
  """Direction along which the agents can move."""
  
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3

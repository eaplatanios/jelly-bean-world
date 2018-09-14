from __future__ import absolute_import, division, print_function

from enum import Enum

__all__ = ['Direction', 'RelativeDirection']

class Direction(Enum):
  """Direction which agent can face."""

  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3


class RelativeDirection(Enum):
  """RelativeDirection along which the agents can move."""

  FORWARD = 0
  BACKWARD = 1
  LEFT = 2
  RIGHT = 3

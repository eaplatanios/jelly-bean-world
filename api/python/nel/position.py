from __future__ import absolute_import, division, print_function

__all__ = ['Position']

class Position(object):
  """Position in the simulation environment."""
  
  def __init__(self, x, y):
    """Creates a new position.

    Arguments:
      x: Horizontal coordinate (integer).
      y: Vertical coordinate (integer).
    """
    self.x = x
    self.y = y

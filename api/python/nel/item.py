from __future__ import absolute_import, division, print_function

from enum import Enum

__all__ = ['Item', 'IntensityFunction', 'InteractionFunction']

class Item(object):
  """Represents an item in the world (e.g., jelly beans)."""

  def __init__(self, name, scent, color, required_item_counts, blocks_movement):
    """Creates a new item.
  
    Arguments:
      name:                 Name, represented as a string.
      scent:                Scent, represented as a list of floats.
      color:                Color, represented as a list of floats.
      required_item_counts: A list whose value at index `i` indicates the
                            minimum number of items of type `i` that need to be
                            collected by the agent in order to automatically
                            collect items of this type.
      blocks_movement:      Whether this item blocks movement of agents.
    """
    self.name = name
    self.scent = scent
    self.color = color
    self.required_item_counts = required_item_counts
    self.blocks_movement = blocks_movement


class IntensityFunction(Enum):
  """Item intensity function used in the Gibbs sampler for map generation.
  See `nel/energy_functions.h` for implementations of these functions."""

  ZERO = 0
  """A function that always outputs zero: f(x) = 0."""

  CONSTANT = 1
  """A function that outputs a constant, for each item type: f(x) = c(t(x))
  where t(x) is the type of the item x, and c(t) is the constant associated
  with item type t. The arguments for this function should be
    [c(t_1), c(t_2), ..., c(t_n)]
  where t_1 is the first item type, t_2 is the second item type, etc."""


class InteractionFunction(Enum):
  """Item interaction function used in the Gibbs sampler for map generation.
  See `nel/energy_functions.h` for implementations of these functions."""

  ZERO = 0
  """A function that always outputs zero: f(x,y) = 0."""

  PIECEWISE_BOX = 1
  """Two rectangular functions of the squared distance between x and y,
  centered at 0. The pseudocode for f(x,y) looks like:

    distance = ||p(x) - p(y)||^2
    if (distance < l_1(t(x),t(y))):
      return c_1(t(x),t(y))
    elif (distance < l_2(t(x),t(y))):
      return c_2(t(x),t(y))
    else:
      return 0

  where p(x) is the position of item x, t(x) is the type of item x. l_1, l_2,
  c_1, and c_2 are functions that take two item types and return a real number.
  The arguments for this function should be
    [n,
     l_1(t_1,t_1), l_2(t_1,t_1), c_1(t_1,t_1), c_2(t_1,t_1),
     l_1(t_1,t_2), l_2(t_1,t_2), c_1(t_1,t_2), c_2(t_2,t_2),
        etc for (t_1,t_3), ..., (t_1,t_n),
     l_1(t_2,t_1), l_2(t_2,t_1), c_1(t_2,t_1), c_2(t_2,t_1),
     l_1(t_2,t_2), l_2(t_2,t_2), c_1(t_2,t_2), c_2(t_2,t_2),
        etc for (t_2,t_3), ..., (t_2,t_n),
        etc until (t_n,t_n)]
  where t_1 is the first item type, t_2 is the second item type, etc.
  """

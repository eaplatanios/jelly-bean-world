from __future__ import absolute_import, division, print_function

from enum import Enum

__all__ = ['Item', 'IntensityFunction', 'InteractionFunction']

class Item(object):
  """Represents an item in the world (e.g., jelly beans)."""

  def __init__(self, name, scent, color, required_item_counts,
               blocks_movement, intensity_fn, intensity_fn_args, interaction_fn_args):
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
      intensity_fn:         The IntensityFunction used by the Gibbs sampler for
                            generating items of this type in the map.
      intensity_fn_args:    A list of float arguments to intensity_fn.
      interaction_fn_args:  A list of n lists, where n is the number of item
                            types. For each sublist interaction_fn_args[i], the
                            first element contains the InteractionFunction
                            between items of this type and items of type i, and
                            the remaining elements of the sublist contain the
                            parameters to this interaction function.
    """
    self.name = name
    self.scent = scent
    self.color = color
    self.required_item_counts = required_item_counts
    self.blocks_movement = blocks_movement
    self.intensity_fn = intensity_fn.value
    self.intensity_fn_args = intensity_fn_args
    self.interaction_fn_args = interaction_fn_args
    assert all([len(l) > 0 and type(l[0]) == InteractionFunction for i in interaction_fn_args]), 'Each sublist in `interaction_fn_args` must contain an InteractionFunction instance as the first element.'
    for l in interaction_fn_args:
      l[0] = l[0].value


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

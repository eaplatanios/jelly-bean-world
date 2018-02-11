from __future__ import absolute_import, division, print_function

from enum import Enum
from nel_c import simulator_c

__all__ = ['Direction', 'Simulator']


class Direction(Enum):
  """Directions along which agents can move."""
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3


class Simulator(object):
  """Environment simulator.

  All agents live within a simulated environment. 
  Therefore, in order to create agents, users must 
  always first create a simulator and provide it to 
  the agent's constructor.
  """
  
  def __init__(self):
    """Creates a new simulator."""
    self._handle = simulator_c.new()

  def __del__(self):
    """Deletes this simulator and deallocates all 
    associated memory. This simulator cannot be used 
    again after it's been deleted."""
    simulator_c.delete(self._handle)
  
  def _add_agent(self):
    """Adds a new agent to this simulator.
    
    Returns:
      The new agent's ID.
    """
    return simulator_c.add_agent(self._handle)

  def _move(self, agent_id, direction, num_steps):
    """"""
    simulator_c.move(self._handle, agent_id, direction, num_steps)

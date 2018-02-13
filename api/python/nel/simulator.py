from __future__ import absolute_import, division, print_function

from enum import Enum
from nel_c import simulator_c

from .item import IntensityFunction, InteractionFunction

__all__ = ['SimulatorStepCallback', 'SimulatorConfig', 'Simulator']


class SimulatorStepCallback(Enum):
  """Step callback function used by the simulator."""

  C_CALLBACK = 0   # This callback uses the C-Python API bdindings to 
                   # interface with the simulator and should be faster 
                   # than the MPI callback. However, this does not allow 
                   # the simulator to run as a separate server process, 
                   # that multiple agent processed can attach to. In 
                   # this case, all agents must be defined and used as 
                   # part of the same Python process that created the 
                   # simulator.
  MPI_CALLBACK = 1 # This callback uses message passing over a socket 
                   # (using TCP) and allows the simulator to run as a 
                   # separate server process that multiple agent processes 
                   # can attach to. However, it may result in slower 
                   # performance than the C callback.


class SimulatorConfig(object):
  """Represents a configuration for a simulator."""

  def __init__(
      self, max_steps_per_movement, vision_range, 
      patch_size, gibbs_num_iter, items, 
      intensity_fn=IntensityFunction.CONSTANT, 
      intensity_fn_args=[-2.0], 
      interaction_fn=InteractionFunction.PIECEWISE_BOX, 
      interaction_fn_args=[40.0, 0.0, 200.0, -40.0]):
    """Creates a new simulator configuration.

    Arguments:
      max_steps_per_movement: Maximum steps allowed for each agent move action.
      vision_range:           Vision range of each agent.
      patch_size:             Size of each patch used by the map generator.
      gibbs_num_iter:         Number of Gibbs sampling iterations performed for 
                              sampling each patch of the map.
      items:                  List of items to include in this world.
      intensity_fn:           Item intensity function used in the Gibbs sampler 
                              for map generation.
      intensity_fn_args:      Arguments to the item intensity function.
      interaction_fn:         Item interaction function used in the Gibbs sampler 
                              for map generation.
      interaction_fn_args:    Arguments to the item interaction function.
    """
    assert(len(items) > 0, "A non-empty list of items must be provided.")
    self.max_steps_per_movement = max_steps_per_movement
    self.scent_num_dims = len(items[0].scent)
    self.color_num_dims = len(items[0].color)
    self.vision_range = vision_range
    self.patch_size = patch_size
    self.gibbs_num_iter = gibbs_num_iter
    self.items = items
    assert(all([len(i.scent) == self.scent_num_dims for i in items]), 
      "All items must use the same dimensionality for the scent vector.")
    assert(all([len(i.color) == self.color_num_dims for i in items]), 
      "All items must use the same dimensionality for the color vector.")
    self.intensity_fn = intensity_fn
    self.intensity_fn_args = intensity_fn_args
    self.interaction_fn = interaction_fn
    self.interaction_fn_args = interaction_fn_args


class Simulator(object):
  """Environment simulator.

  All agents live within a simulated environment. Therefore, in order to create 
  agents, users must always first create a simulator and provide it to the 
  agent's constructor.
  """
  
  def __init__(self, config, step_callback=SimulatorStepCallback.C_CALLBACK):
    """Creates a new simulator.
    
    Arguments:
      config:        Configuration for the new simulator.
      step_callback: Step callback function to use. If intending to use this 
                     simulator as a server, the `MPI_CALLBACK` function should 
                     be used. The default is to use the `C_CALLBACK` function.
    """
    self.config = config
    self.step_callback = step_callback
    self._handle = simulator_c.new(
      config.max_steps_per_movement, config.scent_num_dims, 
      config.color_num_dims, config.vision_range, config.patch_size, 
      config.gibbs_num_iter, 
      [(i.name, i.scent, i.color, i.intensity) for i in config.items], 
      config.intensity_fn, config.intensity_fn_args, 
      config.interaction_fn, config.interaction_fn_args, 
      step_callback)
    self._server_handle = None
  
  def start_server(
      self, address='localhost', port=54353, conn_queue_capacity=256, 
      num_workers=8):
    """Starts the simulator server.
    
    Note that the simulator can only be started as a server if the MPI 
    step callback is used. A call to this function will be ignored if the 
    C step callback is being used.

    Arguments:
      address:             Server address.
      port:                Server port.
      conn_queue_capacity: Connection queue capacity for the server.
      num_workers:         Number of worker threads for the server.
    """
    if self.step_callback is not SimulatorStepCallback.MPI_CALLBACK:
      raise ValueError(
        "Using the simulator as a server is only supported when the " 
        "MPI step callback function is used.")
    if self._server_handle is not None:
      print(
        "Ignoring request to start the simulation server, because one "
        "is already running.")
    self._server_handle = simulator_c.start_server(
      self._handle, address, str(port), conn_queue_capacity, num_workers)

  def stop_server(self):
    """Stops the simulator server.
    
    A call to this function is ignored if no simulator server is currently 
    running.
    """
    if self._server_handle is not None:
      simulator_c.stop_server(self._server_handle)
      self._server_handle = None
    else:
      print(
        "Ignoring request to stop the simulation server, because it "
        "is not currently running.")

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

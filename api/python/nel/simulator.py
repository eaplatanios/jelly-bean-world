from __future__ import absolute_import, division, print_function

from enum import Enum
from nel import simulator_c

from .agent import AgentState
from .item import IntensityFunction, InteractionFunction
from .position import Position

__all__ = ['MovementConflictPolicy', 'SimulatorConfig', 'Simulator']


class MovementConflictPolicy(Enum):
  """Policy used to resolve the conflict when two or more agents request to
     move into the same grid cell."""

  NO_COLLISIONS = 0
  FIRST_COME_FIRST_SERVED = 1
  RANDOM = 2


class SimulatorConfig(object):
  """Represents a configuration for a simulator."""

  def __init__(self, max_steps_per_movement,
      vision_range, patch_size, gibbs_num_iter, items, agent_color,
      collision_policy, decay_param, diffusion_param, deleted_item_lifetime,
      intensity_fn, intensity_fn_args, interaction_fn, interaction_fn_args):
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
    assert len(items) > 0, 'A non-empty list of items must be provided.'
    self.max_steps_per_movement = max_steps_per_movement
    self.scent_num_dims = len(items[0].scent)
    self.color_num_dims = len(items[0].color)
    self.vision_range = vision_range
    self.patch_size = patch_size
    self.gibbs_num_iter = gibbs_num_iter
    self.items = items
    self.agent_color = agent_color
    assert len(agent_color) == self.color_num_dims, 'Agent color must have the same dimension as item colors'
    assert all([len(i.scent) == self.scent_num_dims for i in items]), 'All items must use the same dimensionality for the scent vector.'
    assert all([len(i.color) == self.color_num_dims for i in items]), 'All items must use the same dimensionality for the color vector.'
    self.collision_policy = collision_policy
    self.decay_param = decay_param
    self.diffusion_param = diffusion_param
    self.deleted_item_lifetime = deleted_item_lifetime
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

  def __init__(
      self, sim_config=None, is_server=False, server_address=None,
      port=54353, conn_queue_capacity=256, num_workers=8,
      save_frequency=1000, save_filepath=None, load_filepath=None):
    """Creates a new simulator.

    Arguments:
      sim_config      Configuration for the new simulator.
    """
    self._handle = None
    self._server_handle = None
    self._client_handle = None
    self._time = 0
    if save_frequency <= 0:
      raise ValueError('"save_frequency" must be strictly greater than zero.')
    if sim_config != None:
      # create a local server or simulator
      if load_filepath != None:
        raise ValueError('"load_filepath" must be None if "sim_config" is specified.')
      elif server_address != None:
        raise ValueError('"server_address" must be None if "sim_config" is specified.')
      self._handle = simulator_c.new(
        sim_config.max_steps_per_movement, sim_config.scent_num_dims,
        sim_config.color_num_dims, sim_config.vision_range, sim_config.patch_size,
        sim_config.gibbs_num_iter,
        [(i.name, i.scent, i.color, i.auto_collected) for i in sim_config.items],
        sim_config.agent_color, sim_config.collision_policy.value,
        sim_config.decay_param, sim_config.diffusion_param,
        sim_config.deleted_item_lifetime,
        sim_config.intensity_fn.value, sim_config.intensity_fn_args,
        sim_config.interaction_fn.value, sim_config.interaction_fn_args,
        self._step_callback, save_frequency, save_filepath)
      if is_server:
        self._server_handle = simulator_c.start_server(
          self._handle, port, conn_queue_capacity, num_workers)
    elif server_address != None:
      # connect to a remote server
      if load_filepath != None:
        raise ValueError('"load_filepath" must be None if "server_address" is specified.')
      self._client_handle = simulator_c.start_client(server_address, port, self._step_callback)
    else:
      # load local server or simulator from file
      if load_filepath == None:
        raise ValueError('"load_filepath" must be non-None if "sim_config" and "server_address" are None.')
      self._handle = simulator_c.load(load_filepath, self._step_callback, save_frequency, save_filepath)
      if is_server:
        self._server_handle = simulator_c.start_server(
          self._handle, port, conn_queue_capacity, num_workers)
    self.agents = dict()

  def __del__(self):
    """Deletes this simulator and deallocates all 
    associated memory. This simulator cannot be used 
    again after it's been deleted."""
    if self._client_handle != None:
      simulator_c.stop_client(self._client_handle)
    if self._server_handle != None:
      simulator_c.stop_server(self._server_handle)
    if self._handle != None:
      simulator_c.delete(self._handle)

  def _add_agent(self, py_agent):
    """Adds a new agent to this simulator.

    Arguments:
      py_agent: Python agent to be added to this simulator.

    Returns:
      The new agent's ID.
    """
    id = simulator_c.add_agent(self._handle, self._client_handle)
    self.agents[id] = py_agent
    return id

  def _move(self, agent_id, direction, num_steps):
    """Moves the specified agent in the simulated environment.

    Note that the agent is not moved until the simulator advances by a 
    time step and issues a notification about that event. The simulator 
    only advances the time step once all agents have requested to move.

    Arguments:
      agent_id:  Agent ID.
      direction: Direction along which to move.
      num_steps: Number of steps to take in the specified direction.
    """
    return simulator_c.move(self._handle,
      self._client_handle, agent_id, direction.value, num_steps)

  def _position(self, agent_id):
    return simulator_c.position(self._handle, self._client_handle, agent_id)

  def _scent(self, agent_id):
    return simulator_c.scent(self._handle, self._client_handle, agent_id)

  def _vision(self, agent_id):
    return simulator_c.vision(self._handle, self._client_handle, agent_id)

  def _collected_items(self, agent_id):
    return simulator_c.collected_items(self._handle, self._client_handle, agent_id)

  def _step_callback(self):
    self._time += 1
    for (id, agent) in self.agents.items():
      agent.on_step()

  def time(self):
    return self._time

  def _map(self, bottom_left, top_right):
    return simulator_c.map(self._handle, self._client_handle, bottom_left, top_right)

if __name__ == '__main__':
  # TODO: Parse command line arguments and construct a simulator config.
  # TODO: Start server.
  # TODO: Keep this process alive while the server is running and stop the 
  #       server when it's killed.
  print('TODO')

from __future__ import absolute_import, division, print_function

from enum import Enum
from pydoc import locate
from nel import simulator_c

from .item import IntensityFunction, InteractionFunction

__all__ = ['MPIError', 'MovementConflictPolicy', 'SimulatorConfig', 'Simulator']


class MPIError(Exception):
  pass

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
      self, on_step_callback=None, sim_config=None,
      is_server=False, server_address=None, port=54353,
      conn_queue_capacity=256, num_workers=8,
	  on_lost_connection_callback=None, save_frequency=1000,
      save_filepath=None, load_filepath=None, load_time=-1):
    """Creates a new simulator.

    Arguments:
      sim_config      Configuration for the new simulator.
    """
    self._handle = None
    self._server_handle = None
    self._client_handle = None
    self._save_filepath = save_filepath
    self.agents = dict()
    if on_step_callback == None:
      self._on_step = lambda *args: None
    else:
      self._on_step = on_step_callback
    if on_lost_connection_callback == None:
      on_lost_connection_callback = lambda *args: None

    if save_frequency <= 0:
      raise ValueError('"save_frequency" must be strictly greater than zero.')
    if load_filepath != None and load_time < 0:
      raise ValueError('If "load_filepath" is specified, "load_time" must also be specified as a non-negative integer.')

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
      self._time = 0
    elif server_address != None:
      if load_filepath != None:
        # load agents from file
        self._load_agents(load_filepath)
      # connect to a remote server
      agent_ids = list(self.agents.keys())
      agent_values = list(self.agents.values())
      (self._time, self._client_handle, agent_states) = simulator_c.start_client(
		  server_address, port, self._step_callback, on_lost_connection_callback, agent_ids)
      for i in range(len(agent_ids)):
        agent = agent_values[i]
        (agent._position, agent._scent, agent._vision, agent._items) = agent_states[i]
    else:
      # load local server or simulator from file
      if load_filepath == None:
        raise ValueError('"load_filepath" must be non-None if "sim_config" and "server_address" are None.')
      (self._time, self._handle) = simulator_c.load(load_filepath + str(load_time), self._step_callback, save_frequency, save_filepath)
      # remove the time from the end of the filepath to get the original save_filepath
      self._load_agents(load_filepath)
      if is_server:
        self._server_handle = simulator_c.start_server(
          self._handle, port, conn_queue_capacity, num_workers)

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
    (py_agent._position, py_agent._scent, py_agent._vision, py_agent._items, id) = simulator_c.add_agent(self._handle, self._client_handle)
    self.agents[id] = py_agent
    return id

  def move(self, agent, direction, num_steps):
    """Moves the specified agent in the simulated environment.

    Note that the agent is not moved until the simulator advances by a 
    time step and issues a notification about that event. The simulator 
    only advances the time step once all agents have requested to move.

    Arguments:
      agent:     The agent intending to move.
      direction: Direction along which to move.
      num_steps: Number of steps to take in the specified direction.
    """
    return simulator_c.move(self._handle,
      self._client_handle, agent._id, direction.value, num_steps)

  def _step_callback(self, agent_states, saved):
    self._time += 1
    for agent_state in agent_states:
      (position, scent, vision, items, id) = agent_state
      agent = self.agents[id]
      (agent._position, agent._scent, agent._vision, agent._items) = (position, scent, vision, items)
    if saved:
      self._save_agents()
    self._on_step()

  def time(self):
    return self._time

  def _map(self, bottom_left, top_right):
    return simulator_c.map(self._handle, self._client_handle, bottom_left, top_right)

  def _load_agents(self, load_filepath):
    with open(load_filepath + str(self._time) + '.agent_info', 'rb') as fin:
      for line in fin:
        tokens = line.split(sep=' ')
        agent_id = int(tokens[0])
        agent_class = locate(tokens[1])
        agent = agent_class(self, load_filepath=load_filepath + str(self._time) + '.agent' + str(agent_id))
        agents[agent_id] = agent
        agent._id = agent_id

  def _save_agents(self):
    with open(self._save_filepath + str(self._time) + '.agent_info', 'wb') as fout:
      for agent_id, agent in agents.items():
        agent.save(self._save_filepath + str(self._time) + '.agent' + str(agent_id))
        fout.write(str(agent_id) + ' ' + agent.__module__ + '.' + agent.__name__ + '\n')

if __name__ == '__main__':
  # TODO: Parse command line arguments and construct a simulator config.
  # TODO: Start server.
  # TODO: Keep this process alive while the server is running and stop the 
  #       server when it's killed.
  print('TODO')

# Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

from enum import Enum
from pydoc import locate
from jbw import simulator_c
from .direction import Direction
import os

from .item import IntensityFunction, InteractionFunction

__all__ = ['MPIError', 'MovementConflictPolicy', 'ActionPolicy', 'SimulatorConfig', 'Simulator']


class MPIError(Exception):
  pass

class MovementConflictPolicy(Enum):
  """Policy used to resolve the conflict when two or more agents request to
     move into the same grid cell."""

  NO_COLLISIONS = 0
  FIRST_COME_FIRST_SERVED = 1
  RANDOM = 2

class ActionPolicy(Enum):
  """Policy used to indicate whether each action is allowed, disallowed, or
     ignored. If the action is disallowed, then attempting to perform it will
     immediately fail, preventing the simulator from progressing if the agent
     hasn't performed an action during the current time step. Whereas if the
     action is ignored, then the agent will perform a no-op for that time step."""

  ALLOWED = 0
  DISALLOWED = 0
  IGNORED = 0

class SimulatorConfig(object):
  """Represents a configuration for a simulator."""

  def __init__(self, max_steps_per_movement, allowed_movement_directions,
      allowed_turn_directions, no_op_allowed, vision_range, patch_size,
      mcmc_num_iter, items, agent_color, collision_policy, decay_param,
      diffusion_param, deleted_item_lifetime, seed=0):
    """Creates a new simulator configuration.

    Arguments:
      max_steps_per_movement:      Maximum steps allowed for each agent move
                                   action.
      allowed_movement_directions: A list of ActionPolicies, each element
                                   corresponding to each possible Direction,
                                   that specifies whether the simulator allows,
                                   disallows, or ignores the respective
                                   movement action.
      allowed_turn_directions:     A list of ActionPolicies, each element
                                   corresponding to each possible
                                   RelativeDirection, that specifies whether
                                   the simulator allows, disallows, or ignores
                                   the respective turn.
      no_op_allowed:               Whether or not actions can perform no action.
      vision_range:                Vision range of each agent.
      patch_size:                  Size of each patch used by the map
                                   generator.
      mcmc_num_iter:               Number of Gibbs sampling iterations
                                   performed for sampling each patch of the
                                   map.
      items:                       List of items to include in this world.
      seed:                        The initial seed for the pseudorandom number
                                   generator.
    """
    assert len(items) > 0, 'A non-empty list of items must be provided.'
    self.max_steps_per_movement = max_steps_per_movement
    self.allowed_movement_directions = allowed_movement_directions
    self.allowed_turn_directions = allowed_turn_directions
    self.no_op_allowed = no_op_allowed
    self.scent_num_dims = len(items[0].scent)
    self.color_num_dims = len(items[0].color)
    self.vision_range = vision_range
    self.patch_size = patch_size
    self.mcmc_num_iter = mcmc_num_iter
    self.items = items
    self.agent_color = agent_color
    assert len(agent_color) == self.color_num_dims, 'Agent color must have the same dimension as item colors'
    assert all([len(i.scent) == self.scent_num_dims for i in items]), 'All items must use the same dimensionality for the scent vector.'
    assert all([len(i.color) == self.color_num_dims for i in items]), 'All items must use the same dimensionality for the color vector.'
    assert all([len(i.required_item_counts) == len(items) for i in items]), 'The `required_item_counts` field must be the same dimension as `items`'
    assert all([len(i.required_item_costs) == len(items) for i in items]), 'The `required_item_costs` field must be the same dimension as `items`'
    assert all([len(i.interaction_fns) == len(items) for i in items]), 'The `interaction_fn_args` field must be the same dimension as `items`'
    self.collision_policy = collision_policy
    self.decay_param = decay_param
    self.diffusion_param = diffusion_param
    self.deleted_item_lifetime = deleted_item_lifetime
    self.seed = seed


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
    """This constructor may be used to either: (1) create a new simulator
    locally (local mode), (2) create a new simulator server in the current
    process (server mode), or (3) connect to a remote simulator server (client
    mode). This constructor may also be used to load a simulator from a file,
    and run it in either local mode or server mode.

    To construct a new simulator in local mode, a configuration must be
    specified with `sim_config`, `is_server` must be `False`, `server_address`
    must be unspecified, and `load_filepath` must be unspecified (the latter
    three arguments are default).

    To construct a new simulator in server mode, `sim_config` must be
    specified, `is_server` must be `True`, `server_address` must be
    unspecified, and `load_filepath` must be unspecified (the latter two
    arguments are default). The arguments `port`, `conn_queue_capacity`, and
    `num_workers` may be used to change aspects of the server.

    To connect to an existing simulator server (i.e. construct the simulator in
    client mode), `sim_config` must be unspecified, but `server_address` must
    be specified. If provided, `on_lost_connection_callback` is called if the
    client loses its connection to the server.

    To load a simulator from a file, `sim_config` must be unspecified,
    `load_filepath` is set to the name of file to load **excluding** the
    simulation time. The simulation time to load must be specified with
    `load_time`. In client mode, the agents are loaded from the given filepath.

    Arguments:
      on_step_callback    (all modes) The callback invoked when the simulator
                          completes a step.
      sim_config          (local and server modes) Configuration for the new
                          simulator.
      is_server           Indicates whether this simulation is to be run as a
                          server.
      server_address      (client mode) The address of the simulator server to
                          connect to.
      port                (server mode) The port of the simulator server.
      conn_queue_capacity (server mode) The maximum number of simultaneous
                          connection attempts that the server will attempt to
                          process.
      num_workers         (server mode) The number of worker threads that will
                          be used to process incoming client messages.
      on_lost_connection_callback (client mode) The function that is called
                          when the client loses its connection with the server.
      save_frequency      (local and server modes) Indicates how often the
                          simulator and the agents should be saved to
                          `save_filepath`. Note that if `save_filepath` is
                          unspecified, this parameter has no effect.
      save_filepath       (all modes) The path where the simulator and agents
                          are saved (in client mode, only agents are saved).
                          The directory containing this path is created if it
                          doesn't already exist.
      load_filepath       (all modes) The path from which the simulator and
                          agents are loaded (in client mode, only agents are
                          loaded). This **should not** contain the simulation
                          time. Instead, the time should be specified via
                          `load_time`.
      load_time           (all modes) The simulation time to load. This is used
                          in conjunction with `load_filepath` to determine the
                          precise filenames to load.
    """
    self._handle = None
    self._server_handle = None
    self._client_handle = None
    self._save_filepath = save_filepath
    self._save_frequency = save_frequency
    self.agents = dict()
    if on_step_callback == None:
      self._on_step = lambda *args: None
    else:
      self._on_step = on_step_callback
    if on_lost_connection_callback == None:
      on_lost_connection_callback = lambda *args: None

    if save_filepath != None:
      # make the save directory if it doesn't exist
      directory = os.path.dirname(save_filepath)
      try:
        os.makedirs(directory)
      except OSError:
        if not os.path.exists(directory):
          raise
      # check that the save directory is writable
      if not os.access(directory, os.W_OK):
        raise IOError('The path "' + save_filepath + '" is not writable.')

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
      self._handle = simulator_c.new(sim_config.seed,
        sim_config.max_steps_per_movement, [d.value for d in sim_config.allowed_movement_directions],
        [d.value for d in sim_config.allowed_turn_directions], sim_config.no_op_allowed, sim_config.scent_num_dims,
        sim_config.color_num_dims, sim_config.vision_range, sim_config.patch_size, sim_config.mcmc_num_iter,
        [(i.name, i.scent, i.color, i.required_item_counts, i.required_item_costs, i.blocks_movement, i.intensity_fn, i.intensity_fn_args, i.interaction_fns) for i in sim_config.items],
        sim_config.agent_color, sim_config.collision_policy.value, sim_config.decay_param,
        sim_config.diffusion_param, sim_config.deleted_item_lifetime, self._step_callback)
      if is_server:
        self._server_handle = simulator_c.start_server(
          self._handle, port, conn_queue_capacity, num_workers)
      self._time = 0
    elif server_address != None:
      if load_filepath != None:
        # load agents from file
        self._load_agents(load_filepath, load_time)
      # connect to a remote server
      agent_ids = list(self.agents.keys())
      agent_values = list(self.agents.values())
      (self._time, self._client_handle, agent_states) = simulator_c.start_client(
          server_address, port, self._step_callback, on_lost_connection_callback, agent_ids)
      for i in range(len(agent_ids)):
        (position, direction, scent, vision, items) = agent_states[i]
        agent = agent_values[i]
        (agent._position, agent._direction, agent._scent, agent._vision, agent._items) = (position, Direction(direction), scent, vision, items)
    else:
      # load local server or simulator from file
      if load_filepath == None:
        raise ValueError('"load_filepath" must be non-None if "sim_config" and "server_address" are None.')
      self._load_agents(load_filepath, load_time)
      (self._time, self._handle, agent_states) = simulator_c.load(load_filepath + str(load_time), self._step_callback)
      for agent_state in agent_states:
        (position, direction, scent, vision, items, id) = agent_state
        agent = self.agents[id]
        (agent._position, agent._direction, agent._scent, agent._vision, agent._items) = (position, Direction(direction), scent, vision, items)
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

  def _add_agent(self, agent):
    """Adds a new agent to this simulator and retrieves its state.

    Arguments:
      agent: Python agent to be added to this simulator.

    Returns:
      The new agent's ID.
    """
    (position, direction, scent, vision, items, id) = simulator_c.add_agent(self._handle, self._client_handle)
    self.agents[id] = agent
    (agent._position, agent._direction, agent._scent, agent._vision, agent._items) = (position, Direction(direction), scent, vision, items)
    return id

  def move(self, agent, direction, num_steps=1):
    """Moves the specified agent in the simulated environment.

    Note that the agent is not moved until the simulator advances by a 
    time step and issues a notification about that event. The simulator 
    only advances the time step once all agents have requested to move.

    Arguments:
      agent:     The agent intending to move.
      direction: RelativeDirection to move.
      num_steps: Number of steps to take in the specified direction.

    Returns:
      `True`, if successful; `False`, otherwise.
    """
    return simulator_c.move(self._handle,
      self._client_handle, agent._id, direction.value, num_steps)

  def turn(self, agent, direction):
    """Turns the specified agent in the simulated environment.

    Note that the agent is not turned until the simulator advances by a 
    time step and issues a notification about that event. The simulator 
    only advances the time step once all agents have requested to perform an
    action.

    Arguments:
      agent:     The agent intending to turn.
      direction: Direction to turn.

    Returns:
      `True`, if successful; `False`, otherwise.
    """
    return simulator_c.turn(self._handle,
      self._client_handle, agent._id, direction.value)

  def no_op(self, agent):
    """Instructs the specified agent in the simulated environment to do nothing.

    The simulator only advances the time step once all agents have requested to
    perform an action (or a no-op).

    Arguments:
      agent:     The agent intending to do nothing.

    Returns:
      `True`, if successful; `False`, otherwise.
    """
    return simulator_c.no_op(self._handle, self._client_handle, agent._id)

  def get_agents(self):
    """Retrieves a list of the agents governed by this Simulator. This does not
    include the agents governed by other clients."""
    return list(self.agents.values())

  def _step_callback(self, agent_states):
    """The callback invoked when the simulator has advanced time.

    Arguments:
      agent_states: A list of tuples containing the states of each agent
                    governed by this Simulator. This does not include agents
                    governed by other clients.
    """
    self._time += 1
    for agent_state in agent_states:
      (position, direction, scent, vision, items, id) = agent_state
      agent = self.agents[id]
      (agent._position, agent._direction, agent._scent, agent._vision, agent._items) = (position, Direction(direction), scent, vision, items)
    if self._save_filepath != None and self._time % self._save_frequency == 0:
      simulator_c.save(self._handle, self._save_filepath + str(self._time))
      self._save_agents()
    self._on_step()

  def time(self):
    """Returns the current simulation time."""
    return self._time

  def _map(self, bottom_left, top_right):
    """Returns a list of tuples, each containing the state information of a
    patch in the map. Only the patches visible in the bounding box defined by
    `bottom_left` and `top_right` are returned.

    Arguments:
      bottom_left: A tuple of integers representing the bottom-left corner of
                   the bounding box containing the patches to retrieve.
      top_right:   A tuple of integers representing the top_right corner of the
                   bounding box containing the patches to retrieve.

    Returns:
      A list of tuples, where each tuple contains the state of a patch.
    """
    return simulator_c.map(self._handle, self._client_handle, bottom_left, top_right)

  def set_active(self, agent, active):
    """Sets whether the given agent is active or inactive.

    Arguments:
      agent:    The agent whose active status to set.
      active:   Whether the agent should be set to active or inactive.
    """
    simulator_c.set_active(self._handle, self._client_handle, agent._id, active)

  def is_active(self, agent):
    """Gets whether the given agent is active or inactive.

    Arguments:
      agent:    The agent whose active status to set.

    Returns:
      Whether the agent is active or inactive.
    """
    return simulator_c.is_active(self._handle, self._client_handle, agent._id)

  def _load_agents(self, load_filepath, load_time):
    with open(load_filepath + str(load_time) + '.agent_info', 'rb') as fin:
      for line_bytes in fin:
        line = line_bytes.decode('utf-8')
        tokens = line.split()
        agent_id = int(tokens[0])
        agent_class = locate(tokens[1])
        agent = agent_class(self, load_filepath=load_filepath + str(load_time) + '.agent' + str(agent_id))
        self.agents[agent_id] = agent
        agent._id = agent_id

  def _save_agents(self):
    with open(self._save_filepath + str(self._time) + '.agent_info', 'wb') as fout:
      for agent_id, agent in self.agents.items():
        agent.save(self._save_filepath + str(self._time) + '.agent' + str(agent_id))
        agent_type = type(agent)
        line = str(agent_id) + ' ' + agent_type.__module__ + '.' + agent_type.__name__ + '\n'
        fout.write(line.encode('utf-8'))

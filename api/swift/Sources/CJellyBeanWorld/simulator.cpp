/**
 * Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "include/simulator.h"

#include "gibbs_field.h"
#include "mpi.h"
#include "simulator.h"
#include "status.h"

using namespace core;
using namespace jbw;

void JBW_SetJBWStatusFromStatus(JBW_Status* jbw_s, status s) {
  switch (s) {
    case status::OK: jbw_s->code = JBW_OK;
    case status::OUT_OF_MEMORY: jbw_s->code = JBW_OUT_OF_MEMORY;
    case status::INVALID_AGENT_ID: jbw_s->code = JBW_INVALID_AGENT_ID;
    case status::PERMISSION_ERROR: jbw_s->code = JBW_VIOLATED_PERMISSIONS;
    case status::AGENT_ALREADY_ACTED: jbw_s->code = JBW_AGENT_ALREADY_ACTED;
    case status::AGENT_ALREADY_EXISTS: jbw_s->code = JBW_AGENT_ALREADY_EXISTS;
    case status::SERVER_PARSE_MESSAGE_ERROR: jbw_s->code = JBW_SERVER_PARSE_MESSAGE_ERROR;
    case status::CLIENT_PARSE_MESSAGE_ERROR: jbw_s->code = JBW_CLIENT_PARSE_MESSAGE_ERROR;
    case status::SERVER_OUT_OF_MEMORY: jbw_s->code = JBW_SERVER_OUT_OF_MEMORY;
    case status::CLIENT_OUT_OF_MEMORY: jbw_s->code = JBW_CLIENT_OUT_OF_MEMORY;
  }
}

constexpr AgentSimulationState EMPTY_AGENT_SIM_STATE = { 0 };
constexpr SimulatorInfo EMPTY_SIM_INFO = { 0 };
constexpr SimulationNewClientInfo EMPTY_NEW_CLIENT_INFO = { 0 };
constexpr SimulationClientInfo EMPTY_CLIENT_INFO = { 0 };
constexpr SimulationMap EMPTY_SIM_MAP = { 0 };
constexpr AgentIDList EMPTY_AGENT_ID_LIST = { 0 };

inline Direction to_Direction(direction dir) {
  switch (dir) {
  case direction::UP:    return DirectionUp;
  case direction::DOWN:  return DirectionDown;
  case direction::LEFT:  return DirectionLeft;
  case direction::RIGHT: return DirectionRight;
  case direction::COUNT: break;
  }
  fprintf(stderr, "to_Direction ERROR: Unrecognized direction.\n");
  exit(EXIT_FAILURE);
}


inline direction to_direction(Direction dir) {
  switch (dir) {
  case DirectionUp:    return direction::UP;
  case DirectionDown:  return direction::DOWN;
  case DirectionLeft:  return direction::LEFT;
  case DirectionRight: return direction::RIGHT;
  case DirectionCount: break;
  }
  fprintf(stderr, "to_direction ERROR: Unrecognized Direction.\n");
  exit(EXIT_FAILURE);
}


inline direction to_direction(TurnDirection dir) {
  switch (dir) {
  case TurnDirectionNoChange:    return direction::UP;
  case TurnDirectionReverse:  return direction::DOWN;
  case TurnDirectionLeft:  return direction::LEFT;
  case TurnDirectionRight: return direction::RIGHT;
  }
  fprintf(stderr, "to_direction ERROR: Unrecognized TurnDirection.\n");
  exit(EXIT_FAILURE);
}


inline MovementConflictPolicy to_MovementConflictPolicy(movement_conflict_policy policy) {
  switch (policy) {
  case movement_conflict_policy::NO_COLLISIONS:
    return MovementConflictPolicyNoCollisions;
  case movement_conflict_policy::FIRST_COME_FIRST_SERVED:
    return MovementConflictPolicyFirstComeFirstServe;
  case movement_conflict_policy::RANDOM:
    return MovementConflictPolicyRandom;
  }
  fprintf(stderr, "to_MovementConflictPolicy ERROR: Unrecognized movement_conflict_policy.\n");
  exit(EXIT_FAILURE);
}


inline movement_conflict_policy to_movement_conflict_policy(MovementConflictPolicy policy) {
  switch (policy) {
  case MovementConflictPolicyNoCollisions:
    return movement_conflict_policy::NO_COLLISIONS;
  case MovementConflictPolicyFirstComeFirstServe:
    return movement_conflict_policy::FIRST_COME_FIRST_SERVED;
  case MovementConflictPolicyRandom:
    return movement_conflict_policy::RANDOM;
  }
  fprintf(stderr, "to_movement_conflict_policy ERROR: Unrecognized MovementConflictPolicy.\n");
  exit(EXIT_FAILURE);
}


inline ActionPolicy to_ActionPolicy(action_policy policy) {
  switch (policy) {
  case action_policy::ALLOWED:
    return ActionPolicyAllowed;
  case action_policy::DISALLOWED:
    return ActionPolicyDisallowed;
  case action_policy::IGNORED:
    return ActionPolicyIgnored;
  }
  fprintf(stderr, "to_ActionPolicy ERROR: Unrecognized action_policy.\n");
  exit(EXIT_FAILURE);
}


inline action_policy to_action_policy(ActionPolicy policy) {
  switch (policy) {
  case ActionPolicyAllowed:
    return action_policy::ALLOWED;
  case ActionPolicyDisallowed:
    return action_policy::DISALLOWED;
  case ActionPolicyIgnored:
    return action_policy::IGNORED;
  }
  fprintf(stderr, "to_action_policy ERROR: Unrecognized ActionPolicy.\n");
  exit(EXIT_FAILURE);
}


inline Permissions to_Permissions(const permissions& src) {
  Permissions perms;
  perms.addAgent = src.add_agent;
  perms.removeAgent = src.remove_agent;
  perms.removeClient = src.remove_client;
  perms.setActive = src.set_active;
  perms.getMap = src.get_map;
  perms.getAgentIds = src.get_agent_ids;
  perms.getAgentStates = src.get_agent_states;
  return perms;
}


inline permissions to_permissions(const Permissions& src) {
  permissions perms;
  perms.add_agent = src.addAgent;
  perms.remove_agent = src.removeAgent;
  perms.remove_client = src.removeClient;
  perms.set_active = src.setActive;
  perms.get_map = src.getMap;
  perms.get_agent_ids = src.getAgentIds;
  perms.get_agent_states = src.getAgentStates;
  return perms;
}


inline void init(
  energy_function<intensity_function>& function,
  const IntensityFunction& src,
  JBW_Status* status
) {
  function.fn = get_intensity_fn((intensity_fns) src.id, src.args, src.numArgs);
  function.args = (float*) malloc(sizeof(float) * src.numArgs);
  if (function.args == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  memcpy(function.args, src.args, sizeof(float) * src.numArgs);
  function.arg_count = src.numArgs;
}


inline void init(
  IntensityFunction& function,
  const energy_function<intensity_function>& src,
  JBW_Status* status
) {
  function.id = (unsigned int) get_intensity_fn(src.fn);
  function.args = (float*) malloc(sizeof(float) * src.arg_count);
  if (function.args == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  memcpy(function.args, src.args, sizeof(float) * src.arg_count);
  function.numArgs = src.arg_count;
}

inline void free(IntensityFunction& function) {
  free(function.args);
}


inline void init(
  energy_function<interaction_function>& function,
  const InteractionFunction& src,
  JBW_Status* status
) {
  function.fn = get_interaction_fn((interaction_fns) src.id, src.args, src.numArgs);
  function.args = (float*) malloc(sizeof(float) * src.numArgs);
  if (function.args == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  memcpy(function.args, src.args, sizeof(float) * src.numArgs);
  function.arg_count = src.numArgs;
}


inline void init(
  InteractionFunction& function,
  const energy_function<interaction_function>& src,
  unsigned int item_id,
  JBW_Status* status
) {
  function.id = (unsigned int) get_interaction_fn(src.fn);
  function.args = (float*) malloc(sizeof(float) * src.arg_count);
  if (function.args == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  memcpy(function.args, src.args, sizeof(float) * src.arg_count);
  function.numArgs = src.arg_count;
  function.itemId = item_id;
}

inline void free(InteractionFunction& function) {
  free(function.args);
}


inline void init(
  item_properties& properties,
  const ItemProperties& src,
  unsigned int scent_dimension,
  unsigned int color_dimension,
  unsigned int item_type_count,
  JBW_Status* status
) {
  /* check that `itemId` for `src.energyFunctions.interactionFn` are unique */
  array<unsigned int> item_ids(src.energyFunctions.numInteractionFns);
  for (unsigned int i = 0; i < src.energyFunctions.numInteractionFns; i++)
    item_ids[i] = src.energyFunctions.interactionFns[i].itemId;
  item_ids.length = src.energyFunctions.numInteractionFns;
  if (item_ids.length > 1) {
    sort(item_ids);
    unique(item_ids);
    if (item_ids.length != src.energyFunctions.numInteractionFns) {
      status->code = JBW_INVALID_SIMULATOR_CONFIGURATION;
      return;
    }
  }

  energy_function<intensity_function> intensity_fn;
  init(intensity_fn, src.energyFunctions.intensityFn, status);
  if (status->code != JBW_OK) return;

  array_map<unsigned int, energy_function<interaction_function>> interaction_fns(
    src.energyFunctions.numInteractionFns);
  for (unsigned int i = 0; i < src.energyFunctions.numInteractionFns; i++) {
    interaction_fns.keys[i] = src.energyFunctions.interactionFns[i].itemId;
    init(interaction_fns.values[i], src.energyFunctions.interactionFns[i], status);
    if (status->code != JBW_OK) {
      free(intensity_fn);
      for (unsigned int j = 0; j < i; j++)
        free(interaction_fns.values[j]);
      return;
    }
  }
  interaction_fns.size = src.energyFunctions.numInteractionFns;

  bool success = init(
    properties, src.name, strlen(src.name), src.scent, src.color, src.requiredItemCounts,
    src.requiredItemCosts, src.blocksMovement, intensity_fn, interaction_fns, scent_dimension,
    color_dimension, item_type_count);
  if (!success) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }

  for (auto entry : interaction_fns)
    free(entry.value);
  free(intensity_fn);
}


inline void init(
  ItemProperties& properties,
  const item_properties& src,
  unsigned int scent_dimension,
  unsigned int color_dimension,
  unsigned int item_type_count,
  JBW_Status* status
) {
  properties.name = (char*) malloc(sizeof(char) * (src.name.length + 1));
  if (properties.name == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (unsigned int i = 0; i < src.name.length; i++)
    properties.name[i] = src.name.data[i];
  properties.name[src.name.length] = '\0';

  properties.scent = (float*) malloc(sizeof(float) * scent_dimension);
  if (properties.scent == nullptr) {
    free(properties.name);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (unsigned int i = 0; i < scent_dimension; i++)
    properties.scent[i] = src.scent[i];

  properties.color = (float*) malloc(sizeof(float) * color_dimension);
  if (properties.color == nullptr) {
    free(properties.name);
    free(properties.scent);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (unsigned int i = 0; i < color_dimension; i++)
    properties.color[i] = src.color[i];

  properties.requiredItemCosts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
  if (properties.requiredItemCosts == nullptr) {
    free(properties.name);
    free(properties.scent);
    free(properties.color);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (unsigned int i = 0; i < item_type_count; i++)
    properties.requiredItemCosts[i] = src.required_item_costs[i];

  properties.requiredItemCounts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
  if (properties.requiredItemCounts == nullptr) {
    free(properties.name);
    free(properties.scent);
    free(properties.color);
    free(properties.requiredItemCosts);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (unsigned int i = 0; i < item_type_count; i++)
    properties.requiredItemCounts[i] = src.required_item_counts[i];

  properties.blocksMovement = src.blocks_movement;
  init(properties.energyFunctions.intensityFn, src.intensity_fn, status);
  if (status->code != JBW_OK) {
    free(properties.name);
    free(properties.scent);
    free(properties.color);
    free(properties.requiredItemCosts);
    free(properties.requiredItemCounts);
    return;
  }

  /* count the number of non-zero interaction functions */
  properties.energyFunctions.numInteractionFns = 0;
  for (unsigned int i = 0; i < item_type_count; i++)
    if (src.interaction_fns[i].fn != zero_interaction_fn)
      properties.energyFunctions.numInteractionFns++;

  /* initialize the interaction functions */
  properties.energyFunctions.interactionFns = (InteractionFunction*) malloc(
    sizeof(InteractionFunction) * properties.energyFunctions.numInteractionFns);
  if (properties.energyFunctions.interactionFns == NULL) {
    free(properties.name);
    free(properties.scent);
    free(properties.color);
    free(properties.requiredItemCosts);
    free(properties.requiredItemCounts);
    free(properties.energyFunctions.intensityFn);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  unsigned int index = 0;
  for (unsigned int i = 0; i < item_type_count; i++) {
    if (src.interaction_fns[i].fn == zero_interaction_fn) continue;
    init(properties.energyFunctions.interactionFns[index], src.interaction_fns[i], i, status);
    if (status->code != JBW_OK) {
      free(properties.name);
      free(properties.scent);
      free(properties.color);
      free(properties.requiredItemCosts);
      free(properties.requiredItemCounts);
      free(properties.energyFunctions.intensityFn);
      for (unsigned int j = 0; j < index; j++)
        free(properties.energyFunctions.interactionFns[j]);
      free(properties.energyFunctions.interactionFns);
      return;
    }
    index++;
  }
}


inline void free(ItemProperties& properties) {
  free(properties.name);
  free(properties.scent);
  free(properties.color);
  free(properties.requiredItemCounts);
  free(properties.requiredItemCosts);
  free(properties.energyFunctions.intensityFn);
  free(properties.energyFunctions.interactionFns);
}


inline void init(
  AgentSimulationState& state,
  const agent_state& src,
  const simulator_config& config,
  uint64_t agent_id,
  JBW_Status* status
) {
  state.position.x = src.current_position.x;
  state.position.y = src.current_position.y;
  state.direction = to_Direction(src.current_direction);
  state.id = agent_id;

  state.scent = (float*) malloc(sizeof(float) * config.scent_dimension);
  if (state.scent == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }

  unsigned int vision_size =
    (2 * config.vision_range + 1) * 
    (2 * config.vision_range + 1) *
    config.color_dimension;
  state.vision = (float*) malloc(sizeof(float) * vision_size);
  if (state.vision == nullptr) {
    free(state.scent);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }

  state.collectedItems = (unsigned int*) malloc(sizeof(unsigned int) * config.item_types.length);
  if (state.collectedItems == nullptr) {
    free(state.scent);
    free(state.vision);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }

  for (unsigned int i = 0; i < config.scent_dimension; i++)
    state.scent[i] = src.current_scent[i];
  for (unsigned int i = 0; i < config.item_types.length; i++)
    state.collectedItems[i] = src.collected_items[i];
  memcpy(state.vision, src.current_vision, sizeof(float) * vision_size);
}


inline void free(AgentSimulationState& state) {
  free(state.scent);
  free(state.vision);
  free(state.collectedItems);
}


inline void init(simulator_config& config, const SimulatorConfig& src, JBW_Status* status) {
  config.agent_color = (float*) malloc(sizeof(float) * src.colorDimSize);
  if (config.agent_color == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }

  for (unsigned int i = 0; i < (size_t) DirectionCount; i++)
    config.allowed_movement_directions[i] = to_action_policy(src.allowedMoveDirections[i]);
  for (unsigned int i = 0; i < (size_t) DirectionCount; i++)
    config.allowed_rotations[i] = to_action_policy(src.allowedRotations[i]);
  for (unsigned int i = 0; i < src.colorDimSize; i++)
    config.agent_color[i] = src.agentColor[i];
  config.no_op_allowed = src.noOpAllowed;

  if (!config.item_types.ensure_capacity(max(1u, src.numItemTypes))) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }

  for (unsigned int i = 0; i < src.numItemTypes; i++) {
    init(
      config.item_types[i], src.itemTypes[i], src.scentDimSize, src.colorDimSize,
      src.numItemTypes, status);
    if (status->code != JBW_OK) {
      for (unsigned int j = 0; j < i; j++)
        free(config.item_types[j], src.numItemTypes);
      return;
    }
  }
  config.item_types.length = src.numItemTypes;
  config.max_steps_per_movement = src.maxStepsPerMove;
  config.scent_dimension = src.scentDimSize;
  config.color_dimension = src.colorDimSize;
  config.vision_range = src.visionRange;
  config.patch_size = src.patchSize;
  config.mcmc_iterations = src.mcmcIterations;
  config.collision_policy = to_movement_conflict_policy(src.movementConflictPolicy);
  config.decay_param = src.scentDecay;
  config.diffusion_param = src.scentDiffusion;
  config.deleted_item_lifetime = src.removedItemLifetime;
}


inline void init(
  SimulatorConfig& config,
  const simulator_config& src,
  unsigned int initial_seed,
  JBW_Status* status
) {
  config.randomSeed = initial_seed;
  config.agentColor = (float*) malloc(sizeof(float) * src.color_dimension);
  if (config.agentColor == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }

  for (unsigned int i = 0; i < (size_t) DirectionCount; i++)
    config.allowedMoveDirections[i] = to_ActionPolicy(src.allowed_movement_directions[i]);
  for (unsigned int i = 0; i < (size_t) DirectionCount; i++)
    config.allowedRotations[i] = to_ActionPolicy(src.allowed_rotations[i]);
  for (unsigned int i = 0; i < src.color_dimension; i++)
    config.agentColor[i] = src.agent_color[i];
  config.noOpAllowed = src.no_op_allowed;

  config.itemTypes = (ItemProperties*) malloc(sizeof(ItemProperties) * src.item_types.length);
  if (config.itemTypes == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (unsigned int i = 0; i < src.item_types.length; i++) {
    init(
      config.itemTypes[i], src.item_types[i], src.scent_dimension, src.color_dimension,
      src.item_types.length, status);
    if (status->code != JBW_OK) {
      for (unsigned int j = 0; j < i; j++)
        free(config.itemTypes[j]);
      return;
    }
  }
  config.numItemTypes = src.item_types.length;
  config.maxStepsPerMove = src.max_steps_per_movement;
  config.scentDimSize = src.scent_dimension;
  config.colorDimSize = src.color_dimension;
  config.visionRange = src.vision_range;
  config.patchSize = src.patch_size;
  config.mcmcIterations = src.mcmc_iterations;
  config.movementConflictPolicy = to_MovementConflictPolicy(src.collision_policy);
  config.scentDecay = src.decay_param;
  config.scentDiffusion = src.diffusion_param;
  config.removedItemLifetime = src.deleted_item_lifetime;
}


inline void init(
  SimulationMapPatch& patch,
  const patch_state& src,
  const simulator_config& config,
  JBW_Status* status
) {
  unsigned int n = config.patch_size;
  patch.items = (ItemInfo*) malloc(sizeof(ItemInfo) * src.item_count);
  if (patch.items == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  patch.agents = (AgentInfo*) malloc(sizeof(AgentInfo) * src.agent_count);
  if (patch.agents == nullptr) {
    free(patch.items);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  if (src.scent != nullptr) {
    patch.scent = (float*) malloc(sizeof(float) * n * n * config.scent_dimension);
    if (patch.scent == nullptr) {
      free(patch.items);
      free(patch.agents);
      status->code = JBW_OUT_OF_MEMORY;
      return;
    }
  }
  patch.vision = (float*) malloc(sizeof(float) * n * n * config.color_dimension);
  if (patch.vision == nullptr) {
    free(patch.items);
    free(patch.agents);
    if (src.scent != nullptr) free(patch.scent);
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (unsigned int i = 0; i < src.item_count; i++) {
    patch.items[i].type = src.items[i].item_type;
    patch.items[i].position.x = src.items[i].location.x;
    patch.items[i].position.y = src.items[i].location.y;
  }
  for (unsigned int i = 0; i < src.agent_count; i++) {
    patch.agents[i].position.x = src.agent_positions[i].x;
    patch.agents[i].position.y = src.agent_positions[i].y;
    patch.agents[i].direction = to_Direction(src.agent_directions[i]);
  }
  patch.position.x = src.patch_position.x;
  patch.position.y = src.patch_position.y;
  patch.numItems = src.item_count;
  patch.numAgents = src.agent_count;
  patch.fixed = src.fixed;
  if (src.scent != nullptr) memcpy(patch.scent, src.scent, sizeof(float) * n * n * config.scent_dimension);
  memcpy(patch.vision, src.vision, sizeof(float) * n * n * config.color_dimension);
}


inline void free(SimulationMapPatch& patch) {
  free(patch.items);
  free(patch.agents);
  if (patch.scent != nullptr)
    free(patch.scent);
  free(patch.vision);
}


inline void init(
  SimulationMap& map,
  const array<array<patch_state>>& patches,
  const simulator_config& config,
  JBW_Status* status
) {
  unsigned int patch_count = 0;
  for (const array<patch_state>& row : patches)
    patch_count += row.length;
  unsigned int index = 0;
  map.patches = (SimulationMapPatch*) malloc(
    max((size_t) 1, sizeof(SimulationMapPatch) * patch_count));
  if (map.patches == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (const array<patch_state>& row : patches) {
    for (const patch_state& patch : row) {
      init(map.patches[index], patch, config, status);
      if (status->code != JBW_OK) {
        for (unsigned int i = 0; i < index; i++)
          free(map.patches[i]);
        free(map.patches);
        return;
      }
      index++;
    }
  }
  map.numPatches = patch_count;
}


/**
 * A struct containing additional state information for the simulator. This
 * information includes a pointer to the `async_server` object, if the
 * simulator is run as a server, a pointer to the Swift callback function,
 * the list of agent IDs owned by this simulator (as opposed to other clients),
 * and information for periodically saving the simulator to file.
 */
struct simulator_data {
  async_server server;
  OnStepCallback callback;
  const void* callback_data;
  array<uint64_t> agent_ids;

  simulator_data(
    OnStepCallback callback,
    const void* callback_data
  ) : callback(callback), callback_data(callback_data), agent_ids(16) { }

  static inline void free(simulator_data& data) {
    core::free(data.agent_ids);
    core::free(data.server);
  }
};


/**
 * Initializes `data` by copying the contents from `src`.
 *
 * \param   data      The `simulator_data` structure to initialize.
 * \param   src       The source `simulator_data` structure that will be
 *                    copied to initialize `data`.
 * \param   status    Status pointer used for propagating error information to the caller.
 */
inline bool init(simulator_data& data, const simulator_data& src) {
  if (!array_init(data.agent_ids, src.agent_ids.capacity)) {
    return false;
  }
  data.agent_ids.append(src.agent_ids.data, src.agent_ids.length);
  if (!init(data.server)) { /* async_server is not copyable */
    free(data.agent_ids);
    return false;
  }
  data.callback = src.callback;
  data.callback_data = src.callback_data;
  return true;
}


/**
 * A struct containing additional state information for the client. This
 * information includes responses from the server, pointers to callback
 * functions, and variables for synchronizing communication between the client
 * response listener thread and the calling thread.
 */
struct client_data {
  struct agent_state_array {
    uint64_t* ids;
    agent_state* states;
    size_t length;
  };

  /* storing the server responses */
  status server_response;
  union response_data {
    bool active;
    AgentSimulationState agent_state;
    array<array<patch_state>>* map;
    pair<uint64_t*, size_t> agent_ids;
    agent_state_array agent_states;
  } response_data;

  /* for synchronization */
  bool waiting_for_server;
  std::mutex lock;
  std::condition_variable cv;

  OnStepCallback step_callback;
  LostConnectionCallback lost_connection_callback;
  const void* callback_data;

  static inline void free(client_data& data) {
    data.lock.~mutex();
    data.cv.~condition_variable();
  }
};


inline bool init(client_data& data) {
  data.step_callback = nullptr;
  data.lost_connection_callback = nullptr;
  new (&data.lock) std::mutex();
  new (&data.cv) std::condition_variable();
  return true;
}


/**
 * The callback function invoked by the simulator when time is advanced. This
 * function is only called if the simulator is run locally or as a server. In
 * server mode, the simulator first sends a step response message to all connected
 * clients. Then, it constructs a list of agent states and invokes the
 * callback in `data.callback`.
 *
 * \param   sim     The simulator invoking this function.
 * \param   agents  The underlying array of all agents in `sim`.
 * \param   time    The new simulation time of `sim`.
 */
void on_step(
  simulator<simulator_data>* sim,
  const hash_map<uint64_t, agent_state*>& agents,
  uint64_t time
) {
  // TODO [STATUS]: Find a way to propagate this status code to Swift.
  auto status = JBW_Status { JBW_OK };
  simulator_data& data = sim->get_data();
  if (data.server.status != server_status::STOPPING) {
    /* this simulator is a server, so send a step response to every client */
    if (!send_step_response(data.server, agents, sim->get_config())) {
      fprintf(stderr, "on_step ERROR: send_step_response failed.\n");
      return;
    }
  }

  AgentSimulationState* agent_states = (AgentSimulationState*) malloc(
    sizeof(AgentSimulationState) * data.agent_ids.length);
  if (agent_states == nullptr) {
    fprintf(stderr, "on_step ERROR: Insufficient memory for agent_states.\n");
    return;
  }
  const simulator_config& config = sim->get_config();
  for (size_t i = 0; i < data.agent_ids.length; i++) {
    init(agent_states[i], *agents.get(data.agent_ids[i]), config, data.agent_ids[i], &status);
    if (status.code != JBW_OK) {
      for (size_t j = 0; j < i; j++)
        free(agent_states[j]);
      free(agent_states);
      return;
    }
  }

  /* invoke callback */
  data.callback(data.callback_data, agent_states, data.agent_ids.length);

  for (size_t i = 0; i < data.agent_ids.length; i++)
    free(agent_states[i]);
  free(agent_states);
}


/**
 * Client callback functions.
 */

inline char* concat(const char* first, const char* second) {
  size_t first_length = strlen(first);
  size_t second_length = strlen(second);
  char* buf = (char*) malloc(sizeof(char) * (first_length + second_length + 1));
  if (buf == nullptr) {
    fprintf(stderr, "concat ERROR: Out of memory.\n");
    return nullptr;
  }
  for (unsigned int i = 0; i < first_length; i++)
    buf[i] = first[i];
  for (unsigned int j = 0; j < second_length; j++)
    buf[first_length + j] = second[j];
  buf[first_length + second_length] = '\0';
  return buf;
}

/**
 * The callback invoked when the client receives an add_agent response from the
 * server. This function copies the agent state into an AgentSimulationState
 * object, stores it in `c.data.response_data.agent_state`, and wakes up the
 * parent thread (which should be waiting in the `simulatorAddAgent` function)
 * so that it can return the response.
 *
 * \param   c         The client that received the response.
 * \param   agent_id  The ID of the new agent. This is equal to `UINT64_MAX` if
 *                    the server returned an error.
 * \param   response  The response from the server, containing information
 *                    about any errors.
 * \param   new_agent The state of the new agent.
 */
void on_add_agent(
  client<client_data>& c,
  uint64_t agent_id,
  status response,
  const agent_state& new_agent
) {
  // TODO [STATUS]: Find a way to propagate this status code to Swift.
  auto status = JBW_Status { JBW_OK };
  AgentSimulationState new_agent_state;
  init(new_agent_state, new_agent, c.config, agent_id, &status);
  if (response != status::OK || status.code != JBW_OK)
    new_agent_state = EMPTY_AGENT_SIM_STATE;
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response_data.agent_state = new_agent_state;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a remove_agent response from
 * the server. This function wakes up the parent thread (which should be
 * waiting in the `simulatorRemoveAgent` function) so that it can return the
 * response back to Python.
 *
 * \param   c         The client that received the response.
 * \param   agent_id  The ID of the removed agent.
 * \param   response  The response from the server, containing information
 *                    about any errors.
 */
void on_remove_agent(client<client_data>& c, uint64_t agent_id, status response) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.server_response = response;
  c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a move response from the
 * server. This function copies the result into `c.data.server_response` and
 * wakes up the parent thread (which should be waiting in the
 * `simulatorMoveAgent` function) so that it can return the response.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to move.
 * \param   response        The response from the server, containing
 *                          information about any errors.
 */
void on_move(client<client_data>& c, uint64_t agent_id, status response) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a turn response from the
 * server. This function copies the result into `c.data.server_response` and
 * wakes up the parent thread (which should be waiting in the
 * `simulatorTurnAgent` function) so that it can return the response.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to turn.
 * \param   response        The response from the server, containing
 *                          information about any errors.
 */
void on_turn(client<client_data>& c, uint64_t agent_id, status response) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a do_nothing response from the
 * server. This function copies the result into `c.data.server_response` and
 * wakes up the parent thread (which should be waiting in the
 * `simulatorNoOpAgent` function) so that it can return the response.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to do nothing.
 * \param   response        The response from the server, containing
 *                          information about any errors.
 */
void on_do_nothing(client<client_data>& c, uint64_t agent_id, status response) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a get_map response from the
 * server. This function moves the result into `c.data.response_data.map` and
 * wakes up the parent thread (which should be waiting in the `simulatorMap`
 * function) so that it can return the response back.
 *
 * \param   c        The client that received the response.
 * \param   response The response from the server, containing information about
 *                   any errors.
 * \param   map      A map from patch positions to `patch_state` structures
 *                   containing the state information in each patch.
 */
void on_get_map(client<client_data>& c, status response, array<array<patch_state>>* map) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response_data.map = map;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a get_agent_ids response from
 * the server. This function moves the result into
 * `c.data.response_data.agent_ids` and wakes up the parent thread (which
 * should be waiting in the `simulatorAgentIds` function) so that it can return
 * the response back.
 *
 * \param   c         The client that received the response.
 * \param   response  The response from the server, containing information
 *                    about any errors.
 * \param   agent_ids The array containing the agent IDs.
 * \param   count     The number of IDs in the array `agent_ids`.
 */
void on_get_agent_ids(
  client<client_data>& c, status response,
  uint64_t* agent_ids, size_t count)
{
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response_data.agent_ids = make_pair(agent_ids, count);
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a get_agent_states response
 * from the server. This function moves the result into
 * `c.data.response_data.agent_states` and wakes up the parent thread (which
 * should be waiting in the `simulatorAgentStates` function) so that it can
 * return the response back.
 *
 * \param   c            The client that received the response.
 * \param   response     The response from the server, containing information
 *                       about any errors.
 * \param   agent_ids    The array containing the agent IDs.
 * \param   agent_states The array containing the agent states.
 * \param   count        The length of `agent_ids` and `agent_states`.
 */
void on_get_agent_states(client<client_data>& c,
  status response, uint64_t* agent_ids,
  agent_state* agent_states, size_t count)
{
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response_data.agent_states.ids = agent_ids;
  c.data.response_data.agent_states.states = agent_states;
  c.data.response_data.agent_states.length = count;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a set_active response from the
 * server. This function wakes up the parent thread (which should be waiting in
 * the `simulatorSetActive` function) so that it can return the response back.
 *
 * \param   c        The client that received the response.
 * \param   agent_id The ID of the agent whose active status was set.
 * \param   response The response from the server, containing information about
 *                   any errors.
 */
void on_set_active(client<client_data>& c, uint64_t agent_id, status response) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives an is_active response from the
 * server. This function moves the result into `c.data.response.action_result`
 * and wakes up the parent thread (which should be waiting in the
 * `simulatorIsActive` function) so that it can return the response back.
 *
 * \param   c        The client that received the response.
 * \param   agent_id The ID of the agent whose active status was requested.
 * \param   response The response from the server, containing information about
 *                   any errors.
 * \param   active   Whether or not the agent is active.
 */
void on_is_active(client<client_data>& c, uint64_t agent_id, status response, bool active) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response_data.active = active;
  c.data.server_response = response;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a step response from the
 * server. This function constructs a list of AgentSimulationState objects
 * governed by this client and invokes the function `c.data.step_callback`.
 *
 * \param   c            The client that received the response.
 * \param   response     The response from the server, containing information
 *                       about any errors.
 * \param   agent_ids    An array of agent IDs governed by the client.
 * \param   agent_states An array, parallel to `agent_ids`, containing the
 *                       state information of each agent at the beginning of
 *                       the new time step in the simulation.
 */
void on_step(
  client<client_data>& c,
  status response,
  const array<uint64_t>& agent_ids,
  const agent_state* agent_states
) {
  // TODO [STATUS]: Find a way to propagate this status code to Swift.
  auto status = JBW_Status { JBW_OK };
  JBW_SetJBWStatusFromStatus(&status, response);
  AgentSimulationState* agents = (AgentSimulationState*) malloc(
    sizeof(AgentSimulationState) * agent_ids.length);
  if (agents == nullptr) {
    status.code = JBW_OUT_OF_MEMORY;
    return;
  }
  for (size_t i = 0; i < agent_ids.length; i++) {
    init(agents[i], agent_states[i], c.config, agent_ids[i], &status);
    if (status.code != JBW_OK) {
      for (size_t j = 0; j < i; j++)
        free(agents[j]);
      free(agents);
      return;
    }
  }
  c.data.step_callback(c.data.callback_data, agents, agent_ids.length);
  for (size_t i = 0; i < agent_ids.length; i++)
    free(agents[i]);
  free(agents);
}


/**
 * The callback invoked when the client loses the connection to the server.
 * \param   c       The client whose connection to the server was lost.
 */
void on_lost_connection(client<client_data>& c) {
  c.client_running = false;
  c.data.cv.notify_one();
  c.data.lost_connection_callback(c.data.callback_data);
}


/**
 * This functions waits for a response from the server, and for one of the
 * above client callback functions to be invoked.
 * \param   c       The client expecting a response from the server.
 */
inline void wait_for_server(client<client_data>& c) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  while (c.data.waiting_for_server && c.client_running)
    c.data.cv.wait(lck);
}


void* simulatorCreate(
  const SimulatorConfig* config,
  OnStepCallback onStepCallback,
  JBW_Status* status
) {
  simulator_config sim_config;
  init(sim_config, *config, status);
  if (status->code != JBW_OK) return nullptr;
  simulator_data data(onStepCallback, nullptr);
  simulator<simulator_data>* sim = (simulator<simulator_data>*) malloc(
    sizeof(simulator<simulator_data>));
  if (sim == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return nullptr;
  }
  auto result = init(*sim, sim_config, data, config->randomSeed);
  if (result != status::OK) {
    JBW_SetJBWStatusFromStatus(status, result);
    return nullptr;
  }
  return (void*) sim;
}


void simulatorSave(void* simulatorHandle, const char* filePath, JBW_Status* status) {
  FILE* file = open_file(filePath, "wb");
  if (file == nullptr) {
    status->code = JBW_IO_ERROR;
    return;
  }
  simulator<simulator_data>* sim = (simulator<simulator_data>*) simulatorHandle;
  const simulator_data& data = sim->get_data();
  fixed_width_stream<FILE*> out(file);
  bool result = write(*sim, out)
    && write(data.agent_ids.length, out)
    && write(data.agent_ids.data, out, data.agent_ids.length)
    && write(data.server.state, out);
  fclose(file);
  if (!result) {
    status->code = JBW_IO_ERROR;
  }
}


SimulatorInfo simulatorLoad(
  const char* filePath,
  OnStepCallback onStepCallback,
  JBW_Status* status
) {
  simulator<simulator_data>* sim = (simulator<simulator_data>*) malloc(
    sizeof(simulator<simulator_data>));
  if (sim == nullptr) {
    status->code = JBW_OUT_OF_MEMORY;
    return EMPTY_SIM_INFO;
  }

  simulator_data data(onStepCallback, nullptr);

  FILE* file = open_file(filePath, "rb");
  if (file == nullptr) {
    free(sim);
    status->code = JBW_IO_ERROR;
    return EMPTY_SIM_INFO;
  }

  size_t agent_id_count;
  fixed_width_stream<FILE*> in(file);
  if (!read(*sim, in, data)) {
    free(sim);
    fclose(file);
    status->code = JBW_IO_ERROR;
    return EMPTY_SIM_INFO;
  }
  simulator_data& sim_data = sim->get_data();
  if (!read(agent_id_count, in)
   || !sim_data.agent_ids.ensure_capacity(agent_id_count)
   || !read(sim_data.agent_ids.data, in, agent_id_count)
   || !read(sim_data.server.state, in))
  {
    free(*sim);
    free(sim);
    fclose(file);
    status->code = JBW_IO_ERROR;
    return EMPTY_SIM_INFO;
  }
  sim_data.agent_ids.length = agent_id_count;
  fclose(file);

  agent_state** agent_states = (agent_state**) malloc(sizeof(agent_state*) * agent_id_count);
  if (agent_states == nullptr) {
    free(*sim);
    free(sim);
    status->code = JBW_OUT_OF_MEMORY;
    return EMPTY_SIM_INFO;
  }

  sim->get_agent_states(agent_states, sim_data.agent_ids.data, (unsigned int) agent_id_count);

  const simulator_config& sim_config = sim->get_config();
  auto agents = (AgentSimulationState*) malloc(sizeof(AgentSimulationState) * agent_id_count);
  if (agents == nullptr) {
    free(*sim);
    free(sim);
    free(agent_states);
    status->code = JBW_OUT_OF_MEMORY;
    return EMPTY_SIM_INFO;
  }
  for (size_t i = 0; i < agent_id_count; i++) {
    init(agents[i], *agent_states[i], sim_config, sim_data.agent_ids[i], status);
    if (status->code != JBW_OK) {
      for (size_t j = 0; j < i; j++)
        free(agents[j]);
      free(*sim);
      free(sim);
      free(agent_states);
      free(agents);
      return EMPTY_SIM_INFO;
    }
  }
  free(agent_states);

  SimulatorInfo sim_info;
  sim_info.handle = (void*) sim;
  sim_info.time = sim->time;
  sim_info.agents = agents;
  sim_info.numAgents = (unsigned int) agent_id_count;
  init(sim_info.config, sim_config, sim->get_world().initial_seed, status);
  if (status->code != JBW_OK) {
    for (size_t j = 0; j < agent_id_count; j++)
      free(agents[j]);
    free(*sim);
    free(sim);
    free(agents);
    return EMPTY_SIM_INFO;
  }
  return sim_info;
}


void simulatorDelete(void* simulatorHandle) {
  simulator<simulator_data>* sim = (simulator<simulator_data>*) simulatorHandle;
  free(*sim);
  free(sim);
}

void simulatorSetStepCallbackData(void* simulatorHandle, const void* callbackData) {
  simulator<simulator_data>* sim = (simulator<simulator_data>*) simulatorHandle;
  simulator_data& sim_data = sim->get_data();
  sim_data.callback_data = callbackData;
}

AgentSimulationState simulatorAddAgent(
  void* simulatorHandle,
  void* clientHandle,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call add_agent directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    uint64_t new_agent_id; agent_state* new_agent;
    auto result = sim_handle->add_agent(new_agent_id, new_agent);
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return EMPTY_AGENT_SIM_STATE;
    }

    sim_handle->get_data().agent_ids.add(new_agent_id);

    AgentSimulationState new_agent_state;
    std::unique_lock<std::mutex> lock(new_agent->lock);
    init(new_agent_state, *new_agent, sim_handle->get_config(), new_agent_id, status);
    if (status->code != JBW_OK) return EMPTY_AGENT_SIM_STATE;
    return new_agent_state;
  } else {
    /* this is a client, so send an add_agent message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return EMPTY_AGENT_SIM_STATE;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_add_agent(*client_handle)) {
      status->code = JBW_MPI_ERROR;
      return EMPTY_AGENT_SIM_STATE;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);

    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return EMPTY_AGENT_SIM_STATE;
    }

    return client_handle->data.response_data.agent_state;
  }
}


void simulatorRemoveAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call add_agent directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;

    auto result = sim_handle->remove_agent(agentId);
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return;
    }

    array<uint64_t>& agent_ids = sim_handle->get_data().agent_ids;
    unsigned int index = agent_ids.index_of(agentId);
    if (index != agent_ids.length)
      agent_ids.remove(index);
  } else {
    /* this is a client, so send an add_agent message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_remove_agent(*client_handle, agentId)) {
      status->code = JBW_MPI_ERROR;
      return;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);

    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return;
    }
  }
}


void simulatorMoveAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  Direction direction,
  unsigned int numSteps,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call move directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    auto result = sim_handle->move(agentId, to_direction(direction), numSteps);
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return;
    }
  } else {
    /* this is a client, so send a move message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_move(*client_handle, agentId, to_direction(direction), numSteps)) {
      status->code = JBW_MPI_ERROR;
      return;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);

    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return;
    }
  }
}


void simulatorTurnAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  TurnDirection direction,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call turn directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    auto result = sim_handle->turn(agentId, to_direction(direction));
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return;
    }
  } else {
    /* this is a client, so send a turn message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_turn(*client_handle, agentId, to_direction(direction))) {
      status->code = JBW_MPI_ERROR;
      return;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);

    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return;
    }
  }
}


void simulatorNoOpAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call do_nothing directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    auto result = sim_handle->do_nothing(agentId);
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return;
    }
  } else {
    /* this is a client, so send a do_nothing message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_do_nothing(*client_handle, agentId)) {
      status->code = JBW_MPI_ERROR;
      return;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);

    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return;
    }
  }
}


void simulatorSetActive(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  bool active,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call set_active directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    auto result = sim_handle->set_agent_active(agentId, active);
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return;
    }
  } else {
    /* this is a client, so send a set_active message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_set_active(*client_handle, agentId, active)) {
      status->code = JBW_MPI_ERROR;
      return;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);

    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return;
    }
  }
}


bool simulatorIsActive(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call is_active directly */
    bool active;
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    auto result = sim_handle->is_agent_active(agentId, active);
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return false;
    }
    return active;
  } else {
    /* this is a client, so send a is_active message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return false;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_is_active(*client_handle, agentId)) {
      status->code = JBW_MPI_ERROR;
      return false;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);

    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return false;
    }

    return client_handle->data.response_data.active;
  }
}


const SimulationMap simulatorMap(
  void* simulatorHandle,
  void* clientHandle,
  Position bottomLeftCorner,
  Position topRightCorner,
  bool getScentMap,
  JBW_Status* status
) {
  position bottom_left = position(bottomLeftCorner.x, bottomLeftCorner.y);
  position top_right = position(topRightCorner.x, topRightCorner.y);
  if (clientHandle == nullptr) {
    /* the simulation is local, so call get_map directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    array<array<patch_state>> patches(16);
    auto result = (getScentMap ?
      sim_handle->get_map<true>(bottom_left, top_right, patches) :
      sim_handle->get_map<false>(bottom_left, top_right, patches));
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return EMPTY_SIM_MAP;
    }

    SimulationMap map;
    init(map, patches, sim_handle->get_config(), status);
    if (status->code != JBW_OK)
      map = EMPTY_SIM_MAP;
    for (array<patch_state>& row : patches) {
      for (auto& entry : row)
        free(entry);
      free(row);
    }
    return map;
  } else {
    /* this is a client, so send a get_map message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return EMPTY_SIM_MAP;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_get_map(*client_handle, bottom_left, top_right, getScentMap)) {
      status->code = JBW_MPI_ERROR;
      return EMPTY_SIM_MAP;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);
    SimulationMap map;
    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return EMPTY_SIM_MAP;
    }
    init(map, *client_handle->data.response_data.map, client_handle->config, status);
    if (status->code != JBW_OK)
      map = EMPTY_SIM_MAP;
    for (array<patch_state>& row : *client_handle->data.response_data.map) {
      for (patch_state& entry : row)
        free(entry);
      free(row);
    }
    free(*client_handle->data.response_data.map);
    free(client_handle->data.response_data.map);
    return map;
  }
}


const AgentIDList simulatorAgentIds(
  void* simulatorHandle,
  void* clientHandle,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call get_agent_ids directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    array<uint64_t>& agent_ids = *((array<uint64_t>*) alloca(sizeof(array<uint64_t>)));
    if (!array_init(agent_ids, 16)) {
      status->code = JBW_OUT_OF_MEMORY;
      return EMPTY_AGENT_ID_LIST;
    }
    auto result = sim_handle->get_agent_ids(agent_ids);
    if (result != status::OK) {
      JBW_SetJBWStatusFromStatus(status, result);
      return EMPTY_AGENT_ID_LIST;
    }

    AgentIDList list;
    list.agentIds = agent_ids.data;
    list.numAgents = (unsigned int) agent_ids.length;
    return list;
  } else {
    /* this is a client, so send a get_agent_ids message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return EMPTY_AGENT_ID_LIST;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_get_agent_ids(*client_handle)) {
      status->code = JBW_MPI_ERROR;
      return EMPTY_AGENT_ID_LIST;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);
    AgentIDList list;
    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return EMPTY_AGENT_ID_LIST;
    }
    list.agentIds = client_handle->data.response_data.agent_ids.key;
    list.numAgents = (unsigned int) client_handle->data.response_data.agent_ids.value;
    return list;
  }
}


const AgentSimulationState* simulatorAgentStates(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t* agentIds,
  unsigned int numAgents,
  JBW_Status* status
) {
  if (clientHandle == nullptr) {
    /* the simulation is local, so call get_agent_ids directly */
    simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
    agent_state** agent_states = (agent_state**) malloc(
      max((size_t) 1, sizeof(agent_state*) * numAgents));
    if (agent_states == nullptr) {
      status->code = JBW_OUT_OF_MEMORY;
      return nullptr;
    }
    sim_handle->get_agent_states(agent_states, agentIds, numAgents);

    AgentSimulationState* agent_simulation_states = (AgentSimulationState*) malloc(
      max((size_t) 1, sizeof(AgentSimulationState) * numAgents));
    if (agent_simulation_states == nullptr) {
      status->code = JBW_OUT_OF_MEMORY;
      for (size_t i = 0; i < numAgents; i++)
        if (agent_states[i] != nullptr)
          agent_states[i]->lock.unlock();
      free(agent_states);
      return nullptr;
    }
    const simulator_config& config = sim_handle->get_config();
    for (size_t i = 0; i < numAgents; i++) {
      if (agent_states[i] == nullptr) {
        agent_simulation_states[i] = EMPTY_AGENT_SIM_STATE;
      } else {
        init(agent_simulation_states[i], *agent_states[i], config, agentIds[i], status);
        if (status->code != JBW_OK) {
          for (size_t j = 0; j < i; j++)
            free(agent_simulation_states[j]);
          for (size_t j = i; j < numAgents; j++)
            agent_states[j]->lock.unlock();
          free(agent_simulation_states);
          free(agent_states);
          return nullptr;
        }
        agent_states[i]->lock.unlock();
      }
    }
    return agent_simulation_states;
  } else {
    /* this is a client, so send a get_agent_states message to the server */
    client<client_data>* client_handle = (client<client_data>*) clientHandle;
    if (!client_handle->client_running) {
      status->code = JBW_LOST_CONNECTION;
      return nullptr;
    }

    client_handle->data.waiting_for_server = true;
    if (!send_get_agent_states(*client_handle, agentIds, numAgents)) {
      status->code = JBW_MPI_ERROR;
      return nullptr;
    }

    /* wait for response from server */
    wait_for_server(*client_handle);
    if (client_handle->data.server_response != status::OK) {
      JBW_SetJBWStatusFromStatus(status, client_handle->data.server_response);
      return nullptr;
    }

    AgentSimulationState* agent_simulation_states = (AgentSimulationState*) malloc(
      max((size_t) 1, sizeof(AgentSimulationState) * numAgents));
    if (agent_simulation_states == nullptr) {
      status->code = JBW_OUT_OF_MEMORY;
      for (size_t i = 0; i < client_handle->data.response_data.agent_states.length; i++)
        free(client_handle->data.response_data.agent_states.states[i]);
      free(client_handle->data.response_data.agent_states.ids);
      free(client_handle->data.response_data.agent_states.states);
      return nullptr;
    }
    const simulator_config& config = client_handle->config;
    size_t next_index = 0;
    for (size_t i = 0; i < numAgents; i++) {
      if (next_index == client_handle->data.response_data.agent_states.length
       || client_handle->data.response_data.agent_states.ids[next_index] != agentIds[i])
      {
        agent_simulation_states[i] = EMPTY_AGENT_SIM_STATE;
      } else {
        init(
          agent_simulation_states[i], client_handle->data.response_data.agent_states.states[i],
          config, agentIds[i], status);
        if (status->code != JBW_OK) {
          for (size_t j = i; j < numAgents; j++)
            free(client_handle->data.response_data.agent_states.states[j]);
          free(client_handle->data.response_data.agent_states.ids);
          free(client_handle->data.response_data.agent_states.states);
          free(agent_simulation_states);
          return nullptr;
        }
        free(client_handle->data.response_data.agent_states.states[i]);
        next_index++;
      }
    }
    free(client_handle->data.response_data.agent_states.ids);
    free(client_handle->data.response_data.agent_states.states);
    return agent_simulation_states;
  }
}


void* simulationServerStart(
  void* simulatorHandle,
  unsigned int port,
  unsigned int connectionQueueCapacity,
  unsigned int numWorkers,
  Permissions perms,
  JBW_Status* status
) {
  simulator<simulator_data>* sim_handle = (simulator<simulator_data>*) simulatorHandle;
  async_server& server = sim_handle->get_data().server;
  if (!init_server(
    server, *sim_handle, (uint16_t) port, connectionQueueCapacity,
    numWorkers, to_permissions(perms))
  ) {
    status->code = JBW_MPI_ERROR;
    return nullptr;
  }
  return (void*) &server;
}


void simulationServerStop(void* serverHandle) {
  async_server* server = (async_server*) serverHandle;
  stop_server(*server);
}


Permissions simulationGetPermissions(void* serverHandle, uint64_t clientId) {
  async_server* server = (async_server*) serverHandle;
  return to_Permissions(get_permissions(*server, clientId));
}


void simulationSetPermissions(void* serverHandle, uint64_t clientId, Permissions perms) {
  async_server* server = (async_server*) serverHandle;
  set_permissions(*server, clientId, to_permissions(perms));
}


SimulationNewClientInfo simulationClientConnect(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
  JBW_Status* status
) {
  client<client_data>* new_client = (client<client_data>*) malloc(sizeof(client<client_data>));
  if (new_client == nullptr || !init(*new_client)) {
    if (new_client != nullptr)
      free(new_client);
    status->code = JBW_OUT_OF_MEMORY;
    return EMPTY_NEW_CLIENT_INFO;
  }

  uint64_t client_id;
  uint64_t simulator_time = connect_client(
    *new_client, serverAddress, (uint16_t) serverPort, client_id);
  if (simulator_time == UINT64_MAX) {
    free(*new_client);
    free(new_client);
    status->code = JBW_MPI_ERROR;
    return EMPTY_NEW_CLIENT_INFO;
  }

  new_client->data.step_callback = onStepCallback;
  new_client->data.lost_connection_callback = lostConnectionCallback;
  new_client->data.callback_data = nullptr;

  SimulationNewClientInfo client_info;
  client_info.handle = (void*) new_client;
  client_info.simulationTime = simulator_time;
  client_info.clientId = client_id;
  return client_info;
}


SimulationClientInfo simulationClientReconnect(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
  uint64_t clientId,
  JBW_Status* status
) {
  client<client_data>* new_client = (client<client_data>*) malloc(sizeof(client<client_data>));
  if (new_client == nullptr || !init(*new_client)) {
    status->code = JBW_OUT_OF_MEMORY;
    if (new_client != nullptr)
      free(new_client);
    return EMPTY_CLIENT_INFO;
  }

  uint64_t* agent_ids; agent_state* agent_states;
  unsigned int agent_count;
  uint64_t simulator_time = reconnect_client(
    *new_client, clientId, serverAddress, (uint16_t) serverPort,
    agent_ids, agent_states, agent_count);
  if (simulator_time == UINT64_MAX) {
    free(*new_client);
    free(new_client);
    status->code = JBW_MPI_ERROR;
    return EMPTY_CLIENT_INFO;
  }

  auto agentStates = (AgentSimulationState*) malloc(sizeof(AgentSimulationState) * agent_count);
  if (agentStates == nullptr) {
    for (unsigned int i = 0; i < agent_count; i++)
      free(agent_states[i]);
    free(agent_states);
    free(agent_ids);
    stop_client(*new_client);
    free(*new_client);
    free(new_client);
    status->code = JBW_OUT_OF_MEMORY;
    return EMPTY_CLIENT_INFO;
  }
  for (size_t i = 0; i < agent_count; i++) {
    init(agentStates[i], agent_states[i], new_client->config, agent_ids[i], status);
    if (status->code != JBW_OK) {
      for (size_t j = 0; j < i; j++)
        free(agentStates[j]);
      for (unsigned int j = i; j < agent_count; j++)
        free(agent_states[j]);
      free(agentStates);
      free(agent_states);
      free(agent_ids);
      stop_client(*new_client);
      free(*new_client);
      free(new_client);
      return EMPTY_CLIENT_INFO;
    }
    free(agent_states[i]);
  }
  free(agent_states);

  new_client->data.step_callback = onStepCallback;
  new_client->data.lost_connection_callback = lostConnectionCallback;
  new_client->data.callback_data = nullptr;

  SimulationClientInfo client_info;
  client_info.handle = (void*) new_client;
  client_info.simulationTime = simulator_time;
  client_info.agentIds = agent_ids;
  client_info.agentStates = agentStates;
  client_info.numAgents = agent_count;
  return client_info;
}


void simulationClientStop(void* clientHandle) {
  client<client_data>* client_handle = (client<client_data>*) clientHandle;
  stop_client(*client_handle);
  free(*client_handle);
  free(client_handle);
}


void simulationClientRemove(void* clientHandle, JBW_Status* status) {
  client<client_data>* client_handle = (client<client_data>*) clientHandle;
  if (!remove_client(*client_handle)) {
    status->code = JBW_MPI_ERROR;
  }
  free(*client_handle);
  free(client_handle);
}


void simulatorDeleteSimulatorInfo(SimulatorInfo info) {
  for (unsigned int i = 0; i < info.numAgents; i++)
    free(info.agents[i]);
  free(info.agents);
}


void simulatorDeleteSimulationClientInfo(SimulationClientInfo clientInfo, unsigned int numAgents) {
  for (unsigned int i = 0; i < numAgents; i++)
    free(clientInfo.agentStates[i]);
  free(clientInfo.agentStates);
}


void simulatorDeleteAgentSimulationState(AgentSimulationState agentState) {
  free(agentState);
}


void simulatorDeleteSimulationMap(SimulationMap map) {
  for (unsigned int i = 0; i < map.numPatches; i++)
    free(map.patches[i]);
  free(map.patches);
}

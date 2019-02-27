#include <thread>

#include "include/simulator.h"

#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

using namespace core;
using namespace nel;

constexpr AgentSimulationState EMPTY_AGENT_SIM_STATE = { 0 };
constexpr SimulatorInfo EMPTY_SIM_INFO = { 0 };
constexpr SimulationClientInfo EMPTY_CLIENT_INFO = { 0 };
constexpr SimulationMap EMPTY_SIM_MAP = { 0 };


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


inline bool init(
  energy_function<intensity_function>& function,
  const IntensityFunction& src)
{
  function.fn = get_intensity_fn((intensity_fns) src.id, src.args, src.numArgs);
  function.args = (float*) malloc(sizeof(float) * src.numArgs);
  if (function.args == NULL) {
    /* TODO: communicate out of memory error to swift */
    return false;
  }
  memcpy(function.args, src.args, sizeof(float) * src.numArgs);
  function.arg_count = src.numArgs;
  return true;
}


inline bool init(
  energy_function<interaction_function>& function,
  const InteractionFunction& src)
{
  function.fn = get_interaction_fn((interaction_fns) src.id, src.args, src.numArgs);
  function.args = (float*) malloc(sizeof(float) * src.numArgs);
  if (function.args == NULL) {
    /* TODO: communicate out of memory error to swift */
    return false;
  }
  memcpy(function.args, src.args, sizeof(float) * src.numArgs);
  function.arg_count = src.numArgs;
  return true;
}


inline bool init(
  item_properties& properties, const ItemProperties& src,
  unsigned int scent_dimension, unsigned int color_dimension,
  unsigned int item_type_count)
{
  /* check that `itemId` for `src.energyFunctions.interactionFn` are unique */
  array<unsigned int> item_ids(src.energyFunctions.numInteractionFns);
  for (unsigned int i = 0; i < src.energyFunctions.numInteractionFns; i++)
    item_ids[i] = src.energyFunctions.interactionFns[i].itemId;
  item_ids.length = src.energyFunctions.numInteractionFns;
  if (item_ids.length > 1) {
    sort(item_ids);
    unique(item_ids);
    if (item_ids.length != src.energyFunctions.numInteractionFns) {
      /* TODO: communicate error to swift that the itemIds are not unique */
      return false;
    }
  }

  energy_function<intensity_function> intensity_fn;
  if (!init(intensity_fn, src.energyFunctions.intensityFn))
    return false;

  array_map<unsigned int, energy_function<interaction_function>> interaction_fns(src.energyFunctions.numInteractionFns);
  for (unsigned int i = 0; i < src.energyFunctions.numInteractionFns; i++) {
    interaction_fns.keys[i] = src.energyFunctions.interactionFns[i].itemId;
    if (!init(interaction_fns.values[i], src.energyFunctions.interactionFns[i])) {
      for (unsigned int j = 0; j < i; j++) free(interaction_fns.values[i]);
      free(intensity_fn); return false;
    }
  }
  interaction_fns.size = src.energyFunctions.numInteractionFns;

  bool success = init(properties, src.name, strlen(src.name),
    src.scent, src.color, src.requiredItemCounts,
    src.requiredItemCosts, src.blocksMovement,
    intensity_fn, interaction_fns,
    scent_dimension, color_dimension, item_type_count);
}


inline bool init(
  AgentSimulationState& state,
  const agent_state& src,
  const simulator_config& config,
  uint64_t agent_id)
{
  state.position.x = src.current_position.x;
  state.position.y = src.current_position.y;
  state.direction = to_Direction(src.current_direction);
  state.id = agent_id;

  state.scent = (float*) malloc(sizeof(float) * config.scent_dimension);
  if (state.scent == nullptr) {
    /* TODO: communicate out of memory error to swift */
    return false;
  }
  unsigned int vision_size = (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension;
  state.vision = (float*) malloc(sizeof(float) * vision_size);
  if (state.vision == nullptr) {
    /* TODO: communicate out of memory error to swift */
    free(state.scent); return false;
  }
  state.collectedItems = (unsigned int*) malloc(sizeof(unsigned int) * config.item_types.length);
  if (state.collectedItems == nullptr) {
    /* TODO: communicate out of memory error to swift */
    free(state.scent); free(state.vision); return false;
  }

  for (unsigned int i = 0; i < config.scent_dimension; i++)
    state.scent[i] = src.current_scent[i];
  for (unsigned int i = 0; i < config.item_types.length; i++)
    state.collectedItems[i] = src.collected_items[i];
  memcpy(state.vision, src.current_vision, sizeof(float) * vision_size);
  return true;
}


inline void free(AgentSimulationState& state) {
  free(state.scent);
  free(state.vision);
  free(state.collectedItems);
}


bool init(
  SimulationMapPatch& patch,
  const patch_state& src,
  const simulator_config& config)
{
  unsigned int n = config.patch_size;
  patch.items = (ItemInfo*) malloc(sizeof(ItemInfo) * src.item_count);
  if (patch.items == nullptr) {
    /* TODO: communicate out of memory error to swift */
    return false;
  }
  patch.agents = (AgentInfo*) malloc(sizeof(AgentInfo) * src.agent_count);
  if (patch.agents == nullptr) {
    /* TODO: communicate out of memory error to swift */
    free(patch.items); return false;
  }
  patch.scent = (float*) malloc(sizeof(float) * n * n * config.scent_dimension);
  if (patch.scent == nullptr) {
    /* TODO: communicate out of memory error to swift */
    free(patch.items); free(patch.agents); return false;
  }
  patch.vision = (float*) malloc(sizeof(float) * n * n * config.color_dimension);
  if (patch.vision == nullptr) {
    /* TODO: communicate out of memory error to swift */
    free(patch.items); free(patch.agents);
    free(patch.scent); return false;
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

  memcpy(patch.scent, src.scent, sizeof(float) * n * n * config.scent_dimension);
  memcpy(patch.vision, src.vision, sizeof(float) * n * n * config.color_dimension);
  return true;
}


inline void free(SimulationMapPatch& patch) {
  free(patch.items);
  free(patch.agents);
  free(patch.scent);
  free(patch.vision);
}


bool init(SimulationMap& map,
  const hash_map<position, patch_state>& patches,
  const simulator_config& config)
{
  unsigned int index = 0;
  map.patches = (SimulationMapPatch*) malloc(max((size_t) 1, sizeof(SimulationMapPatch) * patches.table.size));
  if (map.patches == nullptr) {
    /* TODO: communicate out of memory error to swift */
    return false;
  }
  for (const auto& entry : patches) {
    if (!init(map.patches[index], entry.value, config)) {
      for (unsigned int i = 0; i < index; i++) free(map.patches[i]);
      free(map.patches); return false;
    }
    index++;
  }
  map.numPatches = patches.table.size;
  return true;
}


/**
 * A struct containing additional state information for the simulator. This
 * information includes a pointer to the `async_server` object, if the
 * simulator is run as a server, a pointer to the Swift callback function,
 * the list of agent IDs owned by this simulator (as opposed to other clients),
 * and information for periodically saving the simulator to file.
 */
struct simulator_data
{
  char* save_directory;
  unsigned int save_directory_length;
  unsigned int save_frequency;
  async_server* server;
  OnStepCallback callback;
  void* callback_data;

  /* agents owned by the simulator */
  array<uint64_t> agent_ids;

  simulator_data(
      const char* save_filepath,
      unsigned int save_filepath_length,
      unsigned int save_frequency,
      async_server* server,
      OnStepCallback callback,
      void* callback_data) :
    save_frequency(save_frequency), server(server),
    callback(callback), callback_data(callback_data), agent_ids(16)
  {
    if (save_filepath == nullptr) {
      save_directory = nullptr;
    } else {
      save_directory = (char*) malloc(sizeof(char) * save_filepath_length);
      if (save_directory == nullptr) {
        fprintf(stderr, "simulator_data ERROR: Out of memory.\n");
        exit(EXIT_FAILURE);
      }
      save_directory_length = save_filepath_length;
      for (unsigned int i = 0; i < save_filepath_length; i++)
        save_directory[i] = save_filepath[i];
      }
  }

  ~simulator_data() { free_helper(); }

  static inline void free(simulator_data& data) {
    data.free_helper();
    core::free(data.agent_ids);
  }

private:
  inline void free_helper() {
    if (save_directory != nullptr)
      core::free(save_directory);
  }
};


/**
 * Initializes `data` by copying the contents from `src`.
 *
 * \param   data      The `simulator_data` structure to initialize.
 * \param   src       The source `simulator_data` structure that will be
 *                    copied to initialize `data`.
 * \returns `true` if successful; and `false` otherwise.
 */
inline bool init(simulator_data& data, const simulator_data& src)
{
  if (!array_init(data.agent_ids, src.agent_ids.capacity))
    return false;
  data.agent_ids.append(src.agent_ids.data, src.agent_ids.length);

  if (src.save_directory != nullptr) {
    data.save_directory = (char*) malloc(sizeof(char) * max(1u, src.save_directory_length));
    if (data.save_directory == nullptr) {
      fprintf(stderr, "init ERROR: Insufficient memory for simulator_data.save_directory.\n");
      free(data.agent_ids); return false;
    }
    for (unsigned int i = 0; i < src.save_directory_length; i++)
      data.save_directory[i] = src.save_directory[i];
    data.save_directory_length = src.save_directory_length;
  } else {
    data.save_directory = nullptr;
  }
  data.save_frequency = src.save_frequency;
  data.server = src.server;
  data.callback = src.callback;
  return true;
}


/**
 * A struct containing additional state information for the client. This
 * information includes responses from the server, pointers to callback
 * functions, and variables for synchronizing communication between the client
 * response listener thread and the calling thread.
 */
struct client_data {
  /* storing the server responses */
  union response {
    bool action_result;
    AgentSimulationState agent_state;
    hash_map<position, patch_state>* map;
  } response;

  /* for synchronization */
  bool waiting_for_server;
  std::mutex lock;
  std::condition_variable cv;

  OnStepCallback step_callback;
  LostConnectionCallback lost_connection_callback;
  void* callback_data;

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
 * Saves the simulator given by the specified pointer `sim` to the filepath
 * specified by the `simulator_data` structure inside `sim`.
 *
 * \param   sim     The simulator to save.
 * \param   time    The simulation time of `sim`.
 * \returns `true` if successful; and `false` otherwise.
 */
bool save(const simulator<simulator_data>* sim, uint64_t time)
{
  int length = snprintf(nullptr, 0, "%" PRIu64, time);
  if (length < 0) {
    fprintf(stderr, "save ERROR: Error computing filepath to save simulation.\n");
    return false;
  }

  const simulator_data& data = sim->get_data();
  char* filepath = (char*) malloc(sizeof(char) * (data.save_directory_length + length + 1));
  if (filepath == nullptr) {
    fprintf(stderr, "save ERROR: Insufficient memory for filepath.\n");
    return false;
  }

  for (unsigned int i = 0; i < data.save_directory_length; i++)
    filepath[i] = data.save_directory[i];
  snprintf(filepath + data.save_directory_length, length + 1, "%" PRIu64, time);

  FILE* file = open_file(filepath, "wb");
  if (file == nullptr) {
    fprintf(stderr, "save ERROR: Unable to open '%s' for writing. ", filepath);
    perror(""); return false;
  }

  fixed_width_stream<FILE*> out(file);
  bool result = write(*sim, out)
             && write(data.agent_ids.length, out)
             && write(data.agent_ids.data, out, data.agent_ids.length);
  fclose(file);
  return result;
}


/**
 * The callback function invoked by the simulator when time is advanced. This
 * function is only called if the simulator is run locally or as a server. This
 * function first checks if the simulator should be saved to file. Next, in
 * server mode, the simulator sends a step response message to all connected
 * clients. Finally, it constructs a list of agent states and invokes the
 * callback in `data.callback`.
 *
 * \param   sim     The simulator invoking this function.
 * \param   agents  The underlying array of all agents in `sim`.
 * \param   time    The new simulation time of `sim`.
 */
void on_step(const simulator<simulator_data>* sim,
        const array<agent_state*>& agents, uint64_t time)
{
  bool saved = false;
  const simulator_data& data = sim->get_data();
  if (data.save_directory != nullptr && time % data.save_frequency == 0) {
    /* save the simulator to a local file */
    saved = save(sim, time);
  } if (data.server != nullptr) {
    /* this simulator is a server, so send a step response to every client */
    if (!send_step_response(*data.server, agents, sim->get_config(), saved))
      fprintf(stderr, "on_step ERROR: send_step_response failed.\n");
  }

  AgentSimulationState* agent_states = (AgentSimulationState*) malloc(sizeof(AgentSimulationState) * data.agent_ids.length);
  if (agent_states == nullptr) {
    fprintf(stderr, "on_step ERROR: Insufficient memory for agent_states.\n");
    return;
  }
  const simulator_config& config = sim->get_config();
  for (size_t i = 0; i < data.agent_ids.length; i++) {
    if (!init(agent_states[i], *agents[(size_t) data.agent_ids[i]], config, data.agent_ids[i])) {
      fprintf(stderr, "on_step ERROR: Insufficient memory for agent_state.\n");
      for (size_t j = 0; j < i; j++) free(agent_states[i]);
      free(agent_states); return;
    }
  }

  /* invoke callback */
  data.callback(data.callback_data, agent_states, data.agent_ids.length, saved);

  for (size_t i = 0; i < data.agent_ids.length; i++)
    free(agent_states[i]);
  free(agent_states);
}


/**
 * Client callback functions.
 */

/**
 * The callback invoked when the client receives an add_agent response from the
 * server. This function copies the agent state into an AgentSimulationState object,
 * stores it in `c.data.response.agent_state`, and wakes up the parent thread
 * (which should be waiting in the `simulatorAddAgent` function) so that it can
 * return the response.
 *
 * \param   c         The client that received the response.
 * \param   agent_id  The ID of the new agent. This is equal to `UINT64_MAX` if
 *                    the server returned an error.
 * \param   new_agent The state of the new agent.
 */
void on_add_agent(client<client_data>& c,
        uint64_t agent_id, const agent_state& new_agent)
{
  AgentSimulationState new_agent_state;
  if (!init(new_agent_state, new_agent, c.config, agent_id))
    new_agent_state = EMPTY_AGENT_SIM_STATE;

  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response.agent_state = new_agent_state;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a move response from the
 * server. This function copies the result into `c.data.response.action_result`
 * and wakes up the parent thread (which should be waiting in the
 * `simulatorMoveAgent` function) so that it can return the response.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to move.
 * \param   request_success Indicates whether the move request was successfully
 *                          enqueued by the simulator server.
 */
void on_move(client<client_data>& c, uint64_t agent_id, bool request_success) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response.action_result = request_success;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a turn response from the
 * server. This function copies the result into `c.data.response.action_result`
 * and wakes up the parent thread (which should be waiting in the
 * `simulatorTurnAgent` function) so that it can return the response.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to turn.
 * \param   request_success Indicates whether the turn request was successfully
 *                          enqueued by the simulator server.
 */
void on_turn(client<client_data>& c, uint64_t agent_id, bool request_success) {
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response.action_result = request_success;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a get_map response from the
 * server. This function moves the result into `c.data.response.map` and wakes
 * up the parent thread (which should be waiting in the `simulatorMap`
 * function) so that it can return the response back.
 *
 * \param   c       The client that received the response.
 * \param   map     A map from patch positions to `patch_state` structures
 *                  containing the state information in each patch.
 */
void on_get_map(client<client_data>& c,
        hash_map<position, patch_state>* map)
{
  std::unique_lock<std::mutex> lck(c.data.lock);
  c.data.waiting_for_server = false;
  c.data.response.map = map;
  c.data.cv.notify_one();
}


/**
 * The callback invoked when the client receives a step response from the
 * server. This function constructs a list of AgentSimulationState objects governed by
 * this client and invokes the function `c.data.step_callback`.
 *
 * \param   c            The client that received the response.
 * \param   agent_ids    An array of agent IDs governed by the client.
 * \param   agent_states An array, parallel to `agent_ids`, containing the
 *                       state information of each agent at the beginning of
 *                       the new time step in the simulation.
 */
void on_step(client<client_data>& c,
        const array<uint64_t>& agent_ids,
        const agent_state* agent_states)
{
  bool saved;
  if (!read(saved, c.connection)) return;

  AgentSimulationState* agents = (AgentSimulationState*) malloc(sizeof(AgentSimulationState) * agent_ids.length);
  if (agents == nullptr) {
    fprintf(stderr, "on_step ERROR: Insufficient memory for agents.\n");
    return;
  }
  for (size_t i = 0; i < agent_ids.length; i++) {
    if (!init(agents[i], agent_states[i], c.config, agent_ids[i])) {
      fprintf(stderr, "on_step ERROR: Insufficient memory for agent.\n");
      for (size_t j = 0; j < i; j++) free(agents[i]);
      free(agents); return;
    }
  }

  c.data.step_callback(c.data.callback_data, agents, agent_ids.length, saved);

  for (size_t i = 0; i < agent_ids.length; i++)
    free(agents[i]);
  free(agents);
}


/**
 * The callback invoked when the client loses the connection to the server.
 * \param   c       The client whose connection to the server was lost.
 */
void on_lost_connection(client<client_data>& c) {
  fprintf(stderr, "Client lost connection to server.\n");
  c.client_running = false;
  c.data.cv.notify_one();

  /* invoke callback */
  c.data.lost_connection_callback(c.data.callback_data);
}


/**
 * This functions waits for a response from the server, and for one of the
 * above client callback functions to be invoked.
 * \param   c       The client expecting a response from the server.
 */
inline void wait_for_server(client<client_data>& c)
{
    std::unique_lock<std::mutex> lck(c.data.lock);
    while (c.data.waiting_for_server && c.client_running)
        c.data.cv.wait(lck);
}


void* simulatorCreate(
  const SimulatorConfig* config, 
  OnStepCallback onStepCallback,
  void* callbackData,
  unsigned int saveFrequency,
  const char* savePath)
{
  simulator_config sim_config;

  sim_config.agent_color = (float*) malloc(sizeof(float) * config->colorDimSize);
  if (sim_config.agent_color == nullptr)
    /* TODO: how to communicate out of memory errors to swift? */
    return nullptr;

  for (unsigned int i = 0; i < (size_t) DirectionCount; i++)
    sim_config.allowed_movement_directions[i] = config->allowedMoveDirections[i];
  for (unsigned int i = 0; i < (size_t) DirectionCount; i++)
    sim_config.allowed_rotations[i] = config->allowedRotations[i];
  for (unsigned int i = 0; i < config->colorDimSize; i++)
    sim_config.agent_color[i] = config->agentColor[i];

  if (!sim_config.item_types.ensure_capacity(max(1u, config->numItemTypes)))
    /* TODO: how to communicate out of memory errors to swift? */
    return nullptr;
  for (unsigned int i = 0; i < config->numItemTypes; i++) {
    if (!init(sim_config.item_types[i], config->itemTypes[i], config->scentDimSize, config->colorDimSize, config->numItemTypes)) {
      /* TODO: how to communicate out of memory errors to swift? */
      for (unsigned int j = 0; j < i; j++)
        core::free(sim_config.item_types[i], config->numItemTypes);
      return nullptr;
    }
  }
  sim_config.item_types.length = config->numItemTypes;

  sim_config.max_steps_per_movement = config->maxStepsPerMove;
  sim_config.scent_dimension = config->scentDimSize;
  sim_config.color_dimension = config->colorDimSize;
  sim_config.vision_range = config->visionRange;
  sim_config.patch_size = config->patchSize;
  sim_config.gibbs_iterations = config->gibbsIterations;
  sim_config.collision_policy = to_movement_conflict_policy(config->movementConflictPolicy);
  sim_config.decay_param = config->scentDecay;
  sim_config.diffusion_param = config->scentDiffusion;
  sim_config.deleted_item_lifetime = config->removedItemLifetime;

  simulator_data data(savePath,
      (savePath == nullptr) ? 0 : strlen(savePath),
      saveFrequency, nullptr, onStepCallback, callbackData);

  simulator<simulator_data>* sim =
      (simulator<simulator_data>*) malloc(sizeof(simulator<simulator_data>));
  if (sim == nullptr) {
    /* TODO: how to communicate out of memory errors to swift? */
    return nullptr;
  } else if (!init(*sim, sim_config, data, config->randomSeed)) {
    /* TODO: communicate simulator initialization error */
    return nullptr;
  }
  return (void*) sim;
}


SimulatorInfo simulatorLoad(
  const char* filePath, 
  OnStepCallback onStepCallback,
  void* callbackData,
  unsigned int saveFrequency,
  const char* savePath)
{
  simulator<simulator_data>* sim =
      (simulator<simulator_data>*) malloc(sizeof(simulator<simulator_data>));
  if (sim == nullptr) {
    /* TODO: how to communicate out of memory errors to swift? */
    return EMPTY_SIM_INFO;
  }

  simulator_data data(savePath,
      (savePath == nullptr) ? 0 : strlen(savePath),
      saveFrequency, nullptr, onStepCallback, callbackData);

  FILE* file = open_file(filePath, "rb");
  if (file == nullptr) {
    /* TODO: communicate i/o error */
    free(sim); return EMPTY_SIM_INFO;
  }
  size_t agent_id_count;
  fixed_width_stream<FILE*> in(file);
  if (!read(*sim, in, data)) {
    /* TODO: communicate "Failed to load simulator." error */
    free(sim); fclose(file); return EMPTY_SIM_INFO;
  }
  simulator_data& sim_data = sim->get_data();
  if (!read(agent_id_count, in)
   || !sim_data.agent_ids.ensure_capacity(agent_id_count)
   || !read(sim_data.agent_ids.data, in, agent_id_count))
  {
    /* TODO: communicate "Failed to load agent IDs." error */
    free(*sim); free(sim); fclose(file); return EMPTY_SIM_INFO;
  }
  sim_data.agent_ids.length = agent_id_count;
  fclose(file);

  agent_state** agent_states = (agent_state**) malloc(sizeof(agent_state*) * agent_id_count);
  if (agent_states == nullptr) {
    /* TODO: how to communicate out of memory errors to swift? */
    free(*sim); free(sim); return EMPTY_SIM_INFO;
  }

  sim->get_agent_states(agent_states, sim_data.agent_ids.data, (unsigned int) agent_id_count);

  const simulator_config& config = sim->get_config();
  AgentSimulationState* agents = (AgentSimulationState*) malloc(sizeof(AgentSimulationState) * agent_id_count);
  if (agents == nullptr) {
    /* TODO: how to communicate out of memory errors to swift? */
    free(*sim); free(sim); free(agent_states);
    return EMPTY_SIM_INFO;
  }
  for (size_t i = 0; i < agent_id_count; i++) {
    if (!init(agents[i], *agent_states[i], config, sim_data.agent_ids[i])) {
      for (size_t j = 0; j < i; j++) free(agents[i]);
      free(*sim); free(sim); free(agent_states);
      free(agents); return EMPTY_SIM_INFO;
    }
  }
  free(agent_states);

  SimulatorInfo sim_info;
  sim_info.handle = (void*) sim;
  sim_info.time = sim->time;
  sim_info.agents = agents;
  sim_info.numAgents = (unsigned int) agent_id_count;
  return sim_info;
}


void simulatorDelete(
  void* simulatorHandle)
{
  simulator<simulator_data>* sim =
      (simulator<simulator_data>*) simulatorHandle;
  free(*sim); free(sim);
}


AgentSimulationState simulatorAddAgent(
  void* simulatorHandle,
  void* clientHandle)
{
  if (clientHandle == nullptr) {
    /* the simulation is local, so call add_agent directly */
    simulator<simulator_data>* sim_handle =
        (simulator<simulator_data>*) simulatorHandle;
    pair<uint64_t, agent_state*> new_agent = sim_handle->add_agent();
    if (new_agent.key == UINT64_MAX) {
      /* TODO: communicate the error "Failed to add new agent." to swift */
      return EMPTY_AGENT_SIM_STATE;
    }

    sim_handle->get_data().agent_ids.add(new_agent.key);

    AgentSimulationState new_agent_state;
    std::unique_lock<std::mutex> lock(new_agent.value->lock);
    if (!init(new_agent_state, *new_agent.value, sim_handle->get_config(), new_agent.key)) {
      return EMPTY_AGENT_SIM_STATE;
    }
    return new_agent_state;
  } else {
    /* this is a client, so send an add_agent message to the server */
    client<client_data>* client_ptr =
        (client<client_data>*) clientHandle;
    if (!client_ptr->client_running) {
      /* TODO: communicate "Connection to the server was lost." error to swift */
      return EMPTY_AGENT_SIM_STATE;
    }

    client_ptr->data.waiting_for_server = true;
    if (!send_add_agent(*client_ptr)) {
      /* TODO: communicate "Unable to send add_agent request." error to swift */
      return EMPTY_AGENT_SIM_STATE;
    }

    /* wait for response from server */
    wait_for_server(*client_ptr);

    return client_ptr->data.response.agent_state;
  }
}


bool simulatorMove(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  Direction direction,
  unsigned int numSteps)
{
  if (clientHandle == nullptr) {
    /* the simulation is local, so call move directly */
    simulator<simulator_data>* sim_handle =
        (simulator<simulator_data>*) simulatorHandle;
    return sim_handle->move(agentId, to_direction(direction), numSteps);
  } else {
    /* this is a client, so send a move message to the server */
    client<client_data>* client_ptr =
        (client<client_data>*) clientHandle;
    if (!client_ptr->client_running) {
      /* TODO: communicate "Connection to the server was lost." to swift */
      return false;
    }

    client_ptr->data.waiting_for_server = true;
    if (!send_move(*client_ptr, agentId, to_direction(direction), numSteps)) {
      /* TODO: communicate "Unable to send move request." error to swift */
      return false;
    }

    /* wait for response from server */
    wait_for_server(*client_ptr);

    return client_ptr->data.response.action_result;
  }
}


bool simulatorTurn(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  TurnDirection direction)
{
  if (clientHandle == nullptr) {
    /* the simulation is local, so call turn directly */
    simulator<simulator_data>* sim_handle =
        (simulator<simulator_data>*) simulatorHandle;
    return sim_handle->turn(agentId, to_direction(direction));
  } else {
    /* this is a client, so send a turn message to the server */
    client<client_data>* client_ptr =
        (client<client_data>*) clientHandle;
    if (!client_ptr->client_running) {
      /* TODO: communicate "Connection to the server was lost." to swift */
      return false;
    }

    client_ptr->data.waiting_for_server = true;
    if (!send_turn(*client_ptr, agentId, to_direction(direction))) {
      /* TODO: communicate "Unable to send turn request." error to swift */
      return false;
    }

    /* wait for response from server */
    wait_for_server(*client_ptr);

    return client_ptr->data.response.action_result;
  }
}


const SimulationMap simulatorMap(
  void* simulatorHandle,
  void* clientHandle,
  Position bottomLeftCorner,
  Position topRightCorner)
{
  position bottom_left = position(bottomLeftCorner.x, bottomLeftCorner.y);
  position top_right = position(topRightCorner.x, topRightCorner.y);

  if (clientHandle == nullptr) {
    /* the simulation is local, so call get_map directly */
    simulator<simulator_data>* sim_handle =
        (simulator<simulator_data>*) simulatorHandle;
    hash_map<position, patch_state> patches(16, alloc_position_keys);
    if (!sim_handle->get_map(bottom_left, top_right, patches)) {
      /* TODO: communicate "simulator.get_map failed." to swift */
      return EMPTY_SIM_MAP;
    }

    SimulationMap map;
    if (!init(map, patches, sim_handle->get_config()))
      map = EMPTY_SIM_MAP;
    for (auto entry : patches)
      free(entry.value);
    return map;
  } else {
    /* this is a client, so send a get_map message to the server */
    client<client_data>* client_ptr =
        (client<client_data>*) clientHandle;
    if (!client_ptr->client_running) {
      /* TODO: communicate "Connection to the server was lost." error to swift */
      return EMPTY_SIM_MAP;
    }

    client_ptr->data.waiting_for_server = true;
    if (!send_get_map(*client_ptr, bottom_left, top_right)) {
      /* TODO: communicate "Unable to send get_map request." error to swift */
      return EMPTY_SIM_MAP;
    }

    /* wait for response from server */
    wait_for_server(*client_ptr);
    SimulationMap map;
    if (!init(map, *client_ptr->data.response.map, client_ptr->config))
      map = EMPTY_SIM_MAP;
    for (auto entry : *client_ptr->data.response.map)
      free(entry.value);
    free(*client_ptr->data.response.map);
    free(client_ptr->data.response.map);
    return map;
  }
}


void* simulationServerStart(
  void* simulatorHandle,
  unsigned int port,
  unsigned int connectionQueueCapacity,
  unsigned int numWorkers)
{
  simulator<simulator_data>* sim_handle =
      (simulator<simulator_data>*) simulatorHandle;
  async_server* server = (async_server*) malloc(sizeof(async_server));
  if (server == nullptr || !init(*server)) {
    /* TODO: communicate out of memory errors to swift */
    if (server != nullptr) free(server);
    return nullptr;
  } else if (!init_server(*server, *sim_handle, (uint16_t) port, connectionQueueCapacity, numWorkers)) {
    /* TODO: communicate "Unable to initialize MPI server." error to swift */
    free(*server); free(server); return nullptr;
  }
  sim_handle->get_data().server = server;
  return (void*) server;
}


void simulationServerStop(
  void* serverHandle)
{
  async_server* server = (async_server*) serverHandle;
  stop_server(*server);
  free(*server); free(server);
}


SimulationClientInfo simulationClientStart(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
  void* callbackData,
  const uint64_t* agents,
  unsigned int numAgents)
{
  agent_state* agent_states = (agent_state*) malloc(sizeof(agent_state) * numAgents);
  if (agent_states == nullptr) {
    /* TODO: communicate out of memory errors to swift */
    return EMPTY_CLIENT_INFO;
  }

  client<client_data>* new_client =
            (client<client_data>*) malloc(sizeof(client<client_data>));
  if (new_client == nullptr || !init(*new_client)) {
    /* TODO: communicate out of memory errors to swift */
    if (new_client != nullptr) free(new_client);
    free(agent_states); return EMPTY_CLIENT_INFO;
  }

  uint64_t simulator_time = init_client(*new_client, serverAddress,
      (uint16_t) serverPort, agents, agent_states, numAgents);
  if (simulator_time == UINT64_MAX) {
    /* TODO: communicate "Unable to initialize MPI client." error to swift */
    free(*new_client); free(new_client); return EMPTY_CLIENT_INFO;
  }

  AgentSimulationState* agentStates = (AgentSimulationState*) malloc(sizeof(AgentSimulationState) * numAgents);
  if (agentStates == nullptr) {
    /* TODO: how to communicate out of memory errors to swift? */
    for (unsigned int i = 0; i < numAgents; i++) free(agent_states[i]);
    free(agent_states); stop_client(*new_client);
    free(*new_client); free(new_client); return EMPTY_CLIENT_INFO;
  }
  for (size_t i = 0; i < numAgents; i++) {
    if (!init(agentStates[i], agent_states[i], new_client->config, agents[i])) {
      for (size_t j = 0; j < i; j++) free(agentStates[i]);
      for (unsigned int j = i; j < numAgents; j++) free(agent_states[j]);
      free(agentStates); free(agent_states); stop_client(*new_client);
      free(*new_client); free(new_client); return EMPTY_CLIENT_INFO;
    }
    free(agent_states[i]);
  }
  free(agent_states);

  new_client->data.step_callback = onStepCallback;
  new_client->data.lost_connection_callback = lostConnectionCallback;
  new_client->data.callback_data = callbackData;

  SimulationClientInfo client_info;
  client_info.handle = (void*) new_client;
  client_info.simulationTime = simulator_time;
  client_info.agentStates = agentStates;
  return client_info;
}


void simulationClientStop(
  void* clientHandle)
{
  client<client_data>* client_ptr =
      (client<client_data>*) clientHandle;
  stop_client(*client_ptr);
  free(*client_ptr); free(client_ptr);
}


void simulatorDeleteSimulatorInfo(
  SimulatorInfo info)
{
  for (unsigned int i = 0; i < info.numAgents; i++)
    free(info.agents[i]);
  free(info.agents);
}


void simulatorDeleteSimulationClientInfo(
  SimulationClientInfo clientInfo,
  unsigned int numAgents)
{
  for (unsigned int i = 0; i < numAgents; i++)
    free(clientInfo.agentStates[i]);
  free(clientInfo.agentStates);
}


void simulatorDeleteAgentSimulationState(
  AgentSimulationState agentState)
{
  free(agentState);
}


void simulatorDeleteSimulationMap(
  SimulationMap map)
{
  for (unsigned int i = 0; i < map.numPatches; i++)
    free(map.patches[i]);
  free(map.patches);
}

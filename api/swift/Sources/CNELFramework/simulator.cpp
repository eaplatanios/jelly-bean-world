#include <thread>

#include "include/simulator.h"

#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

using namespace core;
using namespace nel;

constexpr SimulatorInfo EMPTY_SIM_INFO = { nullptr, 0, nullptr, 0 };
constexpr SimulationClientInfo EMPTY_CLIENT_INFO = { nullptr, 0, nullptr };


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
  item_properties& properties, const ItemProperties& src,
  unsigned int scent_dimension, unsigned int color_dimension,
  unsigned int item_type_count)
{
  return init(properties, src.name, strlen(src.name),
    src.scent, src.color, src.requiredItemCounts,
    src.requiredItemCosts, src.blocksMovement,
    reinterpret_cast<intensity_function>(src.intensityFn), 
    reinterpret_cast<interaction_function*>(src.interactionFns), 
    src.intensityFnArgs, src.interactionFnArgs,
    src.intensityFnArgCount, src.interactionFnArgCounts,
    scent_dimension, color_dimension, item_type_count);
}


inline bool init(
  AgentState& state,
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


inline void free(AgentState& state) {
  free(state.scent);
  free(state.vision);
  free(state.collectedItems);
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

  /* agents owned by the simulator */
  array<uint64_t> agent_ids;

  simulator_data(
      const char* save_filepath,
      unsigned int save_filepath_length,
      unsigned int save_frequency,
      async_server* server,
      OnStepCallback callback) :
    save_frequency(save_frequency), server(server),
    callback(callback), agent_ids(16)
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
    AgentState* agent_state;
    hash_map<position, patch_state>* map;
  } response;

  /* for synchronization */
  bool waiting_for_server;
  std::mutex lock;
  std::condition_variable cv;

  OnStepCallback step_callback;
  LostConnectionCallback lost_connection_callback;

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

  AgentState* agent_states = (AgentState*) malloc(sizeof(AgentState) * data.agent_ids.length);
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
  data.callback(agent_states, data.agent_ids.length, saved);

  for (size_t i = 0; i < data.agent_ids.length; i++)
    free(agent_states[i]);
  free(agent_states);
}


/**
 * Client callback functions.
 */

/**
 * The callback invoked when the client receives an add_agent response from the
 * server. This function copies the agent state into an AgentState object,
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
  AgentState* new_agent_state = (AgentState*) malloc(sizeof(AgentState));
  if (new_agent_state == nullptr) {
    fprintf(stderr, "on_add_agent ERROR: Out of memory.\n");
  } else if (!init(*new_agent_state, new_agent, c.config, agent_id)) {
    free(new_agent_state);
    new_agent_state = nullptr;
  }

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
 * server. This function constructs a list of AgentState objects governed by
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

  AgentState* agents = (AgentState*) malloc(sizeof(AgentState) * agent_ids.length);
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

  c.data.step_callback(agents, agent_ids.length, saved);

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
  c.data.lost_connection_callback();
}


void* simulatorCreate(
  const SimulatorConfig* config, 
  OnStepCallback onStepCallback,
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
      saveFrequency, nullptr, onStepCallback);

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
      saveFrequency, nullptr, onStepCallback);

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
  AgentState* agents = (AgentState*) malloc(sizeof(AgentState) * agent_id_count);
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
  void* simulator_handle)
{
  simulator<simulator_data>* sim =
      (simulator<simulator_data>*) simulator_handle;
  free(*sim); free(sim);
}


AgentState simulatorAddAgent(
  void* simulator_handle,
  void* client_handle);


bool simulatorMove(
  void* simulator_handle,
  void* client_handle,
  uint64_t agentId,
  Direction direction,
  unsigned int numSteps);


bool simulatorTurn(
  void* simulator_handle,
  void* client_handle,
  uint64_t agentId,
  TurnDirection direction);


const SimulationMap simulatorMap(
  void* simulator_handle,
  void* client_handle,
  const Position* bottomLeftCorner,
  const Position* topRightCorner);


void* simulationServerStart(
  void* simulator_handle,
  unsigned int port,
  unsigned int connectionQueueCapacity,
  unsigned int numWorkers)
{
  simulator<simulator_data>* sim_handle =
      (simulator<simulator_data>*) simulator_handle;
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
  void* server_handle)
{
  async_server* server = (async_server*) server_handle;
  stop_server(*server);
  free(*server); free(server);
}


SimulationClientInfo simulationClientStart(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
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

  AgentState* agentStates = (AgentState*) malloc(sizeof(AgentState) * numAgents);
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

  SimulationClientInfo client_info;
  client_info.handle = (void*) new_client;
  client_info.simulationTime = simulator_time;
  client_info.agentStates = agentStates;
  return client_info;
}


void simulationClientStop(
  void* client_handle)
{
  client<client_data>* client_ptr =
      (client<client_data>*) client_handle;
  stop_client(*client_ptr);
  free(*client_ptr); free(client_ptr);
}

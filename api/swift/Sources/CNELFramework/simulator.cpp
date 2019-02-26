#include <thread>

#include "include/simulator.h"

#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

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
    src.intensityFn, src.interactionFns,
    src.intensityFnArgs, src.interactionFnArgs,
    src.intensityFnArgCount, src.interactionFnArgCounts,
    scent_dimension, color_dimension, item_type_count);
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

    simulator_data(const char* save_filepath,
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

void* simulatorLoad(
  const char* filePath, 
  OnStepCallback onStepCallback,
  unsigned int saveFrequency,
  const char* savePath);

void simulatorDelete(
  void* simulator_handle);

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
  unsigned int numWorkers);

void simulationServerStop(
  void* server_handle);

SimulationClientInfo simulationClientStart(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
  const uint64_t* agents,
  unsigned int numAgents);

void simulationClientStop(
  void* client_handle);

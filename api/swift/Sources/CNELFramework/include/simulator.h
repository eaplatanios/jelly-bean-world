#include <inttypes.h>
#include <stdbool.h>

#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

using namespace core;
using namespace nel;

/** Represents all possible directions of motion 
 * in the environment. */
enum Direction {
  DirectionUp = 0,
  DirectionDown,
  DirectionLeft,
  DirectionRight,
  DirectionCount
};

static_assert((size_t) DirectionCount == (size_t) direction::COUNT, "DirectionCount is not equal to direction::COUNT");

/** Represents all possible directions of turning 
 * in the environment. */
enum TurnDirection {
  TurnDirectionNoChange = 0,
  TurnDirectionReverse,
  TurnDirectionLeft,
  TurnDirectionRight
};

enum MovementConflictPolicy {
  MovementConflictPolicyNoCollisions = 0,
  MovementConflictPolicyFirstComeFirstServe,
  MovementConflictPolicyRandom
};

struct Position {
  int64_t x;
  int64_t y;
};

typedef void (*OnStepCallback)();
typedef void (*LostConnectionCallback)();

/** A structure containing the properties of an item type. */
struct ItemProperties {
  const char* name;

  float* scent;
  float* color;

  unsigned int* requiredItemCounts;
  unsigned int* requiredItemCosts;

  bool blocksMovement;

  intensity_function intensityFn;
  interaction_function* interactionFns;

  float* intensityFnArgs;
  float** interactionFnArgs;
  unsigned int intensityFnArgCount;
  unsigned int* interactionFnArgCounts;
};

struct AgentState {
  Position position;
  Direction direction;
  float* scent;
  float* vision;
  unsigned int* collectedItems;
  uint64_t id;
};

struct SimulatorConfig {
  /* Simulation Parameters */
  unsigned int randomSeed;

  /* Agent Capabilities */
  unsigned int maxStepsPerMove;
  unsigned int scentDimSize;
  unsigned int colorDimSize;
  unsigned int visionRange;
  bool allowedMoveDirections[(size_t) DirectionCount];
  bool allowedRotations[(size_t) DirectionCount];

  /* World Properties */
  unsigned int patchSize;
  unsigned int gibbsIterations;
  unsigned int numItems;
  ItemProperties* itemTypes;
  unsigned int numItemTypes;
  float* agentColor;
  MovementConflictPolicy movementConflictPolicy;

  /* Scent Diffusion Parameters */
  float scentDecay;
  float scentDiffusion;
  unsigned int removedItemLifetime;
};

struct ItemInfo {
  unsigned int type;
  Position position;
};

struct AgentInfo {
  Position position;
  Direction direction;
};

struct SimulationMapPatch {
  Position position;
  bool fixed;
  float* scent;
  float* vision;
  ItemInfo* items;
  unsigned int numItems;
  AgentInfo* agents;
  unsigned int numAgents;
};

struct SimulationMap {
  SimulationMapPatch* patches;
  unsigned int numPatches;
};

struct SimulationClientInfo {
  void* handle;
  uint64_t simulationTime;
  AgentState* agentStates;
  unsigned int numAgents;
};

extern "C" {

void* simulatorCreate(
  const SimulatorConfig* config, 
  OnStepCallback onStepCallback,
  unsigned int saveFrequency,
  const char* savePath);

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

} /* extern "C" */

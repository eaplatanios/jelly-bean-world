#include <inttypes.h>
#include <stdbool.h>

/** Represents all possible directions of motion
 *  in the environment. */
typedef enum Direction {
  DirectionUp = 0,
  DirectionDown,
  DirectionLeft,
  DirectionRight,
  DirectionCount
} Direction;

/** Represents all possible directions of turning
 *  in the environment. */
typedef enum TurnDirection {
  TurnDirectionNoChange = 0,
  TurnDirectionReverse,
  TurnDirectionLeft,
  TurnDirectionRight
} TurnDirection;

typedef enum MovementConflictPolicy {
  MovementConflictPolicyNoCollisions = 0,
  MovementConflictPolicyFirstComeFirstServe,
  MovementConflictPolicyRandom
} MovementConflictPolicy;

typedef struct Position {
  int64_t x;
  int64_t y;
} Position;

typedef struct IntensityFunction {
  unsigned int id;
  const float* args;
  unsigned int numArgs;
} IntensityFunction;

typedef struct InteractionFunction {
  unsigned int id;
  const float* args;
  unsigned int numArgs;
  unsigned int itemId;
} InteractionFunction;

typedef struct EnergyFunctions {
  IntensityFunction intensityFn;
  const InteractionFunction* interactionFns;
} EnergyFunctions;

/** A structure containing the properties of an item type. */
typedef struct ItemProperties {
  const char* name;

  const float* scent;
  const float* color;

  const unsigned int* requiredItemCounts;
  const unsigned int* requiredItemCosts;

  bool blocksMovement;

  EnergyFunctions energyFunctions;
} ItemProperties;

typedef struct AgentSimulationState {
  uint64_t id;
  Position position;
  Direction direction;
  float* scent;
  float* vision;
  unsigned int* collectedItems;
} AgentSimulationState;

typedef void (*OnStepCallback)(void*, const AgentSimulationState*, unsigned int, bool);
typedef void (*LostConnectionCallback)(void*);

typedef struct SimulatorConfig {
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
  const ItemProperties* itemTypes;
  unsigned int numItemTypes;
  const float* agentColor;
  MovementConflictPolicy movementConflictPolicy;

  /* Scent Diffusion Parameters */
  float scentDecay;
  float scentDiffusion;
  unsigned int removedItemLifetime;
} SimulatorConfig;

typedef struct SimulatorInfo {
  void* handle;
  uint64_t time;
  AgentSimulationState* agents;
  unsigned int numAgents;
} SimulatorInfo;

typedef struct ItemInfo {
  unsigned int type;
  Position position;
} ItemInfo;

typedef struct AgentInfo {
  Position position;
  Direction direction;
} AgentInfo;

typedef struct SimulationMapPatch {
  Position position;
  bool fixed;
  float* scent;
  float* vision;
  ItemInfo* items;
  unsigned int numItems;
  AgentInfo* agents;
  unsigned int numAgents;
} SimulationMapPatch;

typedef struct SimulationMap {
  SimulationMapPatch* patches;
  unsigned int numPatches;
} SimulationMap;

typedef struct SimulationClientInfo {
  void* handle;
  uint64_t simulationTime;
  AgentSimulationState* agentStates;
} SimulationClientInfo;

void* simulatorCreate(
  const SimulatorConfig* config,
  OnStepCallback onStepCallback,
  void* callbackData,
  unsigned int saveFrequency,
  const char* savePath);

SimulatorInfo simulatorLoad(
  const char* filePath,
  OnStepCallback onStepCallback,
  void* callbackData,
  unsigned int saveFrequency,
  const char* savePath);

void simulatorDelete(
  void* simulatorHandle);

AgentSimulationState simulatorAddAgent(
  void* simulatorHandle,
  void* clientHandle);

bool simulatorMoveAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  Direction direction,
  unsigned int numSteps);

bool simulatorTurnAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  TurnDirection direction);

SimulationMap simulatorMap(
  const void* simulatorHandle,
  const void* clientHandle,
  Position bottomLeftCorner,
  Position topRightCorner);

void* simulationServerStart(
  void* simulatorHandle,
  unsigned int port,
  unsigned int connectionQueueCapacity,
  unsigned int numWorkers);

void simulationServerStop(
  void* serverHandle);

SimulationClientInfo simulationClientStart(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
  void* callbackData,
  const uint64_t* agents,
  unsigned int numAgents);

void simulationClientStop(
  void* clientHandle);

void simulatorDeleteSimulatorInfo(
  SimulatorInfo info);

void simulatorDeleteSimulationClientInfo(
  SimulationClientInfo clientInfo,
  unsigned int numAgents);

void simulatorDeleteAgentSimulationState(
  AgentSimulationState agentState);

void simulatorDeleteSimulationMap(
  SimulationMap map);

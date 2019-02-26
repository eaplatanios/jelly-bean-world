#include <inttypes.h>
#include <stdbool.h>

/** Represents all possible directions of motion 
 * in the environment. */
typedef enum Direction {
  DirectionUp = 0,
  DirectionDown,
  DirectionLeft,
  DirectionRight,
  DirectionCount
} Direction;

/** Represents all possible directions of turning 
 * in the environment. */
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

typedef void (*OnStepCallback)();
typedef void (*LostConnectionCallback)();
typedef float (*intensityFunction)(const Position, const float*);
typedef float (*interactionFunction)(const Position, const Position, const float*);

typedef struct Item {
  unsigned int type;
  Position position;
  uint64_t creationTime;
  uint64_t deletionTime;
} Item;

/** A structure containing the properties of an item type. */
typedef struct ItemProperties {
  const char* name;

  float* scent;
  float* color;

  unsigned int* requiredItemCounts;
  unsigned int* requiredItemCosts;

  bool blocksMovement;

  intensityFunction intensityFn;
  interactionFunction* interactionFns;

  float* intensityFnArgs;
  float** interactionFnArgs;
  unsigned int intensityFnArgCount;
  unsigned int* interactionFnArgCounts;
} ItemProperties;

typedef struct AgentState {
  Position position;
  Direction direction;
  float* scent;
  float* vision;
  bool acted;
  Position requestedPosition;
  Direction requestedDirection;
  unsigned int* collectedItems;
  // TODO: Do we need to store the mutex here?
} AgentState;

typedef struct SimulatorConfig {
  /* Simulation Parameters */
  int randomSeed;

  /* Agent Capabilities */
  int maxStepsPerMove;
  int scentDimSize;
  int colorDimSize;
  int visionRange;
  bool allowedMoveDirections[(size_t) DirectionCount];
  bool allowedRotations[(size_t) DirectionCount];

  /* World Properties */
  unsigned int patchSize;
  unsigned int gibbsIterations;
  unsigned int numItems;
  ItemProperties* itemTypes;
  float* agentColor;
  MovementConflictPolicy movementConflictPolicy;

  /* Scent Diffusion Parameters */
  float scentDecay;
  float scentDiffusion;
  unsigned int removedItemLifetime;
} SimulatorConfig;

typedef struct Simulator {
  uint64_t simulationTime;
  void* handle;
  AgentState* agentStates;
  unsigned int numAgents;
} Simulator;

typedef struct SimulationServer {
  void* handle;
} SimulationServer;

typedef struct SimulationClient {
  void* handle;
  uint64_t simulationTime;
  AgentState* agentStates;
  unsigned int numAgents;
} SimulationClient;

typedef struct SimulationMapPatch {
  Position position;
  bool fixed;
  float* scent;
  float* vision;
  Item* items;
  unsigned int numItems;
} SimulationMapPatch;

typedef struct SimulationMap {
  SimulationMapPatch* patches;
  unsigned int numPatches;
} SimulationMap;

Simulator simulatorCreate(
  SimulatorConfig* config, 
  OnStepCallback onStepCallback,
  int saveFrequency,
  const char* savePath);

Simulator simulatorLoad(
  const char* filePath, 
  OnStepCallback onStepCallback,
  int saveFrequency,
  const char* savePath);

void simulatorDelete(Simulator* simulator);

AgentState simulatorAddAgent(
  Simulator* simulator,
  SimulationClient* client);

bool simulatorMove(
  Simulator* simulator,
  SimulationClient* client,
  uint64_t agentId,
  Direction direction,
  unsigned int numSteps);

bool simulatorTurn(
  Simulator* simulator,
  SimulationClient* client,
  uint64_t agentId,
  TurnDirection direction);

SimulationMap simulatorMap(
  Simulator* simulator,
  SimulationClient* client,
  Position* bottomLeftCorner,
  Position* topRightCorner);

SimulationServer simulationServerStart(
  Simulator* simulator,
  unsigned int port,
  unsigned int connectionQueueCapacity,
  unsigned int numWorkers);

SimulationServer simulationServerStop(
  SimulationServer* server);

SimulationClient simulationClientStart(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
  uint64_t* agents,
  unsigned int numAgents);

SimulationClient simulationClientStop(
  SimulationClient* client);

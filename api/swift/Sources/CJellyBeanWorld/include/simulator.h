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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

typedef enum ActionPolicy {
  ActionPolicyAllowed,
  ActionPolicyDisallowed,
  ActionPolicyIgnored
} ActionPolicy;

typedef struct Position {
  int64_t x;
  int64_t y;
} Position;

typedef struct IntensityFunction {
  unsigned int id;
  float* args;
  unsigned int numArgs;
} IntensityFunction;

typedef struct InteractionFunction {
  unsigned int id;
  unsigned int itemId;
  float* args;
  unsigned int numArgs;
} InteractionFunction;

typedef struct EnergyFunctions {
  IntensityFunction intensityFn;
  InteractionFunction* interactionFns;
  unsigned int numInteractionFns;
} EnergyFunctions;

/** A structure containing the properties of an item type. */
typedef struct ItemProperties {
  char* name;
  float* scent;
  float* color;
  unsigned int* requiredItemCounts;
  unsigned int* requiredItemCosts;
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

typedef void (*OnStepCallback)(const void*, const AgentSimulationState*, unsigned int);
typedef void (*LostConnectionCallback)(const void*);

typedef struct SimulatorConfig {
  /* Simulation Parameters */
  unsigned int randomSeed;

  /* Agent Capabilities */
  unsigned int maxStepsPerMove;
  unsigned int scentDimSize;
  unsigned int colorDimSize;
  unsigned int visionRange;
  ActionPolicy allowedMoveDirections[(size_t) DirectionCount];
  ActionPolicy allowedRotations[(size_t) DirectionCount];
  bool noOpAllowed;

  /* World Properties */
  unsigned int patchSize;
  unsigned int mcmcIterations;
  ItemProperties* itemTypes;
  unsigned int numItemTypes;
  float* agentColor;
  MovementConflictPolicy movementConflictPolicy;

  /* Scent Diffusion Parameters */
  float scentDecay;
  float scentDiffusion;
  unsigned int removedItemLifetime;
} SimulatorConfig;

typedef struct SimulatorInfo {
  void* handle;
  SimulatorConfig config;
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

typedef struct SimulationNewClientInfo {
  void* handle;
  uint64_t simulationTime;
  uint64_t clientId;
} SimulationNewClientInfo;

typedef struct SimulationClientInfo {
  void* handle;
  uint64_t simulationTime;
  uint64_t* agentIds;
  AgentSimulationState* agentStates;
  unsigned int numAgents;
} SimulationClientInfo;

void* simulatorCreate(
  const SimulatorConfig* config,
  OnStepCallback onStepCallback);

SimulatorInfo simulatorLoad(
  const char* filePath,
  OnStepCallback onStepCallback);

void simulatorDelete(
  void* simulatorHandle);

bool simulatorSave(
  void* simulatorHandle,
  const char* filePath);

void simulatorSetStepCallbackData(
  void* simulatorHandle,
  const void* callbackData);

AgentSimulationState simulatorAddAgent(
  void* simulatorHandle,
  void* clientHandle);

bool simulatorRemoveAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId);

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

bool simulatorNoOpAgent(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId);

void simulatorSetActive(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId,
  bool active);

bool simulatorIsActive(
  void* simulatorHandle,
  void* clientHandle,
  uint64_t agentId);

const SimulationMap simulatorMap(
  void* simulatorHandle,
  void* clientHandle,
  Position bottomLeftCorner,
  Position topRightCorner);

void* simulationServerStart(
  void* simulatorHandle,
  unsigned int port,
  unsigned int connectionQueueCapacity,
  unsigned int numWorkers);

void simulationServerStop(
  void* serverHandle);

SimulationNewClientInfo simulationClientConnect(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback);

SimulationClientInfo simulationClientReconnect(
  const char* serverAddress,
  unsigned int serverPort,
  OnStepCallback onStepCallback,
  LostConnectionCallback lostConnectionCallback,
  uint64_t clientId);

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

#ifdef __cplusplus
} /* extern "C" */
#endif

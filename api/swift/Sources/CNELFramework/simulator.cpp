#include <thread>

#include "include/simulator.h"

#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

void* createSimulator(
    SimulatorConfig config, 
    OnStepCallback onStepCallbackFn,
    int saveFrequency, 
    const char* savePath) {
  return nullptr;
}

void* loadSimulator(
    const char* filePath, 
    OnStepCallback onStepCallbackFn,
    int saveFrequency, 
    const char* savePath) {
  return nullptr;
}

void simulatorDelete(Simulator* simulator) {
  delete simulator;
}

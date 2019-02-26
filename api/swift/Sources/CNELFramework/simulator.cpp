#include <thread>

#include "include/simulator.h"

#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

void* createSimulator(
    SimulatorConfig config, 
    onStepCallback onStepCallbackFn,
    int saveFrequency, 
    const char* savePath) {

}

void* loadSimulator(
    const char* filePath, 
    onStepCallback onStepCallbackFn,
    int saveFrequency, 
    const char* savePath) {

}

#include <Python.h>

#include <nel/simulator.h>

using namespace core;

/** 
 * Creates a new simulator and returns a handle to it.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    No arguments need to be provided for this call. 
 *                  If any are provided, they will be ignored.
 */
static PyObject* simulator_new(PyObject *self, PyObject *args) {
    return PyLong_FromVoidPtr(new simulator());
}

/**
 * Deletes a simulator and frees all memory allocated for that 
 * simulator.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 * \returns None.
 */  
static PyObject* simulator_delete(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    if (!PyArg_ParseTuple(args, "o", &py_sim_handle))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    delete sim_handle;
    Py_INCREF(Py_None);
    return Py_None;
}

/** 
 * Adds a new agent to an existing simulator and returns its ID.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 * \returns ID of the new agent.
 */
static PyObject* simulator_add_agent(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    if (!PyArg_ParseTuple(args, "o", &py_sim_handle))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    int id = sim_handle->add_agent();
    return Py_BuildValue("i", id);
}

/** 
 * Moves the agent in the simulation environment and advances the simulator 
 * by one step.
 * 
 * Note that this function call blocks until all the agents in the current 
 * simulation environment invoke "step".
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Agent ID.
 *                  - Move direction encoded as an integer:
 *                      UP = 0,
 *                      DOWN = 1,
 *                      LEFT = 2,
 *                      RIGHT = 3.
 *                  - Number of steps.
 * \returns None.
 */
static PyObject* simulator_step(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    int* agent_id;
    int* dir;
    int* num_steps;
    if (!PyArg_ParseTuple(args, "oIii", &py_sim_handle, &agent_id, &dir, &num_steps))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    sim_handle->step(*agent_id, static_cast<direction>(*dir), *num_steps);
    Py_INCREF(Py_None);
    return Py_None;
}

/** 
 * Gets the current position of an agent in the simulation environment.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Agent ID.
 * \returns Tuple containing the horizontal and vertical coordinates of the 
 *          agent's current position.
 */
static PyObject* simulator_position(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    int* agent_id;
    if (!PyArg_ParseTuple(args, "oI", &py_sim_handle, &agent_id))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    position pos = sim_handle->get_position(*agent_id);
    return Py_BuildValue("(ii)", pos.x, pos.y);
}

static PyMethodDef SimulatorMethods[] = {
    {"new",  simulator_new, METH_VARARGS, "Creates a new simulator and returns its pointer."},
    {"delete",  simulator_delete, METH_VARARGS, "Deletes an existing simulator."},
    {"add_agent",  simulator_add_agent, METH_VARARGS, "Adds an agent to the simulator and returns its ID."},
    {"step",  simulator_step, METH_VARARGS, "Moves the agent in the simulation environment and advances the simulator by one step."},
    {"position",  simulator_position, METH_VARARGS, "Gets the current position of an agent in the simulation environment."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef simulator_module = {
        PyModuleDef_HEAD_INIT, "simulator_c", "Simulator", -1, SimulatorMethods
    };

    PyMODINIT_FUNC PyInit_simulator_c(void) {
        return PyModule_Create(&simulator_module);
    }
#else
    PyMODINIT_FUNC initsimulator_c(void) {
        (void) Py_InitModule("simulator_c", SimulatorMethods);
    }
#endif

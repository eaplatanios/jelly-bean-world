#include <Python.h>

#include <nel/simulator.h>

namespace nel {

using namespace core;

static float* PyArg_ParseFloatList(PyObject* arg) {
    if (!PyList_Check(arg)) {
        fprintf(stderr, "Expected float list, but got invalid argument.");
        exit(EXIT_FAILURE);
    }
    Py_ssize_t len = PyList_Size(arg);
    float* items = new float[len];
    for (unsigned int i = 0; i < len; i++)
        items[i] = PyFloat_AsDouble(PyList_GetItem(arg, (Py_ssize_t) i))
    return items;
}

/** 
 * Creates a new simulator and returns a handle to it.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    No arguments need to be provided for this call. 
 *                  If any are provided, they will be ignored.
 * \returns Pointer to the new simulator.
 */
static PyObject* simulator_new(PyObject *self, PyObject *args) {
    unsigned int* max_steps_per_movement;
    unsigned int* scent_num_dims;
    unsigned int* color_num_dims;
    unsigned int* vision_range;
    unsigned int* patch_size;
    unsigned int* gibbs_iterations;
    PyObject* py_items;
    if (!PyArg_ParseTuple(
      args, "IIIIIIO", &max_steps_per_movement, &scent_num_dims, &color_num_dims, 
      &vision_range, &patch_size, &gibbs_iterations, &py_items))
        return NULL;
    PyObject *py_items_iter = PyObject_GetIter(py_items);
    if (!py_items_iter)
        return NULL;
    array<item_properties> item_types(8);
    while (true) {
        PyObject *next_py_item = PyIter_Next(py_items_iter);
        if (!next_py_item) break;
        const char* name;
        PyObject* py_scent;
        PyObject* py_color;
        float* intensity;
        if (!PyArg_ParseTuple(args, "sOOf", &name, &py_scent, &py_color, &intensity))
            return NULL;
        float* scent = PyArg_ParseFloatList(py_scent);
        float* color = PyArg_ParseFloatList(py_color);
        item_properties& new_item = item_types[item_types.length];
        init(new_item.name, name);
        new_item.scent = scent;
        new_item.color = color;
        new_item.intensity = *intensity;
        item_types.length += 1
    }
    simulator_config config;
    config.max_steps_per_movement = *max_steps_per_movement;
    config.scent_num_dims = *scent_num_dims;
    config.color_num_dims = *color_num_dims;
    config.vision_range = *vision_range;
    config.patch_size = *patch_size;
    config.gibbs_iterations = *gibbs_iterations;
    config.item_types = item_types;
    // TODO: intensity and interaction functions.
    // TODO: step callback function.
    return PyLong_FromVoidPtr(new simulator(config, nullptr));
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
    if (!PyArg_ParseTuple(args, "O", &py_sim_handle))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    delete sim_handle;
    Py_INCREF(Py_None);
    return Py_None;
}

/** 
 * Adds a new agent to an existing simulator and returns a pointer to its 
 * state.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 * \returns Pointer to the new agent's state.
 */
static PyObject* simulator_add_agent(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    if (!PyArg_ParseTuple(args, "O", &py_sim_handle))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    return PyLong_FromVoidPtr((void*) sim_handle->add_agent());
}

/** 
 * Attempt to move the agent in the simulation environment. If the agent
 * already has an action queued for this turn, this attempt will fail.
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
 * \returns `true` if the move command is successfully queued; `false` otherwise.
 */
static PyObject* simulator_move(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_agt_handle;
    int* dir;
    int* num_steps;
    if (!PyArg_ParseTuple(args, "OOii", &py_sim_handle, &py_agt_handle, &dir, &num_steps))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    agent_state* agt_handle = (agent_state*) PyLong_AsVoidPtr(py_agt_handle);
    bool success = sim_handle->move(agt_handle, static_cast<direction>(*dir), *num_steps);
    return Py_BuildValue("O", success ? Py_True : Py_False);
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
    PyObject* py_agt_handle;
    if (!PyArg_ParseTuple(args, "OO", &py_sim_handle, &py_agt_handle))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    agent_state* agt_handle = (agent_state*) PyLong_AsVoidPtr(py_agt_handle);
    position pos = sim_handle->get_position(agt_handle);
    return Py_BuildValue("(ii)", pos.x, pos.y);
}

} /* namespace nel */

static PyMethodDef SimulatorMethods[] = {
    {"new",  nel::simulator_new, METH_VARARGS, "Creates a new simulator and returns its pointer."},
    {"delete",  nel::simulator_delete, METH_VARARGS, "Deletes an existing simulator."},
    {"add_agent",  nel::simulator_add_agent, METH_VARARGS, "Adds an agent to the simulator and returns its ID."},
    {"move",  nel::simulator_move, METH_VARARGS, "Attempts to move the agent in the simulation environment."},
    {"position",  nel::simulator_position, METH_VARARGS, "Gets the current position of an agent in the simulation environment."},
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

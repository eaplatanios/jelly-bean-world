#include <Python.h>

#include <thread>
#include <nel/gibbs_field.h>
#include <nel/mpi.h>
#include <nel/simulator.h>

namespace nel {

using namespace core;

static pair<float*, Py_ssize_t> PyArg_ParseFloatList(PyObject* arg) {
    if (!PyList_Check(arg)) {
        fprintf(stderr, "Expected float list, but got invalid argument.");
        exit(EXIT_FAILURE);
    }
    Py_ssize_t len = PyList_Size(arg);
    float* items = new float[len];
    for (unsigned int i = 0; i < len; i++)
        items[i] = PyFloat_AsDouble(PyList_GetItem(arg, (Py_ssize_t) i));
    return make_pair(items, len);
}

/* Python callback function for when the simulator advances by a step. */
static PyObject* py_step_callback = NULL;

/** 
 * Sets the step callback function for Python simulators using the C callback method.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to a Python callable acting as the callback.
 * \returns None.
 */
static PyObject* simulator_set_step_callback(PyObject *self, PyObject *args) {
    PyObject *result = NULL;
    PyObject *temp;
    if (PyArg_ParseTuple(args, "O:set_step_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "Parameter must be a callable.");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback. */
        Py_XDECREF(my_callback);  /* Dispose of the previous callback. */
        py_step_callback = temp;       /* Store the new callback. */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}

enum class step_callback_fns {
    C_API = 0, MPI_API = 1
};

void c_api_step_callback_fn(const agent_state& agent) {
    // TODO!!!
    fprintf(stderr, "The C API simulator step callback function has not been implemented yet.");
    exit(EXIT_FAILURE);
}

void mpi_api_step_callback_fn(const agent_state& agent) {
    // TODO!!!
    fprintf(stderr, "The MPI API simulator step callback function has not been implemented yet.");
    exit(EXIT_FAILURE);
}

step_callback get_step_callback_fn(step_callback_fns type) {
    switch (type) {
        case step_callback_fns::C_API  : return c_api_step_callback_fn;
        case step_callback_fns::MPI_API: return mpi_api_step_callback_fn;
    }
}

/** 
 * Creates a new simulator and returns a handle to it.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    
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
    unsigned int* py_intensity_fn;
    PyObject* py_intensity_fn_args;
    unsigned int* py_interaction_fn;
    PyObject* py_interaction_fn_args;
    unsigned int* py_step_callback_fn;
    if (!PyArg_ParseTuple(
      args, "IIIIIIOIOIOI", &max_steps_per_movement, &scent_num_dims, &color_num_dims, 
      &vision_range, &patch_size, &gibbs_iterations, &py_items, 
      &py_intensity_fn, &py_intensity_fn_args, 
      &py_interaction_fn, &py_interaction_fn_args, 
      &py_step_callback_fn)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.new'.");
        exit(EXIT_FAILURE);
    }
    PyObject *py_items_iter = PyObject_GetIter(py_items);
    if (!py_items_iter) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.new'.");
        exit(EXIT_FAILURE);
    }
    array<item_properties> item_types(8);
    while (true) {
        PyObject *next_py_item = PyIter_Next(py_items_iter);
        if (!next_py_item) break;
        char* name;
        PyObject* py_scent;
        PyObject* py_color;
        float* intensity;
        if (!PyArg_ParseTuple(args, "sOOf", &name, &py_scent, &py_color, &intensity))
            return NULL;
        float* scent = PyArg_ParseFloatList(py_scent).key;
        float* color = PyArg_ParseFloatList(py_color).key;
        item_properties& new_item = item_types[item_types.length];
        init(new_item.name, name);
        new_item.scent = scent;
        new_item.color = color;
        new_item.intensity = *intensity;
        item_types.length += 1;
    }
    simulator_config config;
    config.max_steps_per_movement = *max_steps_per_movement;
    config.scent_dimension = *scent_num_dims;
    config.color_dimension = *color_num_dims;
    config.vision_range = *vision_range;
    config.patch_size = *patch_size;
    config.gibbs_iterations = *gibbs_iterations;
    config.item_types = item_types;
    pair<float*, Py_ssize_t> intensity_fn_args = PyArg_ParseFloatList(py_intensity_fn_args);
    config.intensity_fn = get_intensity_fn((intensity_fns) *py_intensity_fn, intensity_fn_args.key, (unsigned int) intensity_fn_args.value);
    pair<float*, Py_ssize_t> interaction_fn_args = PyArg_ParseFloatList(py_interaction_fn_args);
    config.interaction_fn = get_interaction_fn((interaction_fns) *py_interaction_fn, interaction_fn_args.key, (unsigned int) interaction_fn_args.value);
    step_callback step_callback_fn = get_step_callback_fn((step_callback_fns) *py_step_callback_fn);
    return PyLong_FromVoidPtr(new simulator(config, step_callback_fn));
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
 * Starts the simulator server.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Server address (string).
 *                  - Server port (string).
 *                  - Connection queue capacity (integer).
 *                  - Worker count (integer).
 * \returns Handle to the simulator server thread.
 */ 
static PyObject* simulator_start_server(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    const char* address;
    const char* port;
    unsigned int* conn_queue_capacity;
    unsigned int* worker_count;
    if (!PyArg_ParseTuple(
            args, "OssII", &py_sim_handle, &address, &port, &conn_queue_capacity, &worker_count))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    async_server* sim_server = new async_server();
    // TODO: Pass a reference to the simulator.
    init_server(*sim_server, address, port, conn_queue_capacity, worker_count);
    return PyLong_FromVoidPtr(sim_server);
}

/** 
 * Stops the simulator server.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator server thread as a PyLong.
 * \returns None.
 */ 
static PyObject* simulator_stop_server(PyObject *self, PyObject *args) {
    PyObject* py_sim_server;
    if (!PyArg_ParseTuple(args, "O", &py_sim_server))
        return NULL;
    async_server* sim_server = (async_server*) PyLong_AsVoidPtr(py_sim_server);
    stop_server(*sim_server);
    delete sim_server;
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
    bool success = sim_handle->move(*agt_handle, static_cast<direction>(*dir), *num_steps);
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
    position pos = sim_handle->get_position(*agt_handle);
    return Py_BuildValue("(ii)", pos.x, pos.y);
}

} /* namespace nel */

static PyMethodDef SimulatorMethods[] = {
    {"set_step_callback",  nel::simulator_set_step_callback, METH_VARARGS, "Sets the step callback for simulators."},
    {"new",  nel::simulator_new, METH_VARARGS, "Creates a new simulator and returns its pointer."},
    {"delete",  nel::simulator_delete, METH_VARARGS, "Deletes an existing simulator."},
    {"start_server",  nel::simulator_start_server, METH_VARARGS, "Starts the simulator server."},
    {"stop_server",  nel::simulator_stop_server, METH_VARARGS, "Stops the simulator server."},
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

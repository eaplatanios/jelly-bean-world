#include <Python.h>

#include <thread>
#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

namespace nel {

using namespace core;

enum class simulator_type {
    C = 0, MPI = 1
};

struct py_simulator_data
{
    char* save_directory;
    unsigned int save_directory_length;
    unsigned int save_frequency;
    async_server* server;

	static inline void free(py_simulator_data& data) {
        if (data.save_directory != NULL)
            core::free(data.save_directory);
    }
};

inline bool init(py_simulator_data& data, const py_simulator_data& src) {
    if (src.save_directory != NULL) {
        data.save_directory = (char*) malloc(sizeof(char) * max(1u, src.save_directory_length));
        if (data.save_directory == NULL) {
            fprintf(stderr, "init ERROR: Insufficient memory for py_simulator_data.save_directory.\n");
            return false;
        }
        for (unsigned int i = 0; i < src.save_directory_length; i++)
            data.save_directory[i] = src.save_directory[i];
        data.save_directory_length = src.save_directory_length;
    }
    data.save_frequency = src.save_frequency;
    data.server = src.server;
    return true;
}

struct py_client_data {
    /* storing the server responses */
    bool move_result;
    uint64_t agent_id;
    position pos;

    /* for synchronization */
    bool waiting_for_step;
    bool waiting_for_server;
    std::mutex lock;
    std::condition_variable cv;
};

static pair<float*, Py_ssize_t> PyArg_ParseFloatList(PyObject* arg) {
    if (!PyList_Check(arg)) {
        PyErr_SetString(PyExc_ValueError, "Expected float list, but got invalid argument.");
        return make_pair(NULL, 0);
    }
    Py_ssize_t len = PyList_Size(arg);
    float* items = (float*) malloc(sizeof(float) * len);
    if (items == NULL) {
        PyErr_NoMemory();
        return make_pair(NULL, 0);
    }
    for (unsigned int i = 0; i < len; i++)
        items[i] = PyFloat_AsDouble(PyList_GetItem(arg, (Py_ssize_t) i));
    return make_pair(items, len);
}

static PyObject* Py_BuildAgentState(const agent_state* state) {
    PyObject* py_sim_handle = PyLong_FromVoidPtr(sim);
    PyObject* py_scent = PyList_New((Py_ssize_t) sim_config.scent_dimension);
    for (unsigned int i = 0; i < sim_config.scent_dimension; i++)
        PyList_SetItem(py_scent, (Py_ssize_t) i, PyFloat_FromDouble((double) agent.current_scent[i]));
    PyObject* py_vision = PyList_New((Py_ssize_t) sim_config.color_dimension);
    for (unsigned int i = 0; i < sim_config.color_dimension; i++)
        PyList_SetItem(py_vision, (Py_ssize_t) i, PyFloat_FromDouble((double) agent.current_vision[i]));
    return Py_BuildValue("(II)OO", agent.current_position.x, , agent.current_position.y, py_scent, py_vision);
}

/* Python callback function for when the simulator advances by a step. */
static PyObject* py_step_callback = NULL;

/** 
 * Sets the step callback function for Python simulators using the C callback 
 * method.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to a Python callable acting as the callback. This 
 *                    callable should take an agent state as input.
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
        py_step_callback = temp;  /* Store the new callback. */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}

void on_step(const simulator<py_simulator_data>* sim,
        py_simulator_data& data, uint64_t time)
{
    if (data.server != NULL) {
        /* this simulator is a server, so send a step response to every client */
        if (!send_step_response(*data.server))
            fprintf(stderr, "on_step ERROR: send_step_response failed.\n");
    }

    /* TODO: call python callback */

    PyObject* py_agent = Py_BuildAgentState(agent);
    PyObject* args = Py_BuildValue("OIO", py_sim_handle, id, py_agent);
    PyObject_CallObject(py_step_callback, arglist);
    Py_DECREF(args);
}

/**
 * Client callback functions.
 */

void on_add_agent(client<py_client_data>& c, uint64_t agent_id) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.agent_id = agent_id;
    c.data.cv.notify_one();
}

void on_move(client<py_client_data>& c, uint64_t agent_id, bool request_success) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.move_result = request_success;
	c.data.cv.notify_one();
}

void on_get_position(client<py_client_data>& c, uint64_t agent_id, const position& pos) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.pos = pos;
	c.data.cv.notify_one();
}

void on_step(client<py_client_data>& c) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_step = false;
	c.data.cv.notify_one();
}

void on_lost_connection(client<py_client_data>& c) {
	fprintf(stderr, "Client lost connection to server.\n");
	c.client_running = false;
	c.data.cv.notify_one();
}

inline void wait_for_server(std::condition_variable& cv,
		std::mutex& lock, bool& waiting_for_server, bool& client_running)
{
	std::unique_lock<std::mutex> lck(lock);
	while (waiting_for_server && client_running) cv.wait(lck);
}

/**
 * Creates a new simulator and returns a handle to it.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    
 * \returns Pointer to the new simulator.
 */
static PyObject* simulator_new(PyObject *self, PyObject *args)
{
    unsigned int* max_steps_per_movement;
    unsigned int* scent_num_dims;
    unsigned int* color_num_dims;
    unsigned int* vision_range;
    unsigned int* patch_size;
    unsigned int* gibbs_iterations;
    PyObject* py_items;
    PyObject* py_agent_color;
    unsigned int* collision_policy;
    float* decay_param; float* diffusion_param;
    unsigned int* deleted_item_lifetime;
    unsigned int* py_intensity_fn;
    PyObject* py_intensity_fn_args;
    unsigned int* py_interaction_fn;
    PyObject* py_interaction_fn_args;
    unsigned int* save_frequency;
    char* save_filepath;
    if (!PyArg_ParseTuple(
      args, "IIIIIIOOIffIIOIOIIz", &max_steps_per_movement, &scent_num_dims, &color_num_dims, 
      &vision_range, &patch_size, &gibbs_iterations, &py_items, &py_agent_color, &collision_policy,
      &decay_param, &diffusion_param, &deleted_item_lifetime, &py_intensity_fn,
      &py_intensity_fn_args, &py_interaction_fn, &py_interaction_fn_args, &save_frequency, &save_filepath)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.new'.\n");
        return NULL;
    }

    simulator_config config;
    PyObject *py_items_iter = PyObject_GetIter(py_items);
    if (!py_items_iter) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument types in the call to 'simulator_c.new'.");
        return NULL;
    }
    while (true) {
        PyObject *next_py_item = PyIter_Next(py_items_iter);
        if (!next_py_item) break;
        char* name;
        PyObject* py_scent;
        PyObject* py_color;
        int* automatically_collected;
        if (!PyArg_ParseTuple(args, "sOOi", &name, &py_scent, &py_color, &automatically_collected)) {
            fprintf(stderr, "Invalid argument types for item property in call to 'simulator_c.new'.\n");
            return NULL;
        }
        item_properties& new_item = config.item_types[config.item_types.length];
        init(new_item.name, name);
        new_item.scent = PyArg_ParseFloatList(py_scent).key;
        new_item.color = PyArg_ParseFloatList(py_color).key;
        new_item.automatically_collected = (*automatically_collected != 0);
        config.item_types.length += 1;
    }

    config.max_steps_per_movement = *max_steps_per_movement;
    config.scent_dimension = *scent_num_dims;
    config.color_dimension = *color_num_dims;
    config.vision_range = *vision_range;
    config.patch_size = *patch_size;
    config.gibbs_iterations = *gibbs_iterations;
    config.agent_color = PyArg_ParseFloatList(py_agent_color).key;
    config.collision_policy = (movement_conflict_policy) *collision_policy;
    config.decay_param = *decay_param;
    config.diffusion_param = *diffusion_param;
    config.deleted_item_lifetime = *deleted_item_lifetime;
    pair<float*, Py_ssize_t> intensity_fn_args = PyArg_ParseFloatList(py_intensity_fn_args);
    pair<float*, Py_ssize_t> interaction_fn_args = PyArg_ParseFloatList(py_interaction_fn_args);
    config.intensity_fn = get_intensity_fn((intensity_fns) *py_intensity_fn, intensity_fn_args.key, (unsigned int) intensity_fn_args.value);
    config.interaction_fn = get_interaction_fn((interaction_fns) *py_interaction_fn, interaction_fn_args.key, (unsigned int) interaction_fn_args.value);
    config.intensity_fn_args = intensity_fn_args.key;
    config.interaction_fn_args = interaction_fn_args.key;
    config.intensity_fn_arg_count = intensity_fn_args.value;
    config.interaction_fn_arg_count = interaction_fn_args.value;

    if (config.intensity_fn == NULL || config.interaction_fn == NULL
     || config.intensity_fn_args == NULL || config.interaction_fn_args == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid intensity/interaction"
                " function arguments in the call to 'simulator_c.new'.");
        return NULL;
    }

    py_simulator_data data;
    data.save_directory = save_filepath;
    if (save_filepath != NULL)
        data.save_directory_length = strlen(save_filepath);
    data.save_frequency = *save_frequency;
    data.server = NULL;

    simulator<py_simulator_data>* sim =
            (simulator<py_simulator_data>*) malloc(sizeof(simulator<py_simulator_data>));
    if (sim == NULL) {
        PyErr_NoMemory();
        return NULL;
    } else if (!init(sim, config, data)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize simulator.");
        return NULL;
    }
    return PyLong_FromVoidPtr(sim);
}

static PyObject* simulator_load(PyObject *self, PyObject *args)
{
    char* load_filepath;
    unsigned int* save_frequency;
    char* save_filepath;
    if (!PyArg_ParseTuple(args, "sIz", &load_filepath, &save_frequency, &save_filepath)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.load'.\n");
        return NULL;
    }

    simulator<py_simulator_data>* sim =
            (simulator<py_simulator_data>*) malloc(sizeof(simulator<py_simulator_data>));
    if (sim == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    py_simulator_data data;
    data.save_directory = save_filepath;
    if (save_filepath != NULL)
        data.save_directory_length = strlen(save_filepath);
    data.save_frequency = *save_frequency;
    data.server = NULL;

    FILE* file = fopen(load_filepath, "rb");
    if (file == NULL) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }
    fixed_width_stream<FILE*> in(file);
    if (!read(*sim, in, data)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load simulator.");
        free(sim); fclose(file);
        return NULL;
    }
    fclose(file);
    return PyLong_FromVoidPtr(sim);
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
    if (!PyArg_ParseTuple(args, "O", &py_sim_handle)) {
        fprintf(stderr, "Invalid simulator handle argument in the call to 'simulator_c.delete'.\n");
        return NULL;
    }
    simulator<py_simulator_data>* sim_handle =
            (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
    free(*sim_handle); free(sim_handle);
    Py_INCREF(Py_None);
    return Py_None;
}

/**
 * Starts the simulator server.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Server port (integer).
 *                  - Connection queue capacity (integer).
 *                  - Worker count (integer).
 * \returns Handle to the simulator server thread.
 */
static PyObject* simulator_start_server(PyObject *self, PyObject *args)
{
    PyObject* py_sim_handle;
    unsigned int* port;
    unsigned int* connection_queue_capacity;
    unsigned int* num_workers;
    if (!PyArg_ParseTuple(args, "OIII", &py_sim_handle, &port, &connection_queue_capacity, &num_workers)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.start_server'.\n");
        return NULL;
    }

    simulator<py_simulator_data>* sim_handle =
            (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
    async_server* server = (async_server*) malloc(sizeof(async_server));
    if (server == NULL) {
        PyErr_NoMemory();
        return NULL;
    } else if (!init_server(*server, *sim_handle, (uint16_t) port, connection_queue_capacity, num_workers)) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI server.");
        return NULL;
    }
    sim_handle->get_data().server = server;
    return PyLong_FromVoidPtr(server);
}

/**
 * Stops the simulator server.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator server object as a PyLong.
 * \returns None.
 */
static PyObject* simulator_stop_server(PyObject *self, PyObject *args)
{
    PyObject* py_server_handle;
    if (!PyArg_ParseTuple(args, "O", &py_server_handle)) {
        fprintf(stderr, "Invalid server handle argument in the call to 'simulator_c.stop_server'.\n");
        return NULL;
    }
    async_server* server = (async_server*) PyLong_AsVoidPtr(py_server_handle);
    stop_server(*server);
    free(server);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* simulator_start_client(PyObject *self, PyObject *args)
{
    char* server_address;
    unsigned int* port;
    if (!PyArg_ParseTuple(args, "sI", &server_address, &port)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.start_client'.\n");
        return NULL;
    }

    client<py_client_data>* new_client =
            (client<py_client_data>*) malloc(sizeof(client<py_client_data>));
    if (new_client == NULL) {
        PyErr_NoMemory();
        return NULL;
    } else if (!init_client(*new_client, server_address, "54353")) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI client.");
        return NULL;
    }
}

static PyObject* simulator_stop_client(PyObject *self, PyObject *args)
{
    PyObject* py_client_handle;
    if (!PyArg_ParseTuple(args, "O", &py_client_handle)) {
        fprintf(stderr, "Invalid server handle argument in the call to 'simulator_c.stop_client'.\n");
        return NULL;
    }
    client<py_client_data>* client_handle =
            (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
    stop_client(*client_handle);
    free(client_handle);
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
 *                  - Simulator type encoded as an integer:
 *                      C = 0,
 *                      MPI = 1.
 * \returns Pointer to the new agent's state.
 */
static PyObject* simulator_add_agent(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    if (!PyArg_ParseTuple(args, "OO", &py_sim_handle, &py_client_handle)) {
        fprintf(stderr, "Invalid server handle argument in the call to 'simulator_c.add_agent'.\n");
        return NULL;
    }
    simulator<py_simulator_data>* sim_handle =
            (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
    if (py_client_handle == Py_None) {
        /* this is a client, so send an add_agent message to the server */

    } else {
        /* the simulation is local, so call add_agent directly */
        uint64_t id = sim_handle->add_agent();

        PyObject* py_agent = Py_BuildAgentState(sim_handle->agents[id]);
        switch ((simulator_type) *py_sim_type) {
            case simulator_type::C:
                return Py_BuildValue("IO", id, py_agent);
            case simulator_type::MPI:
                /* TODO !!! */
                return NULL;
        }
    }
}

/** 
 * Attempt to move the agent in the simulation environment. If the agent
 * already has an action queued for this turn, this attempt will fail.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Simulator type encoded as an integer:
 *                      C = 0,
 *                      MPI = 1.
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
    unsigned int* py_sim_type;
    unsigned int* agent_id;
    int* dir;
    int* num_steps;
    if (!PyArg_ParseTuple(args, "OIIii", &py_sim_handle, &py_sim_type, &agent_id, &dir, &num_steps))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    agent_state* agt_handle = sim_handle->agents[agent_id];
    switch ((simulator_type) *py_sim_type) {
        case simulator_type::C: 
            bool success = sim_handle->move(*agt_handle, static_cast<direction>(*dir), *num_steps);
            return Py_BuildValue("O", success ? Py_True : Py_False);
        case simulator_type::MPI: 
            /* TODO !!! */ 
            return NULL;
    }
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
    unsigned int* agent_id;
    if (!PyArg_ParseTuple(args, "OI", &py_sim_handle, &agent_id))
        return NULL;
    simulator* sim_handle = (simulator*) PyLong_AsVoidPtr(py_sim_handle);
    agent_state* agt_handle = sim_handle->agents[agent_id];
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

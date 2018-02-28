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
    PyObject* callback;

    ~py_simulator_data() { free(*this); }

	static inline void free(py_simulator_data& data) {
        if (data.save_directory != NULL)
            core::free(data.save_directory);
        if (data.callback != NULL)
            Py_DECREF(data.callback);
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
    } else {
        data.save_directory = NULL;
    }
    data.save_frequency = src.save_frequency;
    data.server = src.server;
    data.callback = src.callback;
    Py_INCREF(data.callback);
    return true;
}

struct py_client_data {
    /* storing the server responses */
    union response {
        bool move_result;
        uint64_t agent_id;
        position pos;
        float* scent;
        float* vision;
        unsigned int* collected_items;
        hash_map<position, patch_state>* map;
    } response;

    /* for synchronization */
    bool waiting_for_step;
    bool waiting_for_server;
    std::mutex lock;
    std::condition_variable cv;

    PyObject* callback;

    static inline void free(py_client_data& data) {
        if (data.callback != NULL)
            Py_DECREF(data.callback);
        data.lock.~mutex();
        data.cv.~condition_variable();
    }
};

inline bool init(py_client_data& data) {
    data.callback = NULL;
    new (&data.lock) std::mutex();
    new (&data.cv) std::condition_variable();
    return true;
}

static pair<float*, Py_ssize_t> PyArg_ParseFloatList(PyObject* arg) {
    if (!PyList_Check(arg)) {
        PyErr_SetString(PyExc_ValueError, "Expected float list, but got invalid argument.");
        return make_pair<float*, Py_ssize_t>(NULL, 0);
    }
    Py_ssize_t len = PyList_Size(arg);
    float* items = (float*) malloc(sizeof(float) * len);
    if (items == NULL) {
        PyErr_NoMemory();
        return make_pair<float*, Py_ssize_t>(NULL, 0);
    }
    for (unsigned int i = 0; i < len; i++)
        items[i] = (float) PyFloat_AsDouble(PyList_GetItem(arg, (Py_ssize_t) i));
    return make_pair(items, len);
}

bool save(const simulator<py_simulator_data>* sim,
        const py_simulator_data& data, uint64_t time)
{
    int length = snprintf(NULL, 0, "%" PRIu64, time);
    if (length < 0) {
        fprintf(stderr, "on_step ERROR: Error computing filepath to save simulation.\n");
        return false;
    }

    char* filepath = (char*) malloc(sizeof(char) * (data.save_directory_length + length + 1));
    if (filepath == NULL) {
        fprintf(stderr, "on_step ERROR: Insufficient memory for filepath.\n");
        return false;
    }

    for (unsigned int i = 0; i < data.save_directory_length; i++)
        filepath[i] = data.save_directory[i];
    snprintf(filepath + data.save_directory_length, length + 1, "%" PRIu64, time);

    FILE* file = fopen(filepath, "wb");
    if (file == NULL) {
        fprintf(stderr, "on_step: Unable to open '%s' for writing", filepath);
        perror(""); return false;
    }

    fixed_width_stream<FILE*> out(file);
    bool result = write(*sim, out);
    fclose(file);
    return result;
}

void on_step(const simulator<py_simulator_data>* sim,
        py_simulator_data& data, uint64_t time)
{
    bool saved = false;
    if (data.server != NULL) {
        /* this simulator is a server, so send a step response to every client */
        if (!send_step_response(*data.server))
            fprintf(stderr, "on_step ERROR: send_step_response failed.\n");
    } if (data.save_directory != NULL && time % data.save_frequency == 0) {
        /* save the simulator to a local file */
        saved = save(sim, data, time);
    }

    /* call python callback */
    PyObject* py_saved = saved ? Py_True : Py_False;
    Py_INCREF(py_saved);
    PyObject* args = Py_BuildValue("(O)", py_saved);
    PyObject* result = PyEval_CallObject(data.callback, args);
    Py_DECREF(args);
    if (result != NULL)
        Py_DECREF(result);
}

/**
 * Client callback functions.
 */

void on_add_agent(client<py_client_data>& c, uint64_t agent_id) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.response.agent_id = agent_id;
    c.data.cv.notify_one();
}

void on_move(client<py_client_data>& c, uint64_t agent_id, bool request_success) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.response.move_result = request_success;
	c.data.cv.notify_one();
}

void on_get_position(client<py_client_data>& c, uint64_t agent_id, const position& pos) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.response.pos = pos;
	c.data.cv.notify_one();
}

void on_get_scent(client<py_client_data>& c, uint64_t agent_id, float* scent)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
    c.data.response.scent = scent;
    c.data.cv.notify_one();
}

void on_get_vision(client<py_client_data>& c, uint64_t agent_id, float* vision)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
    c.data.response.vision = vision;
    c.data.cv.notify_one();
}

void on_get_collected_items(client<py_client_data>& c,
        uint64_t agent_id, unsigned int* collected_items)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
    c.data.response.collected_items = collected_items;
    c.data.cv.notify_one();
}

void on_get_map(client<py_client_data>& c,
        hash_map<position, patch_state>* map)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
    c.data.response.map = map;
    c.data.cv.notify_one();
}

void on_step(client<py_client_data>& c) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_step = false;
	c.data.cv.notify_one();

    /* call python callback */
    PyObject* args = Py_BuildValue("()");
    PyObject* result = PyEval_CallObject(c.data.callback, args);
    Py_DECREF(args);
    if (result != NULL)
        Py_DECREF(result);
}

void on_lost_connection(client<py_client_data>& c) {
	fprintf(stderr, "Client lost connection to server.\n");
	c.client_running = false;
	c.data.cv.notify_one();
}

inline void wait_for_server(client<py_client_data>& c)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	while (c.data.waiting_for_server && c.client_running)
        c.data.cv.wait(lck);
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
    simulator_config config;
    PyObject* py_items;
    PyObject* py_agent_color;
    unsigned int collision_policy;
    unsigned int py_intensity_fn;
    unsigned int py_interaction_fn;
    PyObject* py_intensity_fn_args;
    PyObject* py_interaction_fn_args;
    PyObject* py_callback;
    unsigned int save_frequency;
    char* save_filepath;
    if (!PyArg_ParseTuple(
      args, "IIIIIIOOIffIIOIOOIz", &config.max_steps_per_movement, &config.scent_dimension,
      &config.color_dimension, &config.vision_range, &config.patch_size, &config.gibbs_iterations,
      &py_items, &py_agent_color, &collision_policy, &config.decay_param, &config.diffusion_param,
      &config.deleted_item_lifetime,
      &py_intensity_fn, &py_intensity_fn_args, &py_interaction_fn, &py_interaction_fn_args,
      &py_callback, &save_frequency, &save_filepath)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.new'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable.\n");
        return NULL;
    }

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
        PyObject* automatically_collected;
        if (!PyArg_ParseTuple(next_py_item, "sOOO", &name, &py_scent, &py_color, &automatically_collected)) {
            fprintf(stderr, "Invalid argument types for item property in call to 'simulator_c.new'.\n");
            return NULL;
        }
        item_properties& new_item = config.item_types[config.item_types.length];
        init(new_item.name, name);
        new_item.scent = PyArg_ParseFloatList(py_scent).key;
        new_item.color = PyArg_ParseFloatList(py_color).key;
        new_item.automatically_collected = (automatically_collected == Py_True);
        config.item_types.length += 1;
    }

    config.agent_color = PyArg_ParseFloatList(py_agent_color).key;
    config.collision_policy = (movement_conflict_policy) collision_policy;
    pair<float*, Py_ssize_t> intensity_fn_args = PyArg_ParseFloatList(py_intensity_fn_args);
    pair<float*, Py_ssize_t> interaction_fn_args = PyArg_ParseFloatList(py_interaction_fn_args);
    config.intensity_fn = get_intensity_fn((intensity_fns) py_intensity_fn,
            intensity_fn_args.key, (unsigned int) intensity_fn_args.value, (unsigned int) config.item_types.length);
    config.interaction_fn = get_interaction_fn((interaction_fns) py_interaction_fn,
            interaction_fn_args.key, (unsigned int) interaction_fn_args.value, (unsigned int) config.item_types.length);
    config.intensity_fn_args = intensity_fn_args.key;
    config.interaction_fn_args = interaction_fn_args.key;
    config.intensity_fn_arg_count = (unsigned int) intensity_fn_args.value;
    config.interaction_fn_arg_count = (unsigned int) interaction_fn_args.value;

    if (config.intensity_fn == NULL || config.interaction_fn == NULL
     || config.intensity_fn_args == NULL || config.interaction_fn_args == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid intensity/interaction"
                " function arguments in the call to 'simulator_c.new'.");
        return NULL;
    }

    py_simulator_data data;
    data.save_directory = save_filepath;
    if (save_filepath != NULL)
        data.save_directory_length = (unsigned int) strlen(save_filepath);
    data.save_frequency = save_frequency;
    data.server = NULL;
    data.callback = py_callback;
    Py_INCREF(py_callback);

    simulator<py_simulator_data>* sim =
            (simulator<py_simulator_data>*) malloc(sizeof(simulator<py_simulator_data>));
    if (sim == NULL) {
        PyErr_NoMemory();
        return NULL;
    } else if (!init(*sim, config, data)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize simulator.");
        return NULL;
    }
    return PyLong_FromVoidPtr(sim);
}

static PyObject* simulator_load(PyObject *self, PyObject *args)
{
    char* load_filepath;
    PyObject* py_callback;
    unsigned int save_frequency;
    char* save_filepath;
    if (!PyArg_ParseTuple(args, "sOIz", &load_filepath, &py_callback, &save_frequency, &save_filepath)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.load'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable.\n");
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
        data.save_directory_length = (unsigned int) strlen(save_filepath);
    data.save_frequency = save_frequency;
    data.server = NULL;
    data.callback = py_callback;
    Py_INCREF(py_callback);

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
    unsigned int port;
    unsigned int connection_queue_capacity;
    unsigned int num_workers;
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
    } else if (!init(*server)) {
        PyErr_NoMemory();
        free(server); return NULL;
    } else if (!init_server(*server, *sim_handle, (uint16_t) port, connection_queue_capacity, num_workers)) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI server.");
        free(*server); free(server); return NULL;
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
    free(*server); free(server);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* simulator_start_client(PyObject *self, PyObject *args)
{
    char* server_address;
    unsigned int port;
    PyObject* py_callback;
    if (!PyArg_ParseTuple(args, "sIO", &server_address, &port, &py_callback)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.start_client'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable.\n");
        return NULL;
    }

    client<py_client_data>* new_client =
            (client<py_client_data>*) malloc(sizeof(client<py_client_data>));
    if (new_client == NULL) {
        PyErr_NoMemory();
        return NULL;
    } else if (!init(*new_client)) {
        PyErr_NoMemory();
        free(new_client); return NULL;
    } else if (!init_client(*new_client, server_address, (uint16_t) port)) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI client.");
        free(*new_client); free(new_client); return NULL;
    }

    new_client->data.callback = py_callback;
    Py_INCREF(py_callback);
    return PyLong_FromVoidPtr(new_client);
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
    free(*client_handle); free(client_handle);
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
 *                  - Handle to the native client object as a PyLong.
 * \returns Pointer to the new agent's state.
 */
static PyObject* simulator_add_agent(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    if (!PyArg_ParseTuple(args, "OO", &py_sim_handle, &py_client_handle)) {
        fprintf(stderr, "Invalid server handle argument in the call to 'simulator_c.add_agent'.\n");
        return NULL;
    }
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call add_agent directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        uint64_t id = sim_handle->add_agent();
        if (id == UINT64_MAX) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to add new agent.");
            return NULL;
        }
        return PyLong_FromUnsignedLongLong(id);
    } else {
        /* this is a client, so send an add_agent message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!send_add_agent(*client_handle)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send add_agent request.");
            return NULL;
        }

		/* wait for response from server */
		wait_for_server(*client_handle);

		if (client_handle->data.response.agent_id == UINT64_MAX) {
            /* server returned failure */
            PyErr_SetString(PyExc_RuntimeError, "Failed to add new agent.");
            return NULL;
		}

        return PyLong_FromUnsignedLongLong(client_handle->data.response.agent_id);
    }
}

/** 
 * Attempt to move the agent in the simulation environment. If the agent
 * already has an action queued for this turn, this attempt will fail.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong.
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
    PyObject* py_client_handle;
    unsigned long long agent_id;
    unsigned int dir;
    unsigned int num_steps;
    if (!PyArg_ParseTuple(args, "OOKII", &py_sim_handle, &py_client_handle, &agent_id, &dir, &num_steps))
        return NULL;
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call move directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        if (sim_handle->move(agent_id, (direction) dir, num_steps)) {
            Py_INCREF(Py_True);
            return Py_True;
        } else {
            Py_INCREF(Py_False);
            return Py_False;
        }
    } else {
        /* this is a client, so send a move message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!send_move(*client_handle, agent_id, (direction) dir, num_steps)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send move request.");
            return NULL;
        }

		/* wait for response from server */
		wait_for_server(*client_handle);

        if (client_handle->data.response.move_result) {
            Py_INCREF(Py_True);
            return Py_True;
        } else {
            Py_INCREF(Py_False);
            return Py_False;
        }
    }
}

/** 
 * Gets the current position of an agent in the simulation environment.
 * 
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong.
 *                  - Agent ID.
 * \returns Tuple containing the horizontal and vertical coordinates of the 
 *          agent's current position.
 */
static PyObject* simulator_position(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    if (!PyArg_ParseTuple(args, "OOK", &py_sim_handle, &py_client_handle, &agent_id))
        return NULL;
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call get_position directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        position pos = sim_handle->get_position(agent_id);
        return Py_BuildValue("(LL)", pos.x, pos.y);
    } else {
        /* this is a client, so send a get_position message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!send_get_position(*client_handle, agent_id)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send get_position request.");
            return NULL;
        }

		/* wait for response from server */
		wait_for_server(*client_handle);

        return Py_BuildValue("(LL)", client_handle->data.response.pos.x, client_handle->data.response.pos.y);
    }
}

static PyObject* simulator_scent(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    if (!PyArg_ParseTuple(args, "OOK", &py_sim_handle, &py_client_handle, &agent_id))
        return NULL;

    const float* scent;
    unsigned int scent_dimension;
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call get_scent directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        scent = sim_handle->get_scent(agent_id);
        scent_dimension = sim_handle->get_config().scent_dimension;
    } else {
        /* this is a client, so send a get_scent message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!send_get_scent(*client_handle, agent_id)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send get_scent request.");
            return NULL;
        }

		/* wait for response from server */
		wait_for_server(*client_handle);
        scent = client_handle->data.response.scent;
        scent_dimension = client_handle->config.scent_dimension;
    }

    PyObject* py_scent = PyTuple_New(scent_dimension);
    for (unsigned int i = 0; i < scent_dimension; i++)
        PyTuple_SetItem(py_scent, i, PyFloat_FromDouble(scent[i]));
    if (py_client_handle != Py_None)
        free((float*) scent);
    return py_scent;
}

static PyObject* simulator_vision(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    if (!PyArg_ParseTuple(args, "OOK", &py_sim_handle, &py_client_handle, &agent_id))
        return NULL;

    const float* vision;
    unsigned int color_dimension, vision_range;
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call get_vision directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        vision = sim_handle->get_vision(agent_id);
        vision_range = sim_handle->get_config().vision_range;
        color_dimension = sim_handle->get_config().color_dimension;
    } else {
        /* this is a client, so send a get_vision message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!send_get_vision(*client_handle, agent_id)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send get_vision request.");
            return NULL;
        }

		/* wait for response from server */
		wait_for_server(*client_handle);
        vision = client_handle->data.response.vision;
        color_dimension = client_handle->config.color_dimension;
        vision_range = client_handle->config.vision_range;
    }

    PyObject* py_vision = PyList_New(2 * vision_range + 1);
    for (unsigned int i = 0; i < 2 * vision_range + 1; i++) {
        PyObject* row = PyList_New(2 * vision_range + 1);
        PyList_SetItem(py_vision, i, row);
        for (unsigned int j = 0; j < 2 * vision_range + 1; j++) {
            PyObject* pixel = PyTuple_New(color_dimension);
            PyList_SetItem(row, j, pixel);
            unsigned int offset = (i*(2*vision_range + 1) + j) * color_dimension;
            for (unsigned int k = 0; k < color_dimension; k++)
                PyTuple_SetItem(pixel, k, PyFloat_FromDouble(vision[offset + k]));
        }
    }
    if (py_client_handle != Py_None)
        free((float*) vision);
    return py_vision;
}

static PyObject* simulator_collected_items(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    if (!PyArg_ParseTuple(args, "OOK", &py_sim_handle, &py_client_handle, &agent_id))
        return NULL;

    const unsigned int* items;
    size_t item_type_count;
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call get_scent directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        items = sim_handle->get_collected_items(agent_id);
        item_type_count = sim_handle->get_config().item_types.length;
    } else {
        /* this is a client, so send a get_scent message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!send_get_scent(*client_handle, agent_id)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send get_scent request.");
            return NULL;
        }

		/* wait for response from server */
		wait_for_server(*client_handle);
        items = client_handle->data.response.collected_items;
        item_type_count = client_handle->config.item_types.length;
    }

    PyObject* py_items = PyTuple_New(item_type_count);
    for (size_t i = 0; i < item_type_count; i++)
        PyTuple_SetItem(py_items, i, PyLong_FromSize_t(items[i]));
    if (py_client_handle != Py_None)
        free((float*) items);
    return py_items;
}

static PyObject* build_py_map(
        const hash_map<position, patch_state>& patches,
        const simulator_config& config)
{
    unsigned int index = 0;
    PyObject* list = PyList_New(patches.table.size);
    for (const auto& entry : patches) {
        const patch_state& patch = entry.value;
        PyObject* items = PyList_New(patch.item_count);
        for (unsigned int i = 0; i < patch.item_count; i++)
            PyList_SetItem(items, i, Py_BuildValue("I(LL)",
                    patch.items[i].item_type,
                    patch.items[i].location.x,
                    patch.items[i].location.y));

        PyObject* agents = PyList_New(patch.agent_count);
        for (unsigned int i = 0; i < patch.agent_count; i++)
            PyList_SetItem(agents, i, Py_BuildValue("(LL)", patch.agents[i].x, patch.agents[i].y));

        unsigned int n = config.patch_size;
        PyObject* scent = PyList_New(n);
        PyObject* vision = PyList_New(n);
        for (unsigned int i = 0; i < config.patch_size; i++) {
            PyObject* scent_row = PyList_New(n);
            PyObject* vision_row = PyList_New(n);
            PyList_SetItem(scent, i, scent_row);
            PyList_SetItem(vision, i, vision_row);
            for (unsigned int j = 0; j < config.patch_size; j++) {
                PyObject* scent_cell = PyTuple_New(config.scent_dimension);
                PyObject* vision_cell = PyTuple_New(config.color_dimension);
                PyList_SetItem(scent_row, j, scent_cell);
                PyList_SetItem(vision_row, j, vision_cell);
                for (unsigned int k = 0; k < config.scent_dimension; k++)
                    PyTuple_SetItem(scent_cell, k, PyFloat_FromDouble(patch.scent[(i*n + j)*config.scent_dimension + k]));
                for (unsigned int k = 0; k < config.color_dimension; k++)
                    PyTuple_SetItem(vision_cell, k, PyFloat_FromDouble(patch.vision[(i*n + j)*config.color_dimension + k]));
            }
        }

        PyObject* fixed = patch.fixed ? Py_True : Py_False;
        Py_INCREF(fixed);
        PyObject* py_patch = Py_BuildValue("((LL)OOOOO)", patch.patch_position.x, patch.patch_position.y, fixed, scent, vision, items, agents);
        PyList_SetItem(list, index, py_patch);
        index++;
    }
    return list;
}

static PyObject* simulator_map(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    int64_t py_bottom_left_x, py_bottom_left_y;
    int64_t py_top_right_x, py_top_right_y;
    if (!PyArg_ParseTuple(args, "OO(LL)(LL)", &py_sim_handle, &py_client_handle,
            &py_bottom_left_x, &py_bottom_left_y, &py_top_right_x, &py_top_right_y))
        return NULL;
    position bottom_left = position(py_bottom_left_x, py_bottom_left_y);
    position top_right = position(py_top_right_x, py_top_right_y);

    if (py_client_handle == Py_None) {
        /* the simulation is local, so call get_scent directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        hash_map<position, patch_state> patches(16, alloc_position_keys);
        if (!sim_handle->get_map(bottom_left, top_right, patches)) {
            PyErr_SetString(PyExc_RuntimeError, "simulator.get_map failed.");
            return NULL;
        }
        PyObject* py_map = build_py_map(patches, sim_handle->get_config());
        for (auto entry : patches)
            free(entry.value);
        return py_map;
    } else {
        /* this is a client, so send a get_scent message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!send_get_map(*client_handle, bottom_left, top_right)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send get_map request.");
            return NULL;
        }

		/* wait for response from server */
		wait_for_server(*client_handle);
        PyObject* py_map = build_py_map(*client_handle->data.response.map, client_handle->config);
        for (auto entry : *client_handle->data.response.map)
            free(entry.value);
        free(*client_handle->data.response.map);
        free(client_handle->data.response.map);
        return py_map;
    }
}

} /* namespace nel */

static PyMethodDef SimulatorMethods[] = {
    {"new",  nel::simulator_new, METH_VARARGS, "Creates a new simulator and returns its pointer."},
    {"load",  nel::simulator_load, METH_VARARGS, "Loads a simulator from file and returns its pointer."},
    {"delete",  nel::simulator_delete, METH_VARARGS, "Deletes an existing simulator."},
    {"start_server",  nel::simulator_start_server, METH_VARARGS, "Starts the simulator server."},
    {"stop_server",  nel::simulator_stop_server, METH_VARARGS, "Stops the simulator server."},
    {"start_client",  nel::simulator_start_client, METH_VARARGS, "Starts the simulator client."},
    {"stop_client",  nel::simulator_stop_client, METH_VARARGS, "Stops the simulator client."},
    {"add_agent",  nel::simulator_add_agent, METH_VARARGS, "Adds an agent to the simulator and returns its ID."},
    {"move",  nel::simulator_move, METH_VARARGS, "Attempts to move the agent in the simulation environment."},
    {"position",  nel::simulator_position, METH_VARARGS, "Gets the current position of an agent in the simulation environment."},
    {"scent",  nel::simulator_scent, METH_VARARGS, "Gets the current scent perception of the given agent."},
    {"vision",  nel::simulator_vision, METH_VARARGS, "Gets the current vision perception of the given agent."},
    {"collected_items",  nel::simulator_collected_items, METH_VARARGS, "Gets the counts of the items collected by the given agent."},
    {"map",  nel::simulator_map, METH_VARARGS, "Returns a list of patches within a given bounding box."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef simulator_module = {
        PyModuleDef_HEAD_INIT, "simulator_c", "Simulator", -1, SimulatorMethods, NULL, NULL, NULL, NULL
    };

    PyMODINIT_FUNC PyInit_simulator_c(void) {
        return PyModule_Create(&simulator_module);
    }
#else
    PyMODINIT_FUNC initsimulator_c(void) {
        (void) Py_InitModule("simulator_c", SimulatorMethods);
    }
#endif

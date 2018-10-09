#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL NEL_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <thread>
#include "nel/gibbs_field.h"
#include "nel/mpi.h"
#include "nel/simulator.h"

namespace nel {

using namespace core;

/**
 * A pointer to the AddAgentError Python class. The function `import_errors`
 * must be called before this is useable.
 */
static PyObject* add_agent_error;

/**
 * A pointer to the MPIError Python class. The function `import_errors` must be
 * called before this is useable.
 */
static PyObject* mpi_error;


/**
 * A struct containing additional state information for the simulator. This
 * information includes a pointer to the `async_server` object, if the
 * simulator is run as a server, a pointer to the Python callback function,
 * the list of agent IDs owned by this simulator (as opposed to other clients),
 * and information for periodically saving the simulator to file.
 */
struct py_simulator_data
{
    char* save_directory;
    unsigned int save_directory_length;
    unsigned int save_frequency;
    async_server* server;
    PyObject* callback;

    /* agents owned by the simulator */
    array<uint64_t> agent_ids;

    py_simulator_data(const char* save_filepath,
            unsigned int save_filepath_length,
            unsigned int save_frequency,
            async_server* server,
            PyObject* callback) :
        save_frequency(save_frequency), server(server),
        callback(callback), agent_ids(16)
    {
        if (save_filepath == NULL) {
            save_directory = NULL;
        } else {
            save_directory = (char*) malloc(sizeof(char) * save_filepath_length);
            if (save_directory == NULL) {
                fprintf(stderr, "py_simulator_data ERROR: Out of memory.\n");
                exit(EXIT_FAILURE);
            }
            save_directory_length = save_filepath_length;
            for (unsigned int i = 0; i < save_filepath_length; i++)
                save_directory[i] = save_filepath[i];
        }
        Py_INCREF(callback);
    }

    ~py_simulator_data() { free_helper(); }

    static inline void free(py_simulator_data& data) {
        data.free_helper();
        core::free(data.agent_ids);
    }

private:
    inline void free_helper() {
        if (save_directory != NULL)
            core::free(save_directory);
        if (callback != NULL)
            Py_DECREF(callback);
    }
};

/**
 * Initializes `data` by copying the contents from `src`.
 *
 * \param   data      The `py_simulator_data` structure to initialize.
 * \param   src       The source `py_simulator_data` structure that will be
 *                    copied to initialize `data`.
 * \returns `true` if successful; and `false` otherwise.
 */
inline bool init(py_simulator_data& data, const py_simulator_data& src)
{
    if (!array_init(data.agent_ids, src.agent_ids.capacity))
        return false;
    data.agent_ids.append(src.agent_ids.data, src.agent_ids.length);

    if (src.save_directory != NULL) {
        data.save_directory = (char*) malloc(sizeof(char) * max(1u, src.save_directory_length));
        if (data.save_directory == NULL) {
            fprintf(stderr, "init ERROR: Insufficient memory for py_simulator_data.save_directory.\n");
            free(data.agent_ids); return false;
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

/**
 * A struct containing additional state information for the client. This
 * information includes responses from the server, pointers to Python callback
 * functions, and variables for synchronizing communication between the client
 * response listener thread and the Python thread.
 */
struct py_client_data {
    /* storing the server responses */
    union response {
        bool action_result;
        PyObject* agent_states;
        hash_map<position, patch_state>* map;
    } response;

    /* for synchronization */
    bool waiting_for_server;
    std::mutex lock;
    std::condition_variable cv;

    PyObject* step_callback;
    PyObject* lost_connection_callback;

    static inline void free(py_client_data& data) {
        if (data.step_callback != NULL)
            Py_DECREF(data.step_callback);
        if (data.lost_connection_callback != NULL)
            Py_DECREF(data.lost_connection_callback);
        data.lock.~mutex();
        data.cv.~condition_variable();
    }
};

inline bool init(py_client_data& data) {
    data.step_callback = NULL;
    data.lost_connection_callback = NULL;
    new (&data.lock) std::mutex();
    new (&data.cv) std::condition_variable();
    return true;
}

/**
 * Converts the given Python list of floating points to an native array of
 * floats. The native array and its size are returned as a `core::pair`. If an
 * error occurs, the native array will be returned as `NULL`.
 *
 * \param   arg     Pointer to the Python list of floats.
 * \returns A pair containing a pointer to the native array of floats and its
 *          length. The pointer is `NULL` upon error.
 */
static pair<float*, Py_ssize_t> PyArg_ParseFloatList(PyObject* arg, Py_ssize_t start=0) {
    if (!PyList_Check(arg)) {
        PyErr_SetString(PyExc_ValueError, "Expected float list, but got invalid argument.");
        return make_pair<float*, Py_ssize_t>(NULL, 0);
    }
    Py_ssize_t len = PyList_Size(arg);
    float* items = (float*) malloc(max((size_t) 1, sizeof(float) * (len - start)));
    if (items == NULL) {
        PyErr_NoMemory();
        return make_pair<float*, Py_ssize_t>(NULL, 0);
    }
    for (Py_ssize_t i = start; i < len; i++)
        items[i - start] = (float) PyFloat_AsDouble(PyList_GetItem(arg, i));
    return make_pair(items, len - start);
}

/**
 * Saves the simulator given by the specified pointer `sim` to the filepath
 * specified by the `py_simulator_data` structure inside `sim`.
 *
 * \param   sim     The simulator to save.
 * \param   time    The simulation time of `sim`.
 * \returns `true` if successful; and `false` otherwise.
 */
bool save(const simulator<py_simulator_data>* sim, uint64_t time)
{
    int length = snprintf(NULL, 0, "%" PRIu64, time);
    if (length < 0) {
        fprintf(stderr, "on_step ERROR: Error computing filepath to save simulation.\n");
        return false;
    }

    const py_simulator_data& data = sim->get_data();
    char* filepath = (char*) malloc(sizeof(char) * (data.save_directory_length + length + 1));
    if (filepath == NULL) {
        fprintf(stderr, "on_step ERROR: Insufficient memory for filepath.\n");
        return false;
    }

    for (unsigned int i = 0; i < data.save_directory_length; i++)
        filepath[i] = data.save_directory[i];
    snprintf(filepath + data.save_directory_length, length + 1, "%" PRIu64, time);

    FILE* file = open_file(filepath, "wb");
    if (file == NULL) {
        fprintf(stderr, "on_step: Unable to open '%s' for writing. ", filepath);
        perror(""); return false;
    }

    fixed_width_stream<FILE*> out(file);
    bool result = write(*sim, out)
               && write(data.agent_ids.length, out)
               && write(data.agent_ids.data, out, data.agent_ids.length);
    fclose(file);
    return result;
}

/**
 * Constructs the Python objects `py_position`, `py_scent`, `py_vision`, and
 * `py_items` and stores the state of the given `agent`.
 *
 * \param   agent       The agent whose state is copied into the Python
 *                      objects.
 * \param   config      The configuration of the simulator containing `agent`.
 * \param   py_position An output numpy array of type int64 that will contain
 *                      the position of `agent`.
 * \param   py_scent    An output numpy array of type float that will contain
 *                      the current perceived scent of `agent`. This array will
 *                      have length equal to `config.scent_dimension`.
 * \param   py_vision   The output numpy array of type float that will contain
 *                      the current perceived vision of `agent`. This array
 *                      will have shape
 *                      `(2*config.vision_range + 1, 2*config.vision_range + 1, config.color_dimension)`.
 * \param   py_items    The output numpy array of type uint64 that will contain
 *                      the counts of the collected items. This array is
 *                      parallel to the array of `item_types` in `config`.
 * \returns `true` if successful; and `false` otherwise. Upon failure,
 *          `py_position`, `py_scent`, `py_vision`, and `py_items` are
 *          uninitialized.
 */
static inline bool build_py_agent(
        const agent_state& agent,
        const simulator_config& config,
        PyObject*& py_position,
        PyObject*& py_direction,
        PyObject*& py_scent,
        PyObject*& py_vision,
        PyObject*& py_items)
{
    /* first copy all arrays in 'agent' */
    int64_t* positions = (int64_t*) malloc(sizeof(int64_t) * 2);
    if (positions == NULL) {
        PyErr_NoMemory();
        return false;
    }
    float* scent = (float*) malloc(sizeof(float) * config.scent_dimension);
    if (scent == NULL) {
        PyErr_NoMemory(); free(positions);
        return false;
    }
    unsigned int vision_size = (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension;
    float* vision = (float*) malloc(sizeof(float) * vision_size);
    if (vision == NULL) {
        PyErr_NoMemory(); free(positions); free(scent);
        return false;
    }
    uint64_t* items = (uint64_t*) malloc(sizeof(uint64_t) * config.item_types.length);
    if (items == NULL) {
        PyErr_NoMemory(); free(positions); free(scent); free(vision);
        return false;
    }

    positions[0] = agent.current_position.x;
    positions[1] = agent.current_position.y;
    for (unsigned int i = 0; i < config.scent_dimension; i++)
        scent[i] = agent.current_scent[i];
    for (unsigned int i = 0; i < vision_size; i++)
        vision[i] = agent.current_vision[i];
    for (unsigned int i = 0; i < config.item_types.length; i++)
        items[i] = agent.collected_items[i];

    npy_intp pos_dim[] = {2};
    npy_intp scent_dim[] = {(npy_intp) config.scent_dimension};
    npy_intp vision_dim[] = {
            2 * (npy_intp) config.vision_range + 1,
            2 * (npy_intp) config.vision_range + 1,
            (npy_intp) config.color_dimension};
    npy_intp items_dim[] = {(npy_intp) config.item_types.length};
    py_position = PyArray_SimpleNewFromData(1, pos_dim, NPY_INT64, positions);
    py_direction = PyLong_FromSize_t((size_t) agent.current_direction);
    py_scent = PyArray_SimpleNewFromData(1, scent_dim, NPY_FLOAT, scent);
    py_vision = PyArray_SimpleNewFromData(3, vision_dim, NPY_FLOAT, vision);
    py_items = PyArray_SimpleNewFromData(1, items_dim, NPY_UINT64, items);
    return true;
}

/**
 * Constructs a Python tuple containing the position, current scent perception,
 * current visual perception, the collected item counts, and the ID of the
 * given `agent`.
 *
 * \param   agent    The agent whose state to copy into the Python objects.
 * \param   config   The configuration of the simulator containing `agent`.
 * \param   agent_id The ID of `agent` in the simulator.
 * \returns A pointer to the constructed Python tuple, if successful; `NULL`
 *          otherwise.
 */
static PyObject* build_py_agent(
        const agent_state& agent,
        const simulator_config& config,
        uint64_t agent_id)
{
    PyObject* py_position; PyObject* py_direction;
    PyObject* py_scent; PyObject* py_vision; PyObject* py_items;
    if (!build_py_agent(agent, config, py_position, py_direction, py_scent, py_vision, py_items))
        return NULL;
    return Py_BuildValue("(OOOOOO)", py_position, py_direction, py_scent, py_vision, py_items, PyLong_FromUnsignedLongLong(agent_id));
}

/**
 * Constructs a Python tuple containing the position, current scent perception,
 * current visual perception, and the collected item counts of the given
 * `agent`.
 *
 * \param   agent    The agent whose state to copy into the Python objects.
 * \param   config   The configuration of the simulator containing `agent`.
 * \returns A pointer to the constructed Python tuple, if successful; `NULL`
 *          otherwise.
 */
static PyObject* build_py_agent(
        const agent_state& agent,
        const simulator_config& config)
{
    PyObject* py_position; PyObject* py_direction;
    PyObject* py_scent; PyObject* py_vision; PyObject* py_items;
    if (!build_py_agent(agent, config, py_position, py_direction, py_scent, py_vision, py_items))
        return NULL;
    return Py_BuildValue("(OOOOO)", py_position, py_direction, py_scent, py_vision, py_items);
}

/**
 * The callback function invoked by the simulator when time is advanced. This
 * function is only called if the simulator is run in locally or as a server.
 * This function first checks if the simulator should be saved to file. Next,
 * in server mode, the simulator sends a step response message to all connected
 * clients. Finally, it constructs a Python list of agent states and invokes
 * the Python callback in `data.callback`.
 *
 * \param   sim     The simulator invoking this function.
 * \param   agents  The underlying array of all agents in `sim`.
 * \param   time    The new simulation time of `sim`.
 */
void on_step(const simulator<py_simulator_data>* sim,
        const array<agent_state*>& agents, uint64_t time)
{
    bool saved = false;
    const py_simulator_data& data = sim->get_data();
    if (data.save_directory != NULL && time % data.save_frequency == 0) {
        /* save the simulator to a local file */
        saved = save(sim, time);
    } if (data.server != NULL) {
        /* this simulator is a server, so send a step response to every client */
        if (!send_step_response(*data.server, agents, sim->get_config(), saved))
            fprintf(stderr, "on_step ERROR: send_step_response failed.\n");
    }

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); /* acquire global interpreter lock */
    PyObject* py_states = PyList_New(data.agent_ids.length);
    if (py_states == NULL) {
        fprintf(stderr, "on_step ERROR: PyList_New returned NULL.\n");
        PyGILState_Release(gstate); /* release global interpreter lock */
        return;
    }
    const simulator_config& config = sim->get_config();
    for (size_t i = 0; i < data.agent_ids.length; i++)
        PyList_SetItem(py_states, i, build_py_agent(*agents[(size_t) data.agent_ids[i]], config, data.agent_ids[i]));

    /* call python callback */
    PyObject* py_saved = saved ? Py_True : Py_False;
    Py_INCREF(py_saved);
    PyObject* args = Py_BuildValue("(OO)", py_states, py_saved);
    PyObject* result = PyEval_CallObject(data.callback, args);
    Py_DECREF(args);
    if (result != NULL)
        Py_DECREF(result);
    PyGILState_Release(gstate); /* release global interpreter lock */
}


/**
 * Client callback functions.
 */

/**
 * The callback invoked when the client receives an add_agent response from the
 * server. This function copies the agent state into a Python object, stores
 * it in `c.data.response.agent_states`, and wakes up the Python thread (which
 * should be waiting in the `simulator_add_agent` function) so that it can
 * return the response back to Python.
 *
 * \param   c         The client that received the response.
 * \param   agent_id  The ID of the new agent. This is equal to `UINT64_MAX` if
 *                    the server returned an error.
 * \param   new_agent The state of the new agent.
 */
void on_add_agent(client<py_client_data>& c,
        uint64_t agent_id, const agent_state& new_agent)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); /* acquire global interpreter lock */
    PyObject* agent = (agent_id == UINT64_MAX) ? NULL : build_py_agent(new_agent, c.config, agent_id);
    PyGILState_Release(gstate);

    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.response.agent_states = agent;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a move response from the
 * server. This function copies the result into `c.data.response.action_result`
 * and wakes up the Python thread (which should be waiting in the
 * `simulator_move` function) so that it can return the response back to
 * Python.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to move.
 * \param   request_success Indicates whether the move request was successfully
 *                          enqueued by the simulator server.
 */
void on_move(client<py_client_data>& c, uint64_t agent_id, bool request_success) {
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.response.action_result = request_success;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a turn response from the
 * server. This function copies the result into `c.data.response.action_result`
 * and wakes up the Python thread (which should be waiting in the
 * `simulator_turn` function) so that it can return the response back to
 * Python.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to turn.
 * \param   request_success Indicates whether the turn request was successfully
 *                          enqueued by the simulator server.
 */
void on_turn(client<py_client_data>& c, uint64_t agent_id, bool request_success) {
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.response.action_result = request_success;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a get_map response from the
 * server. This function moves the result into `c.data.response.map` and wakes
 * up the Python thread (which should be waiting in the `simulator_map`
 * function) so that it can return the response back to Python.
 *
 * \param   c       The client that received the response.
 * \param   map     A map from patch positions to `patch_state` structures
 *                  containing the state information in each patch.
 */
void on_get_map(client<py_client_data>& c,
        hash_map<position, patch_state>* map)
{
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.response.map = map;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a step response from the
 * server. This function constructs a Python list of agent states governed by
 * this client and invokes the Python function `c.data.step_callback`.
 *
 * \param   c            The client that received the response.
 * \param   agent_ids    An array of agent IDs governed by the client.
 * \param   agent_states An array, parallel to `agent_ids`, containing the
 *                       state information of each agent at the beginning of
 *                       the new time step in the simulation.
 */
void on_step(client<py_client_data>& c,
        const array<uint64_t>& agent_ids,
        const agent_state* agent_states)
{
    bool saved;
    if (!read(saved, c.connection)) return;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); /* acquire global interpreter lock */
    PyObject* py_states = PyList_New(agent_ids.length);
    if (py_states == NULL) {
        fprintf(stderr, "on_step ERROR: PyList_New returned NULL.\n");
        PyGILState_Release(gstate); /* release global interpreter lock */
        return;
    }
    for (size_t i = 0; i < agent_ids.length; i++)
        PyList_SetItem(py_states, i, build_py_agent(agent_states[i], c.config, agent_ids[i]));

    /* invoke python callback */
    PyObject* py_saved = saved ? Py_True : Py_False;
    Py_INCREF(py_saved);
    PyObject* args = Py_BuildValue("(OO)", py_states, py_saved);
    PyObject* result = PyEval_CallObject(c.data.step_callback, args);
    Py_DECREF(args);
    if (result != NULL)
        Py_DECREF(result);
    PyGILState_Release(gstate); /* release global interpreter lock */
}

/**
 * The callback invoked when the client loses the connection to the server.
 * \param   c       The client whose connection to the server was lost.
 */
void on_lost_connection(client<py_client_data>& c) {
    fprintf(stderr, "Client lost connection to server.\n");
    c.client_running = false;
    c.data.cv.notify_one();

    /* invoke python callback */
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyObject* args = Py_BuildValue("()");
    PyObject* result = PyEval_CallObject(c.data.lost_connection_callback, args);
    Py_DECREF(args);
    if (result != NULL)
        Py_DECREF(result);
    PyGILState_Release(gstate);
}

/**
 * This functions waits for a response from the server, and for one of the
 * above client callback functions to be invoked. Since this waiting is a
 * blocking operation, it releases the Python global interpreter lock, and
 * re-acquires it before returning.
 * \param   c       The client expecting a response from the server.
 */
inline void wait_for_server(client<py_client_data>& c)
{
    /* release the global interpreter lock */
    PyThreadState* python_thread = PyEval_SaveThread();

    std::unique_lock<std::mutex> lck(c.data.lock);
    while (c.data.waiting_for_server && c.client_running)
        c.data.cv.wait(lck);

    /* re-acquire the global interpreter lock */
    PyEval_RestoreThread(python_thread);
}

/**
 * Imports the Python exception classes from the nel module.
 */
static inline void import_errors() {
#if PY_MAJOR_VERSION >= 3
    PyObject* module_name = PyUnicode_FromString("nel");
#else
    PyObject* module_name = PyString_FromString("nel");
#endif
    PyObject* module = PyImport_Import(module_name);
    PyObject* module_dict = PyModule_GetDict(module);
    add_agent_error = PyDict_GetItemString(module_dict, "AddAgentError");
    mpi_error = PyDict_GetItemString(module_dict, "MPIError");
    Py_DECREF(module_name); Py_DECREF(module);
}

/**
 * Creates a new simulator and returns a handle to it.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    A Python tuple containing the arguments to this function:
 *                  - (int) The seed for the pseudorandom number generator.
 *                  - (int) The maximum movement distance per turn for all
 *                  agents.
 *                  - (int) The scent dimension.
 *                  - (int) The color dimension for visual perception.
 *                  - (int) The range of vision for all agents.
 *                  - (int) The patch size.
 *                  - (int) The number of Gibbs sampling iterations when
 *                    initializing items in new patches.
 *                  - (list) A list of the item types.
 *                  - (list of floats) The color of all agents.
 *                  - (int) The movement conflict resolution policy.
 *                  - (float) The scent decay parameter.
 *                  - (float) The scent diffusion parameter.
 *                  - (int) The duration of time for which removed items are
 *                    remembered by the simulation in order to compute their
 *                    scent contribution.
 *                  - (function) The function to invoke when the simulator
 *                    advances time.
 *                  - (int) The frequency by which the simulator is saved to
 *                    file.
 *                  - (string) The filepath to save the simulator (the
 *                    simulator time will be appended to this filepath when
 *                    saving).
 *
 *                  The list of item types must contain tuples containing:
 *                  - (string) The name.
 *                  - (list of floats) The item scent.
 *                  - (list of floats) The item color.
 *                  - (list of ints) The number of items of each type that is
 *                    required to automatically collect items of this type.
 *                  - (list of ints) The number of items of each type that is
 *                    removed from the agent's inventory whenever an item of
 *                    this type is collected.
 *                  - (bool) Whether this item type blocks agent movement.
 *                  - (int) The ID of the intensity function.
 *                  - (list of floats) The arguments to the intensity function.
 *                  - (list of list of floats) The list of interaction
 *                    functions, where the first element in each sublist is the
 *                    ID of the interaction function, and the remaining
 *                    elements are its arguments.
 * \returns Pointer to the new simulator.
 */
static PyObject* simulator_new(PyObject *self, PyObject *args)
{
    simulator_config config;
    PyObject* py_allowed_movement_directions;
    PyObject* py_allowed_turn_directions;
    PyObject* py_items;
    PyObject* py_agent_color;
    unsigned int seed;
    unsigned int collision_policy;
    PyObject* py_callback;
    unsigned int save_frequency;
    char* save_filepath;
    if (!PyArg_ParseTuple(
      args, "IIOOIIIIIOOIffIOIz", &seed, &config.max_steps_per_movement,
      &py_allowed_movement_directions, &py_allowed_turn_directions, &config.scent_dimension,
      &config.color_dimension, &config.vision_range, &config.patch_size, &config.gibbs_iterations,
      &py_items, &py_agent_color, &collision_policy, &config.decay_param, &config.diffusion_param,
      &config.deleted_item_lifetime, &py_callback, &save_frequency, &save_filepath)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.new'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable.\n");
        return NULL;
    } else if (!PyList_Check(py_items)) {
        PyErr_SetString(PyExc_TypeError, "'items' must be a list.\n");
        return NULL;
    } else if (!PyList_Check(py_allowed_movement_directions)) {
        PyErr_SetString(PyExc_TypeError, "'allowed_movement_directions' must be a list.\n");
        return NULL;
    } else if (!PyList_Check(py_allowed_turn_directions)) {
        PyErr_SetString(PyExc_TypeError, "'allowed_turn_directions' must be a list.\n");
        return NULL;
    }

    PyObject *py_items_iter = PyObject_GetIter(py_items);
    if (!py_items_iter) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument types in the call to 'simulator_c.new'.");
        return NULL;
    }
    Py_ssize_t item_type_count = PyList_Size(py_items);
    while (true) {
        PyObject *next_py_item = PyIter_Next(py_items_iter);
        if (!next_py_item) break;

        char* name;
        PyObject* py_scent;
        PyObject* py_color;
        PyObject* py_required_item_counts;
        PyObject* py_required_item_costs;
        PyObject* blocks_movement;
        unsigned int py_intensity_fn;
        PyObject* py_intensity_fn_args;
        PyObject* py_interaction_fn_args;
        if (!PyArg_ParseTuple(next_py_item, "sOOOOOIOO", &name, &py_scent, &py_color, &py_required_item_counts,
          &py_required_item_costs, &blocks_movement, &py_intensity_fn, &py_intensity_fn_args, &py_interaction_fn_args)) {
            fprintf(stderr, "Invalid argument types for item property in call to 'simulator_c.new'.\n");
            return NULL;
        }

        if (!PyList_Check(py_intensity_fn_args) || !PyList_Check(py_interaction_fn_args)) {
            PyErr_SetString(PyExc_TypeError, "'intensity_fn_args' and 'interaction_fn_args' must be lists.\n");
            return NULL;
        }

        item_properties& new_item = config.item_types[config.item_types.length];
        init(new_item.name, name);
        new_item.scent = PyArg_ParseFloatList(py_scent).key;
        new_item.color = PyArg_ParseFloatList(py_color).key;
        new_item.required_item_counts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
        for (Py_ssize_t i = 0; i < item_type_count; i++)
            new_item.required_item_counts[i] = PyLong_AsUnsignedLong(PyList_GetItem(py_required_item_counts, i));
        new_item.required_item_costs = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
        for (Py_ssize_t i = 0; i < item_type_count; i++)
            new_item.required_item_costs[i] = PyLong_AsUnsignedLong(PyList_GetItem(py_required_item_costs, i));
        new_item.blocks_movement = (blocks_movement == Py_True);

        pair<float*, Py_ssize_t> intensity_fn_args = PyArg_ParseFloatList(py_intensity_fn_args);
        new_item.intensity_fn = get_intensity_fn((intensity_fns) py_intensity_fn,
                intensity_fn_args.key, (unsigned int) intensity_fn_args.value);
        if (new_item.intensity_fn == NULL) {
            PyErr_SetString(PyExc_ValueError, "Invalid intensity"
                    " function arguments in the call to 'simulator_c.new'.");
            return NULL;
        }
        new_item.intensity_fn_args = intensity_fn_args.key;
        new_item.intensity_fn_arg_count = (unsigned int) intensity_fn_args.value;
        new_item.interaction_fns = (interaction_function*) malloc(sizeof(interaction_function) * item_type_count);
        new_item.interaction_fn_args = (float**) malloc(sizeof(float*) * item_type_count);
        new_item.interaction_fn_arg_counts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
        for (Py_ssize_t i = 0; i < item_type_count; i++) {
            PyObject* sublist = PyList_GetItem(py_interaction_fn_args, i);
            unsigned int py_interaction_fn = PyLong_AsUnsignedLong(PyList_GetItem(sublist, 0));

            pair<float*, Py_ssize_t> interaction_fn_args = PyArg_ParseFloatList(sublist, 1);
            new_item.interaction_fns[i] = get_interaction_fn((interaction_fns) py_interaction_fn,
                    interaction_fn_args.key, (unsigned int) interaction_fn_args.value);
            new_item.interaction_fn_args[i] = interaction_fn_args.key;
            new_item.interaction_fn_arg_counts[i] = (unsigned int) interaction_fn_args.value;
            if (new_item.interaction_fns[i] == NULL) {
                PyErr_SetString(PyExc_ValueError, "Invalid interaction"
                        " function arguments in the call to 'simulator_c.new'.");
                return NULL;
            }
        }
        config.item_types.length += 1;
    }

    Py_ssize_t allowed_movement_direction_count = PyList_Size(py_allowed_movement_directions);
    Py_ssize_t allowed_turn_direction_count = PyList_Size(py_allowed_turn_directions);
    memset(config.allowed_movement_directions, 0, sizeof(bool) * (size_t) direction::COUNT);
    memset(config.allowed_rotations, 0, sizeof(bool) * (size_t) direction::COUNT);
    for (Py_ssize_t i = 0; i < allowed_movement_direction_count; i++)
        config.allowed_movement_directions[PyLong_AsUnsignedLong(PyList_GetItem(py_allowed_movement_directions, i))] = true;
    for (Py_ssize_t i = 0; i < allowed_turn_direction_count; i++)
        config.allowed_rotations[PyLong_AsUnsignedLong(PyList_GetItem(py_allowed_turn_directions, i))] = true;

    config.agent_color = PyArg_ParseFloatList(py_agent_color).key;
    config.collision_policy = (movement_conflict_policy) collision_policy;

    py_simulator_data data(save_filepath,
            (save_filepath == NULL) ? 0 : strlen(save_filepath),
            save_frequency, NULL, py_callback);

    simulator<py_simulator_data>* sim =
            (simulator<py_simulator_data>*) malloc(sizeof(simulator<py_simulator_data>));
    if (sim == NULL) {
        PyErr_NoMemory();
        return NULL;
    } else if (!init(*sim, config, data, seed)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize simulator.");
        return NULL;
    }
    import_errors();
    return PyLong_FromVoidPtr(sim);
}

/**
 * Loads a simulator from file.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    A Python tuple containing the arguments to this function:
 *                  - (string) The full path to the file from which to load the
 *                    simulator.
 *                  - (function) The callback to invoke whenever the simulator
 *                    advances time.
 *                  - (int) The frequency by which the simulator is saved to
 *                    file.
 *                  - (string) The filepath to save the simulator (the
 *                    simulator time will be appended to this filepath when
 *                    saving).
 * \returns A Python tuple containing:
 *          - The simulation time.
 *          - A pointer to the loaded simulator.
 *          - A list of tuples containing the states of the agents governed by
 *            this simulator (not including agents governed by other clients).
 *            See `build_py_agent` for details on the contents of each tuple.
 */
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
        PyErr_NoMemory(); return NULL;
    }

    py_simulator_data data(save_filepath,
            (save_filepath == NULL) ? 0 : strlen(save_filepath),
            save_frequency, NULL, py_callback);

    FILE* file = open_file(load_filepath, "rb");
    if (file == NULL) {
        PyErr_SetFromErrno(PyExc_OSError);
        free(sim); return NULL;
    }
    size_t agent_id_count;
    fixed_width_stream<FILE*> in(file);
    if (!read(*sim, in, data)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load simulator.");
        free(sim); fclose(file); return NULL;
    }
    py_simulator_data& sim_data = sim->get_data();
    if (!read(agent_id_count, in)
            || !sim_data.agent_ids.ensure_capacity(agent_id_count)
            || !read(sim_data.agent_ids.data, in, agent_id_count))
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load agent IDs.");
        free(*sim); free(sim); fclose(file); return NULL;
    }
    sim_data.agent_ids.length = agent_id_count;
    fclose(file);

    /* parse the list of agent IDs from Python */
    agent_state** agent_states = (agent_state**) malloc(sizeof(agent_state*) * agent_id_count);
    if (agent_states == NULL) {
        PyErr_NoMemory();
        free(*sim); free(sim); fclose(file); return NULL;
    }

    sim->get_agent_states(agent_states, sim_data.agent_ids.data, (unsigned int) agent_id_count);

    const simulator_config& config = sim->get_config();
    PyObject* py_states = PyList_New((Py_ssize_t) agent_id_count);
    if (py_states == NULL) {
        free(agent_states); free(*sim);
        free(sim); fclose(file); return NULL;
    }
    for (size_t i = 0; i < agent_id_count; i++)
        PyList_SetItem(py_states, (Py_ssize_t) i, build_py_agent(*agent_states[i], config, sim_data.agent_ids[i]));
    free(agent_states);

    import_errors();
    return Py_BuildValue("(LOO)", sim->time, PyLong_FromVoidPtr(sim), py_states);
}

/**
 * Deletes a simulator and frees all memory allocated for that simulator.
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
 *                  - (int) Server port.
 *                  - (int) Maximum number of new simultaneous connections.
 *                  - (int) Number of threads to process server messages.
 * \returns Handle to the simulator server.
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
 * Stops the simulator server and frees all associated system resources.
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

/**
 * Starts a client and connects it to the specified simulator server.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - (string) The server address.
 *                  - (int) The server port.
 *                  - (function) The Python function to invoke whenever the
 *                    simulator advances time.
 *                  - (function) The Python function to invoke if the client
 *                    loses its connection to the server.
 *                  - (list of ints) The IDs of the agents governed by this
 *                    client (this should be an empty list if the client
 *                    hasn't added agents yet).
 * \returns A Python tuple containing:
 *          - The simulation time.
 *          - A handle to the client.
 *          - A list of tuples containing the states of the agents governed by
 *            this simulator (not including agents governed by other clients).
 *            See `build_py_agent` for details on the contents of each tuple.
 */
static PyObject* simulator_start_client(PyObject *self, PyObject *args)
{
    char* server_address;
    unsigned int port;
    PyObject* py_step_callback;
    PyObject* py_lost_connection_callback;
    PyObject* py_agent_ids;
    if (!PyArg_ParseTuple(args, "sIOOO", &server_address, &port, &py_step_callback, &py_lost_connection_callback, &py_agent_ids)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.start_client'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_step_callback) || !PyCallable_Check(py_lost_connection_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callbacks must be callable.\n");
        return NULL;
    }

    /* parse the list of agent IDs from Python */
    Py_ssize_t agent_count = PyList_Size(py_agent_ids);
    uint64_t* agent_ids = (uint64_t*) malloc(sizeof(uint64_t) * agent_count);
    agent_state* agent_states = (agent_state*) malloc(sizeof(agent_state) * agent_count);
    if (agent_ids == NULL || agent_states == NULL) {
        if (agent_ids != NULL) free(agent_ids);
        PyErr_NoMemory(); return NULL;
    }
    for (Py_ssize_t i = 0; i < agent_count; i++)
        agent_ids[i] = PyLong_AsUnsignedLongLong(PyList_GetItem(py_agent_ids, i));

    client<py_client_data>* new_client =
            (client<py_client_data>*) malloc(sizeof(client<py_client_data>));
    if (new_client == NULL) {
        PyErr_NoMemory();
        free(agent_ids); free(agent_states);
        return NULL;
    } else if (!init(*new_client)) {
        PyErr_NoMemory();
        free(agent_ids); free(agent_states);
        free(new_client); return NULL;
    }

    uint64_t simulator_time = init_client(*new_client, server_address,
            (uint16_t) port, agent_ids, agent_states, (unsigned int) agent_count);
    if (simulator_time == UINT64_MAX) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI client.");
        free(*new_client); free(new_client); return NULL;
    }

    PyObject* py_states = PyList_New(agent_count);
    if (py_states == NULL) return NULL;
    for (Py_ssize_t i = 0; i < agent_count; i++) {
        PyList_SetItem(py_states, i, build_py_agent(agent_states[i], new_client->config));
        free(agent_states[i]);
    }
    free(agent_states); free(agent_ids);

    new_client->data.step_callback = py_step_callback;
    new_client->data.lost_connection_callback = py_lost_connection_callback;
    Py_INCREF(py_step_callback);
    Py_INCREF(py_lost_connection_callback);
    import_errors();
    return Py_BuildValue("(LOO)", simulator_time, PyLong_FromVoidPtr(new_client), py_states);
}

/**
 * Stops the specified client and frees all associated system resources.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native client object as a PyLong.
 * \returns None.
 */
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
 *                  - Handle to the native client object as a PyLong. If this
 *                    is None, `add_agent` is directly invoked on the simulator
 *                    object. Otherwise, the client sends an add_agent message
 *                    to the server and waits for its response.
 * \returns Pointer to a tuple containing the new agent's state. See
 *          `build_py_agent` for details on the contents of this tuple.
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
        pair<uint64_t, agent_state*> new_agent = sim_handle->add_agent();
        if (new_agent.key == UINT64_MAX) {
            PyErr_SetString(add_agent_error, "Failed to add new agent.");
            return NULL;
        }
        sim_handle->get_data().agent_ids.add(new_agent.key);
        std::unique_lock<std::mutex> lock(new_agent.value->lock);
        return Py_BuildValue("O", build_py_agent(*new_agent.value, sim_handle->get_config(), new_agent.key));
    } else {
        /* this is a client, so send an add_agent message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!client_handle->client_running) {
            PyErr_SetString(mpi_error, "Connection to the server was lost.");
            return NULL;
        }

        client_handle->data.waiting_for_server = true;
        if (!send_add_agent(*client_handle)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send add_agent request.");
            return NULL;
        }

        /* wait for response from server */
        wait_for_server(*client_handle);

        if (client_handle->data.response.agent_states == NULL) {
            /* server returned failure */
            PyErr_SetString(add_agent_error, "Failed to add new agent.");
            return NULL;
        }

        return client_handle->data.response.agent_states;
    }
}

/**
 * Attempt to move the agent in the simulation environment. If the agent
 * already has an action queued for this turn, this attempt will fail.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong. If this
 *                    is None, `move` is directly invoked on the simulator
 *                    object. Otherwise, the client sends a move message to the
 *                    server and waits for its response.
 *                  - Agent ID.
 *                  - Move direction encoded as an integer:
 *                      FORWARD = 0,
 *                      BACKWARD = 1,
 *                      LEFT = 2,
 *                      RIGHT = 3.
 *                  - Number of steps.
 * \returns `True` if the move command is successfully queued; `False`
 *          otherwise.
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

        /* release the global interpreter lock */
        PyThreadState* python_thread = PyEval_SaveThread();
        bool result = sim_handle->move(agent_id, (direction) dir, num_steps);

        /* re-acquire the global interpreter lock and return */
        PyEval_RestoreThread(python_thread);
        PyObject* py_result = (result ? Py_True : Py_False);
        Py_INCREF(py_result); return py_result;
    } else {
        /* this is a client, so send a move message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!client_handle->client_running) {
            PyErr_SetString(mpi_error, "Connection to the server was lost.");
            return NULL;
        }

        client_handle->data.waiting_for_server = true;
        if (!send_move(*client_handle, agent_id, (direction) dir, num_steps)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send move request.");
            return NULL;
        }

        /* wait for response from server */
        wait_for_server(*client_handle);

        PyObject* result = (client_handle->data.response.action_result ? Py_True : Py_False);
        Py_INCREF(result);
        return result;
    }
}

/**
 * Attempt to turn the agent in the simulation environment. If the agent
 * already has an action queued for this turn, this attempt will fail.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong. If this
 *                    is None, `turn` is directly invoked on the simulator
 *                    object. Otherwise, the client sends a turn message to the
 *                    server and waits for its response.
 *                  - Agent ID.
 *                  - Turn direction encoded as an integer:
 *                      NO_CHANGE = 0,
 *                      REVERSE = 1,
 *                      LEFT = 2,
 *                      RIGHT = 3.
 * \returns `True` if the turn command is successfully queued; `False`
 *          otherwise.
 */
static PyObject* simulator_turn(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    unsigned int dir;
    if (!PyArg_ParseTuple(args, "OOKI", &py_sim_handle, &py_client_handle, &agent_id, &dir))
        return NULL;
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call turn directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);

        /* release the global interpreter lock */
        PyThreadState* python_thread = PyEval_SaveThread();
        bool result = sim_handle->turn(agent_id, (direction) dir);

        /* re-acquire the global interpreter lock and return */
        PyEval_RestoreThread(python_thread);
        PyObject* py_result = (result ? Py_True : Py_False);
        Py_INCREF(py_result); return py_result;
    } else {
        /* this is a client, so send a turn message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!client_handle->client_running) {
            PyErr_SetString(mpi_error, "Connection to the server was lost.");
            return NULL;
        }

        client_handle->data.waiting_for_server = true;
        if (!send_turn(*client_handle, agent_id, (direction) dir)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send turn request.");
            return NULL;
        }

        /* wait for response from server */
        wait_for_server(*client_handle);

        PyObject* result = (client_handle->data.response.action_result ? Py_True : Py_False);
        Py_INCREF(result);
        return result;
    }
}

/**
 * Constructs a Python list containing tuples, where each tuple contains the
 * state information of a patch in the given hash_map of patches.
 *
 * \param   patches A hash_map from patch positions to `patch_state` objects.
 * \param   config  The configuration of the simulator in which the patches
 *                  reside.
 * \returns A Python list containing tuples, where each tuple corresponds to a
 *          patch in `patches`, containing:
 *          - (tuple of 2 ints) The patch position.
 *          - (bool) Whether the patch is fixed.
 *          - (numpy array of floats) The scent at each cell in the patch. This
 *            array has shape `(n, n, config.scent_dimension)`.
 *          - (numpy array of floats) The color at each cell in the patch. This
 *            array has shape `(n, n, config.color_dimension)`.
 *          - (list) The list of items in this patch.
 *          - (list) The list of agents in this patch. The list contains tuples
 *            of 3 ints, the first two indicate the position of each agent, and
 *            the third indicates the direction.
 *
 *          The list of items contains a tuple for each item, where each tuple
 *          contains:
 *          - (int) The ID of the item type (which is an index into the array
 *            `config.item_types`).
 *          - (tuple of 2 ints) The position of the item.
 */
static PyObject* build_py_map(
        const hash_map<position, patch_state>& patches,
        const simulator_config& config)
{
    unsigned int index = 0;
    PyObject* list = PyList_New(patches.table.size);
    for (const auto& entry : patches) {
        const patch_state& patch = entry.value;
        PyObject* py_items = PyList_New(patch.item_count);
        for (unsigned int i = 0; i < patch.item_count; i++)
            PyList_SetItem(py_items, i, Py_BuildValue("I(LL)",
                    patch.items[i].item_type,
                    patch.items[i].location.x,
                    patch.items[i].location.y));

        PyObject* py_agents = PyList_New(patch.agent_count);
        for (unsigned int i = 0; i < patch.agent_count; i++)
            PyList_SetItem(py_agents, i, Py_BuildValue("(LLL)", patch.agent_positions[i].x, patch.agent_positions[i].y, (long long) patch.agent_directions[i]));

        npy_intp n = (npy_intp) config.patch_size;
        float* scent = (float*) malloc(sizeof(float) * n * n * config.scent_dimension);
        float* vision = (float*) malloc(sizeof(float) * n * n * config.color_dimension);
        memcpy(scent, patch.scent, sizeof(float) * n * n * config.scent_dimension);
        memcpy(vision, patch.vision, sizeof(float) * n * n * config.color_dimension);

        npy_intp scent_dim[] = {n, n, (npy_intp) config.scent_dimension};
        npy_intp vision_dim[] = {n, n, (npy_intp) config.color_dimension};
        PyObject* py_scent = PyArray_SimpleNewFromData(3, scent_dim, NPY_FLOAT, scent);
        PyObject* py_vision = PyArray_SimpleNewFromData(3, vision_dim, NPY_FLOAT, vision);

        PyObject* fixed = patch.fixed ? Py_True : Py_False;
        Py_INCREF(fixed);
        PyObject* py_patch = Py_BuildValue("((LL)OOOOO)", patch.patch_position.x, patch.patch_position.y, fixed, py_scent, py_vision, py_items, py_agents);
        PyList_SetItem(list, index, py_patch);
        index++;
    }
    return list;
}

/**
 * Retrieves the state of the map within the specified bounding box.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong. If this
 *                    is None, `get_map` is directly invoked on the simulator
 *                    object. Otherwise, the client sends a get_map message to
 *                    the server and waits for its response.
 *                  - (tuple of 2 ints) The bottom-left corner of the bounding
 *                    box containing the patches to retrieve.
 *                  - (tuple of 2 ints) The top-right corner of the bounding
 *                    box containing the patches to retrieve.
 * \returns A Python list of tuples, where each tuple contains the state
 *          information of a patch within the bounding box. See `build_py_map`
 *          for details on the contents of each tuple.
 */
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
        if (!client_handle->client_running) {
            PyErr_SetString(mpi_error, "Connection to the server was lost.");
            return NULL;
        }

        client_handle->data.waiting_for_server = true;
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
    {"turn",  nel::simulator_turn, METH_VARARGS, "Attempts to turn the agent in the simulation environment."},
    {"map",  nel::simulator_map, METH_VARARGS, "Returns a list of patches within a given bounding box."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef simulator_module = {
        PyModuleDef_HEAD_INIT, "simulator_c", "Simulator", -1, SimulatorMethods, NULL, NULL, NULL, NULL
    };

    PyMODINIT_FUNC PyInit_simulator_c(void) {
        import_array();
        return PyModule_Create(&simulator_module);
    }
#else
    PyMODINIT_FUNC initsimulator_c(void) {
        import_array();
        (void) Py_InitModule("simulator_c", SimulatorMethods);
    }
#endif

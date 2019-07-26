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

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <thread>
#include "jbw/gibbs_field.h"
#include "jbw/mpi.h"
#include "jbw/simulator.h"

namespace jbw {

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
    async_server server;
    PyObject* callback;

    /* agents owned by the simulator */
    array<uint64_t> agent_ids;

    py_simulator_data(PyObject* callback) :
        callback(callback), agent_ids(16)
    {
        Py_INCREF(callback);
        server.status = server_status::STOPPING;
    }

    ~py_simulator_data() { free_helper(); }

    static inline void free(py_simulator_data& data) {
        data.free_helper();
        core::free(data.agent_ids);
        core::free(data.server);
    }

private:
    inline void free_helper() {
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

    /* async_server is not copyable */
    if (!init(data.server)) {
        free(data.agent_ids);
        return false;
    }
    data.callback = src.callback;
    Py_INCREF(data.callback);
    data.server.status = server_status::STOPPING;
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
    mpi_response server_response;
    union response_data {
        PyObject* agent_state;
        hash_map<position, patch_state>* map;
    } response_data;

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
    PyObject* py_agent_id = PyLong_FromUnsignedLongLong(agent_id);
    PyObject* py_agent = Py_BuildValue("(OOOOOO)", py_position, py_direction, py_scent, py_vision, py_items, py_agent_id);
    Py_DECREF(py_position); Py_DECREF(py_direction);
    Py_DECREF(py_scent); Py_DECREF(py_vision);
    Py_DECREF(py_items); Py_DECREF(py_agent_id);
    return py_agent;
}

/**
 * The callback function invoked by the simulator when time is advanced. This
 * function is only called if the simulator is run locally or as a server. In
 * server mode, the simulator sends a step response message to all connected
 * clients. Finally, it constructs a Python list of agent states and invokes
 * the Python callback in `data.callback`.
 *
 * \param   sim     The simulator invoking this function.
 * \param   agents  The underlying array of all agents in `sim`.
 * \param   time    The new simulation time of `sim`.
 */
void on_step(simulator<py_simulator_data>* sim,
        const array<agent_state*>& agents, uint64_t time)
{
    py_simulator_data& data = sim->get_data();
    if (data.server.status != server_status::STOPPING) {
        /* this simulator is a server, so send a step response to every client */
        if (!send_step_response(data.server, agents, sim->get_config()))
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
    PyObject* args = Py_BuildValue("(O)", py_states);
    PyObject* result = PyEval_CallObject(data.callback, args);
    Py_DECREF(args);
    Py_DECREF(py_states);
    if (result != NULL)
        Py_DECREF(result);
    PyGILState_Release(gstate); /* release global interpreter lock */
}


/**
 * Client callback functions.
 */

inline char* concat(const char* first, const char* second) {
    size_t first_length = strlen(first);
    size_t second_length = strlen(second);
    char* buf = (char*) malloc(sizeof(char) * (first_length + second_length + 1));
    if (buf == NULL) {
        fprintf(stderr, "concat ERROR: Out of memory.\n");
        return NULL;
    }
    for (unsigned int i = 0; i < first_length; i++)
        buf[i] = first[i];
    for (unsigned int j = 0; j < second_length; j++)
        buf[first_length + j] = second[j];
    buf[first_length + second_length] = '\0';
    return buf;
}

inline void check_response(mpi_response response, const char* prefix) {
    char* message;
    switch (response) {
    case mpi_response::INVALID_AGENT_ID:
        message = concat(prefix, "Invalid agent ID.");
        if (message != NULL) { PyErr_SetString(mpi_error, message); free(message); } break;
    case mpi_response::SERVER_PARSE_MESSAGE_ERROR:
        message = concat(prefix, "Server was unable to parse MPI message from client.");
        if (message != NULL) { PyErr_SetString(mpi_error, message); free(message); } break;
    case mpi_response::CLIENT_PARSE_MESSAGE_ERROR:
        message = concat(prefix, "Client was unable to parse MPI message from server.");
        if (message != NULL) { PyErr_SetString(mpi_error, message); free(message); } break;
    case mpi_response::SUCCESS:
    case mpi_response::FAILURE:
        break;
    }
}

/**
 * The callback invoked when the client receives an add_agent response from the
 * server. This function copies the agent state into a Python object, stores
 * it in `c.data.response_data.agent_state`, and wakes up the Python thread
 * (which should be waiting in the `simulator_add_agent` function) so that it
 * can return the response back to Python.
 *
 * \param   c         The client that received the response.
 * \param   agent_id  The ID of the new agent. This is equal to `UINT64_MAX` if
 *                    the server returned an error.
 * \param   response  The MPI response from the server, containing information
 *                    about any errors.
 * \param   new_agent The state of the new agent.
 */
void on_add_agent(client<py_client_data>& c, uint64_t agent_id,
        mpi_response response, const agent_state& new_agent)
{
    check_response(response, "add_agent: ");
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); /* acquire global interpreter lock */
    PyObject* agent;
    if (response != mpi_response::SUCCESS || agent_id == UINT64_MAX)
        agent = NULL;
    else agent = build_py_agent(new_agent, c.config, agent_id);
    PyGILState_Release(gstate);

    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.response_data.agent_state = agent;
    c.data.server_response = response;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a move response from the
 * server. This function copies the result into `c.data.server_response` and
 * wakes up the Python thread (which should be waiting in the `simulator_move`
 * function) so that it can return the response back to Python.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to move.
 * \param   response        The MPI response from the server, containing
 *                          information about any errors.
 */
void on_move(client<py_client_data>& c, uint64_t agent_id, mpi_response response) {
    check_response(response, "move: ");
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.server_response = response;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a turn response from the
 * server. This function copies the result into `c.data.server_response` and
 * wakes up the Python thread (which should be waiting in the `simulator_turn`
 * function) so that it can return the response back to Python.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to turn.
 * \param   response        The MPI response from the server, containing
 *                          information about any errors.
 */
void on_turn(client<py_client_data>& c, uint64_t agent_id, mpi_response response) {
    check_response(response, "turn: ");
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.server_response = response;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a do_nothing response from the
 * server. This function copies the result into `c.data.server_response` and
 * wakes up the Python thread (which should be waiting in the `simulator_no_op`
 * function) so that it can return the response back to Python.
 *
 * \param   c               The client that received the response.
 * \param   agent_id        The ID of the agent that requested to do nothing.
 * \param   response        The MPI response from the server, containing
 *                          information about any errors.
 */
void on_do_nothing(client<py_client_data>& c, uint64_t agent_id, mpi_response response) {
    check_response(response, "no_op: ");
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.server_response = response;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a get_map response from the
 * server. This function moves the result into `c.data.response_data.map` and
 * wakes up the Python thread (which should be waiting in the `simulator_map`
 * function) so that it can return the response back to Python.
 *
 * \param   c        The client that received the response.
 * \param   response The MPI response from the server, containing information
 *                   about any errors.
 * \param   map      A map from patch positions to `patch_state` structures
 *                   containing the state information in each patch.
 */
void on_get_map(client<py_client_data>& c,
        mpi_response response,
        hash_map<position, patch_state>* map)
{
    check_response(response, "get_map: ");
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.response_data.map = map;
    c.data.server_response = response;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives a set_active response from the
 * server. This function wakes up the Python thread (which should be waiting in
 * the `simulator_set_active` function) so that it can return the response back
 * to Python.
 *
 * \param   c        The client that received the response.
 * \param   agent_id The ID of the agent whose active status was set.
 * \param   response The MPI response from the server, containing information
 *                   about any errors.
 */
void on_set_active(client<py_client_data>& c, uint64_t agent_id, mpi_response response)
{
    check_response(response, "set_active: ");
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.server_response = response;
    c.data.cv.notify_one();
}

/**
 * The callback invoked when the client receives an is_active response from the
 * server. This function moves the result into `c.data.server_response` and
 * wakes up the Python thread (which should be waiting in the
 * `simulator_is_active` function) so that it can return the response back to
 * Python.
 *
 * \param   c        The client that received the response.
 * \param   agent_id The ID of the agent whose active status was requested.
 * \param   response The MPI response from the server, containing information
 *                   about whether the agent is active and any errors.
 */
void on_is_active(client<py_client_data>& c, uint64_t agent_id, mpi_response response)
{
    check_response(response, "is_active: ");
    std::unique_lock<std::mutex> lck(c.data.lock);
    c.data.waiting_for_server = false;
    c.data.server_response = response;
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
        mpi_response response,
        const array<uint64_t>& agent_ids,
        const agent_state* agent_states)
{
    check_response(response, "on_step: ");

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
    PyObject* args = Py_BuildValue("(O)", py_states);
    PyObject* result = PyEval_CallObject(c.data.step_callback, args);
    Py_DECREF(args);
    Py_DECREF(py_states);
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
 * Imports the Python exception classes from the jbw module.
 */
static inline void import_errors() {
#if PY_MAJOR_VERSION >= 3
    PyObject* module_name = PyUnicode_FromString("jbw");
#else
    PyObject* module_name = PyString_FromString("jbw");
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
 *                  - (list of ints) The ActionPolicy for each possible movement.
 *                  - (list of ints) The ActionPolicy for each possible turn.
 *                  - (bool) Whether or not the no-op action is allowed.
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
    PyObject* py_no_op_allowed;
    PyObject* py_items;
    PyObject* py_agent_color;
    unsigned int seed;
    unsigned int collision_policy;
    PyObject* py_callback;
    if (!PyArg_ParseTuple(
      args, "IIOOOIIIIIOOIffIO", &seed, &config.max_steps_per_movement,
      &py_allowed_movement_directions, &py_allowed_turn_directions, &py_no_op_allowed,
      &config.scent_dimension, &config.color_dimension, &config.vision_range,
      &config.patch_size, &config.mcmc_iterations, &py_items, &py_agent_color,
      &collision_policy, &config.decay_param, &config.diffusion_param,
      &config.deleted_item_lifetime, &py_callback)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.new'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable.\n");
        return NULL;
    } else if (!PyList_Check(py_items)) {
        PyErr_SetString(PyExc_TypeError, "'items' must be a list.\n");
        return NULL;
    } else if (!PyList_Check(py_allowed_movement_directions)
            || PyList_Size(py_allowed_movement_directions) != (size_t) direction::COUNT)
    {
        PyErr_SetString(PyExc_TypeError, "'allowed_movement_directions' must"
            " be a list with length equal to the number of possible movement directions.\n");
        return NULL;
    } else if (!PyList_Check(py_allowed_turn_directions)
            || PyList_Size(py_allowed_turn_directions) != (size_t) direction::COUNT)
    {
        PyErr_SetString(PyExc_TypeError, "'allowed_turn_directions' must be a"
            " list with length equal to the number of possible movement directions.\n");
        return NULL;
    }

    PyObject *py_items_iter = PyObject_GetIter(py_items);
    if (!py_items_iter) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument types in the call to 'simulator_c.new'.");
        return NULL;
    }
    Py_ssize_t item_type_count = PyList_Size(py_items);
    if (!config.item_types.ensure_capacity(max((Py_ssize_t) 1, item_type_count))) {
        PyErr_NoMemory();
        return NULL;
    }
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
        new_item.intensity_fn.fn = get_intensity_fn((intensity_fns) py_intensity_fn,
                intensity_fn_args.key, (unsigned int) intensity_fn_args.value);
        if (new_item.intensity_fn.fn == NULL) {
            PyErr_SetString(PyExc_ValueError, "Invalid intensity"
                    " function arguments in the call to 'simulator_c.new'.");
            return NULL;
        }
        new_item.intensity_fn.args = intensity_fn_args.key;
        new_item.intensity_fn.arg_count = (unsigned int) intensity_fn_args.value;
        new_item.interaction_fns = (energy_function<interaction_function>*)
                malloc(sizeof(energy_function<interaction_function>) * item_type_count);
        for (Py_ssize_t i = 0; i < item_type_count; i++) {
            PyObject* sublist = PyList_GetItem(py_interaction_fn_args, i);
            unsigned int py_interaction_fn = PyLong_AsUnsignedLong(PyList_GetItem(sublist, 0));

            pair<float*, Py_ssize_t> interaction_fn_args = PyArg_ParseFloatList(sublist, 1);
            new_item.interaction_fns[i].fn = get_interaction_fn((interaction_fns) py_interaction_fn,
                    interaction_fn_args.key, (unsigned int) interaction_fn_args.value);
            new_item.interaction_fns[i].args = interaction_fn_args.key;
            new_item.interaction_fns[i].arg_count = (unsigned int) interaction_fn_args.value;
            if (new_item.interaction_fns[i].fn == NULL) {
                PyErr_SetString(PyExc_ValueError, "Invalid interaction"
                        " function arguments in the call to 'simulator_c.new'.");
                return NULL;
            }
        }
        config.item_types.length += 1;
    }

    for (unsigned int i = 0; i < (unsigned int) direction::COUNT; i++)
        config.allowed_movement_directions[i] = (action_policy) PyLong_AsUnsignedLong(PyList_GetItem(py_allowed_movement_directions, i));
    for (unsigned int i = 0; i < (unsigned int) direction::COUNT; i++)
        config.allowed_rotations[i] = (action_policy) PyLong_AsUnsignedLong(PyList_GetItem(py_allowed_turn_directions, i));
    config.no_op_allowed = PyObject_IsTrue(py_no_op_allowed);

    config.agent_color = PyArg_ParseFloatList(py_agent_color).key;
    config.collision_policy = (movement_conflict_policy) collision_policy;

    py_simulator_data data(py_callback);

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
 * Saves a simulator to file.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    A Python tuple containing the arguments to this function:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - (string) The full path to the file to which to save the
 *                    simulator.
 * \returns `True` if successful; `False` otherwise.
 */
static PyObject* simulator_save(PyObject *self, PyObject *args)
{
    PyObject* py_sim_handle;
    char* save_filepath;
    if (!PyArg_ParseTuple(args, "Os", &py_sim_handle, &save_filepath)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.save'.\n");
        return NULL;
    }
    FILE* file = open_file(save_filepath, "wb");
    if (file == nullptr) {
        fprintf(stderr, "save ERROR: Unable to open '%s' for writing. ", save_filepath);
        perror(nullptr); Py_INCREF(Py_False);
        return Py_False;
    }

    simulator<py_simulator_data>* sim_handle =
            (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
    const py_simulator_data& data = sim_handle->get_data();
    fixed_width_stream<FILE*> out(file);
    bool result = write(*sim_handle, out)
               && write(data.agent_ids.length, out)
               && write(data.agent_ids.data, out, data.agent_ids.length)
               && write(data.server.state, out);
    fclose(file);

    PyObject* py_result = (result ? Py_True : Py_False);
    Py_INCREF(py_result); return py_result;
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
    if (!PyArg_ParseTuple(args, "sO", &load_filepath, &py_callback)) {
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

    py_simulator_data data(py_callback);

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
    server_state& state = *((server_state*) alloca(sizeof(server_state)));
    py_simulator_data& sim_data = sim->get_data();
    if (!read(agent_id_count, in)
     || !sim_data.agent_ids.ensure_capacity(agent_id_count)
     || !read(sim_data.agent_ids.data, in, agent_id_count)
     || !read(state, in))
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load agent IDs and server state.");
        free(*sim); free(sim); fclose(file); return NULL;
    }
    sim_data.agent_ids.length = agent_id_count;
    swap(state, sim_data.server.state);
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
    PyObject* py_sim = PyLong_FromVoidPtr(sim);
    PyObject* to_return = Py_BuildValue("(KOO)", sim->time, py_sim, py_states);
    Py_DECREF(py_sim); Py_DECREF(py_states);
    return to_return;
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
    async_server& server = sim_handle->get_data().server;
    if (!init_server(server, *sim_handle, (uint16_t) port, connection_queue_capacity, num_workers)) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI server.");
        return NULL;
    }
    return PyLong_FromVoidPtr(&server);
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
 * \returns A Python tuple containing:
 *          - The simulation time.
 *          - A handle to the client.
 *          - The ID assigned to the client by the server.
 */
static PyObject* simulator_connect_client(PyObject *self, PyObject *args)
{
    char* server_address;
    unsigned int port;
    PyObject* py_step_callback;
    PyObject* py_lost_connection_callback;
    if (!PyArg_ParseTuple(args, "sIOO", &server_address, &port, &py_step_callback, &py_lost_connection_callback)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.start_client'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_step_callback) || !PyCallable_Check(py_lost_connection_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callbacks must be callable.\n");
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
    }

    uint64_t client_id;
    uint64_t simulator_time = connect_client(*new_client, server_address, (uint16_t) port, client_id);
    if (simulator_time == UINT64_MAX) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI client.");
        free(*new_client); free(new_client); return NULL;
    }

    new_client->data.step_callback = py_step_callback;
    new_client->data.lost_connection_callback = py_lost_connection_callback;
    Py_INCREF(py_step_callback);
    Py_INCREF(py_lost_connection_callback);
    import_errors();
    PyObject* py_new_client = PyLong_FromVoidPtr(new_client);
    PyObject* to_return = Py_BuildValue("(KOK)", simulator_time, py_new_client, client_id);
    Py_DECREF(py_new_client);
    return to_return;
}

/**
 * Reconnects a client to the specified simulator server.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - (string) The server address.
 *                  - (int) The server port.
 *                  - (function) The Python function to invoke whenever the
 *                    simulator advances time.
 *                  - (function) The Python function to invoke if the client
 *                    loses its connection to the server.
 *                  - (int) The ID of the client assigned by the server.
 * \returns A Python tuple containing:
 *          - The simulation time.
 *          - A handle to the client.
 *          - A list of tuples containing the states of the agents governed by
 *            this simulator (not including agents governed by other clients).
 *            See `build_py_agent` for details on the contents of each tuple.
 */
static PyObject* simulator_reconnect_client(PyObject *self, PyObject *args)
{
    char* server_address;
    unsigned int port;
    PyObject* py_step_callback;
    PyObject* py_lost_connection_callback;
    unsigned long long client_id;
    if (!PyArg_ParseTuple(args, "sIOOK", &server_address, &port, &py_step_callback, &py_lost_connection_callback, &client_id)) {
        fprintf(stderr, "Invalid argument types in the call to 'simulator_c.start_client'.\n");
        return NULL;
    }

    if (!PyCallable_Check(py_step_callback) || !PyCallable_Check(py_lost_connection_callback)) {
        PyErr_SetString(PyExc_TypeError, "Callbacks must be callable.\n");
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
    }

    uint64_t* agent_ids;
    agent_state* agent_states;
    unsigned int agent_count;
    uint64_t simulator_time = reconnect_client(*new_client, (uint64_t) client_id,
            server_address, (uint16_t) port, agent_ids, agent_states, agent_count);
    if (simulator_time == UINT64_MAX) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initialize MPI client.");
        free(*new_client); free(new_client); return NULL;
    }

    PyObject* py_states = PyList_New(agent_count);
    if (py_states == NULL) return NULL;
    for (Py_ssize_t i = 0; i < agent_count; i++) {
        PyList_SetItem(py_states, i, build_py_agent(agent_states[i], new_client->config, agent_ids[i]));
        free(agent_states[i]);
    }
    free(agent_states); free(agent_ids);

    new_client->data.step_callback = py_step_callback;
    new_client->data.lost_connection_callback = py_lost_connection_callback;
    Py_INCREF(py_step_callback);
    Py_INCREF(py_lost_connection_callback);
    import_errors();
    PyObject* py_new_client = PyLong_FromVoidPtr(new_client);
    PyObject* to_return = Py_BuildValue("(KOO)", simulator_time, py_new_client, py_states);
    Py_DECREF(py_new_client); Py_DECREF(py_states);
    return to_return;
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
        PyObject* py_agent = build_py_agent(*new_agent.value, sim_handle->get_config(), new_agent.key);
        PyObject* to_return = Py_BuildValue("O", py_agent);
        Py_DECREF(py_agent);
        return to_return;
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

        if (client_handle->data.response_data.agent_state == NULL) {
            /* server returned failure */
            PyErr_SetString(add_agent_error, "Failed to add new agent.");
            return NULL;
        }

        return client_handle->data.response_data.agent_state;
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

        PyObject* result = (client_handle->data.server_response == mpi_response::SUCCESS ? Py_True : Py_False);
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

        PyObject* result = (client_handle->data.server_response == mpi_response::SUCCESS ? Py_True : Py_False);
        Py_INCREF(result);
        return result;
    }
}

/**
 * Attempt to instruct the agent in the simulation environment to do nothing.
 * If the agent already has an action queued for this turn, this attempt will
 * fail.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong. If this
 *                    is None, `do_nothing` is directly invoked on the
 *                    simulator object. Otherwise, the client sends a
 *                    do_nothing message to the server and waits for its
 *                    response.
 *                  - Agent ID.
 * \returns `True` if the turn command is successfully queued; `False`
 *          otherwise.
 */
static PyObject* simulator_no_op(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    if (!PyArg_ParseTuple(args, "OOK", &py_sim_handle, &py_client_handle, &agent_id))
        return NULL;
    if (py_client_handle == Py_None) {
        /* the simulation is local, so call do_nothing directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);

        /* release the global interpreter lock */
        PyThreadState* python_thread = PyEval_SaveThread();
        bool result = sim_handle->do_nothing(agent_id);

        /* re-acquire the global interpreter lock and return */
        PyEval_RestoreThread(python_thread);
        PyObject* py_result = (result ? Py_True : Py_False);
        Py_INCREF(py_result); return py_result;
    } else {
        /* this is a client, so send a do_nothing message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!client_handle->client_running) {
            PyErr_SetString(mpi_error, "Connection to the server was lost.");
            return NULL;
        }

        client_handle->data.waiting_for_server = true;
        if (!send_do_nothing(*client_handle, agent_id)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send do_nothing request.");
            return NULL;
        }

        /* wait for response from server */
        wait_for_server(*client_handle);

        PyObject* result = (client_handle->data.server_response == mpi_response::SUCCESS ? Py_True : Py_False);
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
        Py_DECREF(fixed);
        Py_DECREF(py_scent);
        Py_DECREF(py_vision);
        Py_DECREF(py_items);
        Py_DECREF(py_agents);
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
 *          for details on the contents of each tuple. If an error occurs, None
 *          is returned, instead.
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
        /* the simulation is local, so call get_map directly */
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
        /* this is a client, so send a get_map message to the server */
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
        if (client_handle->data.server_response != mpi_response::SUCCESS) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        PyObject* py_map = build_py_map(*client_handle->data.response_data.map, client_handle->config);
        for (auto entry : *client_handle->data.response_data.map)
            free(entry.value);
        free(*client_handle->data.response_data.map);
        free(client_handle->data.response_data.map);
        return py_map;
    }
}

/**
 * Sets whether the agent is active or inactive.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong. If this
 *                    is None, `set_active` is directly invoked on the
 *                    simulator object. Otherwise, the client sends a
 *                    set_active message to the server and waits for its
 *                    response.
 *                  - Agent ID.
 *                  - A boolean indicating whether to make this agent active.
 * \returns None.
 */
static PyObject* simulator_set_active(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    PyObject* py_active;
    if (!PyArg_ParseTuple(args, "OOKO", &py_sim_handle, &py_client_handle, &agent_id, &py_active))
        return NULL;
    int result = PyObject_IsTrue(py_active);
    bool active;
    if (result == 1) active = true;
    else if (result == 0) active = false;
    else {
        PyErr_SetString(PyExc_TypeError, "`active` must be boolean.\n");
        return NULL;
    }

    if (py_client_handle == Py_None) {
        /* the simulation is local, so call get_map directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        sim_handle->set_agent_active(agent_id, active);
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        /* this is a client, so send a get_map message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!client_handle->client_running) {
            PyErr_SetString(mpi_error, "Connection to the server was lost.");
            return NULL;
        }

        client_handle->data.waiting_for_server = true;
        if (!send_set_active(*client_handle, agent_id, active)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send set_active request.");
            return NULL;
        }

        /* wait for response from server */
        wait_for_server(*client_handle);
        Py_INCREF(Py_None);
        return Py_None;
    }
}

/**
 * Gets whether the agent is active or inactive.
 *
 * \param   self    Pointer to the Python object calling this method.
 * \param   args    Arguments:
 *                  - Handle to the native simulator object as a PyLong.
 *                  - Handle to the native client object as a PyLong. If this
 *                    is None, `set_active` is directly invoked on the
 *                    simulator object. Otherwise, the client sends a
 *                    set_active message to the server and waits for its
 *                    response.
 *                  - Agent ID.
 * \returns `True` if the agent is active; `False` if it's inactive, and `None`
 *          if an error occurred.
 */
static PyObject* simulator_is_active(PyObject *self, PyObject *args) {
    PyObject* py_sim_handle;
    PyObject* py_client_handle;
    unsigned long long agent_id;
    if (!PyArg_ParseTuple(args, "OOK", &py_sim_handle, &py_client_handle, &agent_id))
        return NULL;

    if (py_client_handle == Py_None) {
        /* the simulation is local, so call get_map directly */
        simulator<py_simulator_data>* sim_handle =
                (simulator<py_simulator_data>*) PyLong_AsVoidPtr(py_sim_handle);
        bool active = sim_handle->is_agent_active(agent_id);
        PyObject* py_result = (active ? Py_True : Py_False);
        Py_INCREF(py_result); return py_result;
    } else {
        /* this is a client, so send a get_map message to the server */
        client<py_client_data>* client_handle =
                (client<py_client_data>*) PyLong_AsVoidPtr(py_client_handle);
        if (!client_handle->client_running) {
            PyErr_SetString(mpi_error, "Connection to the server was lost.");
            return NULL;
        }

        client_handle->data.waiting_for_server = true;
        if (!send_is_active(*client_handle, agent_id)) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to send is_active request.");
            return NULL;
        }

        /* wait for response from server */
        wait_for_server(*client_handle);
        PyObject* py_result;
        if (client_handle->data.server_response == mpi_response::SUCCESS) {
            py_result = Py_True;
        } else if (client_handle->data.server_response == mpi_response::FAILURE) {
            py_result = Py_False;
        } else {
            py_result = Py_None;
        }
        Py_INCREF(py_result); return py_result;
    }
}

} /* namespace jbw */

static PyMethodDef SimulatorMethods[] = {
    {"new",  jbw::simulator_new, METH_VARARGS, "Creates a new simulator and returns its pointer."},
    {"save",  jbw::simulator_save, METH_VARARGS, "Saves a simulator to file."},
    {"load",  jbw::simulator_load, METH_VARARGS, "Loads a simulator from file and returns its pointer."},
    {"delete",  jbw::simulator_delete, METH_VARARGS, "Deletes an existing simulator."},
    {"start_server",  jbw::simulator_start_server, METH_VARARGS, "Starts the simulator server."},
    {"stop_server",  jbw::simulator_stop_server, METH_VARARGS, "Stops the simulator server."},
    {"connect_client",  jbw::simulator_connect_client, METH_VARARGS, "Connects a new simulator client to a server."},
    {"reconnect_client",  jbw::simulator_reconnect_client, METH_VARARGS, "Reconnects an existing client to a server."},
    {"stop_client",  jbw::simulator_stop_client, METH_VARARGS, "Stops the simulator client."},
    {"add_agent",  jbw::simulator_add_agent, METH_VARARGS, "Adds an agent to the simulator and returns its ID."},
    {"move",  jbw::simulator_move, METH_VARARGS, "Attempts to move the agent in the simulation environment."},
    {"turn",  jbw::simulator_turn, METH_VARARGS, "Attempts to turn the agent in the simulation environment."},
    {"no_op",  jbw::simulator_no_op, METH_VARARGS, "Attempts to instruct the agent to do nothing (a no-op) in the simulation environment."},
    {"map",  jbw::simulator_map, METH_VARARGS, "Returns a list of patches within a given bounding box."},
    {"set_active",  jbw::simulator_set_active, METH_VARARGS, "Sets whether the agent is active or inactive."},
    {"is_active",  jbw::simulator_is_active, METH_VARARGS, "Gets whether the agent is active or inactive."},
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

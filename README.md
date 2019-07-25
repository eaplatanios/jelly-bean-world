# Never-Ending Learning Framework

A framework for experimenting with never-ending learning.

## Requirements

- GCC 4.9+, Clang 5+, or Visual C++ 14+
- Python 2.7 or 3.5+ and Numpy (for the Python API)

## Using Python

### Installation Instructions

Assuming that you have Python installed in your system and
that you are located in the root directory of this
repository, run the following commands:

```bash
git submodule update --init --recursive
cd api/python
python setup.py install
```

Note that if you plan to use this framework with OpenAI gym
you should also install `gym` using `pip install gym`.

### Usage

We first present a typical workflow in Python, without 
using OpenAI gym and we then show how to use this framework 
within OpenAI gym.

The typical workflow is as follows:

  1. Extend the `Agent` class to implement custom agents.
  2. Create a Simulator object.
  3. Construct agent instances in this simulator.
  4. Issue action commands for each agent.

The following is a simple example where a simulator is constructed locally
(within the same process) and a single agent continuously moves forward. Note
that the agent's decision-making logic goes in the `do_next_action` method.

```python
import nel

class SimpleAgent(nel.Agent):
  def __init__(self, simulator, load_filepath=None):
    super(SimpleAgent, self).__init__(simulator, load_filepath)

  def do_next_action(self):
    self.move(nel.RelativeDirection.FORWARD)

  def save(self, filepath):
    pass

  def _load(self, filepath):
    pass


# specify the item types
items = []
items.append(nel.Item("banana", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0], [0], False,
        intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=[-2.0],
        interaction_fns=[[nel.InteractionFunction.PIECEWISE_BOX, 40.0, 200.0, 0.0, -40.0]]))

# construct the simulator configuration
config = nel.SimulatorConfig(max_steps_per_movement=1, vision_range=1,
  allowed_movement_directions=[nel.ActionPolicy.ALLOWED, nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.DISALLOWED],
  allowed_turn_directions=[nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.ALLOWED, nel.ActionPolicy.ALLOWED],
  no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
  collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
  decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000)

# create a local simulator
sim = nel.Simulator(sim_config=config)

# add one agent to the simulation
agent = SimpleAgent(sim)

# start the main loop
for t in range(10000):
  agent.do_next_action()
```

See [api/python/test/simulator_test.py](api/python/test/simulator_test.py) for
an example with more types of items as well as a visualization using the
`MapVisualizer` class.

It is straightforward to create the simulator in *server mode*, where other
clients can connect to it:
```python
sim = nel.Simulator(sim_config=config, is_server=True)
```

To connect to existing server (i.e. create the simulator in *client mode*), for
example running on `localhost`:
```python
sim = nel.Simulator(server_address="localhost")
```

See [api/python/test/server_test.py](api/python/test/server_test.py) and
[api/python/test/client_test.py](api/python/test/client_test.py) for an example
of simulators running in server and client modes (using MPI to communicate).

#### Using with OpenAI Gym

We also provide a NEL environment for OpenAI gym, which is 
implemented in [api/python/nel/environment.py](api/python/nel/environment.py). 

The action space consists of three actions:
  - `0`: Move forward.
  - `1`: Turn left.
  - `2`: Turn right.

The observation space consists of a dictionary:
  - `scent`: Vector with shape `[S]`, where `S` is the 
    scent dimensionality.
  - `vision`: Matrix with shape `[2R+1, 2R+1, V]`, 
    where `R` is the vision range and `V` is the 
    vision/color dimensionality.
  - `moved`: Binary value indicating whether the last 
    action resulted in the agent moving.

After installing the `nel` framework and `gym`, the 
provided environment can be used as follows:

```python
import gym
import nel

# Use 'NEL-render-v0' to include rendering support.
# Otherwise, use 'NEL-v0', which should be much faster.
env = gym.make('NEL-render-v0')

# The created environment can then be used as any other 
# OpenAI gym environment. For example:
for t in range(10000):
  # Render the current environment.
  env.render()
  # Sample a random action.
  action = env.action_space.sample()
  # Run a simulation step using the sampled action.
  observation, reward, _, _ = env.step(action)
```

Environments with different configurations can be 
registered as shown in [api/python/nel/environments.py](api/python/nel/environments.py) 
and used as shown above.

### Agent class

Agents have a very simple interface. They have an abstract `do_next_action`
method which should contain the decision-making logic of the agent and call
methods such as `self.move` or `self.turn` to perform the next action. The
Agent class has an abstract `save` method that users can implement to save the
state of an agent to a file. `save` is called by Simulator whenever the
simulator is saved. The Agent class also has an abstract `_load` method which
is called by the Agent constructor to load the agent's state from a given
filepath.

Agents also have a private fields that store state information, such as the
agent's position, direction, current scent perception, current visual
perception, etc.

## Using C++

The typical workflow is as follows:

  1. Create a Simulator object.
  2. Add agents to this simulator.
  3. Issue action commands for each agent.

The following is a simple example where a simulator is constructed locally
(within the same process) and a single agent continuously moves forward.

```c++
#include "network.h"
#include "simulator.h"

using namespace nel;

/** A helper function to set interaction function parameters. */
inline void set_interaction_args(
    item_properties* item_types, unsigned int first_item_type,
    unsigned int second_item_type, interaction_function interaction,
    std::initializer_list<float> args)
{
  item_types[first_item_type].interaction_fns[second_item_type].fn = interaction;
  item_types[first_item_type].interaction_fns[second_item_type].arg_count = (unsigned int) args.size();
  item_types[first_item_type].interaction_fns[second_item_type].args = (float*) malloc(max((size_t) 1, sizeof(float) * args.size()));

  unsigned int counter = 0;
  for (auto i = args.begin(); i != args.end(); i++)
    item_types[first_item_type].interaction_fns[second_item_type].args[counter++] = *i;
}

void on_step(const simulator<empty_data>* sim, const array<agent_state*>& agents, uint64_t time) { }

int main(int argc, const char** argv) {
  /* construct the simulator configuration */
  simulator_config config;
  config.max_steps_per_movement = 1;
  config.scent_dimension = 3;
  config.color_dimension = 3;
  config.vision_range = 10;
  for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
    config.allowed_movement_directions[i] = action_policy::ALLOWED;
  for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
    config.allowed_rotations[i] = action_policy::ALLOWED;
  config.no_op_allowed = false;
  config.patch_size = 32;
  config.mcmc_iterations = 4000;
  config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
  config.agent_color[2] = 1.0f;
  config.collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
  config.decay_param = 0.5f;
  config.diffusion_param = 0.12f;
  config.deleted_item_lifetime = 2000;

  /* configure item types */
  config.item_types.ensure_capacity(1);
  config.item_types[0].name = "banana";
  config.item_types[0].scent = (float*) calloc(config.scent_dimension, sizeof(float));
  config.item_types[0].color = (float*) calloc(config.color_dimension, sizeof(float));
  config.item_types[0].required_item_counts = (unsigned int*) calloc(1, sizeof(unsigned int));
  config.item_types[0].required_item_costs = (unsigned int*) calloc(1, sizeof(unsigned int));
  config.item_types[0].scent[0] = 1.0f;
  config.item_types[0].color[0] = 1.0f;
  config.item_types[0].blocks_movement = false;
  config.item_types.length = 1;

  /* specify the intensity and interaction function parameters */
  config.item_types[0].intensity_fn.fn = constant_intensity_fn;
  config.item_types[0].intensity_fn.arg_count = 1;
  config.item_types[0].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
  config.item_types[0].intensity_fn.args[0] = -2.0f;
  config.item_types[0].interaction_fns = (energy_function<interaction_function>*)
    malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
  set_interaction_args(config.item_types.data, 0, 0, piecewise_box_interaction_fn, {40.0f, 200.0f, 0.0f, -40.0f});

  /* create a local simulator */
  simulator<empty_data>& sim = *((simulator<empty_data>*) alloca(sizeof(simulator<empty_data>)));
  if (!init(sim, config, empty_data())) {
    fprintf(stderr, "ERROR: Unable to initialize simulator.\n");
    return EXIT_FAILURE;
  }

  /* add one agent to the simulation */
  pair<uint64_t, agent_state*> agent = sim.add_agent();
  if (agent.key == UINT64_MAX) {
    fprintf(stderr, "ERROR: Unable to add new agent.\n");
    return EXIT_FAILURE;
  }

  /* the main simulation loop */
  for (unsigned int t = 0; t < 10000; t++) {
    if (!sim.move(agent.key, direction::UP, 1)) {
      fprintf(stderr, "ERROR: Unable to move agent.\n");
      return EXIT_FAILURE;
    }
  }
  free(sim);
}
```

See [nel/simulator_test.cpp](nel/simulator_test.cpp) for an example with more
types of items, as well as a multithreaded example and an MPI example.

To setup an MPI server for a simulator `sim`, the `init_server` function in
[nel/mpi.h](nel/mpi.h) may be used:
```c++
  /* NOTE: this blocks during the lifetime of the server */
  if (!init_server(sim, 54353, 256, 8)) { /* process error */ }
```

To set up an asynchronous MPI server (where `init_server` will not block), the
`async_server` class in [nel/mpi.h](nel/mpi.h) is used:
```c++
  async_server new_server;
  if (!init_server(new_server, sim, 54353, 256, 8)) { /* process error */ }
```

To connect to an existing server, for example at `localhost:54353`, we use the
`client` class defined in [nel/mpi.h](nel/mpi.h):
```c++
  client<empty_data> new_client;
  if (!init_client(new_client, "localhost", 54353, NULL, NULL, 0)) { /* process error */ }
```
The commands may be sent to the server using the functions `send_add_agent`,
`send_move`, `send_turn`, etc. When the client receives responses from the
server, the functions `on_add_agent`, `on_move`, `on_turn`, etc will be
invoked, which must be defined by the user.

## Design

The center of our design is the **simulator**. The simulator handles everything
that happens to our artificial environment. It does by controlling the map
generation process, as well as the agent-environment interaction. **Agents** are
part of the design too, but in a very limited way. This is intentional, and it
is done in order to allow for flexibility in the design of custom agents.

### Simulator

The simulator controls the following things:

  - Incremental map generation, based on the movement of the agents.
  - Passage of time.
  - Allowed agent-environment interactions.

Under the current design:

  - Each simulator *owns* a set of agents.
  - Users can easily add/register new agents to an existing simulator.
  - Each agent interacts with the simulator by deciding
    **when and where to move**.
  - Once all agents have requested to perform an action, the simulator
    progresses by one time step and notifies invokes a callback function.
  - Some items in the world are automatically collected by the agents. The
    collected items are available in each agent's state information.

**NOTE:** Note that the agent is not moved until the simulator advances the
time step and issues a notification about that event. The simulator only
advances the time step once all agents have requested to act.

We provide a message-passing interface (using TCP) to allow the simulator to
run remotely, and agents can issue move commands to the server. In Python, the
Simulator can be constructed as a server with the appropriate constructor
arguments. If a server address is provided to the Simulator constructor, it
will try to connect to the Simulator running at the specified remote address,
and all calls to the Simulator class will be issued as commands to the server.
In C++, [nel/mpi.h](nel/mpi.h) provides the functionality to run the server and
clients. See [nel/simulator_test.cpp](nel/simulator_test.cpp) for an example.


#### Mechanics

We simulate an infinite map by dividing it into a collection of `n x n`
*patches*. When agents move around, they may move into new patches. The
simulator ensures that when an agent approaches a new patch, that patch is
appropriately initialized (if it wasn't previously). When new patches are
initialized, we fill them with items.

The items are distributed according to a pairwise interaction point process on
the 2-dimensional grid of integers. The probability of a collection of points
`X = {X_1, X_2, ...}` is given by:
```
    p(X) = c * exp{sum over i of f(X_i) + sum over j of g(X_i, X_j)}.
```
Here, `c` is a normalizing constant. `f(x)` is the **intensity** function, that
controls the likelihood of generating a point at `x` independent of other
points. `g(x,y)` is the **interaction** function, which controls the likelihood
of generating the point at `x` given the existence of a point at `y`.
Metropolis-Hastings sampling is used to sample from this distribution.

It is through the interaction function that we can control whether items of one
type are "attracted to" or "repelled by" items of another type. We allow the
user to specify which intensity/interaction functions they wish to use, for
each item.

Vision is implemented straightforwardly: within the visual field of each agent,
empty cells are rendered with a single color. Then for each item within the
visual field of the agent, we render the corresponding pixel with the color of
the item.

Scent is modeled as a simple diffusion system on the grid, where each cell is
given a vector of scents (where each dimension can be used to model
orthogonal/unrelated scents). More precisely, if `S(x,y,t)` is the scent at
location `(x,y)` at time `t`, then
```
    S(x,y,t+1) = lambda*S(x,y,t) + C(x,y,t+1) + alpha*(S(x-1,y,t) + S(x+1,y,t) + S(x,y-1,t) + S(x,y+1,t)),
```
where `lambda` is the rate of decay of the scent at the current location,
`alpha` is the rate of diffusion from neighboring cells, and `C(x,y,t)` is the
scent of any items located at `(x,y)` at time `t` (this is zero if there are no
items at that position).

Our simulator keeps track of the creation and destruction times of each item in
the world, and so the scent can be computed correctly, even as items are
created and destroyed (collected) in the world. The simulation ensures the
scent (or lack thereof) diffuses correctly.

### Implementation

The core library is implemented in **C++** and has no dependencies. It should
be able to run on Mac, Linux, and Windows. We already provide a **Python** API
that is quite simple to use and extend. APIs for other languages should also be
easy to implement.

## Troubleshooting

### Repository initialization, publickey
If you get the message `Permission denied (publickey).` when initializing the
repository by calling `git submodule update --init --recursive` make sure you
have your public key set correctly in https://github.com/settings/keys. You can
see this example http://zeeelog.blogspot.com/2017/08/the-authenticity-of-host-githubcom.html
to generate a new one.

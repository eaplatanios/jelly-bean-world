# Jelly Bean World

A framework for experimenting with never-ending learning.
If you use this framework in your research, please cite as:

```bibtex
@inproceedings{jbw:2020,
  title         = {{Jelly Bean World: A Testbed for Never-Ending Learning}},
  author        = {Emmanouil Antonios Platanios and Abulhair Saparov and Tom Mitchell},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  url           = {https://arxiv.org/abs/2002.06306},
  year          = {2020},
}
```


This file is split in multiple sections:
  
  - [Requirements:](#requirements) Describes the
    prerequisite software that needs to be installed before
    you can use the Jelly Bean World.
  - [Using Swift:](#using-swift) Describes how to setup and
    use our Swift API, and how to reproduce the experiments
    as detailed in our [paper](https://arxiv.org/abs/2002.06306).
  - [Using Python:](#using-python) Describes how to setup
    and use our Python API.
    - [Installing with Pip:](#installing-with-pip) How to install
      using Pip.
  - [Using C++:](#using-c) Describes how to setup and use
    our C++ API, and how to run the greedy agent as
    detailed in our [paper](https://arxiv.org/abs/2002.06306).
  - [Using the Visualizer:](#using-the-visualizer)
    Describes how to use build and use the Vulkan-based
    visualizer.
  - [Design:](#design) Provides a brief description of the
    Jelly Bean World design. More information can found in
    our [paper](https://arxiv.org/abs/2002.06306).
  - [Troubleshooting:](#troubleshooting) Discusses common
    issues.

## Requirements

- GCC 4.9+, Clang 5+, or Visual C++ 14+
- **Python API:**
  - Python 2.7 or 3.5+
  - Numpy
- **Swift API:**
  - Swift for TensorFlow 0.8 toolchain
- **Visualizer:**
  - Vulkan
  - GLFW
  - Make (for Mac or Linux), or Visual Studio 2017+ (for
    Windows)

## Using Swift

### Running Experiments

Assuming you have the Swift for TensorFlow 0.8 toolchain
installed in your system, you can run Jelly Bean World
experiments using commands like the following:

```bash
swift run -c release JellyBeanWorldExperiments run \
  --reward collectJellyBeans \
  --agent ppo \
  --observation vision \
  --network plain
```

Specifically, to reproduce the results presented in our
[paper](https://arxiv.org/abs/2002.06306) you can use 
the scripts located in the [scripts/experiments](scripts/experiments)
directory. A good way to start using the Swift API is to
play around by modifying these scripts and observing how
the results change.

Files that aggregate the experiments results will be 
generated in the [temp/results](temp/results) directory.

After running Swift experiments using the aforementioned
command, you can plot the experiment results using commands
like the following:

```bash
swift run -c release JellyBeanWorldExperiments plot \
  --reward collectJellyBeans \
  --agent ppo
```

The plots will (by default) be generated in the
[temp/results](temp/results) directory.

### Using the Swift API directly

Using the Jelly Bean World Swift API typically consists of
the following steps:

  1. Create a [simulator configuration](api/swift/Sources/JellyBeanWorld/Simulator.swift#L423).
     This configuration contains information about the 
     types of items that exist and their distribution, as
     well as about the mechanics of the environment (e.g.,
     vision and scent dimensionality, allowed agent
     actions, etc.). You can also create multiple 
     configurations, which will allow you to concurrently
     run multiple instances of the Jelly Bean World
     simulator.
  2. Create a [reward schedule](api/swift/Sources/JellyBeanWorld/Rewards.swift#L100).
     This schedule defines the reward function that will
     be used at each point in time. Our library already 
     provides some pre-existing reward schedules that you
     can use, but you should feel free to create new ones.
     The reward schedules that we used for our
     [paper](https://arxiv.org/abs/2002.06306) can be found
     in [the experiments module](api/swift/Sources/JellyBeanWorldExperiments/Experiment.swift#L165).
  3. Create a Jelly Bean World [environment](api/swift/Sources/JellyBeanWorld/Environment.swift#L19).
  4. Create an agent. We do not impose any constraints on
     how you design your agents. They will be the ones
     interacting with the environment in a similar manner 
     how agents interact with OpenAI Gym environments.
     Specifically, the agents mainly interact with the 
     environment through the
     [environment step function](api/swift/Sources/JellyBeanWorld/Environment.swift#L71).

A good reference point for starting is the 
[the experiments module.](api/swift/Sources/JellyBeanWorldExperiments/Experiment.swift#L165)
that we built and used to run all experiments that were 
presented in our [paper](https://arxiv.org/abs/2002.06306).

## Using Python

### Installing with Pip

With Pip, simply run:
```bash
pip install 'git+ssh://git@github.com/eaplatanios/jelly-bean-world#subdirectory=api/python'
```
(thanks to [Chris Chow](https://github.com/ckchow) for adding support for this)

### Installing from Source

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
import jbw
from math import pi

class SimpleAgent(jbw.Agent):
  def __init__(self, simulator, load_filepath=None):
    super(SimpleAgent, self).__init__(simulator, load_filepath)

  def do_next_action(self):
    self.move(jbw.RelativeDirection.FORWARD)

  def save(self, filepath):
    pass

  def _load(self, filepath):
    pass


# specify the item types
items = []
items.append(jbw.Item("banana", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0], [0], False, 0.0,
        intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-2.0],
        interaction_fns=[[jbw.InteractionFunction.PIECEWISE_BOX, 40.0, 200.0, 0.0, -40.0]]))

# construct the simulator configuration
config = jbw.SimulatorConfig(max_steps_per_movement=1, vision_range=1,
  allowed_movement_directions=[jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED],
  allowed_turn_directions=[jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.ALLOWED],
  no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
  collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED, agent_field_of_view=2*pi,
  decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000)

# create a local simulator
sim = jbw.Simulator(sim_config=config)

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
sim = jbw.Simulator(sim_config=config, is_server=True, default_client_permissions=jbw.GRANT_ALL_PERMISSIONS)
```

To connect to existing server (i.e. create the simulator in *client mode*), for
example running on `localhost`:
```python
sim = jbw.Simulator(server_address="localhost")
```

See [api/python/test/server_test.py](api/python/test/server_test.py) and
[api/python/test/client_test.py](api/python/test/client_test.py) for an example
of simulators running in server and client modes (using MPI to communicate).

#### Using with OpenAI Gym

We also provide a JBW environment for OpenAI gym, which is 
implemented in [api/python/src/jbw/environment.py](api/python/src/jbw/environment.py). 

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

After installing the `jbw` framework and `gym`, the 
provided environment can be used as follows:

```python
import gym
import jbw

# Use 'JBW-render-v0' to include rendering support.
# Otherwise, use 'JBW-v0', which should be much faster.
env = gym.make('JBW-render-v0')

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
registered as shown in [api/python/src/jbw/environments.py](api/python/src/jbw/environments.py) 
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
#include <jbw/network.h>
#include <jbw/simulator.h>

using namespace jbw;

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

void on_step(simulator<empty_data>* sim, const hash_map<uint64_t, agent_state*>& agents, uint64_t time) { }

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
  config.agent_field_of_view = 2.09f;
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
	config.item_types[0].visual_occlusion = 0.0;
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
  if (init(sim, config, empty_data()) != status::OK) {
    fprintf(stderr, "ERROR: Unable to initialize simulator.\n");
    return EXIT_FAILURE;
  }

  /* add one agent to the simulation */
  uint64_t new_agent_id; agent_state* new_agent;
  status result = sim.add_agent(new_agent_id, new_agent);
  if (result != status::OK) {
    fprintf(stderr, "ERROR: Unable to add new agent.\n");
    return EXIT_FAILURE;
  }

  /* the main simulation loop */
  for (unsigned int t = 0; t < 10000; t++) {
    if (sim.move(new_agent_id, direction::UP, 1) != status::OK) {
      fprintf(stderr, "ERROR: Unable to move agent.\n");
      return EXIT_FAILURE;
    }
  }
  free(sim);
}
```

See [jbw/tests/simulator_test.cpp](jbw/tests/simulator_test.cpp) for an example
with more types of items, as well as a multithreaded example and an MPI example.

To setup an MPI server for a simulator `sim`, the `init_server` function in
[jbw/mpi.h](jbw/mpi.h) may be used:
```c++
  /* NOTE: this blocks during the lifetime of the server */
  sync_server new_server;
  if (!init_server(new_server, sim, 54353, 256, 8, permissions::grant_all())) { /* process error */ }
```

To set up an asynchronous MPI server (where `init_server` will not block), the
`async_server` class in [jbw/mpi.h](jbw/mpi.h) is used:
```c++
  async_server new_server;
  if (!init_server(new_server, sim, 54353, 256, 8, permissions::grant_all())) { /* process error */ }
```

To connect to an existing server, for example at `localhost:54353`, we use the
`client` class defined in [jbw/mpi.h](jbw/mpi.h):
```c++
  client<empty_data> new_client;
  uint64_t client_id;
  uint64_t simulation_time = connect_client(new_client, "localhost", 54353, client_id);
  if (simulation_time == UINT64_MAX) { /* process error */ }
```
The commands may be sent to the server using the functions `send_add_agent`,
`send_move`, `send_turn`, etc. When the client receives responses from the
server, the functions `on_add_agent`, `on_move`, `on_turn`, etc will be
invoked, which must be defined by the user.

### Greedy Agent

The greedy visual agent is implemented in [jbw/agents/greedy_visual_agent.cpp](jbw/agents/greedy_visual_agent.cpp).

#### Compiling on Mac or Linux:
```bash
git submodule update --init --recursive
make greedy_visual_agent
```
The compiled executable will be located in `bin/greedy_visual_agent`.

#### Compiling on Windows:
First make sure the submodules are initialized by running
`git submodule update --init --recursive`. Next, build the Visual Studio
project in `vs/greedy_visual_agent.vcxproj`. The compiled executable will be
located in `bin/greedy_visual_agent.exe`.

Feel free to experiment with the environment configuration in [jbw/agents/greedy_visual_agent.cpp](jbw/agents/greedy_visual_agent.cpp).

## Using the Visualizer

We provide a real-time interactive visualizer, located in [jbw/visualizer/jbw_visualizer.cpp](jbw/visualizer/jbw_visualizer.cpp),
built using Vulkan and GLFW.

#### Mac or Linux:
```bash
git submodule update --init --recursive
make visualizer
```
The executable will be located in `bin/jbw_visualizer`. To run the visualizer:
```bash
cd bin
./jbw_visualizer
```

#### Windows:
First make sure the submodules are initialized by running
`git submodule update --init --recursive`. Next, open the Visual Studio project
`vs/jbw_visualizer.vcxproj` and make sure the include and library paths have
the correct Vulkan and GLFW directories on your machine. To do so, in the
Solution Explorer, right click on the `jbw_visualizer` project and select
"Properties". Select "C\C++ > General", and modify "Additional Include
Directories" as needed. For the library paths, select "Linker > General" and
modify "Additional Library Directories" as needed. Then build the project. The
executable will be located in `bin/jbw_visualizer.exe`. To run the visualizer:
```bash
cd bin
jbw_visualizer
```

#### Instructions:
Running the visualizer without arguments will print the help output, detailing
how to use the visualizer. The visualizer connects to a simulation server, as
specified by an address command-line argument, and begins to render the
connected simulation. The user is able to move the camera in the environment,
zoom in and out, increase or decrease the simulation update rate, track agents,
and take screenshots. A local simulation can be started without needing to
connect to a server by running `./jbw_visualizer --local`, which uses a
hard-coded configuration in [jbw/visualizer/jbw_visualizer.cpp](jbw/visualizer/jbw_visualizer.cpp).

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
In C++, [jbw/mpi.h](jbw/mpi.h) provides the functionality to run the server
and clients. See [jbw/tests/simulator_test.cpp](jbw/tests/simulator_test.cpp)
for an example.

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
be able to run on Mac, Linux, and Windows. We already provide **Python** and
**Swift** APIs that are quite simple to use and extend. APIs for other languages
are also easy to implement.

## Troubleshooting

### Repository initialization, publickey

If you get the message `Permission denied (publickey).` when initializing the
repository by calling `git submodule update --init --recursive` make sure you
have your public key set correctly in https://github.com/settings/keys. You can
see this example http://zeeelog.blogspot.com/2017/08/the-authenticity-of-host-githubcom.html
to generate a new one.

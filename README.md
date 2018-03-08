# Never-Ending Learning Framework

A framework for experimenting with never-ending learning.

## Requirements

- GCC 5+, Clang 5+, or Visual C++ 14+
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

### Usage

The typical workflow is as follows:

  1. Extend the `Agent` class to implement custom agents.
  2. Create a Simulator object.
  3. Construct agent instances in this simulator.
  4. Issue move commands for each agent.

The following is a simple example where a simulator is constructed locally
(within the same process) and a single agent continuously moves east. Note that
the agent's decision-making logic goes in the `next_move` method.

```python
import nel

class EasterlyAgent(nel.Agent):
  def __init__(self, simulator, load_filepath=None):
    super(EasterlyAgent, self).__init__(simulator, load_filepath)

  def next_move(self):
    return nel.Direction.RIGHT

  def save(self, filepath):
    pass

  def _load(self, filepath):
    pass


items = []
items.append(nel.Item("banana", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], True))

intensity_fn_args = [-2.0]
interaction_fn_args = [len(items)]
interaction_fn_args.extend([40.0, 200.0, 0.0, -40.0]) # parameters for interaction between item 0 and item 0

config = nel.SimulatorConfig(max_steps_per_movement=1, vision_range=1,
  patch_size=32, gibbs_num_iter=10, items=items, agent_color=[0.0, 0.0, 1.0],
  collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
  decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000,
  intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=intensity_fn_args,
  interaction_fn=nel.InteractionFunction.PIECEWISE_BOX, interaction_fn_args=interaction_fn_args)

sim = nel.Simulator(sim_config=config)

agent = EasterlyAgent(sim)

for t in range(10000):
  sim._move(agent._id, agent.next_move(), 1)
```

See [api/python/test/simulator_test.py](api/python/test/simulator_test.py) for
an example with more types of items as well as a visualization using the
MapVisualizer class.

### Agent class

Agents have a very simple interface. They have an abstract `next_move` method
which should contain the decision-making logic of the agent and return the
direction in which the agent has decided to move. The Agent class has an
abstract `save` method that users can implement to save the state of an agent
to a file. `save` is called by Simulator whenever the simulator is saved. The
Agent class also has an abstract `_load` method which is called by the Agent
constructor to load the agent's state from a given filepath.

Agents also have a private fields that store state information, such as the
agent's position, current scent perception, current visual perception, etc.

## Using C++

The typical workflow is as follows:

  1. Create a Simulator object.
  2. Add agents to this simulator.
  3. Issue move commands for each agent.

The following is a simple example where a simulator is constructed locally
(within the same process) and a single agent continuously moves east.

```c++
#include "simulator.h"

using namespace nel;

struct empty_data {
  static inline void free(empty_data& data) { }
};

constexpr bool init(empty_data& data, const empty_data& src) { return true; }

inline void set_interaction_args(float* args, unsigned int item_type_count,
    unsigned int first_item_type, unsigned int second_item_type,
    float first_cutoff, float second_cutoff, float first_value, float second_value)
{
  args[4 * (first_item_type * item_type_count + second_item_type) + 1] = first_cutoff;
  args[4 * (first_item_type * item_type_count + second_item_type) + 2] = second_cutoff;
  args[4 * (first_item_type * item_type_count + second_item_type) + 3] = first_value;
  args[4 * (first_item_type * item_type_count + second_item_type) + 4] = second_value;
}

void on_step(const simulator<empty_data>* sim, empty_data& data, uint64_t time) { }

int main(int argc, const char** argv) {
  simulator_config config;
  config.max_steps_per_movement = 1;
  config.scent_dimension = 3;
  config.color_dimension = 3;
  config.vision_range = 10;
  config.patch_size = 32;
  config.gibbs_iterations = 10;
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
  config.item_types[0].scent[0] = 1.0f;
  config.item_types[0].color[0] = 1.0f;
  config.item_types[0].automatically_collected = true;
  config.item_types.length = 1;

  config.intensity_fn_arg_count = (unsigned int) config.item_types.length;
  config.interaction_fn_arg_count = (unsigned int) (4 * config.item_types.length * config.item_types.length + 1);
  config.intensity_fn = constant_intensity_fn;
  config.interaction_fn = piecewise_box_interaction_fn;
  config.intensity_fn_args = (float*) malloc(sizeof(float) * config.intensity_fn_arg_count);
  config.interaction_fn_args = (float*) malloc(sizeof(float) * config.interaction_fn_arg_count);
  config.intensity_fn_args[0] = -2.0f;
  config.interaction_fn_args[0] = (float) config.item_types.length;
  set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 0, 0, 40.0f, 200.0f, 0.0f, -40.0f);

  simulator<empty_data>& sim = *((simulator<empty_data>*) alloca(sizeof(simulator<empty_data>)));
  if (!init(sim, config, empty_data())) {
    fprintf(stderr, "ERROR: Unable to initialize simulator.\n");
    return EXIT_FAILURE;
  }

  uint64_t agent_id = sim.add_agent();
  if (agent_id == UINT64_MAX) {
    fprintf(stderr, "ERROR: Unable to add new agent.\n");
    return EXIT_FAILURE;
  }

  for (unsigned int t = 0; t < 10000; t++) {
    if (!sim.move(agent_id, direction::RIGHT, 1)) {
      fprintf(stderr, "ERROR: Unable to move agent.\n");
      return EXIT_FAILURE;
    }
  }
  free(sim);
}
```

See [nel/simulator_test.cpp](nel/simulator_test.cpp) for an example with more
types of items, as well as a multithreaded example and an MPI example.

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
  - Once all agents have requested to move, the simulator progresses by one 
    time step and notifies invokes a callback function.
  - Some items in the world are automatically collected by the agents. The
    collected items are available in each agent's state information.

**NOTE:** Note that the agent is not moved until the simulator advances the
time step and issues a notification about that event. The simulator only 
advances the time step once all agents have requested to move.

We provide a message-passing interface (using TCP) to allow the simulator to
run remotely, and agents can issue move commands to the server. In Python, the
Simulator can be constructed as a server with the appropriate constructor
arguments. If a server address is provided to the Simulator constructor, it
will try to connect to the Simulator running at the specified remote address,
and all calls to the Simulator class will be issued as commands to the server.
In C++, `nel/mpi.h` provides the functionality to run the server and clients.
See `nel/simulator_test.cpp` for an example.


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
of generating the point at `x` given the existence of a point at `y`. Gibbs
sampling is used to sample from this distribution, and only requires a small
number of iterations to mix for non-pathological intensity/interaction
functions.

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

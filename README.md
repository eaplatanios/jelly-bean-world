# Never-Ending Learning Framework

A framework for experimenting with never-ending learning.

## Requirements

- GCC 5+ or Clang 5+
- Python 2.7 or 3.6 (for the Python API)

## Installation Instructions

### Python API

Assuming that you have Python installed in your system and 
that you are located in the root directory of this 
repository, run the following commands:

```bash
git submodule update --init --recursive
cd api/python
python setup.py install
```

You can proceed with the [description of a typical workflow](#workflow), for 
some instructions on how to use this library.

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
    time step and notifies all the agents (by invoking the `on_step()` method 
    that they implement).
  - All items in the world are automatically collected by the agents and are 
    available in their *current state* information. <!-- TODO: !!! -->

**NOTE:** Note that the agent is not moved until the simulator advances the  
time step and issues a notification about that event. The simulator only 
advances the time step once all agents have requested to move.

**NOTE:** Contention is not currently being handled and thus there may be issues 
with multi-agent systems. We plan to fix this very soon.

Simulators currently support two modes of operation:

  - **C Mode:** Uses the C-Python API bindings to interface with the simulator 
    and should be faster than the MPI mode. However, this mode does not allow 
    the simulator to run as a separate server process, that multiple agent 
    processes can attach to. In this case, all agents must be defined and used 
    as part of the same Python process that creates the simulator.
  - **MPI Mode:** Uses message passing over a socket (using TCP) and allows the 
    simulator to run as a separate server process that multiple agent processes 
    can attach to. However, it may result in slower performance than the C mode. 
    In this case, the simulator-agent interaction is done asynchronously using 
    message passing.

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
    p(X) = c * exp{sum from i to inf of f(X_i) * sum from j to inf of g(X_i, X_j)}.
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
cells with items are rendered with a single color. Then for each item within
the visual field of the agent, we render the corresponding pixel with the color
of the item.

Scent is modeled as a simple diffusion system on the grid, where each cell is
given a vector of scents (where each dimension can be used to model
orthogonal/unrelated scents). More precisely, if `S(x,y,t)` is the scent at
location `(x,y)` at time `t`, then
```
    S(x,y,t+1) = lambda*S(x,y,t) + C(x,y,t) + alpha*(S(x-1,y,t) + S(x+1,y,t) + S(x,y-1,t) + S(x,y+1,t)).
```
<!-- TODO: add details about modeling scent in inactive patches -->

### Agents

Agents have a very simple interface. They implement a `move` method that can 
be used to request a move from the simulator and they also have an abstract 
`on_step()` method that users can implement based on what they want their agent 
to do on each time step.

Agents also have a private field named `_current_state` that contains their 
current state in the simulation.

### Implementation

The core library is implemented in **C++** and has no dependencies. It should 
be able to run on Mac and Linux (and maybe even Windows, but we have not tested 
that case). We already provide a **Python** API that is quite simple to use and 
extend. APIs for other languages should also be easy to implement.

## Workflow

The typical workflow for this library is as follows:

  1. Extend the `Agent` class to implement custom agents.
  2. Create a simulator (if using the MPI mode, `start_server(...)` should be 
     called before this server can be used).
  3. Create agent instances in that simulator.
  4. *Optionally:* Allow for some way to monitor the agent's performance in its 
     `on_step()` method.

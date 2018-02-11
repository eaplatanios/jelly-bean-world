#include "../core/array.h"
#include "../map.h"

using namespace core;
using namespace map;

struct agent_state {
    position current_position;
};

struct cell_state {
    float scent[SCENT_DIMENSION];
    float color[3];
};

enum class direction { up = 0, down = 1, left = 2, right = 3 };

class simulator {
    array<agent_state> agents;


public:
    simulator() : agents(2), time(0) { }
    
    unsigned int time;
    
    inline unsigned int add_agent() {
        unsigned int id = array.length;
        agents.add({{0, 0}});
        return id;
    }

    inline void step(unsigned int agent_id, direction direction, int num_steps) {
        position current_position = agents[agent_id].current_position;
        switch (direction) {
            case up   : current_position.y += num_steps; break;
            case down : current_position.y -= num_steps; break;
            case left : current_position.x -= num_steps; break;
            case right: current_position.x += num_steps; break;
        }
        // TODO: Update the agent's state (i.e., position, items, etc.).
        // TODO: Block until all agents have invoked "step".
    }

    inline position position(unsigned int agent_id) {
        return agents[agent_id].current_position;
    }
};

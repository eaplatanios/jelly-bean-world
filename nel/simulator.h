#include <core/array.h>
#include "map.h"

using namespace core;
using namespace nel;

enum class direction { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

constexpr unsigned int SCENT_DIMENSION = 3;
constexpr unsigned int VISION_RANGE = 1;

struct agent_state {
    /* sensor properties */
    position current_position;
    float current_scent[SCENT_DIMENSION];
    float current_vision[3 * (2*VISION_RANGE + 1) * (2*VISION_RANGE + 1)]; /* in row-major order */

    /* effector properties */
    bool agent_acted; /* has the agent acted this turn? */
    direction next_move;
    unsigned int num_steps;
};

inline bool init(agent_state& new_agent) {
    new_agent.current_position = {0, 0};
    /* TODO: compute scent and vision */
    new_agent.agent_acted = false;
    return true;
}

class simulator {
    array<agent_state> agents;
    unsigned int acted_agent_count;
    std::mutex lock;

public:
    simulator() : agents(2), acted_agent_count(0), time(0) { }

    unsigned int time;

    inline unsigned int add_agent() {
        lock.lock();
        agents.ensure_capacity(agents.length + 1);

        unsigned int id = agents.length;
        init(agents[agents.length]);
        agents.length++;
        lock.unlock();
        return id;
    }

    inline bool move(unsigned int agent_id, direction direction, unsigned int num_steps) {
        lock.lock();
        agent_state& agent = agents[agent_id];
        if (agent.agent_acted) {
            lock.unlock(); return false;
        }
        agent.next_move = direction;
        agent.num_steps = num_steps;
        agent.agent_acted = true;
        acted_agent_count++;

        if (acted_agent_count == agents.length)
            step(); /* advance the simulation by one time step */

        lock.unlock(); return true;
    }

    inline position get_position(unsigned int agent_id) {
        lock.lock();
        position location = agents[agent_id].current_position;
        lock.unlock();
        return location;
    }

private:
    /* Precondition: The mutex is locked. This function does not release the mutex. */
    inline void step() {
        for (agent_state& agent : agents) {
            if (!agent.agent_acted) continue;

            /* compute agent motion */
            position& current_position = agent.current_position;
            switch (agent.next_move) {
                case direction::UP   : current_position.y += agent.num_steps; break;
                case direction::DOWN : current_position.y -= agent.num_steps; break;
                case direction::LEFT : current_position.x -= agent.num_steps; break;
                case direction::RIGHT: current_position.x += agent.num_steps; break;
            }

            /* TODO: compute new scent and vision for agent */
            agent.agent_acted = false;
        }
        acted_agent_count = 0;
        time++;
    }
};

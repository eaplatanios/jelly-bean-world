#ifndef NEL_SIMULATOR_H_
#define NEL_SIMULATOR_H_

#include <core/array.h>
#include "config.h"
#include "map.h"

using namespace core;
using namespace nel;

enum class direction { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

struct agent_state {
    /* sensor properties */
    position current_position;
    float* current_scent;

    /* 'pixels' are in row-major order, each pixel is a
       contiguous chunk of D floats where D is the color dimension */
    float* current_vision;

    /* effector properties */
    bool agent_acted; /* has the agent acted this turn? */
    direction next_move;
    unsigned int num_steps;

    std::mutex lock;

    inline static void free(agent_state& agent) {
        core::free(agent.current_scent);
        core::free(agent.current_vision);
        lock.~mutex();
    }
};

inline bool init(agent_state& new_agent, map& world,
        unsigned int color_dimension, unsigned int vision_range,
        unsigned int scent_dimension)
{
    new_agent.current_position = {0, 0};
    new_agent.current_scent = (float*) malloc(sizeof(float) * scent_dimension);
    if (new_agent.current_scent == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.current_scent.\n");
        return false;
    }
    new_agent.current_vision = (float*) malloc(sizeof(float)
        * (2*vision_range + 1) * (2*vision_range + 1) * color_dimension);
    if (new_agent.current_vision == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.current_vision.\n");
        free(new_agent.current_scent); return false;
    }

    new_agent.agent_acted = false;
    new (&new_agent.lock) std::mutex();


    return true;
}

class simulator {
    map world;
    array<agent_state*> agents;
    std::atomic<unsigned int> acted_agent_count;
    std::mutex agent_array_lock;

    simulator_config config;

public:
    simulator(const simulator_config& config) :
        map(config.patch_size, config.item_types.length, config.gibbs_iterations),
        agents(16), acted_agent_count(0), config(config), time(0) { }

    unsigned int time;

    inline agent_state* add_agent() {
        agent_array_lock.lock();
        agents.ensure_capacity(agents.length + 1);
        agent_state* new_agent = (agent_state*) malloc(sizeof(agent_state));
        agents.add(new_agent);
        agent_array_lock.unlock();

        init(*new_agent, map);
        return new_agent;
    }

    inline bool move(agent_state& agent, direction direction, unsigned int num_steps) {
        if (num_steps > config.max_steps_per_movement)
            return false;

        agent.lock.lock();
        if (agent.agent_acted) {
            agent.lock.unlock(); return false;
        }
        agent.next_move = direction;
        agent.num_steps = num_steps;
        agent.agent_acted = true;

        if (++acted_agent_count == agents.length)
            step(); /* advance the simulation by one time step */

        agent->lock.unlock(); return true;
    }

    inline position get_position(agent_state& agent) {
        agent.lock.lock();
        position location = agent.current_position;
        agent.lock.unlock();
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

    inline void get_scent(position world_position) {
        /* TODO: continue here (what is the closed form steady state of the diffusion difference equation?) */
    }
};

#endif /* NEL_SIMULATOR_H_ */

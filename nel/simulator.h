#ifndef NEL_SIMULATOR_H_
#define NEL_SIMULATOR_H_

#include <core/array.h>
#include <core/utility.h>
#include <atomic>
#include "map.h"

namespace nel {

using namespace core;

/** Represents all possible directions of motion in the environment. */
enum class direction { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

template<typename Stream>
inline bool read(direction& dir, Stream& in) {
    uint8_t c;
    if (!read(c, in)) return false;
    dir = (direction) c;
    return true;
}

template<typename Stream>
inline bool write(const direction& dir, Stream& out) {
    return write((uint8_t) dir, out);
}

/** Represents the state of an agent in the simulator. */
struct agent_state {
    /* Current position of the agent. */
    position current_position;

    /* Scent at the current position. */
    float* current_scent;

    /** 
     * Visual field at the current position. Consists of 'pixels' 
     * in row-major order, where each pixel is a contiguous chunk 
     * of D floats (where D is the color dimension). 
     */
    float* current_vision;
    
    /**
     * `true` if the agent has already acted (i.e., moved) in the 
     * current turn. 
     */
    bool agent_acted;

    /**
     * Direction for the agent's next move. This is updated when 
     * the agent requests to move and is reset once the simulator 
     * progresses by one time step and moves the agent.
     */
    direction next_move;

    /* Number of steps for the agent's next move. */
    unsigned int num_steps;

    /** 
     * Lock used by the simulator to prevent simultaneous updates 
     * to an agent's state.
     */
    std::mutex lock;

    /** Frees all allocated memory associated with this agent state. */
    inline static void free(agent_state& agent) {
        core::free(agent.current_scent);
        core::free(agent.current_vision);
        agent.lock.~mutex();
    }
};

/**
 * Initializes an agent's state in the provided world.
 * 
 * \param   agent_state     Agent state to initialize.
 * \param   world           Map of the world in which the agent is initialized.
 * \param   color_dimension Size of the color sense dimesion.
 * \param   vision_range    Range of the vision sense 
 *                          (i.e., number of grid cells).
 * \param   scent_dimension Size of the scent sense dimesion.
 */
inline bool init(agent_state& agent_state, map& world,
        unsigned int color_dimension, unsigned int vision_range,
        unsigned int scent_dimension)
{
    agent_state.current_position = {0, 0};
    agent_state.current_scent = (float*) malloc(sizeof(float) * scent_dimension);
    if (agent_state.current_scent == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.current_scent.\n");
        return false;
    }
    agent_state.current_vision = (float*) malloc(sizeof(float)
        * (2*vision_range + 1) * (2*vision_range + 1) * color_dimension);
    if (agent_state.current_vision == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.current_vision.\n");
        free(agent_state.current_scent); return false;
    }

    /* TODO: Initialize vision and scent. */

    agent_state.agent_acted = false;
    new (&agent_state.lock) std::mutex();

    return true;
}

struct item_properties {
	string name;

	float* scent;
	float* color;

	/* energy function parameters for the Gibbs field */
	float intensity;

	static inline void free(item_properties& properties) {
		core::free(properties.name);
		core::free(properties.scent);
		core::free(properties.color);
	}
};

inline bool init(
		item_properties& properties, const item_properties& src,
		unsigned int scent_dimension, unsigned int color_dimension)
{
	properties.name = src.name;
	properties.scent = (float*) malloc(sizeof(float) * scent_dimension);
	if (properties.scent == NULL) {
		fprintf(stderr, "init ERROR: Insufficient memory for item_properties.scent.\n");
		return false;
	}
	properties.color = (float*) malloc(sizeof(float) * color_dimension);
	if (properties.color == NULL) {
		fprintf(stderr, "init ERROR: Insufficient memory for item_properties.scent.\n");
		free(properties.scent); return false;
	}

	for (unsigned int i = 0; i < scent_dimension; i++)
		properties.scent[i] = src.scent[i];
	for (unsigned int i = 0; i < color_dimension; i++)
		properties.color[i] = src.color[i];
	properties.intensity = src.intensity;
	return true;
}

struct simulator_config {
	/* agent capabilities */
	unsigned int max_steps_per_movement;
	unsigned int scent_dimension;
	unsigned int color_dimension;
	unsigned int vision_range;

	/* world properties */
	unsigned int patch_size;
	unsigned int gibbs_iterations;
	array<item_properties> item_types;

	intensity_function intensity_fn;
	interaction_function interaction_fn;

	// We assume that the length of the args arrays is known at this point and has been checked.
    float* intensity_fn_args;
    float* interaction_fn_args;

	simulator_config() : item_types(8) { }

	simulator_config(const simulator_config& src) :
		max_steps_per_movement(src.max_steps_per_movement),
		scent_dimension(src.scent_dimension), color_dimension(src.color_dimension),
		vision_range(src.vision_range), patch_size(src.patch_size),
		gibbs_iterations(src.gibbs_iterations), item_types(src.item_types.length), 
		intensity_fn(src.intensity_fn), interaction_fn(src.interaction_fn), 
        intensity_fn_args(src.intensity_fn_args), interaction_fn_args(src.interaction_fn_args)
	{
		for (unsigned int i = 0; i < src.item_types.length; i++)
			init(item_types[i], src.item_types[i], scent_dimension, color_dimension);
		item_types.length = src.item_types.length;
    }

	~simulator_config() {
		for (item_properties& properties : item_types)
			core::free(properties);
    }
};

typedef void (*step_callback)(
    const simulator* sim, const unsigned int id, 
    const agent_state&, const simulator_config&);

/**
 * Simulator that forms the core of our experimentation framework.
 * 
 * \tparam  StepCallback    Callback function type for when the simulator 
 *                          advances a time step.
 */
class simulator {
    /* Map of the world managed by this simulator. */
    map world;

    /* Agents managed by this simulator. */
    array<agent_state*> agents;

    /** 
     * Atomic counter for how many agents have acted during each time step. 
     * This counter is used to force the simulator to wait until all agents 
     * have acted, before advancing the simulation time step.
     */
    std::atomic<unsigned int> acted_agent_count;

    /* Lock for the agents array, used to prevent simultaneous updates. */
    std::mutex agent_array_lock;

    /* Configuration for this simulator. */
    simulator_config config;

    /* Callback function for when the simulator advances a time step. */
    step_callback step_callback_fn;

public:
    simulator(const simulator_config& config, const step_callback step_callback) :
        world(config.patch_size, config.item_types.length, config.gibbs_iterations, 
            config.intensity_fn, config.intensity_fn_args, 
            config.interaction_fn, config.interaction_fn_args),
        agents(16), acted_agent_count(0), config(config), 
        step_callback_fn(step_callback), time(0) { }

    /* Current simulation time step. */
    unsigned int time;
    
    /** 
     * Adds a new agent to this simulator and returns its initial state.
     * 
     * \returns Initial state of the new agent.
     */
    inline agent_state* add_agent() {
        agent_array_lock.lock();
        unsigned int id = agents.length;
        agents.ensure_capacity(agents.length + 1);
        agent_state* new_agent = (agent_state*) malloc(sizeof(agent_state));
        agents.add(new_agent);
        agent_array_lock.unlock();

        init(*new_agent, world, config.color_dimension, 
            config.vision_range, config.scent_dimension);
        return id;
    }

    /** 
     * Moves an agent.
     * 
     * Note that the agent is only actually moved when the simulation time step 
     * advances, and only if the agent has not already acted for the current 
     * time step.
     * 
     * \param   agent     Agent to move.
     * \param   direction Direction along which to move.
     * \param   num_steps Number of steps to take in the specified direction.
     * \returns `true` if the move was successful, and `false` otherwise.
     */
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

        agent.lock.unlock(); return true;
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
        for (agent_state* agent : agents) {
            if (!agent->agent_acted) continue;

            /* compute agent motion */
            position& current_position = agent->current_position;
            switch (agent->next_move) {
                case direction::UP   : current_position.y += agent->num_steps; break;
                case direction::DOWN : current_position.y -= agent->num_steps; break;
                case direction::LEFT : current_position.x -= agent->num_steps; break;
                case direction::RIGHT: current_position.x += agent->num_steps; break;
            }

            /* TODO: compute new scent and vision for agent */
            agent->agent_acted = false;
        }
        acted_agent_count = 0;
        time++;

        // Invoke the step callback function for each agent.
        for (unsigned int id = 0; id < agents.length; id++)
            step_callback_fn(this, id, *agents[id], config);            
    }

    inline void get_scent(position world_position) {
        /* TODO: continue here (what is the closed form steady state of the diffusion difference equation?) */
    }
};

} /* namespace nel */

#endif /* NEL_SIMULATOR_H_ */

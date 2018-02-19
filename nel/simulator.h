#ifndef NEL_SIMULATOR_H_
#define NEL_SIMULATOR_H_

#include <core/array.h>
#include <core/utility.h>
#include <atomic>
#include "map.h"
#include "diffusion.h"

namespace nel {

using namespace core;

/* forward declarations */
class simulator;
struct agent_state;

/** Represents all possible directions of motion in the environment. */
enum class direction : uint8_t { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

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

template<typename Stream>
inline bool print(const direction& dir, Stream& out) {
    switch (dir) {
    case direction::UP:    return core::print("UP", out);
    case direction::DOWN:  return core::print("DOWN", out);
    case direction::LEFT:  return core::print("LEFT", out);
    case direction::RIGHT: return core::print("RIGHT", out);
    }
    fprintf(stderr, "print ERROR: Unrecognized direction.\n");
    return false;
}

enum class movement_conflict_policy {
    NO_COLLISION = 0,
    FIRST_COME_FIRST_SERVED = 1,
    RANDOM = 2
};

struct item_properties {
	string name;

	float* scent;
	float* color;

    bool automatically_collected;

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
    properties.automatically_collected = src.automatically_collected;
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
    float* agent_color;
    movement_conflict_policy collision_policy;

    /* parameters for scent diffusion */
    float decay_param, diffusion_param;
    unsigned int deleted_item_lifetime;

	intensity_function intensity_fn;
	interaction_function interaction_fn;

    float* intensity_fn_args;
    float* interaction_fn_args;
    unsigned int intensity_fn_arg_count;
    unsigned int interaction_fn_arg_count;

	simulator_config() : item_types(8) { }

    simulator_config(const simulator_config& src) : item_types(src.item_types.length) {
        if (!init_helper(src))
            exit(EXIT_FAILURE);
    }

    ~simulator_config() { free_helper(); }

    static inline void free(simulator_config& config) {
		config.free_helper();
        core::free(config.item_types);
    }

private:
    inline bool init_helper(const simulator_config& src)
    {
        agent_color = (float*) malloc(sizeof(float) * src.color_dimension);
        if (agent_color == NULL) {
            fprintf(stderr, "simulator_config.init_helper ERROR: Insufficient memory for agent_color.\n");
            return false;
        }
        intensity_fn_args = (float*) malloc(sizeof(float) * src.intensity_fn_arg_count);
        if (intensity_fn_args == NULL) {
            fprintf(stderr, "simulator_config.init_helper ERROR: Insufficient memory for intensity_fn_args.\n");
            core::free(agent_color); return false;
        }
        interaction_fn_args = (float*) malloc(sizeof(float) * src.interaction_fn_arg_count);
        if (interaction_fn_args == NULL) {
            fprintf(stderr, "simulator_config.init_helper ERROR: Insufficient memory for interaction_fn_args.\n");
            core::free(agent_color); core::free(intensity_fn_args); return false;
        }

        for (unsigned int i = 0; i < src.color_dimension; i++)
            agent_color[i] = src.agent_color[i];
        for (unsigned int i = 0; i < src.intensity_fn_arg_count; i++)
            intensity_fn_args[i] = src.intensity_fn_args[i];
        for (unsigned int i = 0; i < src.interaction_fn_arg_count; i++)
            interaction_fn_args[i] = src.interaction_fn_args[i];

        for (unsigned int i = 0; i < src.item_types.length; i++) {
            if (!init(item_types[i], src.item_types[i], src.scent_dimension, src.color_dimension)) {
                for (unsigned int j = 0; j < i; j++) core::free(item_types[i]);
                core::free(intensity_fn_args); core::free(interaction_fn_args);
                core::free(agent_color); return false;
            }
        }
        item_types.length = src.item_types.length;

        max_steps_per_movement = src.max_steps_per_movement;
        scent_dimension = src.scent_dimension;
        color_dimension = src.color_dimension;
        vision_range = src.vision_range;
        patch_size = src.patch_size;
        gibbs_iterations = src.gibbs_iterations;
        collision_policy = src.collision_policy;
        decay_param = src.decay_param;
        diffusion_param = src.diffusion_param;
        deleted_item_lifetime = src.deleted_item_lifetime;
        intensity_fn_arg_count = src.intensity_fn_arg_count;
        interaction_fn_arg_count = src.interaction_fn_arg_count;
        return true;
    }

    inline void free_helper() {
        for (item_properties& properties : item_types)
			core::free(properties);
        core::free(agent_color);
        core::free(intensity_fn_args);
        core::free(interaction_fn_args);
    }

    friend bool init(simulator_config&, const simulator_config&);
};

inline bool init(simulator_config& config, const simulator_config& src)
{
    if (!array_init(config.item_types, src.item_types.length)) {
        return false;
    } else if (!config.init_helper(src)) {
        free(config.item_types); return false;
    }
    return true;
}

struct patch_data {
    std::mutex patch_lock;
    array<agent_state*> agents;

    static inline void move(const patch_data& src, patch_data& dst) {
        core::move(src.agents, dst.agents);
        src.patch_lock.~mutex();
        new (&dst.patch_lock) std::mutex();
    }

    static inline void free(patch_data& data) {
        core::free(data.agents);
        data.patch_lock.~mutex();
    }
};

inline bool init(patch_data& data) {
    if (!array_init(data.agents, 4))
        return false;
    new (&data.patch_lock) std::mutex();
    return true;
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
     * The position which the agent requested to move this turn.
     */
    position requested_position;

    /** Number of items of each type in the agent's storage. */
    unsigned int* collected_items;

    /** 
     * Lock used by the simulator to prevent simultaneous updates 
     * to an agent's state.
     */
    std::mutex lock;

    inline void add_color(
            position relative_position, unsigned int vision_range,
            const float* color, unsigned int color_dimension)
    {
        unsigned int x = (unsigned int) (relative_position.x + vision_range);
        unsigned int y = (unsigned int) (relative_position.x + vision_range);
        unsigned int offset = (x*(2*vision_range + 1) + y) * color_dimension;
        for (unsigned int i = 0; i < color_dimension; i++)
            current_vision[offset + i] += color[i];
    }

    inline void add_scent(const float* scent, unsigned int scent_dimension, float value) {
        for (unsigned int i = 0; i < scent_dimension; i++)
            current_scent[i] += scent[i] * value;
    }

    template<typename T>
    inline void update_state(
            patch<patch_data>* neighborhood[4],
            const diffusion<T>& scent_model,
            const simulator_config& config,
            uint64_t current_time)
    {
        /* first zero out both current scent and vision */
        for (unsigned int i = 0; i < config.scent_dimension; i++)
            current_scent[i] = 0.0f;
        for (unsigned int i = 0; i < (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension; i++)
            current_vision[i] = 0.0f;

        for (unsigned int i = 0; i < 4; i++) {
            /* iterate over neighboring items, and add their contributions to scent and vision */
            for (unsigned int j = 0; j < neighborhood[i]->items.length; j++) {
                const item& item = neighborhood[i]->items[j];

                /* check if the item is too old; if so, delete it */
                if (item.deletion_time > 0 && current_time > item.deletion_time + config.deleted_item_lifetime) {
                    neighborhood[i]->items.remove(j); j--;
                }

                /* compute item position in agent coordinates */
                position relative_position = item.location - current_position;

                /* if the item is in the visual field, add its color to the appropriate pixel */
                if (item.deletion_time != 0
                 && abs(relative_position.x) <= config.vision_range
                 && abs(relative_position.y) <= config.vision_range) {
                    add_color(relative_position, config.vision_range,
                            config.item_types[item.item_type].color, config.color_dimension);
                }

                /* if the item is within scent range, add its contribution */
                if (abs(relative_position.x) < scent_model.radius
                 && abs(relative_position.y) < scent_model.radius) {
                    unsigned int creation_t = config.deleted_item_lifetime - 1;
                    if (item.creation_time > 0)
                        creation_t = min(creation_t, (unsigned int) (current_time - item.creation_time));
                    add_scent(config.item_types[item.item_type].scent, config.scent_dimension,
                            (float) scent_model.get_value(creation_t, (int) relative_position.x, (int) relative_position.y));

                    if (item.deletion_time > 0) {
                        unsigned int deletion_t = (unsigned int) (current_time - item.deletion_time);
                        add_scent(config.item_types[item.item_type].scent, config.scent_dimension,
                            (float) -scent_model.get_value(deletion_t, (int) relative_position.x, (int) relative_position.y));
                    }
                }
            }

            /* iterate over neighboring agents, and add their contributions to scent and vision */
            for (agent_state* agent : neighborhood[i]->data.agents) {
                /* compute neighbor position in agent coordinates */
                position relative_position = agent->current_position - current_position;

                /* if the neighbor is in the visual field, add its color to the appropriate pixel */
                if (abs(relative_position.x) <= config.vision_range
                 && abs(relative_position.y) <= config.vision_range) {
                    add_color(relative_position, config.vision_range,
                            config.agent_color, config.color_dimension);
                }
            }
        }
    }

    /** Frees all allocated memory associated with this agent state. */
    inline static void free(agent_state& agent) {
        core::free(agent.current_scent);
        core::free(agent.current_vision);
        core::free(agent.collected_items);
        agent.lock.~mutex();
    }
};

/**
 * Initializes an agent's state in the provided world.
 * 
 * \param   agent_state     Agent state to initialize.
 * \param   world           Map of the world in which the agent is initialized.
 * \param   scent_model     The scent diffusion model.
 * \param   config          The configuration for this simulation.
 * \param   current_time    The current simulation time.
 *
 * \tparam  T               The arithmetic type for the scent diffusion model.
 */
template<typename T>
inline bool init(
        agent_state& agent,
        map<patch_data>& world,
        const diffusion<T>& scent_model,
        const simulator_config& config,
        uint64_t& current_time)
{
    agent.current_position = {0, 0};
    agent.current_scent = (float*) malloc(sizeof(float) * config.scent_dimension);
    if (agent.current_scent == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.current_scent.\n");
        return false;
    }
    agent.current_vision = (float*) malloc(sizeof(float)
        * (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension);
    if (agent.current_vision == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.current_vision.\n");
        free(agent.current_scent); return false;
    }
    agent.collected_items = (unsigned int*) calloc(config.item_types.length, sizeof(unsigned int));
    if (agent.collected_items == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.collected_items.\n");
        free(agent.current_scent); free(agent.current_vision); return false;
    }

    agent.agent_acted = false;
    new (&agent.lock) std::mutex();

    /* initialize the scent and vision of the current agent */
    patch<patch_data>* neighborhood[4]; position patch_positions[4];
    unsigned int index = world.get_fixed_neighborhood(agent.current_position, neighborhood, patch_positions);
    agent.update_state(neighborhood, scent_model, config, current_time);

    neighborhood[index]->data.patch_lock.lock();
    if (config.collision_policy != movement_conflict_policy::NO_COLLISION) {
        for (const agent_state* neighbor : neighborhood[index]->data.agents) {
            if (agent.current_position == neighbor->current_position)
            {
                /* there is already an agent at this position */
				FILE* out = stderr;
                core::print("init ERROR: An agent already occupies position ", out);
                print(agent.current_position, out); core::print(".\n", out);
                free(agent.current_scent); free(agent.current_vision);
                free(agent.collected_items); agent.lock.~mutex();
                neighborhood[index]->data.patch_lock.unlock();
                return false;
            }
        }
    }
    neighborhood[index]->data.agents.add(&agent);
    neighborhood[index]->data.patch_lock.unlock();

    /* update the scent and vision of nearby agents */
    for (unsigned int i = 0; i < 4; i++) {
        for (agent_state* neighbor : neighborhood[i]->data.agents) {
            if (neighbor == &agent) continue;

            patch<patch_data>* other_neighborhood[4];
            world.get_fixed_neighborhood(neighbor->current_position, other_neighborhood, patch_positions);
            neighbor->update_state(other_neighborhood, scent_model, config, current_time);
        }
    }
    return true;
}

typedef void (*step_callback)(
    const simulator* sim, unsigned int id, 
    const agent_state&, const simulator_config&);

/**
 * Simulator that forms the core of our experimentation framework.
 * 
 * \tparam  StepCallback    Callback function type for when the simulator
 *                          advances a time step.
 */
class simulator {
    /* Map of the world managed by this simulator. */
    map<patch_data> world;

    /* The diffusion model to simulate scent. */
    diffusion<double> scent_model;

    /* Agents managed by this simulator. */
    array<agent_state*> agents;

    /* Lock for the agents array, used to prevent simultaneous updates. */
    std::mutex agent_array_lock;

    /* A map from positions to a list of agents that request to move there. */
    hash_map<position, array<agent_state*>> requested_moves;

    /* Lock for the requested_moves map, used to prevent simultaneous updates. */
    std::mutex requested_move_lock;

    /** 
     * Counter for how many agents have acted during each time step. This
     * counter is used to force the simulator to wait until all agents have
     * acted, before advancing the simulation time step.
     */
    unsigned int acted_agent_count;

    /* Configuration for this simulator. */
    simulator_config config;

    /* Callback function for when the simulator advances a time step. */
    step_callback step_callback_fn;

    typedef patch<patch_data> patch_type;

public:
    simulator(const simulator_config& config,
            const step_callback step_callback) :
        world(config.patch_size,
            (unsigned int) config.item_types.length,
            config.gibbs_iterations,
            config.intensity_fn, config.intensity_fn_args,
            config.interaction_fn, config.interaction_fn_args),
        agents(16), requested_moves(32, alloc_position_keys),
        acted_agent_count(0), config(config),
        step_callback_fn(step_callback), time(0)
    {
        if (!init(scent_model, (double) config.diffusion_param,
                (double) config.decay_param, config.patch_size, config.deleted_item_lifetime)) {
            fprintf(stderr, "simulator ERROR: Unable to initialize scent_model.\n");
            exit(EXIT_FAILURE);
        }
    }

    ~simulator() { free_helper(); }

    /* Current simulation time step. */
    uint64_t time;

    /** 
     * Adds a new agent to this simulator and returns its initial state.
     * 
     * \returns Initial state of the new agent, or NULL if there is an error.
     */
    inline agent_state* add_agent() {
        agent_array_lock.lock();
        agents.ensure_capacity(agents.length + 1);
        agent_state* new_agent = (agent_state*) malloc(sizeof(agent_state));
        if (new_agent == NULL) {
            fprintf(stderr, "simulator.add_agent ERROR: Insufficient memory for new agent.\n");
            return NULL;
        } else if (!agents.add(new_agent)) {
            core::free(new_agent);
            return NULL;
        }
        agent_array_lock.unlock();

        if (!init(*new_agent, world, scent_model, config, time)) {
            agent_array_lock.lock();
            agents.remove(agents.index_of(new_agent));
            core::free(new_agent);
            agent_array_lock.unlock();
            return NULL;
        }
        return new_agent;
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
    inline bool move(agent_state& agent, direction dir, unsigned int num_steps) {
        if (num_steps > config.max_steps_per_movement) return false;

        agent.lock.lock();
        if (agent.agent_acted) {
            agent.lock.unlock(); return false;
        }
        agent.agent_acted = true;

        agent.requested_position = agent.current_position;
        switch (dir) {
            case direction::UP   : agent.requested_position.y += num_steps; break;
            case direction::DOWN : agent.requested_position.y -= num_steps; break;
            case direction::LEFT : agent.requested_position.x -= num_steps; break;
            case direction::RIGHT: agent.requested_position.x += num_steps; break;
        }
        agent.lock.unlock();

        /* add the agent's move to the list of requested moves */
        request_new_position(agent);

        agent_array_lock.lock();
        if (++acted_agent_count == agents.length)
            step(); /* advance the simulation by one time step */
        agent_array_lock.unlock();
        return true;
    }

    inline position get_position(agent_state& agent) {
        agent.lock.lock();
        position location = agent.current_position;
        agent.lock.unlock();
        return location;
    }

    static inline void free(simulator& s) {
        s.free_helper();
        core::free(s.agents);
        core::free(s.requested_moves);
        core::free(s.config);
        core::free(s.scent_model);
        core::free(s.world);
        s.agent_array_lock.~mutex();
        s.requested_move_lock.~mutex();
    }

private:
    /* Precondition: The mutex is locked. This function does not release the mutex. */
    inline void step()
    {
        requested_move_lock.lock();
        if (config.collision_policy == movement_conflict_policy::RANDOM) {
            for (auto entry : requested_moves) {
                array<agent_state*>& conflicts = entry.value;
                unsigned int result = sample_uniform((unsigned int) conflicts.length);
                core::swap(conflicts[0], conflicts[result]);
            }
        }

        /* need to ensure agents don't move into positions where other agents failed to move */
        if (config.collision_policy != movement_conflict_policy::NO_COLLISION) {
            array<position> occupied_positions(16);
            for (auto entry : requested_moves) {
                array<agent_state*>& conflicts = entry.value;
                for (unsigned int i = 1; i < conflicts.length; i++)
                    occupied_positions.add(conflicts[i]->current_position);
            }

            bool contains;
            while (occupied_positions.length > 0) {
                array<agent_state*>& conflicts = requested_moves.get(occupied_positions.pop(), contains);
                if (!contains || conflicts[0] == NULL) {
                    continue;
                } else if (contains) {
                    for (unsigned int i = 0; i < conflicts.length; i++)
                        occupied_positions.add(conflicts[i]->current_position);
                    conflicts[0] = NULL; /* prevent any agent from moving here */
                }
            }
        }

        time++;
        acted_agent_count = 0;
        for (agent_state* agent : agents) {
            if (!agent->agent_acted) continue;

            /* check if this agent moved, in accordance with the collision policy */
            position old_patch_position;
            world.world_to_patch_coordinates(agent->current_position, old_patch_position);
            if (config.collision_policy == movement_conflict_policy::NO_COLLISION
             || (agent == requested_moves.get(agent->requested_position)[0]))
            {
                agent->current_position = agent->requested_position;

                /* delete any items that are automatically picked up at this cell */
                patch_type* neighborhood[4]; position patch_positions[4];
                unsigned int index = world.get_fixed_neighborhood(agent->current_position, neighborhood, patch_positions);
                patch_type& current_patch = *neighborhood[index];
                for (item& item : current_patch.items) {
                    if (item.location == agent->current_position) {
                        /* there is an item at our new position */
                        if (config.item_types[item.item_type].automatically_collected) {
                            /* collect this item */
                            item.deletion_time = time;
                            agent->collected_items[item.item_type]++;
                        }
                    }
                }

                if (old_patch_position != patch_positions[index]) {
                    patch_type& prev_patch = world.get_existing_patch(old_patch_position);
                    prev_patch.data.patch_lock.lock();
                    prev_patch.data.agents.remove(prev_patch.data.agents.index_of(agent));
                    prev_patch.data.patch_lock.unlock();
                    current_patch.data.patch_lock.lock();
                    current_patch.data.agents.add(agent);
                    current_patch.data.patch_lock.unlock();
                }
            }
            agent->agent_acted = false;
        }

#if !defined(NDEBUG)
		/* check for collisions, if there aren't supposed to be any */
		if (config.collision_policy != movement_conflict_policy::NO_COLLISION) {
			for (unsigned int i = 0; i < agents.length; i++)
				for (unsigned int j = i + 1; j < agents.length; j++)
					if (agents[i]->current_position == agents[j]->current_position)
						fprintf(stderr, "simulator.step WARNING: Agents %u and %u are at the same position.\n", i, j);
		}
#endif

        /* reset the requested moves */
        for (auto entry : requested_moves)
            core::free(entry.value);
        requested_moves.clear();
        requested_move_lock.unlock();

        /* compute new scent and vision for each agent */
        for (agent_state* agent : agents) {
            patch_type* neighborhood[4]; position patch_positions[4];
            world.get_fixed_neighborhood(agent->current_position, neighborhood, patch_positions);
            agent->update_state(neighborhood, scent_model, config, time);
        }

        /* Invoke the step callback function for each agent. */
        for (unsigned int id = 0; id < agents.length; id++)
            step_callback_fn(this, id, *agents[id], config);
    }

    inline void request_new_position(agent_state& agent)
    {
        if (config.collision_policy == movement_conflict_policy::NO_COLLISION)
            return;

        bool contains; unsigned int bucket;
        requested_move_lock.lock();
        requested_moves.check_size(alloc_position_keys);
        array<agent_state*>& agents = requested_moves.get(agent.requested_position, contains, bucket);
        if (!contains) {
            array_init(agents, 8);
            requested_moves.table.keys[bucket] = agent.requested_position;
            requested_moves.table.size++;
        }
        agents.add(&agent);
        requested_move_lock.unlock();
    }

    inline void free_helper() {
        for (auto entry : requested_moves)
            core::free(entry.value);
        for (agent_state* agent : agents) {
            core::free(*agent);
            core::free(agent);
        }
    }

    friend bool init(simulator&, const simulator_config&, const step_callback);
};

bool init(simulator& sim, 
        const simulator_config& config,
        const step_callback step_callback)
{
    sim.step_callback_fn = step_callback;
    sim.time = 0;
    sim.acted_agent_count = 0;
    if (!array_init(sim.agents, 16)) {
        return false;
    } else if (!hash_map_init(sim.requested_moves, 32, alloc_position_keys)) {
        free(sim.agents); return false;
    } else if (!init(sim.config, config)) {
        free(sim.agents); free(sim.requested_moves);
        return false;
    } else if (!init(sim.scent_model, (double) config.diffusion_param,
            (double) config.decay_param, config.patch_size, config.deleted_item_lifetime)) {
        free(sim.config); free(sim.agents);
        free(sim.requested_moves); return false;
    } else if (!init(sim.world, config.patch_size,
            (unsigned int) config.item_types.length,
            config.gibbs_iterations,
            config.intensity_fn, config.intensity_fn_args,
            config.interaction_fn, config.interaction_fn_args)) {
        free(sim.config); free(sim.agents);
        free(sim.requested_moves);
        free(sim.scent_model); return false;
    }
    new (&sim.agent_array_lock) std::mutex();
    new (&sim.requested_move_lock) std::mutex();
    return true;
}

} /* namespace nel */

#endif /* NEL_SIMULATOR_H_ */

#ifndef NEL_SIMULATOR_H_
#define NEL_SIMULATOR_H_

#include <core/array.h>
#include <core/utility.h>
#include <atomic>
#include <mutex>
#include "map.h"
#include "diffusion.h"

namespace nel {

using namespace core;

/* forward declarations */
template<typename SimulatorData> class simulator;
struct agent_state;

/** Represents all possible directions of motion in the environment. */
enum class direction : uint8_t { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, COUNT };

/**
 * Reads the given direction `dir` from the input stream `in`.
 */
template<typename Stream>
inline bool read(direction& dir, Stream& in) {
    uint8_t c;
    if (!read(c, in)) return false;
    dir = (direction) c;
    return true;
}

/**
 * Writes the given direction `dir` to the output stream `out`.
 */
template<typename Stream>
inline bool write(const direction& dir, Stream& out) {
    return write((uint8_t) dir, out);
}

/**
 * Prints the given direction `dir` to the output stream `out`.
 */
template<typename Stream>
inline bool print(const direction& dir, Stream& out) {
    switch (dir) {
    case direction::UP:    return core::print("UP", out);
    case direction::DOWN:  return core::print("DOWN", out);
    case direction::LEFT:  return core::print("LEFT", out);
    case direction::RIGHT: return core::print("RIGHT", out);
    case direction::COUNT: break;
    }
    fprintf(stderr, "print ERROR: Unrecognized direction.\n");
    return false;
}

/**
 * An enum representing the simulator policy for resolving the case when
 * multiple agents request to move into the same position.
 */
enum class movement_conflict_policy : uint8_t {
    NO_COLLISIONS = 0,
    FIRST_COME_FIRST_SERVED = 1,
    RANDOM = 2
};

/**
 * Reads the given movement_conflict_policy `policy` from the stream `in`.
 */
template<typename Stream>
inline bool read(movement_conflict_policy& policy, Stream& in) {
    uint8_t c;
    if (!read(c, in)) return false;
    policy = (movement_conflict_policy) c;
    return true;
}

/**
 * Writes the given movement_conflict_policy `policy` to the stream `out`.
 */
template<typename Stream>
inline bool write(const movement_conflict_policy& policy, Stream& out) {
    return write((uint8_t) policy, out);
}

/**
 * A structure containing the properties of an item type.
 */
struct item_properties {
    string name;

    float* scent;
    float* color;

    unsigned int* required_item_counts;
    unsigned int* required_item_costs;

    bool blocks_movement;

    intensity_function intensity_fn;
    interaction_function* interaction_fns;

    float* intensity_fn_args;
    float** interaction_fn_args;
    unsigned int intensity_fn_arg_count;
    unsigned int* interaction_fn_arg_counts;

    static inline void free(item_properties& properties, unsigned int item_type_count) {
        core::free(properties.name);
        core::free(properties.scent);
        core::free(properties.color);
        core::free(properties.required_item_counts);
        core::free(properties.required_item_costs);
        core::free(properties.intensity_fn_args);
        for (unsigned int i = 0; i < item_type_count; i++)
            core::free(properties.interaction_fn_args[i]);
        core::free(properties.interaction_fn_args);
        core::free(properties.interaction_fn_arg_counts);
        core::free(properties.interaction_fns);
    }
};

/**
 * Initializes the given item_properties `properties` with the properties given as arguments.
 */
inline bool init(
        item_properties& properties, const char* name, unsigned int name_length,
        const float* scent, const float* color, unsigned int* required_item_counts,
        unsigned int* required_item_costs, bool blocks_movement,
        intensity_function intensity_fn, const interaction_function* interaction_fns,
        const float* intensity_fn_args, const float* const* interaction_fn_args,
        unsigned int intensity_fn_arg_count, const unsigned int* interaction_fn_arg_counts,
        unsigned int scent_dimension, unsigned int color_dimension,
        unsigned int item_type_count)
{
    if (!init(properties.name, name, name_length))
        return false;
    properties.scent = (float*) malloc(sizeof(float) * scent_dimension);
    if (properties.scent == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.scent.\n");
        core::free(properties.name); return false;
    }
    properties.color = (float*) malloc(sizeof(float) * color_dimension);
    if (properties.color == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.scent.\n");
        core::free(properties.name); free(properties.scent); return false;
    }
    properties.required_item_counts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.required_item_counts == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.required_item_counts.\n");
        core::free(properties.name); free(properties.scent);
        free(properties.color); return false;
    }
    properties.required_item_costs = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.required_item_costs == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.required_item_costs.\n");
        core::free(properties.name); free(properties.scent);
        free(properties.color); free(properties.required_item_counts); return false;
    }
    properties.interaction_fns = (interaction_function*) malloc(sizeof(interaction_function) * item_type_count);
    if (properties.interaction_fns == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.interaction_fns.\n");
        core::free(properties.name); free(properties.scent);
        free(properties.color); free(properties.required_item_counts);
        free(properties.required_item_costs); return false;
    }
    properties.intensity_fn_arg_count = intensity_fn_arg_count;
    properties.intensity_fn_args = (float*) malloc(max((size_t) 1, sizeof(float) * intensity_fn_arg_count));
    if (properties.intensity_fn_args == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.intensity_fn_arg_count.\n");
        core::free(properties.name); free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); return false;
    }
    properties.interaction_fn_args = (float**) malloc(sizeof(float*) * item_type_count);
    if (properties.interaction_fn_args == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.interaction_fn_args.\n");
        core::free(properties.name); free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); free(properties.intensity_fn_args); return false;
    }
    properties.interaction_fn_arg_counts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.interaction_fn_arg_counts == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.interaction_fn_arg_counts.\n");
        core::free(properties.name); free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); free(properties.intensity_fn_args);
        free(properties.interaction_fn_args); return false;
    }

    for (unsigned int i = 0; i < item_type_count; i++) {
        properties.interaction_fns[i] = interaction_fns[i];
        properties.interaction_fn_arg_counts[i] = interaction_fn_arg_counts[i];
        properties.interaction_fn_args[i] = (float*) malloc(max((size_t) 1, sizeof(float) * interaction_fn_arg_counts[i]));
        if (properties.interaction_fn_args[i] == NULL) {
            fprintf(stderr, "init ERROR: Insufficient memory for item_properties.interaction_fn_args[%u].\n", i);
            for (unsigned int j = 0; j < i; j++) free(properties.interaction_fn_args[j]);
            core::free(properties.name); free(properties.scent); free(properties.color);
            free(properties.required_item_counts); free(properties.required_item_costs);
            free(properties.interaction_fns); free(properties.intensity_fn_args);
            free(properties.interaction_fn_args); return false;
        }
    }

    for (unsigned int i = 0; i < scent_dimension; i++)
        properties.scent[i] = scent[i];
    for (unsigned int i = 0; i < color_dimension; i++)
        properties.color[i] = color[i];
    for (unsigned int i = 0; i < item_type_count; i++)
        properties.required_item_counts[i] = required_item_counts[i];
    for (unsigned int i = 0; i < item_type_count; i++)
        properties.required_item_costs[i] = required_item_costs[i];
    properties.blocks_movement = blocks_movement;
    properties.intensity_fn = intensity_fn;
    memcpy(properties.intensity_fn_args, intensity_fn_args, sizeof(float) * intensity_fn_arg_count);
    for (unsigned int i = 0; i < item_type_count; i++)
        memcpy(properties.interaction_fn_args[i], interaction_fn_args[i], sizeof(float) * interaction_fn_arg_counts[i]);
    return true;
}

/**
 * Initializes the given item_properties `properties` by copying from `src`.
 */
inline bool init(
        item_properties& properties, const item_properties& src,
        unsigned int scent_dimension, unsigned int color_dimension,
        unsigned int item_type_count)
{
    return init(properties, src.name.data, src.name.length,
        src.scent, src.color, src.required_item_counts,
        src.required_item_costs, src.blocks_movement,
        src.intensity_fn, src.interaction_fns,
        src.intensity_fn_args, src.interaction_fn_args,
        src.intensity_fn_arg_count, src.interaction_fn_arg_counts,
        scent_dimension, color_dimension, item_type_count);
}

/**
 * Reads the given item_properties structure `properties` from the stream `in`.
 */
template<typename Stream>
inline bool read(item_properties& properties, Stream& in,
        unsigned int scent_dimension, unsigned int color_dimension,
        unsigned int item_type_count)
{
    if (!read(properties.name, in)) return false;

    properties.scent = (float*) malloc(sizeof(float) * scent_dimension);
    if (properties.scent == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.scent.\n");
        free(properties.name); return false;
    }
    properties.color = (float*) malloc(sizeof(float) * color_dimension);
    if (properties.color == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.scent.\n");
        free(properties.name); free(properties.scent); return false;
    }
    properties.required_item_counts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.required_item_counts == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.required_item_counts.\n");
        free(properties.scent); free(properties.color); return false;
    }
    properties.required_item_costs = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.required_item_costs == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.required_item_costs.\n");
        free(properties.scent); free(properties.color);
        free(properties.required_item_counts); return false;
    }
    properties.interaction_fns = (interaction_function*) malloc(sizeof(interaction_function) * item_type_count);
    if (properties.interaction_fns == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.interaction_fns.\n");
        free(properties.scent); free(properties.color);
        free(properties.required_item_counts);
        free(properties.required_item_costs); return false;
    }
    properties.interaction_fn_args = (float**) malloc(sizeof(float*) * item_type_count);
    if (properties.interaction_fn_args == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.interaction_fn_args.\n");
        free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); return false;
    }
    properties.interaction_fn_arg_counts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.interaction_fn_arg_counts == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.interaction_fn_arg_counts.\n");
        free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); free(properties.interaction_fn_args); return false;
    }

    if (!read(properties.scent, in, scent_dimension)
     || !read(properties.color, in, color_dimension)
     || !read(properties.required_item_counts, in, item_type_count)
     || !read(properties.required_item_costs, in, item_type_count)
     || !read(properties.blocks_movement, in)
     || !read(properties.intensity_fn, in)
     || !read(properties.intensity_fn_arg_count, in)
     || !read(properties.interaction_fns, in, item_type_count)
     || !read(properties.interaction_fn_arg_counts, in, item_type_count)) {
        free(properties.name); free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); free(properties.interaction_fn_args); return false;
    }

    properties.intensity_fn_args = (float*) malloc(max((size_t) 1, sizeof(float) * properties.intensity_fn_arg_count));
    if (properties.intensity_fn_args == NULL
     || !read(properties.intensity_fn_args, in, properties.intensity_fn_arg_count))
    {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.intensity_fn_args.\n");
        if (properties.intensity_fn_args != NULL) free(properties.intensity_fn_args);
        free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); free(properties.interaction_fn_args);
        free(properties.interaction_fn_arg_counts); return false;
    }

    for (unsigned int i = 0; i < item_type_count; i++) {
        properties.interaction_fn_args[i] = (float*) malloc(max((size_t) 1, sizeof(float) * properties.interaction_fn_arg_counts[i]));
        if (properties.interaction_fn_args[i] == NULL
         || !read(properties.interaction_fn_args[i], in, properties.interaction_fn_arg_counts[i]))
        {
            fprintf(stderr, "read ERROR: Insufficient memory for item_properties.interaction_fn_args.\n");
            for (unsigned int j = 0; j < i; j++) free(properties.interaction_fn_args[j]);
            if (properties.interaction_fn_args[i] != NULL) free(properties.interaction_fn_args[i]);
            free(properties.scent); free(properties.color);
            free(properties.required_item_counts); free(properties.required_item_costs);
            free(properties.interaction_fns); free(properties.interaction_fn_args);
            free(properties.interaction_fn_arg_counts); free(properties.intensity_fn_args); return false;
        }
    }
    return true;
}

/**
 * Writes the given item_properties structure `properties` to the stream `out`.
 */
template<typename Stream>
inline bool write(const item_properties& properties, Stream& out,
        unsigned int scent_dimension, unsigned int color_dimension,
        unsigned int item_type_count)
{
    if (!write(properties.name, out)
     || !write(properties.scent, out, scent_dimension)
     || !write(properties.color, out, color_dimension)
     || !write(properties.required_item_counts, out, item_type_count)
     || !write(properties.required_item_costs, out, item_type_count)
     || !write(properties.blocks_movement, out)
     || !write(properties.intensity_fn, out)
     || !write(properties.intensity_fn_arg_count, out)
     || !write(properties.interaction_fns, out, item_type_count)
     || !write(properties.interaction_fn_arg_counts, out, item_type_count)
     || !write(properties.intensity_fn_args, out, properties.intensity_fn_arg_count))
        return false;

    for (unsigned int i = 0; i < item_type_count; i++)
        if (!write(properties.interaction_fn_args[i], out, properties.interaction_fn_arg_counts[i])) return false;
    return true;
}

/**
 * Represents the configuration of a simulator. 
 */
struct simulator_config {
    /* agent capabilities */
    unsigned int max_steps_per_movement;
    unsigned int scent_dimension;
    unsigned int color_dimension;
    unsigned int vision_range;
    bool allowed_movement_directions[(size_t) direction::COUNT];
    bool allowed_rotations[(size_t) direction::COUNT];

    /* world properties */
    unsigned int patch_size;
    unsigned int gibbs_iterations;
    array<item_properties> item_types;
    float* agent_color;
    movement_conflict_policy collision_policy;

    /* parameters for scent diffusion */
    float decay_param, diffusion_param;
    unsigned int deleted_item_lifetime;

    simulator_config() : item_types(8), agent_color(NULL) { }

    simulator_config(const simulator_config& src) : item_types(src.item_types.length) {
        if (!init_helper(src))
            exit(EXIT_FAILURE);
    }

    ~simulator_config() { free_helper(); }

    static inline void swap(simulator_config& first, simulator_config& second) {
        for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
            core::swap(first.allowed_movement_directions[i], second.allowed_movement_directions[i]);
        for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
            core::swap(first.allowed_rotations[i], second.allowed_rotations[i]);
        core::swap(first.max_steps_per_movement, second.max_steps_per_movement);
        core::swap(first.scent_dimension, second.scent_dimension);
        core::swap(first.color_dimension, second.color_dimension);
        core::swap(first.vision_range, second.vision_range);
        core::swap(first.patch_size, second.patch_size);
        core::swap(first.gibbs_iterations, second.gibbs_iterations);
        core::swap(first.item_types, second.item_types);
        core::swap(first.agent_color, second.agent_color);
        core::swap(first.collision_policy, second.collision_policy);
        core::swap(first.decay_param, second.decay_param);
        core::swap(first.diffusion_param, second.diffusion_param);
        core::swap(first.deleted_item_lifetime, second.deleted_item_lifetime);
    }

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

        for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
            allowed_movement_directions[i] = src.allowed_movement_directions[i];
        for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
            allowed_rotations[i] = src.allowed_rotations[i];
        for (unsigned int i = 0; i < src.color_dimension; i++)
            agent_color[i] = src.agent_color[i];

        for (unsigned int i = 0; i < src.item_types.length; i++) {
            if (!init(item_types[i], src.item_types[i], src.scent_dimension, src.color_dimension, (unsigned int) src.item_types.length)) {
                for (unsigned int j = 0; j < i; j++)
                    core::free(item_types[i], (unsigned int) src.item_types.length);
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
        return true;
    }

    inline void free_helper() {
        for (item_properties& properties : item_types)
            core::free(properties, (unsigned int) item_types.length);
        if (agent_color != NULL)
            core::free(agent_color);
    }

    friend bool init(simulator_config&, const simulator_config&);
};

/**
 * Initializes the given simulator_config with a NULL `agent_color`,
 * `intensity_fn_args`, `interaction_fn_args`, and an empty `item_types`. This
 * function does not initialize any other fields.
 */
inline bool init(simulator_config& config) {
    config.agent_color = NULL;
    return array_init(config.item_types, 8);
}

/**
 * Initializes the given simulator_config `config` by copying from `src`.
 */
inline bool init(simulator_config& config, const simulator_config& src)
{
    if (!array_init(config.item_types, src.item_types.length)) {
        return false;
    } else if (!config.init_helper(src)) {
        free(config.item_types); return false;
    }
    return true;
}

/**
 * Reads the given simulator_config `config` from the input stream `in`.
 */
template<typename Stream>
bool read(simulator_config& config, Stream& in) {
    if (!read(config.max_steps_per_movement, in)
     || !read(config.scent_dimension, in)
     || !read(config.color_dimension, in)
     || !read(config.vision_range, in)
     || !read(config.allowed_movement_directions, in)
     || !read(config.allowed_rotations, in)
     || !read(config.patch_size, in)
     || !read(config.gibbs_iterations, in)
     || !read(config.item_types.length, in))
        return false;

    config.item_types.data = (item_properties*) malloc(max((size_t) 1, sizeof(item_properties) * config.item_types.length));
    if (config.item_types.data == NULL
     || !read(config.item_types.data, in, config.item_types.length, config.scent_dimension, config.color_dimension, (unsigned int) config.item_types.length)) {
        fprintf(stderr, "read ERROR: Insufficient memory for simulator_config.item_types.data.\n");
        return false;
    }

    config.agent_color = (float*) malloc(sizeof(float) * config.color_dimension);
    if (config.agent_color == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for simulator_config.agent_color.\n");
        for (item_properties& properties : config.item_types)
            free(properties, (unsigned int) config.item_types.length);
        free(config.item_types); return false;
    }

    if (!read(config.agent_color, in, config.color_dimension)
     || !read(config.collision_policy, in)
     || !read(config.decay_param, in)
     || !read(config.diffusion_param, in)
     || !read(config.deleted_item_lifetime, in)) {
        for (item_properties& properties : config.item_types)
            free(properties, (unsigned int) config.item_types.length);
        free(config.agent_color); free(config.item_types); return false;
    }
    return true;
}

/**
 * Writes the given simulator_config `config` to the output stream `out`.
 */
template<typename Stream>
bool write(const simulator_config& config, Stream& out) {
    return write(config.max_steps_per_movement, out)
        && write(config.scent_dimension, out)
        && write(config.color_dimension, out)
        && write(config.vision_range, out)
        && write(config.allowed_movement_directions, out)
        && write(config.allowed_rotations, out)
        && write(config.patch_size, out)
        && write(config.gibbs_iterations, out)
        && write(config.item_types.length, out)
        && write(config.item_types.data, out, config.item_types.length, config.scent_dimension, config.color_dimension, (unsigned int) config.item_types.length)
        && write(config.agent_color, out, config.color_dimension)
        && write(config.collision_policy, out)
        && write(config.decay_param, out)
        && write(config.diffusion_param, out)
        && write(config.deleted_item_lifetime, out);
}

/**
 * A structure that is used to store additional state information in the map
 * structure. So far, this structure stores an array of agents that inhabit the
 * associated patch, as well as a lock for accessing this array.
 */
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

/**
 * Initializes the given patch_data `data`, where `agents` is empty.
 */
inline bool init(patch_data& data) {
    if (!array_init(data.agents, 4))
        return false;
    new (&data.patch_lock) std::mutex();
    return true;
}

/**
 * Reads the given patch_data `data` structure from the input stream `in`.
 */
template<typename Stream>
bool read(patch_data& data, Stream& in, array<agent_state*>& agents) {
    size_t agent_count = 0;
    if (!read(agent_count, in)
     || !array_init(data.agents, max((size_t) 2, agent_count)))
        return false;
    for (unsigned int i = 0; i < agent_count; i++) {
        unsigned int index;
        if (!read(index, in)) {
            free(data.agents); return false;
        }
        data.agents[i] = agents[index];
    }
    data.agents.length = agent_count;
    new (&data.patch_lock) std::mutex();
    return true;
}

/**
 * Writes the given patch_data `data` structure to the output stream `out`.
 */
template<typename Stream>
bool write(const patch_data& data, Stream& out,
        hash_map<const agent_state*, unsigned int>& agents)
{
    if (!write(data.agents.length, out))
        return false;
    for (const agent_state* agent : data.agents)
        if (!write(agents.get(agent), out)) return false;
    return true;
}

inline void add_scent(float* dst, const float* scent, unsigned int scent_dimension, float value) {
    for (unsigned int i = 0; i < scent_dimension; i++)
        dst[i] += scent[i] * value;
}

template<typename T>
void compute_scent_contribution(
        const diffusion<T>& scent_model, const item& item,
        position pos, uint64_t current_time,
        const simulator_config& config, float* dst)
{
    /* compute item position in agent coordinates */
    position relative_position = item.location - pos;

    /* if the item is within scent range, add its contribution */
    if ((unsigned int) abs(relative_position.x) < scent_model.radius
     && (unsigned int) abs(relative_position.y) < scent_model.radius)
    {
        unsigned int creation_t = config.deleted_item_lifetime - 1;
        if (item.creation_time > 0)
            creation_t = min(creation_t, (unsigned int) (current_time - item.creation_time));
        add_scent(dst, config.item_types[item.item_type].scent, config.scent_dimension,
                (float) scent_model.get_value(creation_t, (int) relative_position.x, (int) relative_position.y));

        if (item.deletion_time > 0) {
            unsigned int deletion_t = (unsigned int) (current_time - item.deletion_time);
            add_scent(dst, config.item_types[item.item_type].scent, config.scent_dimension,
                (float) -scent_model.get_value(deletion_t, (int) relative_position.x, (int) relative_position.y));
        }
    }
}

/** Represents the state of an agent in the simulator. */
struct agent_state {
    /* Current position of the agent. */
    position current_position;

    /* Current direction of the agent. */
    direction current_direction;

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

    /**
     * The direction which the agent requested to rotate to this turn.
     */
    direction requested_direction;

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
        switch (current_direction) {
        case direction::UP: break;
        case direction::DOWN:
            relative_position.x *= -1;
            relative_position.y *= -1;
            break;
        case direction::LEFT:
            core::swap(relative_position.x, relative_position.y);
            relative_position.y *= -1; break;
        case direction::RIGHT:
            core::swap(relative_position.x, relative_position.y);
            relative_position.x *= -1; break;
        case direction::COUNT: break;
        }
        unsigned int x = (unsigned int) (relative_position.x + vision_range);
        unsigned int y = (unsigned int) (relative_position.y + vision_range);
        unsigned int offset = (x*(2*vision_range + 1) + y) * color_dimension;
        for (unsigned int i = 0; i < color_dimension; i++)
            current_vision[offset + i] += color[i];
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
                if (item.deletion_time > 0 && current_time >= item.deletion_time + config.deleted_item_lifetime) {
                    neighborhood[i]->items.remove(j); j--; continue;
                }

                compute_scent_contribution(scent_model, item, current_position, current_time, config, current_scent);

                /* if the item is in the visual field, add its color to the appropriate pixel */
                position relative_position = item.location - current_position;
                if (item.deletion_time == 0
                 && (unsigned int) abs(relative_position.x) <= config.vision_range
                 && (unsigned int) abs(relative_position.y) <= config.vision_range) {
                    add_color(relative_position, config.vision_range,
                            config.item_types[item.item_type].color, config.color_dimension);
                }
            }

            /* iterate over neighboring agents, and add their contributions to scent and vision */
            for (agent_state* agent : neighborhood[i]->data.agents) {
                /* compute neighbor position in agent coordinates */
                position relative_position = agent->current_position - current_position;

                /* if the neighbor is in the visual field, add its color to the appropriate pixel */
                if ((unsigned int) abs(relative_position.x) <= config.vision_range
                 && (unsigned int) abs(relative_position.y) <= config.vision_range) {
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
        map<patch_data, item_properties>& world,
        const diffusion<T>& scent_model,
        const simulator_config& config,
        uint64_t& current_time)
{
    agent.current_position = {0, 0};
    agent.current_direction = direction::UP;
    agent.requested_position = {0, 0};
    agent.requested_direction = direction::UP;
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

    patch<patch_data>* neighborhood[4]; position patch_positions[4];
    unsigned int index = world.get_fixed_neighborhood(agent.current_position, neighborhood, patch_positions);
    neighborhood[index]->data.patch_lock.lock();
    if (config.collision_policy != movement_conflict_policy::NO_COLLISIONS) {
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

    /* initialize the scent and vision of the current agent */
    agent.update_state(neighborhood, scent_model, config, current_time);

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

/**
 * Reads the given agent_state `agent` from the input stream `in`. 
 */
template<typename Stream>
inline bool read(agent_state& agent, Stream& in, const simulator_config& config)
{
    agent.current_scent = (float*) malloc(sizeof(float) * config.scent_dimension);
    if (agent.current_scent == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for agent_state.current_scent.\n");
        return false;
    }
    agent.current_vision = (float*) malloc(sizeof(float)
        * (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension);
    if (agent.current_vision == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for agent_state.current_vision.\n");
        free(agent.current_scent); return false;
    }
    agent.collected_items = (unsigned int*) malloc(sizeof(unsigned int) * config.item_types.length);
    if (agent.collected_items == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for agent_state.collected_items.\n");
        free(agent.current_scent); free(agent.current_vision); return false;
    }
    new (&agent.lock) std::mutex();

    if (!read(agent.current_position, in)
     || !read(agent.current_direction, in)
     || !read(agent.current_scent, in, config.scent_dimension)
     || !read(agent.current_vision, in, (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension)
     || !read(agent.agent_acted, in)
     || !read(agent.requested_position, in)
     || !read(agent.requested_direction, in)
     || !read(agent.collected_items, in, (unsigned int) config.item_types.length)) {
         free(agent.current_scent); free(agent.current_vision);
         free(agent.collected_items); return false;
     }
     return true;
}

/**
 * Writes the given agent_state `agent` to the output stream `out`. 
 */
template<typename Stream>
inline bool write(const agent_state& agent, Stream& out, const simulator_config& config)
{
    return write(agent.current_position, out)
        && write(agent.current_direction, out)
        && write(agent.current_scent, out, config.scent_dimension)
        && write(agent.current_vision, out, (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension)
        && write(agent.agent_acted, out)
        && write(agent.requested_position, out)
        && write(agent.requested_direction, out)
        && write(agent.collected_items, out, (unsigned int) config.item_types.length);
}

/**
 * This structure contains full information about a patch. This is more than we
 * need for simulation, but it is useful for visualization.
 */
struct patch_state {
    position patch_position;
    bool fixed;
    float* scent;
    float* vision;
    item* items;
    unsigned int item_count;
    position* agent_positions;
    direction* agent_directions;
    unsigned int agent_count;

    static inline void move(const patch_state& src, patch_state& dst) {
        core::move(src.patch_position, dst.patch_position);
        core::move(src.fixed, dst.fixed);
        core::move(src.scent, dst.scent);
        core::move(src.vision, dst.vision);
        core::move(src.items, dst.items);
        core::move(src.item_count, dst.item_count);
        core::move(src.agent_positions, dst.agent_positions);
        core::move(src.agent_directions, dst.agent_directions);
        core::move(src.agent_count, dst.agent_count);
    }

    static inline void free(patch_state& patch) {
        core::free(patch.scent);
        core::free(patch.vision);
        core::free(patch.items);
        core::free(patch.agent_positions);
        core::free(patch.agent_directions);
    }

    inline bool init_helper(unsigned int n,
            unsigned int scent_dimension, unsigned int color_dimension,
            unsigned int item_count, unsigned int agent_count)
    {
        scent = (float*) calloc(n * n * scent_dimension, sizeof(float));
        if (scent == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for scent.\n");
            return false;
        }
        vision = (float*) calloc(n * n * color_dimension, sizeof(float));
        if (vision == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for vision.\n");
            core::free(scent); return false;
        }
        items = (item*) malloc(sizeof(item) * item_count);
        if (items == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for items.\n");
            core::free(scent); core::free(vision); return false;
        }
        agent_positions = (position*) malloc(sizeof(position) * agent_count);
        if (agent_positions == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for agent_positions.\n");
            core::free(scent); core::free(vision);
            core::free(items); return false;
        }
        agent_directions = (direction*) malloc(sizeof(direction) * agent_count);
        if (agent_directions == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for agent_directions.\n");
            core::free(scent); core::free(vision);
            core::free(items); return false;
        }
        return true;
    }
};

/**
 * Initializes the memory of the given patch_state `patch`. This function does
 * not initialize the contents of any of the fields, except for `scent` and
 * `vision`, which are initialized to zeros.
 */
inline bool init(patch_state& patch, unsigned int n,
        unsigned int scent_dimension, unsigned int color_dimension,
        unsigned int item_count, unsigned int agent_count)
{
    patch.item_count = item_count;
    patch.agent_count = agent_count;
    return patch.init_helper(n, scent_dimension, color_dimension, item_count, agent_count);
}

/**
 * Reads the given patch_state `patch` from the input stream `in`.
 */
template<typename Stream>
bool read(patch_state& patch, Stream& in, const simulator_config& config) {
    unsigned int n = config.patch_size;
    return read(patch.patch_position, in) && read(patch.fixed, in)
        && read(patch.item_count, in) && read(patch.agent_count, in)
        && patch.init_helper(n, config.scent_dimension, config.color_dimension, patch.item_count, patch.agent_count)
        && read(patch.scent, in, n * n * config.scent_dimension)
        && read(patch.vision, in, n * n * config.color_dimension)
        && read(patch.items, in, patch.item_count)
        && read(patch.agent_positions, in, patch.agent_count)
        && read(patch.agent_directions, in, patch.agent_count);
}

/**
 * Writes the given patch_state `patch` to the output stream `out`.
 */
template<typename Stream>
bool write(const patch_state& patch, Stream& out, const simulator_config& config) {
    unsigned int n = config.patch_size;
    return write(patch.patch_position, out) && write(patch.fixed, out)
        && write(patch.item_count, out) && write(patch.agent_count, out)
        && write(patch.scent, out, n * n * config.scent_dimension)
        && write(patch.vision, out, n * n * config.color_dimension)
        && write(patch.items, out, patch.item_count)
        && write(patch.agent_positions, out, patch.agent_count)
        && write(patch.agent_directions, out, patch.agent_count);
}

/**
 * Simulator that forms the core of our experimentation framework.
 *
 * \tparam  SimulatorData   Type to store additional state in the simulation.
 */
template<typename SimulatorData>
class simulator {
    /* Configuration for this simulator. */
    simulator_config config;

    /* Map of the world managed by this simulator. */
    map<patch_data, item_properties> world;

    /* The diffusion model to simulate scent. */
    diffusion<double> scent_model;

    /* Agents managed by this simulator. */
    array<agent_state*> agents;

    /* Lock for the agents array and their state (not including their requested
       actions), used to prevent simultaneous updates. */
    std::mutex agent_states_lock;

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

    /* For storing additional state in the simulation. */
    SimulatorData data;

    typedef patch<patch_data> patch_type;

public:
    /**
     * Constructs a new simulator with the given simulator_config `conf` and
     * SimulatorData `data`, calling the copy constructor for `data`.
     */
    simulator(const simulator_config& conf,
            const SimulatorData& data,
            uint_fast32_t seed) :
        config(conf),
        world(config.patch_size,
            config.gibbs_iterations,
            config.item_types.data,
            (unsigned int) config.item_types.length, seed),
        agents(16), requested_moves(32, alloc_position_keys),
        acted_agent_count(0), data(data), time(0)
    {
        if (!init(scent_model, (double) config.diffusion_param,
                (double) config.decay_param, config.patch_size, config.deleted_item_lifetime)) {
            fprintf(stderr, "simulator ERROR: Unable to initialize scent_model.\n");
            exit(EXIT_FAILURE);
        }
    }

    /**
     * Constructs a new simulator with the given simulator_config `conf` and
     * SimulatorData `data`, calling the copy constructor for `data`.
     */
    simulator(const simulator_config& conf,
            const SimulatorData& data) :
#if !defined(NDEBUG)
        simulator(conf, data, 0) { }
#else
        simulator(conf, data, (uint_fast32_t) milliseconds()) { }
#endif

    ~simulator() { free_helper(); }

    /* Current simulation time step. */
    uint64_t time;

    /** 
     * Adds a new agent to this simulator and returns its ID and initial state.
     * 
     * \returns A pair containing the ID of the new agent and its state.
     */
    inline pair<uint64_t, agent_state*> add_agent() {
        agent_states_lock.lock();
        agents.ensure_capacity(agents.length + 1);
        agent_state* new_agent = (agent_state*) malloc(sizeof(agent_state));
        uint64_t id = agents.length;
        if (new_agent == NULL) {
            fprintf(stderr, "simulator.add_agent ERROR: Insufficient memory for new agent.\n");
            return make_pair(UINT64_MAX, (agent_state*) NULL);
        } else if (!agents.add(new_agent)) {
            core::free(new_agent);
            return make_pair(UINT64_MAX, (agent_state*) NULL);
        }
        agent_states_lock.unlock();

        if (!init(*new_agent, world, scent_model, config, time)) {
            agent_states_lock.lock();
            agents.remove((size_t) id);
            core::free(new_agent);
            agent_states_lock.unlock();
            return make_pair(UINT64_MAX, (agent_state*) NULL);
        }
        return make_pair(id, new_agent);
    }

    /** 
     * Moves an agent.
     *
     * Note that the agent is only actually moved when the simulation time step 
     * advances, and only if the agent has not already acted for the current 
     * time step.
     *
     * \param   agent_id  ID of the agent to move.
     * \param   dir       Direction along which to move, *relative* to the
     *                    agent's current direction.
     * \param   num_steps Number of steps to take in the specified direction.
     * \returns `true` if the move was successful, and `false` otherwise.
     */
    inline bool move(uint64_t agent_id, direction dir, unsigned int num_steps)
    {
        if (num_steps > config.max_steps_per_movement
         || !config.allowed_movement_directions[(size_t) dir])
            return false;

        agent_states_lock.lock();
        agent_state& agent = *agents[(size_t) agent_id];
        agent_states_lock.unlock();

        agent.lock.lock();
        if (agent.agent_acted) {
            agent.lock.unlock(); return false;
        }
        agent.agent_acted = true;

        agent.requested_position = agent.current_position;
        agent.requested_direction = agent.current_direction;
        position diff(0, 0);
        switch (dir) {
        case direction::UP   : diff.x = 0; diff.y = num_steps; break;
        case direction::DOWN : diff.x = 0; diff.y = -((int64_t) num_steps); break;
        case direction::LEFT : diff.x = -((int64_t) num_steps); diff.y = 0; break;
        case direction::RIGHT: diff.x = num_steps; diff.y = 0; break;
        case direction::COUNT: break;
        }

        switch (agent.current_direction) {
        case direction::UP: break;
        case direction::DOWN: diff.x *= -1; diff.y *= -1; break;
        case direction::LEFT:
            core::swap(diff.x, diff.y);
            diff.x *= -1; break;
        case direction::RIGHT:
            core::swap(diff.x, diff.y);
            diff.y *= -1; break;
        case direction::COUNT: break;
        }

        agent.requested_position += diff;
        agent.lock.unlock();

        /* add the agent's move to the list of requested moves */
        request_position(agent);

        agent_states_lock.lock();
        if (++acted_agent_count == agents.length)
            step(); /* advance the simulation by one time step */
        agent_states_lock.unlock();
        return true;
    }

    /** 
     * Turns an agent.
     *
     * Note that the agent is only actually turned when the simulation time
     * step advances, and only if the agent has not already acted for the
     * current time step.
     *
     * \param   agent_id ID of the agent to move.
     * \param   dir      Direction to turn, *relative* to the agent's current
     *                   direction.
     * \returns `true` if the turn was successful, and `false` otherwise.
     */
    inline bool turn(uint64_t agent_id, direction dir)
    {
        if (!config.allowed_rotations[(size_t) dir])
            return false;

        agent_states_lock.lock();
        agent_state& agent = *agents[(size_t) agent_id];
        agent_states_lock.unlock();

        agent.lock.lock();
        if (agent.agent_acted) {
            agent.lock.unlock(); return false;
        }
        agent.agent_acted = true;

        agent.requested_position = agent.current_position;
        agent.requested_direction = agent.current_direction;

        switch (dir) {
        case direction::UP: break;
        case direction::DOWN:
            if (agent.current_direction == direction::UP) agent.requested_direction = direction::DOWN;
            else if (agent.current_direction == direction::DOWN) agent.requested_direction = direction::UP;
            else if (agent.current_direction == direction::LEFT) agent.requested_direction = direction::RIGHT;
            else if (agent.current_direction == direction::RIGHT) agent.requested_direction = direction::LEFT;
            break;
        case direction::LEFT:
            if (agent.current_direction == direction::UP) agent.requested_direction = direction::LEFT;
            else if (agent.current_direction == direction::DOWN) agent.requested_direction = direction::RIGHT;
            else if (agent.current_direction == direction::LEFT) agent.requested_direction = direction::DOWN;
            else if (agent.current_direction == direction::RIGHT) agent.requested_direction = direction::UP;
            break;
        case direction::RIGHT:
            if (agent.current_direction == direction::UP) agent.requested_direction = direction::RIGHT;
            else if (agent.current_direction == direction::DOWN) agent.requested_direction = direction::LEFT;
            else if (agent.current_direction == direction::LEFT) agent.requested_direction = direction::UP;
            else if (agent.current_direction == direction::RIGHT) agent.requested_direction = direction::DOWN;
            break;
        case direction::COUNT: break;
        }
        agent.lock.unlock();

        /* add the agent's move to the list of requested moves */
        request_position(agent);

        agent_states_lock.lock();
        if (++acted_agent_count == agents.length)
            step(); /* advance the simulation by one time step */
        agent_states_lock.unlock();
        return true;
    }

    /**
     * Retrieves an array of pointers to agent_state structures, storing them
     * in `states`, which is parallel to the specified `agent_ids` array, and
     * has length `agent_count`.
     *
     * \param      states The output array of agent_state pointers.
     * \param   agent_ids The array of agent IDs whose states to retrieve.
     * \param agent_count The length of `states` and `agent_ids`.
     */
    inline void get_agent_states(agent_state** states,
            uint64_t* agent_ids, unsigned int agent_count)
    {
        agent_states_lock.lock();
        for (unsigned int i = 0; i < agent_count; i++)
            states[i] = agents[(size_t) agent_ids[i]];
        agent_states_lock.unlock();
    }

    /**
     * Returns a SimulatorData reference associated with this simulator.
     */
    inline SimulatorData& get_data() {
        return data;
    }

    /**
     * Returns a SimulatorData reference associated with this simulator.
     */
    inline const SimulatorData& get_data() const {
        return data;
    }

    /**
     * Returns the simulator configuration used to construct this simulator.
     */
    inline const simulator_config& get_config() const {
        return config;
    }

    inline map<patch_data, item_properties>& get_world() {
        return world;
    }

    /**
     * Retrieves the set of patches of the map within the bounding box defined
     * by `bottom_left_corner` and `top_right_corner`. The patches are stored
     * in the map `patches`.
     *
     * \param bottom_left_corner The bottom-left corner of the bounding box in
     *      which to retrieve the map patches.
     * \param top_right_corner The top-right corner of the bounding box in
     *      which to retrieve the map patches.
     * \param patches The output map from patch position to patch_state
     *      structures which will contain the state of the retrieved patches.
     * \returns `true` if successful; `false` otherwise.
     */
    bool get_map(
            position bottom_left_corner,
            position top_right_corner,
            hash_map<position, patch_state>& patches)
    {
        auto process_patch = [&](const patch_type& patch, position patch_position)
        {
            bool contains; unsigned int bucket;
            if (!patches.check_size(alloc_position_keys)) return false;
            patch_state& state = patches.get(patch_position, contains, bucket);
            if (!init(state, config.patch_size,
                    config.scent_dimension, config.color_dimension,
                    (unsigned int) patch.items.length,
                    (unsigned int) patch.data.agents.length))
                return false;
            patches.table.keys[bucket] = patch_position;
            patches.table.size++;

            state.patch_position = patch_position;
            state.item_count = 0;
            state.fixed = patch.fixed;
            for (unsigned int i = 0; i < patch.items.length; i++) {
                if (patch.items[i].deletion_time == 0) {
                    state.items[state.item_count] = patch.items[i];
                    state.item_count++;
                }
            }

            for (unsigned int i = 0; i < patch.data.agents.length; i++) {
                state.agent_positions[i] = patch.data.agents[i]->current_position;
                state.agent_directions[i] = patch.data.agents[i]->current_direction;
            }
            return true;
        };

        std::unique_lock<std::mutex> lock(agent_states_lock);
        position bottom_left_patch_position, top_right_patch_position;
        if (!world.get_state(bottom_left_corner, top_right_corner,
                process_patch, bottom_left_patch_position, top_right_patch_position))
            return false;

        for (int64_t x = bottom_left_patch_position.x - 1; x < top_right_patch_position.x + 1; x++) {
            for (int64_t y = bottom_left_patch_position.y - 1; y < top_right_patch_position.y + 1; y++) {
                const patch_type* patch = world.get_patch_if_exists({x, y});
                if (patch == NULL) continue;

                /* consider all patches in the neighborhood of 'patch' */
                for (int diff_x = -1; diff_x <= 1; diff_x++) {
                    for (int diff_y = -1; diff_y <= 1; diff_y++) {
                        bool contains;
                        patch_state& neighbor = patches.get({x + diff_x, y + diff_y}, contains);
                        position world_position = neighbor.patch_position * config.patch_size;
                        if (!contains) continue;

                        /* for every item in 'patch', add its scent/vision contribution to 'neighbor' */
                        for (unsigned int i = 0; i < patch->items.length; i++) {
                            const item& item = patch->items[i];

                            /* check if the item is too old; if so, ignore it */
                            if (item.deletion_time > 0 && time >= item.deletion_time + config.deleted_item_lifetime)
                                continue;

                            for (unsigned int a = 0; a < config.patch_size; a++)
                                for (unsigned int b = 0; b < config.patch_size; b++)
                                    compute_scent_contribution(scent_model, item, world_position + position(a, b), time,
                                            config, neighbor.scent + ((a*config.patch_size + b)*config.scent_dimension));
                        }
                    }
                }

                /* add color contribution from the items and agents in the current patch */
                bool contains;
                patch_state& state = patches.get({x, y}, contains);
                if (!contains) continue;
                for (const item& item : patch->items) {
                    if (item.deletion_time != 0) continue;
                    position relative_position = item.location - position(x, y) * config.patch_size;
                    float* pixel = state.vision + ((relative_position.x*config.patch_size + relative_position.y)*config.color_dimension);
                    for (unsigned int i = 0; i < config.color_dimension; i++)
                        pixel[i] += config.item_types[item.item_type].color[i];
                }

                for (const agent_state* agent : patch->data.agents) {
                    position relative_position = agent->current_position - position(x, y) * config.patch_size;
                    float* pixel = state.vision + ((relative_position.x*config.patch_size + relative_position.y)*config.color_dimension);
                    for (unsigned int i = 0; i < config.color_dimension; i++)
                        pixel[i] += config.agent_color[i];
                }
            }
        }

        return true;
    }

    static inline void free(simulator& s) {
        s.free_helper();
        core::free(s.agents);
        core::free(s.requested_moves);
        core::free(s.config);
        core::free(s.scent_model);
        core::free(s.world);
        core::free(s.data);
        s.agent_states_lock.~mutex();
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
                if (conflicts[0]->current_position == entry.key) continue; /* give preference to agents that don't move */
                unsigned int result = sample_uniform((unsigned int) conflicts.length);
                core::swap(conflicts[0], conflicts[result]);
            }
        }

        /* check for items that block movement */
        array<position> occupied_positions(16);
        for (auto entry : requested_moves) {
            patch_type* neighborhood[4]; position patch_positions[4];
            unsigned int index = world.get_fixed_neighborhood(entry.key, neighborhood, patch_positions);
            patch_type& current_patch = *neighborhood[index];
            for (item& item : current_patch.items) {
                if (item.location == entry.key && item.deletion_time == 0 && config.item_types[item.item_type].blocks_movement) {
                    /* there is an item at our new position that blocks movement */
                    array<agent_state*>& conflicts = entry.value;
                    occupied_positions.add(conflicts[0]->current_position);
                    conflicts[0] = NULL; /* prevent any agent from moving here */
                }
            }
        }

        /* need to ensure agents don't move into positions where other agents failed to move */
        if (config.collision_policy != movement_conflict_policy::NO_COLLISIONS) {
            array<position> occupied_positions(16);
            for (auto entry : requested_moves) {
                array<agent_state*>& conflicts = entry.value;
                for (unsigned int i = 1; i < conflicts.length; i++)
                    occupied_positions.add(conflicts[i]->current_position);
            }
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

        time++;
        acted_agent_count = 0;
        for (agent_state* agent : agents) {
            if (!agent->agent_acted) continue;

            agent->current_direction = agent->requested_direction;

            /* check if this agent moved, in accordance with the collision policy */
            position old_patch_position;
            world.world_to_patch_coordinates(agent->current_position, old_patch_position);
            if (config.collision_policy == movement_conflict_policy::NO_COLLISIONS
             || (agent == requested_moves.get(agent->requested_position)[0]))
            {
                agent->current_position = agent->requested_position;

                /* delete any items that are automatically picked up at this cell */
                patch_type* neighborhood[4]; position patch_positions[4];
                unsigned int index = world.get_fixed_neighborhood(agent->current_position, neighborhood, patch_positions);
                patch_type& current_patch = *neighborhood[index];
                for (item& item : current_patch.items) {
                    if (item.location == agent->current_position && item.deletion_time == 0) {
                        /* there is an item at our new position */
                        bool collect = true;
                        for (unsigned int i = 0; i < config.item_types.length; i++) {
                            if (agent->collected_items[i] < config.item_types[item.item_type].required_item_counts[i]) {
                                collect = false; break;
                            }
                        }

                        if (collect) {
                            /* collect this item */
                            item.deletion_time = time;
                            agent->collected_items[item.item_type]++;

                            for (unsigned int i = 0; i < config.item_types.length; i++) {
                                if (agent->collected_items[i] < config.item_types[item.item_type].required_item_costs[i])
                                    agent->collected_items[i] = 0;
                                else agent->collected_items[i] -= config.item_types[item.item_type].required_item_costs[i];
                            }
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
        if (config.collision_policy != movement_conflict_policy::NO_COLLISIONS) {
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
        update_agent_scent_and_vision();

        /* Invoke the step callback function for each agent. */
        on_step((const simulator<SimulatorData>*) this, (const array<agent_state*>&) agents, time);
    }

    inline void update_agent_scent_and_vision() {
        for (agent_state* agent : agents) {
            patch_type* neighborhood[4]; position patch_positions[4];
            world.get_fixed_neighborhood(agent->current_position, neighborhood, patch_positions);
            agent->update_state(neighborhood, scent_model, config, time);
        }
    }

    inline void request_position(agent_state& agent)
    {
        /* check for collisions with other agents */
        if (config.collision_policy == movement_conflict_policy::NO_COLLISIONS)
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
        if (agent.current_position == agent.requested_position)
            core::swap(agents[0], agents.last());
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

    template<typename A> friend bool init(simulator<A>&, const simulator_config&, const A&, uint_fast32_t);
    template<typename A, typename B> friend bool read(simulator<A>&, B&, const A&);
    template<typename A, typename B> friend bool write(const simulator<A>&, B&);
};

/**
 * Constructs a new simulator with the given simulator_config `config` and
 * SimulatorData `data`, calling the
 * `bool init(SimulatorData&, const SimulatorData&)` function to initialize
 * `data`.
 *
 * \returns `true` if successful; `false` otherwise.
 */
template<typename SimulatorData>
bool init(simulator<SimulatorData>& sim, 
        const simulator_config& config,
        const SimulatorData& data,
        uint_fast32_t seed)
{
    sim.time = 0;
    sim.acted_agent_count = 0;
    if (!init(sim.data, data)) {
        return false;
    } else if (!array_init(sim.agents, 16)) {
        free(sim.data); return false;
    } else if (!hash_map_init(sim.requested_moves, 32, alloc_position_keys)) {
        free(sim.data); free(sim.agents); return false;
    } else if (!init(sim.config, config)) {
        free(sim.data); free(sim.agents);
        free(sim.requested_moves); return false;
    } else if (!init(sim.scent_model, (double) sim.config.diffusion_param,
            (double) sim.config.decay_param, sim.config.patch_size, sim.config.deleted_item_lifetime)) {
        free(sim.data); free(sim.config); free(sim.agents);
        free(sim.requested_moves); return false;
    } else if (!init(sim.world, sim.config.patch_size,
            sim.config.gibbs_iterations,
            sim.config.item_types.data,
            (unsigned int) sim.config.item_types.length, seed)) {
        free(sim.config); free(sim.data);
        free(sim.agents); free(sim.requested_moves);
        free(sim.scent_model); return false;
    }
    new (&sim.agent_states_lock) std::mutex();
    new (&sim.requested_move_lock) std::mutex();
    return true;
}

/**
 * Constructs a new simulator with the given simulator_config `config` and
 * SimulatorData `data`, calling the
 * `bool init(SimulatorData&, const SimulatorData&)` function to initialize
 * `data`.
 *
 * \returns `true` if successful; `false` otherwise.
 */
template<typename SimulatorData>
inline bool init(simulator<SimulatorData>& sim, 
        const simulator_config& config,
        const SimulatorData& data)
{
#if !defined(NDEBUG)
    uint_fast32_t seed = 0;
#else
    uint_fast32_t seed = (uint_fast32_t) milliseconds();
#endif
    return init(sim, config, data, seed);
}

template<typename Stream>
inline bool read(agent_state*& agent,
        Stream& in, array<agent_state*>& agents)
{
    unsigned int index;
    if (!read(index, in)) return false;
    agent = agents[index];
    return true;
}

template<typename Stream>
inline bool write(const agent_state* agent, Stream& out,
        hash_map<const agent_state*, unsigned int>& agent_indices)
{
    return write(agent_indices.get(agent), out);
}

/**
 * Reads the given simulator `sim` from the input stream `in`. The
 * SimulatorData of `sim` is not read from `in`. Rather, it is initialized by
 * the given `data` argument.
 *
 * \returns `true` if successful; `false` otherwise.
 */
template<typename SimulatorData, typename Stream>
bool read(simulator<SimulatorData>& sim, Stream& in, const SimulatorData& data)
{
    if (!init(sim.data, data)) {
        return false;
    } if (!read(sim.config, in)) {
        free(sim.data); return false;
    }

    size_t agent_count = 0;
    if (!read(agent_count, in)
     || !array_init(sim.agents, ((size_t) 1) << (core::log2(agent_count) + 2))) {
        free(sim.data); free(sim.config); return false;
    }
    for (unsigned int i = 0; i < agent_count; i++) {
        sim.agents[i] = (agent_state*) malloc(sizeof(agent_state));
        if (sim.agents[i] == NULL || !read(*sim.agents[i], in, sim.config)) {
            fprintf(stderr, "read ERROR: Insufficient memory for agent_state in simulator.\n");
            for (unsigned int j = 0; j < i; j++) {
                free(*sim.agents[j]); free(sim.agents[j]);
            }
            if (sim.agents[i] != NULL) free(sim.agents[i]);
            free(sim.data); free(sim.agents);
            free(sim.config); return false;
        }
    }
    sim.agents.length = agent_count;

    if (!read(sim.world, in, sim.config.item_types.data, (unsigned int) sim.config.item_types.length, sim.agents)) {
        for (unsigned int j = 0; j < agent_count; j++) {
            free(*sim.agents[j]); free(sim.agents[j]);
        }
        free(sim.data); free(sim.agents);
        free(sim.config); return false;
    }

    default_scribe scribe;
    if (!read(sim.requested_moves, in, alloc_position_keys, scribe, sim.agents)) {
        for (unsigned int j = 0; j < agent_count; j++) {
            free(*sim.agents[j]); free(sim.agents[j]);
        }
        free(sim.data); free(sim.agents);
        free(sim.config); free(sim.world); return false;
    }

    /* reinitialize the scent model */
    if (!read(sim.time, in) || !read(sim.acted_agent_count, in)
     || !init(sim.scent_model, (double) sim.config.diffusion_param,
            (double) sim.config.decay_param, sim.config.patch_size,
            sim.config.deleted_item_lifetime))
    {
        for (unsigned int j = 0; j < agent_count; j++) {
            free(*sim.agents[j]); free(sim.agents[j]);
        }
        for (auto entry : sim.requested_moves)
            free(entry.value);
        free(sim.data); free(sim.world); free(sim.agents);
        free(sim.requested_moves); free(sim.config);
        return false;
    }
    new (&sim.agent_states_lock) std::mutex();
    new (&sim.requested_move_lock) std::mutex();
    return true;
}

/**
 * Writes the given simulator `sim` to the output stream `out`.
 *
 * **NOTE:** this function assumes the variables in the simulator are not
 *      modified during writing.
 *
 * \returns `true` if successful; `false` otherwise.
 */
template<typename SimulatorData, typename Stream>
bool write(const simulator<SimulatorData>& sim, Stream& out)
{
    if (!write(sim.config, out))
        return false;

    hash_map<const agent_state*, unsigned int> agent_indices((unsigned int) sim.agents.length * RESIZE_THRESHOLD_INVERSE);
    if (!write(sim.agents.length, out)) return false;
    for (unsigned int i = 0; i < sim.agents.length; i++) {
        agent_indices.put(sim.agents[i], i);
        if (!write(*sim.agents[i], out, sim.config)) return false;
    }

    default_scribe scribe;
    return write(sim.world, out, agent_indices)
        && write(sim.requested_moves, out, scribe, agent_indices)
        && write(sim.time, out)
        && write(sim.acted_agent_count, out);
}

} /* namespace nel */

#endif /* NEL_SIMULATOR_H_ */

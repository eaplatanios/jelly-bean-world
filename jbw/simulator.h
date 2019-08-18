/**
 * Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef JBW_SIMULATOR_H_
#define JBW_SIMULATOR_H_

#include <core/array.h>
#include <core/utility.h>
#include <atomic>
#include <math.h>
#include <mutex>
#include "map.h"
#include "diffusion.h"
#include "status.h"

namespace jbw {

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

template<typename FunctionType>
struct energy_function {
    FunctionType fn;
    float* args;
    unsigned int arg_count;

    static inline void free(energy_function<FunctionType>& info) {
        core::free(info.args);
    }
};

template<typename FunctionType>
inline bool init(
        energy_function<FunctionType>& info,
        const energy_function<FunctionType>& src)
{
    info.args = (float*) malloc(max((size_t) 1, sizeof(float) * src.arg_count));
    if (info.args == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for energy_function.args.\n");
        return false;
    }
    memcpy(info.args, src.args, sizeof(float) * src.arg_count);
    info.fn = src.fn;
    info.arg_count = src.arg_count;
    return true;
}

template<typename FunctionType, typename Stream>
bool read(energy_function<FunctionType>& info, Stream& in) {
    if (!read(info.fn, in) || !read(info.arg_count, in))
        return false;
    info.args = (float*) malloc(max((size_t) 1, sizeof(float) * info.arg_count));
    if (info.args == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for energy_function.args.\n");
        return false;
    }
    if (!read(info.args, in, info.arg_count)) {
        free(info.args); return false;
    }
    return true;
}

template<typename FunctionType, typename Stream>
bool write(const energy_function<FunctionType>& info, Stream& out) {
    return write(info.fn, out)
        && write(info.arg_count, out)
        && write(info.args, out, info.arg_count);
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
    float visual_occlusion;

    energy_function<intensity_function> intensity_fn;
    energy_function<interaction_function>* interaction_fns;

    static inline void free(item_properties& properties, unsigned int item_type_count) {
        core::free(properties.name);
        core::free(properties.scent);
        core::free(properties.color);
        core::free(properties.required_item_counts);
        core::free(properties.required_item_costs);
        core::free(properties.intensity_fn);
        for (unsigned int i = 0; i < item_type_count; i++)
            core::free(properties.interaction_fns[i]);
        core::free(properties.interaction_fns);
    }
};

inline bool init_interaction_fns(
        energy_function<interaction_function>* fns,
        const energy_function<interaction_function>* src,
        unsigned int item_type_count)
{
    for (unsigned int i = 0; i < item_type_count; i++) {
        if (!init(fns[i], src[i])) {
            for (unsigned int j = 0; j < i; j++) free(fns[j]);
            return false;
        }
    }
    return true;
}

/* NOTE: this function assumes `fns` is zero-initialized */
inline bool init_interaction_fns(
        energy_function<interaction_function>* fns,
        const array_map<unsigned int, energy_function<interaction_function>>& src,
        unsigned int item_type_count)
{
    for (const auto& entry : src) {
        if (!init(fns[entry.key], entry.value)) {
            for (unsigned int i = 0; i < item_type_count; i++)
                if (fns[i].args != NULL) free(fns[i]);
            return false;
        }
    }

    /* replace the empty functions with the zero interaction function */
    for (unsigned int i = 0; i < item_type_count; i++) {
        if (fns[i].args == NULL) {
            fns[i].fn = zero_interaction_fn;
            fns[i].args = (float*) malloc(1);
            if (fns[i].args == NULL) {
                fprintf(stderr, "init_interaction_fns ERROR: Out of memory.\n");
                for (unsigned int i = 0; i < item_type_count; i++)
                    if (fns[i].args != NULL) free(fns[i]);
                return false;
            }
            fns[i].arg_count = 0;
        }
    }
    return true;
}

/**
 * Initializes the given item_properties `properties` with the properties given as arguments.
 */
template<typename InteractionFunctionInfo>
inline bool init(
        item_properties& properties, const char* name, unsigned int name_length,
        const float* scent, const float* color, const unsigned int* required_item_counts,
        const unsigned int* required_item_costs, bool blocks_movement, float visual_occlusion,
        const energy_function<intensity_function>& intensity_fn, const InteractionFunctionInfo& interaction_fns,
        unsigned int scent_dimension, unsigned int color_dimension, unsigned int item_type_count)
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
        core::free(properties.name); core::free(properties.scent); return false;
    }
    properties.required_item_counts = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.required_item_counts == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.required_item_counts.\n");
        core::free(properties.name); core::free(properties.scent);
        core::free(properties.color); return false;
    }
    properties.required_item_costs = (unsigned int*) malloc(sizeof(unsigned int) * item_type_count);
    if (properties.required_item_costs == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.required_item_costs.\n");
        core::free(properties.name); core::free(properties.scent);
        core::free(properties.color); core::free(properties.required_item_counts);
        return false;
    }
    if (!init(properties.intensity_fn, intensity_fn)) {
        core::free(properties.name); core::free(properties.scent);
        core::free(properties.color); core::free(properties.required_item_counts);
        core::free(properties.required_item_costs); return false;
    }
    properties.interaction_fns = (energy_function<interaction_function>*)
            calloc(item_type_count, sizeof(energy_function<interaction_function>));
    if (properties.interaction_fns == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for item_properties.interaction_fns.\n");
        core::free(properties.name); core::free(properties.scent);
        core::free(properties.color); core::free(properties.required_item_counts);
        core::free(properties.required_item_costs);
        core::free(properties.intensity_fn); return false;
    }
    if (!init_interaction_fns(properties.interaction_fns, interaction_fns, item_type_count)) {
        core::free(properties.name); core::free(properties.scent); core::free(properties.color);
        core::free(properties.required_item_counts); core::free(properties.required_item_costs);
        core::free(properties.intensity_fn); core::free(properties.interaction_fns); return false;
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
    properties.visual_occlusion = visual_occlusion;
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
        src.required_item_costs, src.blocks_movement, src.visual_occlusion,
        src.intensity_fn, src.interaction_fns,
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
    properties.interaction_fns = (energy_function<interaction_function>*)
            malloc(sizeof(energy_function<interaction_function>) * item_type_count);
    if (properties.interaction_fns == NULL) {
        fprintf(stderr, "read ERROR: Insufficient memory for item_properties.interaction_fns.\n");
        free(properties.scent); free(properties.color);
        free(properties.required_item_counts);
        free(properties.required_item_costs); return false;
    }

    if (!read(properties.scent, in, scent_dimension)
     || !read(properties.color, in, color_dimension)
     || !read(properties.required_item_counts, in, item_type_count)
     || !read(properties.required_item_costs, in, item_type_count)
     || !read(properties.blocks_movement, in)
     || !read(properties.visual_occlusion, in)
     || !read(properties.intensity_fn, in))
    {
        free(properties.name); free(properties.scent); free(properties.color);
        free(properties.required_item_counts); free(properties.required_item_costs);
        free(properties.interaction_fns); return false;
    }

    for (unsigned int i = 0; i < item_type_count; i++) {
        if (!read(properties.interaction_fns[i], in)) {
            for (unsigned int j = 0; j < i; j++) free(properties.interaction_fns[j]);
            free(properties.name); free(properties.scent); free(properties.color);
            free(properties.required_item_counts); free(properties.required_item_costs);
            free(properties.interaction_fns); return false;
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
     || !write(properties.visual_occlusion, out)
     || !write(properties.intensity_fn, out))
        return false;

    for (unsigned int i = 0; i < item_type_count; i++)
        if (!write(properties.interaction_fns[i], out)) return false;
    return true;
}

typedef uint8_t action_policy_type;
enum class action_policy : action_policy_type {
    ALLOWED,
    DISALLOWED,
    IGNORED
};

/**
 * Reads a action_policy from `in` and stores the result in `type`.
 */
template<typename Stream>
inline bool read(action_policy& type, Stream& in) {
    action_policy_type v;
    if (!read(v, in)) return false;
    type = (action_policy) v;
    return true;
}

/**
 * Writes the given action_policy `type` to the stream `out`.
 */
template<typename Stream>
inline bool write(const action_policy& type, Stream& out) {
    return write((action_policy_type) type, out);
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
    float agent_field_of_view;
    action_policy allowed_movement_directions[(size_t) direction::COUNT];
    action_policy allowed_rotations[(size_t) direction::COUNT];
    bool no_op_allowed;

    /* world properties */
    unsigned int patch_size;
    unsigned int mcmc_iterations;
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
        core::swap(first.no_op_allowed, second.no_op_allowed);
        core::swap(first.max_steps_per_movement, second.max_steps_per_movement);
        core::swap(first.scent_dimension, second.scent_dimension);
        core::swap(first.color_dimension, second.color_dimension);
        core::swap(first.vision_range, second.vision_range);
        core::swap(first.patch_size, second.patch_size);
        core::swap(first.mcmc_iterations, second.mcmc_iterations);
        core::swap(first.item_types, second.item_types);
        core::swap(first.agent_color, second.agent_color);
        core::swap(first.agent_field_of_view, second.agent_field_of_view);
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
        no_op_allowed = src.no_op_allowed;

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
        mcmc_iterations = src.mcmc_iterations;
        agent_field_of_view = src.agent_field_of_view;
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
     || !read(config.no_op_allowed, in)
     || !read(config.patch_size, in)
     || !read(config.mcmc_iterations, in)
     || !read(config.item_types.length, in)
     || !read(config.agent_field_of_view, in))
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
        && write(config.no_op_allowed, out)
        && write(config.patch_size, out)
        && write(config.mcmc_iterations, out)
        && write(config.item_types.length, out)
        && write(config.item_types.data, out, config.item_types.length, config.scent_dimension, config.color_dimension, (unsigned int) config.item_types.length)
        && write(config.agent_color, out, config.color_dimension)
        && write(config.agent_field_of_view, out)
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
bool read(patch_data& data, Stream& in,
        const hash_map<uint64_t, agent_state*>& agents)
{
    size_t agent_count = 0;
    if (!read(agent_count, in)
     || !array_init(data.agents, max((size_t) 2, agent_count)))
        return false;
    for (unsigned int i = 0; i < agent_count; i++) {
        uint64_t id;
        if (!read(id, in)) {
            free(data.agents); return false;
        }
        data.agents[i] = agents.get(id);
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
        const hash_map<const agent_state*, uint64_t>& agents)
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
     * If this is `true`, the simulator will wait for this agent
     * to act before advancing the simulation.
     */
    bool agent_active;

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

    inline void occlude_color(
            position relative_position, unsigned int vision_range,
            unsigned int color_dimension, const float occlusion)
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
            current_vision[offset + i] = current_vision[offset + i] * (1.0f - occlusion) + occlusion;
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

        array<item> visual_field_items(16);

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
                    visual_field_items.add(item);
                    add_color(
                        relative_position, config.vision_range,
                        config.item_types[item.item_type].color,
                        config.color_dimension);
                 }
            }

            /* iterate over neighboring agents, and add their contributions to scent and vision */
            for (agent_state* agent : neighborhood[i]->data.agents) {
                /* compute neighbor position in agent coordinates */
                position relative_position = agent->current_position - current_position;

                /* if the neighbor is in the visual field, add its color to the appropriate pixel */
                if ((unsigned int) abs(relative_position.x) <= config.vision_range
                 && (unsigned int) abs(relative_position.y) <= config.vision_range) {
                    add_color(
                        relative_position, config.vision_range,
                        config.agent_color, config.color_dimension);
                }
            }
        }

        /* Compute the agent's field of view. */
        float fov_left_angle = 0.0f;
        float fov_right_angle = 0.0f;
        switch (current_direction) {
        case direction::UP:
            fov_left_angle = (M_PI + config.agent_field_of_view) / 2;
            fov_right_angle = (M_PI - config.agent_field_of_view) / 2;
            break;
        case direction::DOWN:
            fov_left_angle = -(M_PI - config.agent_field_of_view) / 2;
            fov_right_angle = -(M_PI + config.agent_field_of_view) / 2;
            break;
        case direction::LEFT:
            fov_left_angle = -M_PI + config.agent_field_of_view / 2;
            fov_right_angle = M_PI - config.agent_field_of_view / 2;
            break;
        case direction::RIGHT:
            fov_left_angle = config.agent_field_of_view / 2;
            fov_right_angle = -config.agent_field_of_view / 2;
            break;
        case direction::COUNT: return;
        }

        constexpr float circle_radius = 0.5f;
        auto circle_tangent_angles = [circle_radius](float x, float y, float& left_angle, float& right_angle) {
            const float dd = sqrt(x * x + y * y);
            const float a = asin(circle_radius / dd);
            const float b = atan2(y, x);
            left_angle = b + a;
            right_angle = b - a;
        };

        /* Apply visual occlusion. */
        int64_t V = (int64_t) config.vision_range;
        for (int64_t i = -V; i <= V; i++) {
            const float cell_x = (float) i;
            for (int64_t j = -V; j <= V; j++) {
                const float cell_y = (float) j;
                const position relative_position = { i, j };
                const float distance = (float) relative_position.squared_length();
                float cell_left_angle, cell_right_angle;
                circle_tangent_angles(cell_x, cell_y, cell_left_angle, cell_right_angle);
                const float cell_angle = abs(cell_left_angle - cell_right_angle);

                /* Check if this cell is outside the agent's field of view. */
                if (config.agent_field_of_view < 2 * M_PI) {
                    float overlap = angle_overlap(
                        fov_left_angle, fov_right_angle,
                        cell_left_angle, cell_right_angle);
                    const float occlusion = 1.0f - min(1.0f, overlap / cell_angle);
                    occlude_color(
                        relative_position, config.vision_range,
                        config.color_dimension, occlusion);
                    if (occlusion == 1.0f) continue;
                }

                /* Check if this cell is occluded by any items. */
                for (item& item : visual_field_items) {
                    const position relative_location = item.location - current_position;
                    float item_distance = (float) relative_location.squared_length();
                    if (item_distance + 1.0f > distance) continue;

                    const float x = (float) relative_location.x;
                    const float y = (float) relative_location.y;
                    float left_angle, right_angle;
                    circle_tangent_angles(x, y, left_angle, right_angle);

                    float overlap = angle_overlap(
                        left_angle, right_angle,
                        cell_left_angle, cell_right_angle);
                    if (overlap > 0.0f) {
                        const float scaling_factor = min(1.0f, overlap / cell_angle);
                        float occlusion = config.item_types[item.item_type].visual_occlusion * scaling_factor;
                        if (occlusion > 0.0f) {
                            occlude_color(
                                relative_position, config.vision_range,
                                config.color_dimension, occlusion);
                        }
                    }
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

    /** Removes this agent from the world and frees all allocated memory. */
    template<typename T>
    inline static void free(agent_state& agent,
            map<patch_data, item_properties>& world,
            const diffusion<T>& scent_model,
            const simulator_config& config,
            uint64_t& current_time)
    {
        patch<patch_data>* neighborhood[4]; position patch_positions[4];
        unsigned int index = world.get_fixed_neighborhood(agent.current_position, neighborhood, patch_positions);
        neighborhood[index]->data.patch_lock.lock();
        unsigned j = neighborhood[index]->data.agents.index_of(&agent);
        neighborhood[index]->data.agents.remove(j);
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

        free(agent);
    }

private:
    static inline float angle_overlap(float al, float ar, float bl, float br) {
        al = al < 0 ? 2 * M_PI + al : al;
        ar = ar < 0 ? 2 * M_PI + ar : ar;
        bl = bl < 0 ? 2 * M_PI + bl : bl;
        br = br < 0 ? 2 * M_PI + br : br;
        if (al < ar) {
            return angle_overlap(al, 0.0f, bl, br) + angle_overlap(2 * M_PI, ar, bl, br);
        } else if (bl < br) {
            return angle_overlap(al, ar, bl, 0.0f) + angle_overlap(al, ar, 2 * M_PI, br);       
        } else {
            if (al > bl) {
                if (ar > bl) return 0.0f;
                else if (ar > br) return bl - ar;
                else return bl - br;
            } else {
                if (br > al) return 0.0f;
                else if (br > ar) return al - br;
                else return al - ar;
            }
        }
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
inline status init(
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
        return status::OUT_OF_MEMORY;
    }
    agent.current_vision = (float*) malloc(sizeof(float)
        * (2*config.vision_range + 1) * (2*config.vision_range + 1) * config.color_dimension);
    if (agent.current_vision == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.current_vision.\n");
        free(agent.current_scent); return status::OUT_OF_MEMORY;
    }
    agent.collected_items = (unsigned int*) calloc(config.item_types.length, sizeof(unsigned int));
    if (agent.collected_items == NULL) {
        fprintf(stderr, "init ERROR: Insufficient memory for agent_state.collected_items.\n");
        free(agent.current_scent); free(agent.current_vision); return status::OUT_OF_MEMORY;
    }

    agent.agent_acted = false;
    agent.agent_active = true;
    new (&agent.lock) std::mutex();

    patch<patch_data>* neighborhood[4]; position patch_positions[4];
    world.mcmc_iterations *= 10; /* TODO: should this be configurable? */
    unsigned int index = world.get_fixed_neighborhood(
        agent.current_position, neighborhood, patch_positions);
    world.mcmc_iterations /= 10;
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
                return status::AGENT_ALREADY_EXISTS;
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
            world.get_fixed_neighborhood(
                neighbor->current_position, other_neighborhood, patch_positions);
            neighbor->update_state(other_neighborhood, scent_model, config, current_time);
        }
    }
    return status::OK;
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
     || !read(agent.agent_active, in)
     || !read(agent.requested_position, in)
     || !read(agent.requested_direction, in)
     || !read(agent.collected_items, in, (unsigned int) config.item_types.length))
    {
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
        && write(agent.agent_active, out)
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
        if (patch.scent != nullptr)
            core::free(patch.scent);
        core::free(patch.vision);
        core::free(patch.items);
        core::free(patch.agent_positions);
        core::free(patch.agent_directions);
    }

    template<bool InitializeScent>
    inline bool init_helper(unsigned int n,
            unsigned int scent_dimension, unsigned int color_dimension,
            unsigned int item_count, unsigned int agent_count)
    {
        if (InitializeScent) {
            scent = (float*) calloc(n * n * scent_dimension, sizeof(float));
            if (scent == NULL) {
                fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for scent.\n");
                return false;
            }
        } else {
            scent = nullptr;
        }
        vision = (float*) calloc(n * n * color_dimension, sizeof(float));
        if (vision == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for vision.\n");
            if (InitializeScent) core::free(scent);
            return false;
        }
        items = (item*) malloc(sizeof(item) * item_count);
        if (items == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for items.\n");
            if (InitializeScent) core::free(scent);
            core::free(vision); return false;
        }
        agent_positions = (position*) malloc(sizeof(position) * agent_count);
        if (agent_positions == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for agent_positions.\n");
            if (InitializeScent) core::free(scent);
            core::free(vision); core::free(items);
            return false;
        }
        agent_directions = (direction*) malloc(sizeof(direction) * agent_count);
        if (agent_directions == NULL) {
            fprintf(stderr, "patch_state.init_helper ERROR: Insufficient memory for agent_directions.\n");
            if (InitializeScent) core::free(scent);
            core::free(vision); core::free(items);
            core::free(agent_positions);
            return false;
        }
        return true;
    }
};

/**
 * Initializes the memory of the given patch_state `patch`. This function does
 * not initialize the contents of any of the fields, except for `scent` and
 * `vision`, which are initialized to zeros.
 */
template<bool InitializeScent>
inline bool init(patch_state& patch, unsigned int n,
        unsigned int scent_dimension, unsigned int color_dimension,
        unsigned int item_count, unsigned int agent_count)
{
    patch.item_count = item_count;
    patch.agent_count = agent_count;
    return patch.init_helper<InitializeScent>(n, scent_dimension, color_dimension, item_count, agent_count);
}

/**
 * Reads the given patch_state `patch` from the input stream `in`.
 */
template<typename Stream>
bool read(patch_state& patch, Stream& in, const simulator_config& config) {
    bool has_scent;
    unsigned int n = config.patch_size;
    return read(patch.patch_position, in) && read(patch.fixed, in)
        && read(patch.item_count, in) && read(patch.agent_count, in)
        && read(has_scent, in)
        && (!has_scent || patch.init_helper<true>(n, config.scent_dimension, config.color_dimension, patch.item_count, patch.agent_count))
        && (has_scent || patch.init_helper<false>(n, config.scent_dimension, config.color_dimension, patch.item_count, patch.agent_count))
        && (!has_scent || read(patch.scent, in, n * n * config.scent_dimension))
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
        && write(patch.scent != nullptr, out)
        && (patch.scent == nullptr || write(patch.scent, out, n * n * config.scent_dimension))
        && write(patch.vision, out, n * n * config.color_dimension)
        && write(patch.items, out, patch.item_count)
        && write(patch.agent_positions, out, patch.agent_count)
        && write(patch.agent_directions, out, patch.agent_count);
}

void* alloc_position_keys(size_t n, size_t element_size) {
    position* keys = (position*) malloc(sizeof(position) * n);
    if (keys == NULL) return NULL;
    for (unsigned int i = 0; i < n; i++)
        position::set_empty(keys[i]);
    return (void*) keys;
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
    hash_map<uint64_t, agent_state*> agents;

    /* Semaphores in this simulator. */
    hash_map<uint64_t, bool> semaphores;

    /* A counter for assigning IDs to new agents and semaphores. */
    uint64_t id_counter;

    /* Lock for the agent and semaphore tables, used to prevent simultaneous updates. */
    std::mutex simulator_lock;

    /* A map from positions to a list of agents that request to move there. */
    hash_map<position, array<agent_state*>> requested_moves;

    /* Lock for the requested_moves map, used to prevent simultaneous updates. */
    std::mutex requested_move_lock;

    /**
     * Counter for how many agents have acted and how many semaphores have
     * signaled during each time step. This counter is used to force the
     * simulator to wait until all agents have acted and all semaphores have
     * signaled, before advancing the simulation time step.
     */
    unsigned int acted_agent_count;

    /**
     * The number of active agents and semaphores in the simulation. The
     * simulation only waits for agents with `agent_active` set to `true`
     * before advancing time.
     */
    unsigned int active_agent_count;

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
            config.mcmc_iterations,
            config.item_types.data,
            (unsigned int) config.item_types.length, seed),
        agents(32), semaphores(8), id_counter(1), requested_moves(32, alloc_position_keys),
        acted_agent_count(0), active_agent_count(0), data(data), time(0)
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
     * Adds a new agent to this simulator. Upon success, `new_agent` and
     * `new_agent_id` will contain the ID of the new agent as well as a pointer
     * to its state.
     */
    inline status add_agent(uint64_t& new_agent_id, agent_state*& new_agent) {
        simulator_lock.lock();
        if (!agents.check_size()) {
            simulator_lock.unlock();
            fprintf(stderr, "simulator.add_agent ERROR: Failed to expand agent table.\n");
            return status::OUT_OF_MEMORY;
        }

        unsigned int bucket = agents.table.index_to_insert(id_counter);
        new_agent = (agent_state*) malloc(sizeof(agent_state));
        new_agent_id = id_counter;
        if (new_agent == nullptr) {
            simulator_lock.unlock();
            return status::OUT_OF_MEMORY;
        }

        status init_status = init(*new_agent, world, scent_model, config, time);
        if (init_status != status::OK) {
            core::free(new_agent);
            simulator_lock.unlock();
            return init_status;
        }
        agents.table.keys[bucket] = id_counter;
        agents.values[bucket] = new_agent;
        agents.table.size++;
        active_agent_count++;
        id_counter++;
        simulator_lock.unlock();
        return status::OK;
    }

    /**
     * Removes the given agent from this simulator.
     *
     * \param   agent_id  ID of the agent to remove.
     */
    inline status remove_agent(uint64_t agent_id) {
        simulator_lock.lock();
        bool contains; unsigned int bucket;
        agent_state* agent = agents.get(agent_id, contains, bucket);
        if (!contains) {
            simulator_lock.unlock();
            return status::INVALID_AGENT_ID;
        }
        agents.remove_at(bucket);
        agent->lock.lock();
        if (agent->agent_acted) {
            unrequest_position(*agent);
            --acted_agent_count;
        }
        if (agent->agent_active)
            --active_agent_count;
        agent->lock.unlock();
        core::free(*agent, world, scent_model, config, time);
        core::free(agent);

        if (acted_agent_count == active_agent_count)
            step(); /* advance the simulation by one time step */
        simulator_lock.unlock();

        return status::OK;
    }

    /**
     * Adds a new semaphore for this simulator. Upon success,
     * `new_semaphore_id` will contain the ID of the new semaphore.
     */
    inline status add_semaphore(uint64_t& new_semaphore_id) {
        simulator_lock.lock();
        if (!semaphores.check_size()) {
            simulator_lock.unlock();
            fprintf(stderr, "simulator.create_semaphore ERROR: Failed to expand semaphore table.\n");
            return status::OUT_OF_MEMORY;
        }

        unsigned int bucket = semaphores.table.index_to_insert(id_counter);
        new_semaphore_id = id_counter;
        semaphores.table.keys[bucket] = id_counter;
        semaphores.values[bucket] = false;
        semaphores.table.size++;
        active_agent_count++;
        id_counter++;
        simulator_lock.unlock();
        return status::OK;
    }

    /**
     * Removes the given semaphore from this simulator.
     *
     * \param   semaphore_id  ID of the semaphore to remove.
     */
    inline status remove_semaphore(uint64_t semaphore_id) {
        simulator_lock.lock();
        bool contains; unsigned int bucket;
        bool signaled = semaphores.get(semaphore_id, contains, bucket);
        if (!contains) {
            simulator_lock.unlock();
            return status::INVALID_SEMAPHORE_ID;
        }
        semaphores.remove_at(bucket);
        if (signaled)
            --acted_agent_count;
        --active_agent_count;

        if (acted_agent_count == active_agent_count)
            step(); /* advance the simulation by one time step */
        simulator_lock.unlock();

        return status::OK;
    }

    /**
     * Signals a semaphore in the simulation.
     *
     * \param   semaphore_id  ID of the semaphore to signal.
     */
    inline status signal_semaphore(uint64_t semaphore_id) {
        bool contains;
        simulator_lock.lock();
        bool& signaled = semaphores.get(semaphore_id, contains);
        if (!contains) {
            simulator_lock.unlock();
            return status::INVALID_SEMAPHORE_ID;
        } else if (signaled) {
            simulator_lock.unlock();
            return status::SEMAPHORE_ALREADY_SIGNALED;
        }
        signaled = true;
        if (++acted_agent_count == active_agent_count)
            step(); /* advance the simulation by one time step */
        simulator_lock.unlock();
        return status::OK;
    }

    /**
     * Sets whether the agent with the given ID is active.
     */
    inline status set_agent_active(uint64_t agent_id, bool active) {
        bool contains;
        simulator_lock.lock();
        agent_state* agent_ptr = agents.get(agent_id, contains);
        if (!contains) {
            simulator_lock.unlock();
            return status::INVALID_AGENT_ID;
        }
        agent_state& agent = *agent_ptr;
        agent.lock.lock();
        simulator_lock.unlock();

        if (agent.agent_active && !active) {
            agent.agent_active = false;
            agent.lock.unlock();

            simulator_lock.lock();
            if (acted_agent_count == --active_agent_count)
                step(); /* advance the simulation by one time step */
            simulator_lock.unlock();
        } else if (!agent.agent_active && active) {
            agent.agent_active = true;
            agent.lock.unlock();

            simulator_lock.lock();
            active_agent_count++;
            simulator_lock.unlock();
        } else {
            agent.lock.unlock();
        }
        return status::OK;
    }

    /**
     * Sets whether the agent with the given ID is active.
     */
    inline status is_agent_active(uint64_t agent_id, bool& active) {
        bool contains;
        std::unique_lock<std::mutex> lock(simulator_lock);
        agent_state* agent_ptr = agents.get(agent_id, contains);
        if (!contains)
            return status::INVALID_AGENT_ID;
        active = agent_ptr->agent_active;
        return status::OK;
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
     */
    inline status move(uint64_t agent_id, direction dir, unsigned int num_steps)
    {
        if (num_steps > config.max_steps_per_movement
         || config.allowed_movement_directions[(size_t) dir] == action_policy::DISALLOWED)
            return status::PERMISSION_ERROR;

        bool contains;
        simulator_lock.lock();
        agent_state& agent = *agents.get(agent_id, contains);
        if (!contains) {
            simulator_lock.unlock();
            return status::INVALID_AGENT_ID;
        }
        agent.lock.lock();
        simulator_lock.unlock();

        if (agent.agent_acted) {
            agent.lock.unlock();
            return status::AGENT_ALREADY_ACTED;
        }
        agent.agent_acted = true;

        agent.requested_position = agent.current_position;
        agent.requested_direction = agent.current_direction;
        if (config.allowed_movement_directions[(size_t) dir] != action_policy::IGNORED) {
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
        }

        /* add the agent's move to the list of requested moves */
        request_position(agent);

        if (agent.agent_active) {
            agent.lock.unlock();
            simulator_lock.lock();
            if (++acted_agent_count == active_agent_count)
                step(); /* advance the simulation by one time step */
            simulator_lock.unlock();
        } else {
            agent.lock.unlock();
        }
        return status::OK;
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
     */
    inline status turn(uint64_t agent_id, direction dir)
    {
        if (config.allowed_rotations[(size_t) dir] == action_policy::DISALLOWED)
            return status::PERMISSION_ERROR;

        bool contains;
        simulator_lock.lock();
        agent_state& agent = *agents.get(agent_id, contains);
        if (!contains) {
            simulator_lock.unlock();
            return status::INVALID_AGENT_ID;
        }
        agent.lock.lock();
        simulator_lock.unlock();

        if (agent.agent_acted) {
            agent.lock.unlock();
            return status::AGENT_ALREADY_ACTED;
        }
        agent.agent_acted = true;

        agent.requested_position = agent.current_position;
        agent.requested_direction = agent.current_direction;

        if (config.allowed_rotations[(size_t) dir] != action_policy::IGNORED) {
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
        }

        /* add the agent's move to the list of requested moves */
        request_position(agent);

        if (agent.agent_active) {
            agent.lock.unlock();
            simulator_lock.lock();
            if (++acted_agent_count == active_agent_count)
                step(); /* advance the simulation by one time step */
            simulator_lock.unlock();
        } else {
            agent.lock.unlock();
        }
        return status::OK;
    }

    /**
     * Instructs the agent to do nothing this turn.
     *
     * \param   agent_id ID of the agent.
     */
    inline status do_nothing(uint64_t agent_id)
    {
        if (!config.no_op_allowed) return status::PERMISSION_ERROR;

        bool contains;
        simulator_lock.lock();
        agent_state& agent = *agents.get(agent_id, contains);
        if (!contains) {
            simulator_lock.unlock();
            return status::INVALID_AGENT_ID;
        }
        agent.lock.lock();
        simulator_lock.unlock();

        if (agent.agent_acted) {
            agent.lock.unlock();
            return status::AGENT_ALREADY_ACTED;
        }
        agent.agent_acted = true;

        agent.requested_position = agent.current_position;
        agent.requested_direction = agent.current_direction;

        /* add the agent's move to the list of requested moves */
        request_position(agent);

        if (agent.agent_active) {
            agent.lock.unlock();
            simulator_lock.lock();
            if (++acted_agent_count == active_agent_count)
                step(); /* advance the simulation by one time step */
            simulator_lock.unlock();
        } else {
            agent.lock.unlock();
        }
        return status::OK;
    }

    /**
     * Retrieves an array of pointers to agent_state structures, storing them
     * in `states`, which is parallel to the specified `agent_ids` array, and
     * has length `agent_count`. For any invalid agent ID, the corresponding
     * agent_state is set to nullptr.
     *
     * NOTE: This function will lock each non-null agent in `states`. The
     *       caller must unlock them afterwards.
     *
     * \param      states The output array of agent_state pointers.
     * \param   agent_ids The array of agent IDs whose states to retrieve.
     * \param agent_count The length of `states` and `agent_ids`.
     */
    inline void get_agent_states(agent_state** states,
            uint64_t* agent_ids, unsigned int agent_count)
    {
        std::unique_lock<std::mutex> lock(simulator_lock);
        for (unsigned int i = 0; i < agent_count; i++) {
            bool contains;
            states[i] = agents.get(agent_ids[i], contains);
            if (contains) states[i]->lock.lock();
            else states[i] = nullptr;
        }
    }

    /**
     * Retrieves an array of IDs of all agents in this simulation.
     *
     * \param   agent_ids The array that will be populated with agent IDs.
     */
    inline status get_agent_ids(array<uint64_t>& agent_ids)
    {
        std::unique_lock<std::mutex> lock(simulator_lock);
        if (!agent_ids.ensure_capacity(agent_ids.length + agents.table.size)) {
            simulator_lock.unlock();
            return status::OUT_OF_MEMORY;
        }
        for (const auto& entry : agents)
            agent_ids[agent_ids.length++] = entry.key;
        return status::OK;
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
     * \param patches The output array of array of patch_state structures which
     *      will contain the state of the retrieved patches. Each inner array
     *      represents a row of patches that all share the same `y` value in
     *      their patch positions;
     */
    template<bool GetScentMap>
    status get_map(
            position bottom_left_corner,
            position top_right_corner,
            array<array<patch_state>>& patches)
    {
        position bottom_left_patch_position, top_right_patch_position;
        world.world_to_patch_coordinates(bottom_left_corner, bottom_left_patch_position);
        world.world_to_patch_coordinates(top_right_corner, top_right_patch_position);

        simulator_lock.lock();

        status result = status::OK;
        apply_contiguous(world.patches, bottom_left_patch_position.y - 1,
            (unsigned int) (top_right_patch_position.y - bottom_left_patch_position.y + 2),
            [&](const array_map<int64_t, patch_type>& row, int64_t y)
        {
            if (!patches.ensure_capacity(patches.length + 1)) {
                result = status::OUT_OF_MEMORY;
                return false;
            }
            array<patch_state>& current_row = patches[patches.length];
            if (!array_init(current_row, 16)) {
                result = status::OUT_OF_MEMORY;
                return false;
            }
            patches.length++;

            apply_contiguous(row, bottom_left_patch_position.x - 1,
                (unsigned int) (top_right_patch_position.x - bottom_left_patch_position.x + 2),
                [&](const patch_type& patch, int64_t x)
            {
                if (!current_row.ensure_capacity(current_row.length + 1)) {
                    result = status::OUT_OF_MEMORY;
                    return false;
                }
                patch_state& state = current_row[current_row.length];
                if (!init<GetScentMap>(state, config.patch_size,
                    config.scent_dimension, config.color_dimension,
                    (unsigned int) patch.items.length,
                    (unsigned int) patch.data.agents.length))
                {
                    result = status::OUT_OF_MEMORY;
                    return false;
                }
                current_row.length++;

                state.patch_position = position(x, y);
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

                /* consider all patches in the neighborhood of 'patch' */
                position patch_world_position = position(x, y) * config.patch_size;
                if (GetScentMap) {
                    for (unsigned int a = 0; a < config.patch_size; a++) {
                        for (unsigned int b = 0; b < config.patch_size; b++) {
                            position current_position = patch_world_position + position(a, b);
                            patch_type* neighborhood[4]; position patch_positions[4];
                            unsigned int patch_count = world.get_neighborhood(current_position, neighborhood, patch_positions);
                            for (unsigned int i = 0; i < patch_count; i++) {
                                /* iterate over neighboring items, and add their contributions to scent and vision */
                                for (unsigned int j = 0; j < neighborhood[i]->items.length; j++) {
                                    const item& item = neighborhood[i]->items[j];

                                    /* check if the item is too old; if so, ignore it */
                                    if (item.deletion_time > 0 && time >= item.deletion_time + config.deleted_item_lifetime)
                                        continue;

                                    compute_scent_contribution(scent_model, item, current_position, time,
                                            config, state.scent + ((a*config.patch_size + b)*config.scent_dimension));
                                }
                            }
                        }
                    }
                }

                for (const item& item : patch.items) {
                    if (item.deletion_time != 0) continue;
                    position relative_position = item.location - patch_world_position;
                    float* pixel = state.vision + ((relative_position.x*config.patch_size + relative_position.y)*config.color_dimension);
                    for (unsigned int i = 0; i < config.color_dimension; i++)
                        pixel[i] += config.item_types[item.item_type].color[i];
                }

                for (const agent_state* agent : patch.data.agents) {
                    position relative_position = agent->current_position - patch_world_position;
                    float* pixel = state.vision + ((relative_position.x*config.patch_size + relative_position.y)*config.color_dimension);
                    for (unsigned int i = 0; i < config.color_dimension; i++)
                        pixel[i] += config.agent_color[i];
                }

                return true;
            });

            if (current_row.length == 0) {
                core::free(current_row);
                patches.length--;
            }
            return true;
        });

        simulator_lock.unlock();
        return result;
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

    static inline void free(simulator& s) {
        s.free_helper();
        core::free(s.agents);
        core::free(s.semaphores);
        core::free(s.requested_moves);
        core::free(s.config);
        core::free(s.scent_model);
        core::free(s.world);
        core::free(s.data);
        s.simulator_lock.~mutex();
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
                unsigned int result = sample_uniform((unsigned int)conflicts.length);
                core::swap(conflicts[0], conflicts[result]);
            }
        }

        /* check for items that block movement */
        array<position> occupied_positions(16);
        for (auto entry : requested_moves) {
            patch_type* neighborhood[4]; position patch_positions[4];
            unsigned int index = world.get_fixed_neighborhood(
                entry.key, neighborhood, patch_positions);
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
            }
            else if (contains) {
                for (unsigned int i = 0; i < conflicts.length; i++)
                    occupied_positions.add(conflicts[i]->current_position);
                conflicts[0] = NULL; /* prevent any agent from moving here */
            }
        }

        time++;
        acted_agent_count = 0;
        for (auto entry : agents) {
            agent_state* agent = entry.value;
            agent->lock.lock();
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
                unsigned int index = world.get_fixed_neighborhood(
                    agent->current_position, neighborhood, patch_positions);
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
            for (unsigned int i = 0; i < agents.table.capacity; i++) {
                if (is_empty(agents.table.keys[i])) continue;
                for (unsigned int j = i + 1; j < agents.table.capacity; j++) {
                    if (is_empty(agents.table.keys[j])) continue;
                    if (agents.values[i]->current_position == agents.values[j]->current_position)
                        fprintf(stderr, "simulator.step WARNING: Agents %u and %u are at the same position.\n", i, j);
                }
            }
        }
#endif

        /* compute new scent and vision for each agent */
        update_agent_scent_and_vision();

        /* reset the requested moves */
        for (auto entry : requested_moves)
            core::free(entry.value);
        requested_moves.clear();
        requested_move_lock.unlock();

        /* reset all semaphores to their non-signaled state */
        for (auto entry : semaphores)
            entry.value = false;

        /* Invoke the step callback function for each agent. */
        on_step((simulator<SimulatorData>*) this, (const hash_map<uint64_t, agent_state*>&) agents, time);
    }

    /* Precondition: This thread has all agent locks, which it will release. */
    inline void update_agent_scent_and_vision() {
        for (auto entry : agents) {
            agent_state* agent = entry.value;
            patch_type* neighborhood[4]; position patch_positions[4];
            world.get_fixed_neighborhood(
                agent->current_position, neighborhood, patch_positions);
            agent->update_state(neighborhood, scent_model, config, time);
            agent->lock.unlock();
        }
    }

    inline void request_position(agent_state& agent)
    {
        /* check for collisions with other agents */
        if (config.collision_policy == movement_conflict_policy::NO_COLLISIONS)
            return;

        bool contains; unsigned int bucket;
        std::unique_lock<std::mutex> lock(requested_move_lock);
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
    }

    inline void unrequest_position(agent_state& agent)
    {
        if (!agent.agent_acted || config.collision_policy == movement_conflict_policy::NO_COLLISIONS)
            return;

        bool contains; unsigned int bucket;
        std::unique_lock<std::mutex> lock(requested_move_lock);
        array<agent_state*>& agents = requested_moves.get(agent.requested_position, contains, bucket);
        if (!contains) return;
        unsigned int index = agents.index_of(&agent);
        if (index != agents.length)
            agents.remove(index);
    }

    inline void free_helper() {
        for (auto entry : requested_moves)
            core::free(entry.value);
        for (auto entry : agents) {
            core::free(*entry.value);
            core::free(entry.value);
        }
    }

    template<typename A> friend status init(simulator<A>&, const simulator_config&, const A&, uint_fast32_t);
    template<typename A, typename B> friend bool read(simulator<A>&, B&, const A&);
    template<typename A, typename B> friend bool write(const simulator<A>&, B&);
};

/**
 * Constructs a new simulator with the given simulator_config `config` and
 * SimulatorData `data`, calling the
 * `bool init(SimulatorData&, const SimulatorData&)` function to initialize
 * `data`.
 */
template<typename SimulatorData>
status init(simulator<SimulatorData>& sim, 
        const simulator_config& config,
        const SimulatorData& data,
        uint_fast32_t seed)
{
    sim.time = 0;
    sim.acted_agent_count = 0;
    sim.active_agent_count = 0;
    sim.id_counter = 1;
    if (!init(sim.data, data)) {
        return status::OUT_OF_MEMORY;
    } else if (!hash_map_init(sim.agents, 32)) {
        free(sim.data); return status::OUT_OF_MEMORY;
    } else if (!hash_map_init(sim.semaphores, 8)) {
        free(sim.data); free(sim.agents); return status::OUT_OF_MEMORY;
    } else if (!hash_map_init(sim.requested_moves, 32, alloc_position_keys)) {
        free(sim.data); free(sim.agents);
        free(sim.semaphores); return status::OUT_OF_MEMORY;
    } else if (!init(sim.config, config)) {
        free(sim.data); free(sim.agents); free(sim.semaphores);
        free(sim.requested_moves); return status::OUT_OF_MEMORY;
    } else if (!init(sim.scent_model, (double) sim.config.diffusion_param,
            (double) sim.config.decay_param, sim.config.patch_size, sim.config.deleted_item_lifetime)) {
        free(sim.data); free(sim.config);
        free(sim.agents); free(sim.semaphores);
        free(sim.requested_moves); return status::OUT_OF_MEMORY;
    } else if (!init(sim.world, sim.config.patch_size,
            sim.config.mcmc_iterations,
            sim.config.item_types.data,
            (unsigned int) sim.config.item_types.length, seed)) {
        free(sim.config); free(sim.data);
        free(sim.agents); free(sim.semaphores);
        free(sim.requested_moves); free(sim.scent_model);
        return status::OUT_OF_MEMORY;
    }
    new (&sim.simulator_lock) std::mutex();
    new (&sim.requested_move_lock) std::mutex();
    return status::OK;
}

/**
 * Constructs a new simulator with the given simulator_config `config` and
 * SimulatorData `data`, calling the
 * `bool init(SimulatorData&, const SimulatorData&)` function to initialize
 * `data`.
 */
template<typename SimulatorData>
inline status init(simulator<SimulatorData>& sim, 
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
inline bool read(agent_state*& agent, Stream& in,
        const hash_map<uint64_t, agent_state*>& agents)
{
    uint64_t id;
    if (!read(id, in)) return false;
    agent = agents.get(id);
    return true;
}

template<typename Stream>
inline bool write(const agent_state* agent, Stream& out,
        const hash_map<const agent_state*, uint64_t>& agent_ids)
{
    return write(agent_ids.get(agent), out);
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

    unsigned int agent_count;
    if (!read(agent_count, in)
     || !hash_map_init(sim.agents, ((size_t)1) << (core::log2(agent_count) + 1) * RESIZE_THRESHOLD_INVERSE))
    {
        free(sim.data); free(sim.config);
        return false;
    }

    for (unsigned int i = 0; i < agent_count; i++) {
        uint64_t id;
        agent_state* agent = (agent_state*) malloc(sizeof(agent_state));
        if (agent == nullptr || !read(id, in) || !read(*agent, in, sim.config)) {
            if (agent != nullptr) free(agent);
            for (auto entry : sim.agents) {
                free(*entry.value); free(entry.value);
            }
            free(sim.data); free(sim.agents);
            free(sim.config); return false;
        }
        sim.agents.put(id, agent);
    }

    if (!read(sim.semaphores, in)) {
        for (auto entry : sim.agents) {
            free(*entry.value); free(entry.value);
        }
        free(sim.data); free(sim.agents);
        free(sim.config); return false;
    }

    if (!read(sim.world, in, sim.config.item_types.data, (unsigned int) sim.config.item_types.length, sim.agents)) {
        for (auto entry : sim.agents) {
            free(*entry.value); free(entry.value);
        }
        free(sim.semaphores);
        free(sim.data); free(sim.agents);
        free(sim.config); return false;
    }

    default_scribe scribe;
    if (!read(sim.requested_moves, in, alloc_position_keys, scribe, sim.agents)) {
        for (auto entry : sim.agents) {
            free(*entry.value); free(entry.value);
        }
        free(sim.semaphores);
        free(sim.data); free(sim.agents);
        free(sim.config); free(sim.world);
        return false;
    }

    /* reinitialize the scent model */
    if (!read(sim.time, in)
     || !read(sim.acted_agent_count, in)
     || !read(sim.active_agent_count, in)
     || !read(sim.id_counter, in)
     || !init(sim.scent_model, (double) sim.config.diffusion_param,
            (double) sim.config.decay_param, sim.config.patch_size,
            sim.config.deleted_item_lifetime))
    {
        for (auto entry : sim.agents) {
            free(*entry.value); free(entry.value);
        }
        for (auto entry : sim.requested_moves)
            free(entry.value);
        free(sim.semaphores);
        free(sim.data); free(sim.world); free(sim.agents);
        free(sim.requested_moves); free(sim.config);
        return false;
    }
    new (&sim.simulator_lock) std::mutex();
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

    hash_map<const agent_state*, uint64_t> agent_ids((unsigned int) sim.agents.table.size * RESIZE_THRESHOLD_INVERSE);
    if (!write(sim.agents.table.size, out)) return false;
    for (const auto& entry : sim.agents) {
        if (!agent_ids.put(entry.value, entry.key)
         || !write(entry.key, out) || !write(*entry.value, out, sim.config))
        {
            return false;
        }
    }

    default_scribe scribe;
    return write(sim.semaphores, out)
        && write(sim.world, out, agent_ids)
        && write(sim.requested_moves, out, scribe, agent_ids)
        && write(sim.time, out)
        && write(sim.acted_agent_count, out)
        && write(sim.active_agent_count, out)
        && write(sim.id_counter, out);
}

} /* namespace jbw */

#endif /* JBW_SIMULATOR_H_ */

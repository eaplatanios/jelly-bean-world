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

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float pixel_density;
    uint patch_size_texels;
    vec3 agent_color;
} ubo;

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    float line_width = min(2.4f / ubo.pixel_density, 0.06f);
    if (frag_color.x >= 4.0f) {
        /* this is an agent */
        float dist = min(min(frag_tex_coord.x, frag_tex_coord.y), (1.0f - frag_tex_coord.y - frag_tex_coord.x));

        float fill_weight;
        if (dist < line_width) {
            fill_weight = dist / line_width;
        } else {
            fill_weight = 1.0f;
        }

        out_color = vec4(frag_color - 4.0f, fill_weight);

    } else if (frag_color.x >= 2.0f) {
        /* this item blocks movement, so render it as a square */
        out_color = vec4(frag_color - 2.0f, 1.0f);

    } else {
        /* this item does not block movement, so render it as a circle */
        vec2 diff = frag_tex_coord - vec2(0.5f, 0.5f);
        float dist = 0.5f - sqrt(dot(diff, diff));

        float fill_weight;
        if (dist < 0.0f) {
            fill_weight = 0.0f;
        } else if (dist < line_width) {
            fill_weight = dist / line_width;
        } else {
            fill_weight = 1.0f;
        }

        out_color = vec4(frag_color, fill_weight);
    }
}

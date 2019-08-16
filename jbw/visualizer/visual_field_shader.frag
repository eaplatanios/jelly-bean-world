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
    float patch_size_texels;
} ubo;

layout(binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    vec2 grid = abs(fract(uv + 0.1f));
    float line_weight;
    if (min(grid.x, grid.y) < 0.2f) {
		line_weight = texture(tex_sampler, frag_tex_coord / ubo.patch_size_texels).w;
	} else {
		line_weight = 1.0f;
    }
    out_color = (1.0f - line_weight) * vec4(0.0f, 0.0f, 0.0f, 1.0f) + line_weight * vec4(texture(tex_sampler, frag_tex_coord).xyz, 1.0);
}

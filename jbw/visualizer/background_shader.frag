#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float pixel_density;
    float patch_size;
} ubo;

layout(binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    vec2 grid = abs(fract(uv + 0.1f)) / fwidth(uv);
    float line_weight = clamp(min(grid.x, grid.y) - (0.2f * ubo.pixel_density - 1.0f), texture(tex_sampler, frag_tex_coord / ubo.patch_size).w, 1.0f);
    out_color = (1.0f - line_weight) * vec4(0.0f, 0.0f, 0.0f, 1.0f) + line_weight * vec4(texture(tex_sampler, frag_tex_coord).xyz, 1.0);
}

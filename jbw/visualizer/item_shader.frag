#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float pixel_density;
    float patch_size_texels;
} ubo;

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    float line_width = min(1.0f / ubo.pixel_density, 0.06f);
    if (frag_color.x >= 4.0f) {
        /* this is an agent */
        float dist = min(min(frag_tex_coord.x, frag_tex_coord.y), (1.0f - frag_tex_coord.y - frag_tex_coord.x));

        float border_weight;
        if (dist < 0.06f) {
            border_weight = 1.0f;
        } else if (dist < 0.06f + line_width) {
            border_weight = 1.0f - ((dist - 0.06f) / line_width);
        } else {
            border_weight = 0.0f;
        }

        out_color = border_weight * vec4(0.0f, 0.0f, 0.0f, 1.0f) + (1.0f - border_weight) * vec4(frag_color - 4.0f, 1.0f);

    } else if (frag_color.x >= 2.0f) {
        /* this item blocks movement, so render it as a square */
        float dist = min(min(min(frag_tex_coord.x, frag_tex_coord.y), 1.0f - frag_tex_coord.x), 1.0f - frag_tex_coord.y);

        float border_weight;
        if (dist < 0.06f) {
            border_weight = 1.0f;
        } else if (dist < 0.06f + line_width) {
            border_weight = 1.0f - ((dist - 0.06f) / line_width);
        } else {
            border_weight = 0.0f;
        }

        out_color = border_weight * vec4(0.0f, 0.0f, 0.0f, 1.0f) + (1.0f - border_weight) * vec4(frag_color - 2.0f, 1.0f);

    } else {
        /* this item does not block movement, so render it as a circle */
        vec2 diff = frag_tex_coord - vec2(0.5f, 0.5f);
        float dist = 0.5f - sqrt(dot(diff, diff));

        float border_weight;
        float fill_weight;
        if (dist < 0.0f) {
            border_weight = 0.0f;
            fill_weight = 0.0f;
        } else if (dist < line_width) {
            border_weight = dist / line_width;
            fill_weight = 0.0f;
        } else if (dist < 0.07f) {
            border_weight = 1.0f;
            fill_weight = 0.0f;
        } else if (dist < 0.07f + line_width) {
            border_weight = 1.0f - ((dist - 0.07f) / line_width);
            fill_weight = 1.0f - border_weight;
        } else {
            border_weight = 0.0f;
            fill_weight = 1.0f;
        }

        out_color = fill_weight * vec4(frag_color, 1.0f) + border_weight * vec4(0.0f, 0.0f, 0.0f, 1.0f) + (1.0 - fill_weight - border_weight) * vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

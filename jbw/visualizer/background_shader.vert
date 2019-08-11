#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float pixel_density;
    float patch_size;
} ubo;

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_tex_coord;

layout(location = 0) out vec2 uv;
layout(location = 1) out vec2 frag_tex_coord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(in_position, 0.0, 1.0);
    uv = (ubo.model * vec4(in_position, 0.0, 1.0)).xy;
    frag_tex_coord = in_tex_coord;
}

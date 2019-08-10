#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "vulkan_renderer.h"
#include <stdlib.h>
#include <core/utility.h>
#include <core/timer.h>
#include <math.h>

using namespace core;
using namespace mirage;

struct vertex {
	float position[2];
	float tex_coord[2];
};

struct pixel {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
};

struct model_view_matrix {
	float model[16];
	float view[16];
	float projection[16];
};

bool resized = false;

inline void on_framebuffer_resize(GLFWwindow* window, int width, int height) {
	resized = true;
}

inline void cleanup(GLFWwindow* window) {
	glfwDestroyWindow(window);
	glfwTerminate();
}

inline void cleanup(GLFWwindow* window,
		vulkan_renderer& renderer,
		shader& vertex_shader,
		shader& fragment_shader)
{
	renderer.delete_shader(vertex_shader);
	renderer.delete_shader(fragment_shader);
	cleanup(window);
}

inline void cleanup_renderer(
		vulkan_renderer& renderer,
		graphics_pipeline& pipeline,
		frame_buffer& fb,
		command_buffer& cb,
		uniform_buffer& ub,
		descriptor_set& set,
		descriptor_pool& pool)
{
	renderer.delete_command_buffer(cb);
	renderer.delete_uniform_buffer(ub);
	renderer.delete_descriptor_set(set);
	renderer.delete_descriptor_pool(pool);
	renderer.delete_frame_buffer(fb);
	renderer.delete_graphics_pipeline(pipeline);
}

inline void cross(float (&out)[3],
		float (&first)[3], float (&second)[3])
{
	out[0] = first[1]*second[2] - first[2]*second[1];
	out[1] = first[2]*second[0] - first[0]*second[2];
	out[2] = first[0]*second[1] - first[1]*second[0];
}

inline float dot(float (&first)[3], float(&second)[3]) {
	return first[0]*second[0] + first[1]*second[1] + first[2]*second[2];
}

inline void make_view_matrix(float (&view)[16],
		float (&forward)[3], float (&up)[3],
		float (&position)[3])
{
	float s[3];
	cross(s, forward, up);
	/* we assume `forward` and `up` have length 1 */
	float u[3];
	cross(u, s, forward);

	view[0] = s[0]; view[4] = s[1]; view[8] = s[2]; view[12] = -dot(s, position);
	view[1] = u[0]; view[5] = u[1]; view[9] = u[2]; view[13] = -dot(u, position);
	view[2] = -forward[0]; view[6] = -forward[1]; view[10] = -forward[2]; view[14] = dot(forward, position);
	view[15] = 1.0f;
}

inline void make_orthographic_projection(float(&proj)[16],
	float fLeft, float fRight, float fBottom, float fTop, float fNear, float fFar)
{
	proj[0] = 2 / (fRight - fLeft);
	proj[5] = -2 / (fTop - fBottom); /* make the positive y axis direction point upwards */
	proj[10] = 2 / (fNear - fFar);
	proj[12] = (fLeft + fRight) / (fLeft - fRight);
	proj[13] = (fBottom + fTop) / (fBottom - fTop);
	proj[14] = (fNear + fFar) / (fNear - fFar);
	proj[15] = 1.0f;
}

inline void make_perspective_projection(float (&proj)[16],
		float fov, float aspect_ratio, float fNear, float fFar)
{
	float tan_half_fov = tan(fov / 2);
	proj[0] = 1.0f / (aspect_ratio * tan_half_fov);
	proj[5] = -1.0f / tan_half_fov; /* make the positive y axis direction point upwards */
	proj[10] = (fNear + fFar) / (fNear - fFar);
	proj[11] = -1.0f;
	proj[14] = 2 * fFar * fNear / (fNear - fFar);
}

template<size_t N>
bool setup_renderer(vulkan_renderer& renderer,
		shader& vertex_shader, shader& fragment_shader,
		graphics_pipeline& pipeline,
		frame_buffer& fb, command_buffer& cb,
		uniform_buffer& ub, const vertex_buffer& vb,
		descriptor_pool& pool,
		descriptor_set& ub_set,
		const binding_description& binding,
		const attribute_descriptions<N>& attributes,
		const descriptor_set_layout& layout,
		const dynamic_texture_image& texture,
		const sampler& sampler)
{
	float clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	vertex_buffer vertex_buffers[] = { vb };
	uint64_t offsets[] = { 0 };
	uint32_t ub_binding = 0;
	uint32_t texture_binding = 1;
	descriptor_type pool_types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
	if (!renderer.create_graphics_pipeline(pipeline, vertex_shader, "main", fragment_shader, "main", primitive_topology::TRIANGLE_STRIP, binding, attributes, &layout, 1)) {
		return false;
	} else if (!renderer.create_frame_buffer(fb, pipeline)) {
		renderer.delete_graphics_pipeline(pipeline);
		return false;
	} else if (!renderer.create_uniform_buffer(ub, sizeof(model_view_matrix))) {
		renderer.delete_frame_buffer(fb);
		renderer.delete_graphics_pipeline(pipeline);
		return false;
	} else if (!renderer.create_descriptor_pool(pool, pool_types, 2)) {
		renderer.delete_uniform_buffer(ub);
		renderer.delete_frame_buffer(fb);
		renderer.delete_graphics_pipeline(pipeline);
		return false;
	} else if (!renderer.create_descriptor_set(ub_set, &ub, &ub_binding, 1, nullptr, nullptr, 0, &texture, &texture_binding, 1, &sampler, layout, pool)) {
		renderer.delete_uniform_buffer(ub);
		renderer.delete_descriptor_pool(pool);
		renderer.delete_frame_buffer(fb);
		renderer.delete_graphics_pipeline(pipeline);
		return false;
	} else if (!renderer.create_command_buffer(cb)) {
		renderer.delete_uniform_buffer(ub);
		renderer.delete_descriptor_set(ub_set);
		renderer.delete_descriptor_pool(pool);
		renderer.delete_frame_buffer(fb);
		renderer.delete_graphics_pipeline(pipeline);
		return false;
	} else if (!renderer.record_command_buffer(cb, fb, pipeline, clear_color, 4, 0, vertex_buffers, offsets, &ub_set, 1)) {
		cleanup_renderer(renderer, pipeline, fb, cb, ub, ub_set, pool);
		return false;
	}
	return true;
}

int main(int argc, const char** argv)
{
	size_t vertex_shader_size;
	char* vertex_shader_src = read_file<true>("vert.spv", vertex_shader_size);
	if (vertex_shader_src == nullptr) {
		fprintf(stderr, "ERROR: Unable to read 'vert.spv'.\n");
		return EXIT_FAILURE;
	}

	size_t fragment_shader_size;
	char* fragment_shader_src = read_file<true>("frag.spv", fragment_shader_size);
	if (fragment_shader_src == nullptr) {
		fprintf(stderr, "ERROR: Unable to read 'frag.spv'.\n");
		free(vertex_shader_src);
		return EXIT_FAILURE;
	}

	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	uint32_t window_width = 800, window_height = 800;
	GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Renderer Test", nullptr, nullptr);
	glfwSetFramebufferSizeCallback(window, on_framebuffer_resize);

	uint32_t extension_count = 0;
	const char** required_extensions = glfwGetRequiredInstanceExtensions(&extension_count);
	vulkan_renderer renderer("Renderer Test", 0, "no engine", 0,
			required_extensions, extension_count, device_selector::FIRST_ANY,
			glfw_surface(window), window_width, window_height, 2, false);

	shader vertex_shader, fragment_shader;
	if (!renderer.create_shader(vertex_shader, vertex_shader_src, vertex_shader_size)) {
		free(vertex_shader_src); free(fragment_shader_src);
		cleanup(window); return EXIT_FAILURE;
	} if (!renderer.create_shader(fragment_shader, fragment_shader_src, fragment_shader_size)) {
		renderer.delete_shader(vertex_shader);
		free(vertex_shader_src); free(fragment_shader_src);
		cleanup(window); return EXIT_FAILURE;
	}
	free(vertex_shader_src); free(fragment_shader_src);

	vertex vertices[] = {
		{{-0.5f, -0.5f}, {1.0f, 0.0f}},
		{{-0.5f, 0.5f}, {1.0f, 1.0f}},
		{{0.5f, -0.5f}, {0.0f, 0.0f}},
		{{0.5f, 0.5f}, {0.0f, 1.0f}}
	};

	binding_description binding(0, sizeof(vertex));
	attribute_descriptions<2> attributes;
	attributes.set<0>(0, 0, attribute_type::FLOAT2, offsetof(vertex, position));
	attributes.set<1>(0, 1, attribute_type::FLOAT2, offsetof(vertex, tex_coord));

	vertex_buffer vb;
	if (!renderer.create_vertex_buffer(vb, sizeof(vertex) * 4)) {
		cleanup(window, renderer, vertex_shader, fragment_shader);
		return EXIT_FAILURE;
	}
	renderer.fill_vertex_buffer(vb, (void*) vertices, sizeof(vertex) * 4);

	graphics_pipeline pipeline;
	frame_buffer fb;
	command_buffer cb;
	descriptor_set_layout layout;
	descriptor_pool pool;
	uniform_buffer ub;
	descriptor_set ub_set;
	sampler sampler;
	dynamic_texture_image texture;

	uint32_t binding_indices[] = { 0, 1 };
	descriptor_type types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
	uint32_t descriptor_counts[] = { 1, 1 };
	shader_stage visibilities[] = { shader_stage::VERTEX, shader_stage::FRAGMENT };
	if (!renderer.create_descriptor_set_layout(layout, binding_indices, types, descriptor_counts, visibilities, 2)) {
		renderer.delete_vertex_buffer(vb);
		cleanup(window, renderer, vertex_shader, fragment_shader);
		return EXIT_FAILURE;
	}

	size_t image_size = sizeof(pixel) * 32 * 32;
	if (!renderer.create_dynamic_texture_image(texture, image_size, 32, 32)) {
		renderer.delete_descriptor_set_layout(layout);
		renderer.delete_vertex_buffer(vb);
		cleanup(window, renderer, vertex_shader, fragment_shader);
		return EXIT_FAILURE;
	}

	if (!renderer.create_sampler(sampler, filter::NEAREST, filter::NEAREST,
			sampler_address_mode::CLAMP_TO_EDGE, sampler_address_mode::CLAMP_TO_EDGE, sampler_address_mode::CLAMP_TO_EDGE, false, 1.0f))
	{
		renderer.delete_dynamic_texture_image(texture);
		renderer.delete_descriptor_set_layout(layout);
		renderer.delete_vertex_buffer(vb);
		cleanup(window, renderer, vertex_shader, fragment_shader);
		return EXIT_FAILURE;
	}

	auto reset_command_buffers = [&]() {
		cleanup_renderer(renderer, pipeline, fb, cb, ub, ub_set, pool);
		return setup_renderer(renderer, vertex_shader, fragment_shader, pipeline, fb, cb, ub, vb, pool, ub_set, binding, attributes, layout, texture, sampler);
	};

	auto get_window_dimensions = [&](uint32_t& width, uint32_t& height) {
		int new_width, new_height;
		glfwGetFramebufferSize(window, &new_width, &new_height);
		width = (uint32_t) new_width;
		height = (uint32_t) new_height;
		window_width = width;
		window_height = height;
	};

	if (!setup_renderer(renderer, vertex_shader, fragment_shader, pipeline, fb, cb, ub, vb, pool, ub_set, binding, attributes, layout, texture, sampler)) {
		renderer.delete_descriptor_set_layout(layout);
		renderer.delete_vertex_buffer(vb);
		cleanup_renderer(renderer, pipeline, fb, cb, ub, ub_set, pool);
		return EXIT_FAILURE;
	}

	model_view_matrix transform = { 0 };
	for (unsigned int i = 0; i < 4; i++)
		/* set the model matrix to the identity */
		transform.model[i * 4 + i] = 1.0f;

	timer stopwatch;
	uint64_t elapsed = 0;
	uint64_t frame_count = 0;
	bool resized = false;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		/* set the texture data */
		pixel* texture_data = (pixel*) texture.mapped_memory;
		uint8_t value =  128 + (uint8_t) (127 * cos(frame_count / 50.0f));
		for (unsigned int i = 0; i < 32; i++) {
			for (unsigned int j = 0; j < 32; j++) {
				texture_data[i * 32 + j].r = ((i + j) % 2 == 0) ? value : (255 - value);
				texture_data[i * 32 + j].g = ((i + j) % 2 == 0) ? value : (255 - value);
				texture_data[i * 32 + j].b = ((i + j) % 2 == 0) ? value : (255 - value);
				texture_data[i * 32 + j].a = 255;
			}
		}
		renderer.transfer_dynamic_texture_image(texture);

		/* construct the view matrix */
		float up[] = { 0.0f, 1.0f, 0.0f };
		float forward[] = { 0.0f, 0.0f, -1.0f };
		float camera_position[] = { 2.0f, 0.0f, 2.0f };
		make_view_matrix(transform.view, forward, up, camera_position);

		make_orthographic_projection(transform.projection,
				window_width / -240.0f, window_width / 240.0f,
				window_height / -240.0f, window_height / 240.0f,
				-100.0f, 100.0f);

		void* transform_data = (void*) &transform;
		renderer.draw_frame(cb, resized, reset_command_buffers, get_window_dimensions, &ub, &transform_data, 1);
		frame_count++;

		if (stopwatch.milliseconds() >= 1000) {
			elapsed += stopwatch.milliseconds();
			printf("framerate: %lf\n", ((double) frame_count / elapsed) * 1000);
			stopwatch.start();
		}
	}

	renderer.wait_until_idle();
	renderer.delete_sampler(sampler);
	renderer.delete_dynamic_texture_image(texture);
	renderer.delete_descriptor_set_layout(layout);
	renderer.delete_vertex_buffer(vb);
	cleanup_renderer(renderer, pipeline, fb, cb, ub, ub_set, pool);
	cleanup(window, renderer, vertex_shader, fragment_shader);
	return EXIT_SUCCESS;
}

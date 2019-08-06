#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "vulkan_renderer.h"
#include "../simulator.h"

namespace jbw {

using namespace core;
using namespace mirage;

/* forward declaration */
template<typename SimulatorData> class visualizer;

template<typename SimulatorData>
inline void on_framebuffer_resize(GLFWwindow* window, int width, int height) {
	visualizer<SimulatorData>* v = (visualizer<SimulatorData>*) glfwGetWindowUserPointer(window);
	v->resized = true;
}

template<typename SimulatorData>
class visualizer
{
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

	GLFWwindow* window;
	bool resized;
	uint32_t width;
	uint32_t height;
	float camera_position[2];
	float pixel_density;

	simulator<SimulatorData>& sim;

	vulkan_renderer renderer;
	shader vertex_shader, fragment_shader;
	graphics_pipeline pipeline;
	frame_buffer fb;
	command_buffer cb;
	descriptor_set_layout layout;
	descriptor_pool pool;
	descriptor_set set;
	uniform_buffer ub;
	dynamic_texture_image texture;
	sampler tex_sampler;
	vertex_buffer scent_quad_buffer;
	model_view_matrix transform;
	binding_description binding;
	attribute_descriptions<2> attributes;

public:
	visualizer(simulator<SimulatorData>& sim, uint32_t window_width, uint32_t window_height) :
			resized(false), width(window_width), height(window_height), sim(sim), binding(0, sizeof(vertex))
	{
		camera_position[0] = 0.0f;
		camera_position[1] = 0.0f;
		pixel_density = 6.0f;
		transform = { 0 };
		make_identity(transform.model);

		size_t vertex_shader_size = 0;
		char* vertex_shader_src = read_file<true>("vert.spv", vertex_shader_size);
		if (vertex_shader_src == nullptr)
			throw new std::runtime_error("visualizer ERROR: Failed to load vertex shader from file.");

		size_t fragment_shader_size = 0;
		char* fragment_shader_src = read_file<true>("frag.spv", fragment_shader_size);
		if (fragment_shader_src == nullptr) {
			free(vertex_shader_src);
			throw new std::runtime_error("visualizer ERROR: Failed to load fragment shader from file.");
		}

		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		window = glfwCreateWindow(window_width, window_height, "Renderer Test", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, on_framebuffer_resize<SimulatorData>);

		uint32_t extension_count = 0;
		const char** required_extensions = glfwGetRequiredInstanceExtensions(&extension_count);
		if (!init(renderer, "Renderer Test", 0, "no engine", 0,
				required_extensions, extension_count, device_selector::FIRST_ANY,
				glfw_surface(window), window_width, window_height, 2, false))
		{
			free(vertex_shader_src); free(fragment_shader_src);
			throw new std::runtime_error("visualizer ERROR: Failed to initializer renderer.");
		}

		if (!renderer.create_shader(vertex_shader, vertex_shader_src, vertex_shader_size)) {
			free(vertex_shader_src); free(fragment_shader_src);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create vertex shader.");
		} if (!renderer.create_shader(fragment_shader, fragment_shader_src, fragment_shader_size)) {
			renderer.delete_shader(vertex_shader);
			free(vertex_shader_src); free(fragment_shader_src);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create fragment shader.");
		}
		free(vertex_shader_src); free(fragment_shader_src);

		attributes.set<0>(0, 0, attribute_type::FLOAT2, offsetof(vertex, position));
		attributes.set<1>(0, 1, attribute_type::FLOAT2, offsetof(vertex, tex_coord));

		if (!renderer.create_vertex_buffer(scent_quad_buffer, sizeof(vertex) * 4)) {
			renderer.delete_shader(vertex_shader);
			renderer.delete_shader(fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to vertex buffer for scent textured quad.");
		}

		uint32_t binding_indices[] = { 0, 1 };
		descriptor_type types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
		uint32_t descriptor_counts[] = { 1, 1 };
		shader_stage visibilities[] = { shader_stage::VERTEX, shader_stage::FRAGMENT };
		if (!renderer.create_descriptor_set_layout(layout, binding_indices, types, descriptor_counts, visibilities, 2)) {
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_shader(vertex_shader);
			renderer.delete_shader(fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create descriptor_set_layout.");
		}

		size_t image_size = sizeof(pixel) * window_width * window_height;
		if (!renderer.create_dynamic_texture_image(texture, image_size, window_width, window_height)) {
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_shader(vertex_shader);
			renderer.delete_shader(fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create texture.");
		}

		if (!renderer.create_sampler(tex_sampler, filter::NEAREST, filter::NEAREST,
				sampler_address_mode::CLAMP_TO_EDGE, sampler_address_mode::CLAMP_TO_EDGE,
				sampler_address_mode::CLAMP_TO_EDGE, false, 1.0f))
		{
			renderer.delete_dynamic_texture_image(texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_shader(vertex_shader);
			renderer.delete_shader(fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to initialize texture sampler.");
		}

		if (!setup_renderer()) {
			renderer.delete_sampler(tex_sampler);
			renderer.delete_dynamic_texture_image(texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_shader(vertex_shader);
			renderer.delete_shader(fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to initialize rendering pipeline.");
		}
	}

	~visualizer() {
		renderer.wait_until_idle();
		renderer.delete_sampler(tex_sampler);
		renderer.delete_dynamic_texture_image(texture);
		renderer.delete_descriptor_set_layout(layout);
		renderer.delete_vertex_buffer(scent_quad_buffer);
		cleanup_renderer();
		renderer.delete_shader(vertex_shader);
		renderer.delete_shader(fragment_shader);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	inline bool is_window_closed() {
		return (bool) glfwWindowShouldClose(window);
	}

	inline bool draw_frame()
	{
		glfwPollEvents();

		float left = camera_position[0] - 0.5f * (width / pixel_density);
		float right = camera_position[0] + 0.5f * (width / pixel_density);
		float bottom = camera_position[1] - 0.5f * (height / pixel_density);
		float top = camera_position[1] + 0.5f * (height / pixel_density);

		array<array<patch_state>> patches(64);
		if (!sim.get_map({(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, patches)) {
			fprintf(stderr, "visualizer.draw_frame ERROR: Unable to get map from simulator.\n");
			return false;
		}

		pixel* texture_data = (pixel*) texture.mapped_memory;
		unsigned int patch_size = sim.get_config().patch_size;
		unsigned int scent_dimension = sim.get_config().scent_dimension;
		if (patches.length > 0) {
			/* find position of the bottom-left corner and the top-right corner */
			/* TODO: do we need the top-right corner? */
			position bottom_left_corner(0, 0), top_right_corner(0, 0);
			bottom_left_corner.y = patches[0][0].patch_position.y;
			top_right_corner.y = patches.last().last().patch_position.y;
			for (const array<patch_state>& row : patches) {
				bottom_left_corner.x = min(bottom_left_corner.x, row[0].patch_position.x);
				top_right_corner.x = max(top_right_corner.x, row.last().patch_position.x);
			}

			unsigned int texture_width = (unsigned int) (top_right_corner.x - bottom_left_corner.x) * patch_size;
			unsigned int texture_height = (unsigned int) (top_right_corner.y - bottom_left_corner.y) * patch_size;

			unsigned int y_index = 0;
			for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y++) {
				if (y != patches[y_index][0].patch_position.y) {
					/* fill the patches in this row with empty pixels */
					const int64_t offset_y = (y - bottom_left_corner.y) * patch_size;
					for (unsigned int b = 0; b < patch_size; b++) {
						for (unsigned int a = 0; a < texture_width; a++) {
							position texture_position = position(a, b + offset_y);
							pixel& p = texture_data[texture_position.y * width + texture_position.x];
							p.r = 255; p.g = 255; p.b = 255; p.a = 255;
						}
					}
					continue;
				}
				const array<patch_state>& row = patches[y_index++];

				unsigned int x_index = 0;
				for (int64_t x = bottom_left_corner.x; x <= top_right_corner.x; x++) {
					const position offset = (position(x, y) - bottom_left_corner) * patch_size;
					if (x != row[x_index].patch_position.x) {
						/* fill this patch with empty pixels */
						for (unsigned int b = 0; b < patch_size; b++) {
							for (unsigned int a = 0; a < patch_size; a++) {
								position texture_position = position(a, b) + offset;
								pixel& p = texture_data[texture_position.y * width + texture_position.x];
								p.r = 255; p.g = 255; p.b = 255; p.a = 255;
							}
						}
						continue;
					}
					const patch_state& patch = row[x_index++];

					for (unsigned int b = 0; b < patch_size; b++) {
						for (unsigned int a = 0; a < patch_size; a++) {
							position texture_position = position(a, b) + offset;
							pixel& current_pixel = texture_data[texture_position.y * width + texture_position.x];

							float* cell_scent = patch.scent + ((a*patch_size + b)*scent_dimension);
							current_pixel = scent_to_color(cell_scent);
						}
					}
				}
			}
			renderer.transfer_dynamic_texture_image(texture);

			vertex vertices[] = {
				{{(float) bottom_left_corner.x * patch_size, (float) bottom_left_corner.y * patch_size}, {(float) texture_width / width, 0.0f}},
				{{(float) bottom_left_corner.x * patch_size, (float) top_right_corner.y * patch_size}, {(float) texture_width / width, (float) texture_height / height}},
				{{(float) top_right_corner.x * patch_size, (float) bottom_left_corner.x * patch_size}, {0.0f, 0.0f}},
				{{(float) top_right_corner.x * patch_size, (float) top_right_corner.y * patch_size}, {0.0f, (float) texture_height / height}}
			};
			renderer.fill_vertex_buffer(scent_quad_buffer, vertices, sizeof(vertex) * 4);

		} else {
			/* no patches are visible, so move the quad outside the view */
			vertex vertices[] = {
				{{top + 10.0f, top + 10.0f}, {1.0f, 0.0f}},
				{{top + 10.0f, top + 10.0f}, {1.0f, 1.0f}},
				{{top + 10.0f, top + 10.0f}, {0.0f, 1.0f}},
				{{top + 10.0f, top + 10.0f}, {0.0f, 0.0f}}
			};
			renderer.fill_vertex_buffer(scent_quad_buffer, vertices, sizeof(vertex) * 4);
		}

		for (array<patch_state>& row : patches) {
			for (patch_state& patch : row) free(patch);
			free(row);
		}

		/* construct the model view matrix */
		float up[] = { 0.0f, 1.0f, 0.0f };
		float forward[] = { 0.0f, 0.0f, -1.0f };
		float camera_position[] = { 0.0f, 0.0f, 2.0f };
		make_identity(transform.model);
		make_view_matrix(transform.view, forward, up, camera_position);
		make_orthographic_projection(transform.projection, left, right, bottom, top, -100.0f, 100.0f);

		auto reset_command_buffers = [&]() {
			cleanup_renderer();
			return setup_renderer();
		};

		auto get_window_dimensions = [&](uint32_t& out_width, uint32_t& out_height) {
			int new_width, new_height;
			glfwGetFramebufferSize(window, &new_width, &new_height);
			out_width = (uint32_t) new_width;
			out_height = (uint32_t) new_height;
			width = out_width;
			height = out_height;
		};

		void* transform_data = (void*) &transform;
		return renderer.draw_frame(cb, resized, reset_command_buffers, get_window_dimensions, &ub, &transform_data, 1);
	}

private:
	bool setup_renderer() {
		float clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		vertex_buffer vertex_buffers[] = { scent_quad_buffer };
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
		} else if (!renderer.create_descriptor_set(set, &ub, &ub_binding, 1, nullptr, nullptr, 0, &texture, &texture_binding, 1, tex_sampler, layout, pool)
				|| !renderer.create_command_buffer(cb))
		{
			renderer.delete_uniform_buffer(ub);
			renderer.delete_descriptor_pool(pool);
			renderer.delete_frame_buffer(fb);
			renderer.delete_graphics_pipeline(pipeline);
			return false;
		} else if (!renderer.record_command_buffer(cb, fb, pipeline, clear_color, 4, 0, vertex_buffers, offsets, &set, 1)) {
			cleanup_renderer();
			return false;
		}
		return true;
	}

	inline void cleanup_renderer()
	{
		renderer.delete_command_buffer(cb);
		renderer.delete_uniform_buffer(ub);
		renderer.delete_descriptor_pool(pool);
		renderer.delete_frame_buffer(fb);
		renderer.delete_graphics_pipeline(pipeline);
	}

	static inline pixel scent_to_color(const float* cell_scent) {
		float x = max(0.0f, min(1.0f, log(pow(cell_scent[0], 0.4f) + 1.0f) / 0.9f));
		float y = max(0.0f, min(1.0f, log(pow(cell_scent[1], 0.4f) + 1.0f) / 0.9f));
		float z = max(0.0f, min(1.0f, log(pow(cell_scent[2], 0.4f) + 1.0f) / 0.9f));

		pixel out;
		out.r = 255 - (uint8_t) (255 * ((y + z) / 2));
		out.g = 255 - (uint8_t) (255 * ((x + z) / 2));
		out.b = 255 - (uint8_t) (255 * ((x + y) / 2));
		out.a = 255;
		return out;
	}

	static inline void cross(float (&out)[3],
			float (&first)[3], float (&second)[3])
	{
		out[0] = first[1]*second[2] - first[2]*second[1];
		out[1] = first[2]*second[0] - first[0]*second[2];
		out[2] = first[0]*second[1] - first[1]*second[0];
	}

	static inline float dot(float (&first)[3], float(&second)[3]) {
		return first[0]*second[0] + first[1]*second[1] + first[2]*second[2];
	}

	static inline void make_identity(float (&mat)[16]) {
		mat[0] = 1.0f; mat[5] = 1.0f; mat[10] = 1.0f; mat[15] = 1.0f;
	}

	static inline void make_view_matrix(float (&view)[16],
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

	static inline void make_orthographic_projection(float (&proj)[16],
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

	template<typename A> friend void on_framebuffer_resize(GLFWwindow*, int, int);
};

} /* namespace jbw */

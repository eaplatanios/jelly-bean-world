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
inline void cursor_position_callback(GLFWwindow* window, double x, double y)
{
	visualizer<SimulatorData>* v = (visualizer<SimulatorData>*) glfwGetWindowUserPointer(window);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
		v->left_mouse_button_pressed = false;
		return;
	}

	if (!v->left_mouse_button_pressed) {
		v->left_mouse_button_pressed = true;
	} else {
		v->camera_position[0] += (v->last_cursor_x - x) / v->pixel_density;
		v->camera_position[1] -= (v->last_cursor_y - y) / v->pixel_density;
	}

	v->last_cursor_x = x;
	v->last_cursor_y = y;
}

template<typename SimulatorData>
inline void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	visualizer<SimulatorData>* v = (visualizer<SimulatorData>*) glfwGetWindowUserPointer(window);

	if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_MINUS) {
			if (v->target_pixel_density / 1.3 <= 1.0f) {
				/* TODO: handle the case where the pixel density is smaller than 1 (we segfault currently since the texture for the scent visualization could become too small) */
				fprintf(stderr, "Zoom beyond the point where the pixel density is smaller than 1 is unsupported.\n");
			} else {
				v->zoom_animation_start_time = milliseconds();
				v->zoom_start_pixel_density = v->pixel_density;
				v->target_pixel_density /= 1.3;
			}
		} else if (key == GLFW_KEY_EQUAL) {
			v->zoom_animation_start_time = milliseconds();
			v->zoom_start_pixel_density = v->pixel_density;
			v->target_pixel_density *= 1.3;
		}
	}
}

template<typename SimulatorData>
class visualizer
{
	struct vertex {
		float position[2];
		float tex_coord[2];
	};

	struct item_vertex {
		float position[2];
		float color[3];
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

	struct uniform_buffer_data {
		model_view_matrix mvp;
		float pixel_density;
		float patch_size;
	};

	GLFWwindow* window;
	bool resized;
	uint32_t width;
	uint32_t height;
	uint32_t texture_width;
	uint32_t texture_height;
	float camera_position[2];
	float pixel_density;

	simulator<SimulatorData>& sim;

	vulkan_renderer renderer;
	shader background_vertex_shader, background_fragment_shader;
	shader item_vertex_shader, item_fragment_shader;
	render_pass pass;
	graphics_pipeline scent_map_pipeline, item_pipeline;
	frame_buffer fb;
	command_buffer cb;
	descriptor_set_layout layout;
	descriptor_pool pool;
	descriptor_set set;
	uniform_buffer ub;
	dynamic_texture_image scent_map_texture;
	sampler tex_sampler;
	vertex_buffer scent_quad_buffer;
	dynamic_vertex_buffer item_quad_buffer;
	size_t item_quad_buffer_capacity;
	uniform_buffer_data uniform_data;
	binding_description background_binding;
	attribute_descriptions<2> background_shader_attributes;
	binding_description item_binding;
	attribute_descriptions<3> item_shader_attributes;

	bool left_mouse_button_pressed;
	double last_cursor_x, last_cursor_y;

	float zoom_start_pixel_density;
	float target_pixel_density;
	unsigned long long zoom_animation_start_time;

public:
	visualizer(simulator<SimulatorData>& sim, uint32_t window_width, uint32_t window_height) :
			resized(false), width(window_width), height(window_height), sim(sim),
			background_binding(0, sizeof(vertex)), item_binding(0, sizeof(item_vertex))
	{
		camera_position[0] = 0.0f;
		camera_position[1] = 0.0f;
		pixel_density = 6.0f;
		target_pixel_density = pixel_density;
		zoom_start_pixel_density = pixel_density;
		zoom_animation_start_time = milliseconds();
		uniform_data = {{{0},{0},{0}}, 0};
		make_identity(uniform_data.mvp.model);

		size_t vertex_shader_size = 0;
		char* vertex_shader_src = read_file<true>("background_vertex_shader.spv", vertex_shader_size);
		if (vertex_shader_src == nullptr)
			throw new std::runtime_error("visualizer ERROR: Failed to load vertex shader from file.");

		size_t fragment_shader_size = 0;
		char* fragment_shader_src = read_file<true>("background_fragment_shader.spv", fragment_shader_size);
		if (fragment_shader_src == nullptr) {
			free(vertex_shader_src);
			throw new std::runtime_error("visualizer ERROR: Failed to load fragment shader from file.");
		}

		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		window = glfwCreateWindow(window_width, window_height, "Renderer Test", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, on_framebuffer_resize<SimulatorData>);
		glfwSetCursorPosCallback(window, cursor_position_callback<SimulatorData>);
		glfwSetKeyCallback(window, key_callback<SimulatorData>);

		uint32_t extension_count = 0;
		const char** required_extensions = glfwGetRequiredInstanceExtensions(&extension_count);
		if (!init(renderer, "JBW Visualizer", 0, "no engine", 0,
				required_extensions, extension_count, device_selector::FIRST_ANY,
				glfw_surface(window), window_width, window_height, 2, false, false, true))
		{
			free(vertex_shader_src); free(fragment_shader_src);
			throw new std::runtime_error("visualizer ERROR: Failed to initializer renderer.");
		}

		if (!renderer.create_shader(background_vertex_shader, vertex_shader_src, vertex_shader_size)) {
			free(vertex_shader_src); free(fragment_shader_src);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create vertex shader.");
		} if (!renderer.create_shader(background_fragment_shader, fragment_shader_src, fragment_shader_size)) {
			renderer.delete_shader(background_vertex_shader);
			free(vertex_shader_src); free(fragment_shader_src);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create fragment shader.");
		}
		free(vertex_shader_src); free(fragment_shader_src);

		background_shader_attributes.set<0>(0, 0, attribute_type::FLOAT2, offsetof(vertex, position));
		background_shader_attributes.set<1>(0, 1, attribute_type::FLOAT2, offsetof(vertex, tex_coord));

		vertex_shader_src = read_file<true>("item_vertex_shader.spv", vertex_shader_size);
		if (vertex_shader_src == nullptr) {
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to load vertex shader from file.");
		}

		fragment_shader_src = read_file<true>("item_fragment_shader.spv", fragment_shader_size);
		if (fragment_shader_src == nullptr) {
			free(vertex_shader_src);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to load fragment shader from file.");
		}

		if (!renderer.create_shader(item_vertex_shader, vertex_shader_src, vertex_shader_size)) {
			free(vertex_shader_src); free(fragment_shader_src);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create vertex shader.");
		} if (!renderer.create_shader(item_fragment_shader, fragment_shader_src, fragment_shader_size)) {
			free(vertex_shader_src); free(fragment_shader_src);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create fragment shader.");
		}
		free(vertex_shader_src); free(fragment_shader_src);

		item_shader_attributes.set<0>(0, 0, attribute_type::FLOAT2, offsetof(item_vertex, position));
		item_shader_attributes.set<1>(0, 1, attribute_type::FLOAT3, offsetof(item_vertex, color));
		item_shader_attributes.set<2>(0, 2, attribute_type::FLOAT2, offsetof(item_vertex, tex_coord));

		if (!renderer.create_vertex_buffer(scent_quad_buffer, sizeof(vertex) * 4)) {
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to vertex buffer for scent textured quad.");
		}

		item_quad_buffer_capacity = 4 * width * height;
		if (!renderer.create_dynamic_vertex_buffer(item_quad_buffer, item_quad_buffer_capacity * sizeof(item_vertex))) {
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to vertex buffer for scent textured quad.");
		}

		uint32_t binding_indices[] = { 0, 1 };
		descriptor_type types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
		uint32_t descriptor_counts[] = { 1, 1 };
		shader_stage visibilities[] = { shader_stage::ALL, shader_stage::FRAGMENT };
		if (!renderer.create_descriptor_set_layout(layout, binding_indices, types, descriptor_counts, visibilities, 2)) {
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create descriptor_set_layout.");
		}

		texture_width = window_width + 2 * sim.get_config().patch_size;
		texture_height = window_height + 2 * sim.get_config().patch_size;
		size_t image_size = sizeof(pixel) * texture_width * texture_height;
		if (!renderer.create_dynamic_texture_image(scent_map_texture, image_size, texture_width, texture_height, image_format::R8G8B8A8_UNORM)) {
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create `scent_map_texture`.");
		}

		if (!renderer.create_sampler(tex_sampler, filter::NEAREST, filter::NEAREST,
				sampler_address_mode::CLAMP_TO_EDGE, sampler_address_mode::CLAMP_TO_EDGE,
				sampler_address_mode::CLAMP_TO_EDGE, false, 1.0f))
		{
			renderer.delete_dynamic_texture_image(scent_map_texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to initialize texture sampler.");
		}

		if (!setup_renderer()) {
			renderer.delete_sampler(tex_sampler);
			renderer.delete_dynamic_texture_image(scent_map_texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to initialize rendering pipeline.");
		}
	}

	~visualizer() {
		renderer.wait_until_idle();
		renderer.delete_sampler(tex_sampler);
		renderer.delete_dynamic_texture_image(scent_map_texture);
		renderer.delete_descriptor_set_layout(layout);
		renderer.delete_vertex_buffer(scent_quad_buffer);
		renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
		cleanup_renderer();
		renderer.delete_shader(item_vertex_shader);
		renderer.delete_shader(item_fragment_shader);
		renderer.delete_shader(background_vertex_shader);
		renderer.delete_shader(background_fragment_shader);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	inline bool is_window_closed() {
		return (bool) glfwWindowShouldClose(window);
	}

	inline bool draw_frame()
	{
		glfwPollEvents();

		/* compute `pixel_density` according to the zoom animation */
		float animation_t = max(0.0f, min(1.0f, (milliseconds() - zoom_animation_start_time) / 300.0f));
		float easing = animation_t * (2 - animation_t);
		pixel_density = easing * target_pixel_density + (1.0f - easing) * zoom_start_pixel_density;

		float left = camera_position[0] - 0.5f * (width / pixel_density);
		float right = camera_position[0] + 0.5f * (width / pixel_density);
		float bottom = camera_position[1] - 0.5f * (height / pixel_density);
		float top = camera_position[1] + 0.5f * (height / pixel_density);

		array<array<patch_state>> patches(64);
		if (sim.get_map({(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, patches) != status::OK) {
			fprintf(stderr, "visualizer.draw_frame ERROR: Unable to get map from simulator.\n");
			return false;
		}

		unsigned int item_vertex_count = 0;
		pixel* scent_map_texture_data = (pixel*) scent_map_texture.mapped_memory;
		const unsigned int patch_size = sim.get_config().patch_size;
		const unsigned int scent_dimension = sim.get_config().scent_dimension;
		const array<item_properties>& item_types = sim.get_config().item_types;
		const float* agent_color = sim.get_config().agent_color;
		if (patches.length > 0) {
			/* find position of the bottom-left corner and the top-right corner */
			size_t required_item_vertices = 0;
			position bottom_left_corner(INT64_MAX, INT64_MAX), top_right_corner(INT64_MIN, INT64_MIN);
			bottom_left_corner.y = patches[0][0].patch_position.y;
			top_right_corner.y = patches.last().last().patch_position.y;
			for (const array<patch_state>& row : patches) {
				bottom_left_corner.x = min(bottom_left_corner.x, row[0].patch_position.x);
				top_right_corner.x = max(top_right_corner.x, row.last().patch_position.x);
				for (const patch_state& patch : row)
					required_item_vertices += 6 * patch.item_count + 3 * patch.agent_count;
			}

			if (required_item_vertices > item_quad_buffer_capacity) {
				size_t new_capacity = 2 * item_quad_buffer_capacity;
				while (required_item_vertices > new_capacity)
					new_capacity *= 2;
				renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
				if (!renderer.create_dynamic_vertex_buffer(item_quad_buffer, new_capacity * sizeof(item_vertex))) {
					fprintf(stderr, "visualizer.draw_frame ERROR: Unable to expand `item_quad_buffer`.\n");
					for (array<patch_state>& row : patches) {
						for (patch_state& patch : row) free(patch);
						free(row);
					}
					return false;
				}
			}

			unsigned int texture_width_cells = (unsigned int) (top_right_corner.x - bottom_left_corner.x + 1) * patch_size;
			unsigned int texture_height_cells = (unsigned int) (top_right_corner.y - bottom_left_corner.y + 1) * patch_size;

			unsigned int y_index = 0;
			item_vertex* item_vertices = (item_vertex*) item_quad_buffer.mapped_memory;
			for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y++) {
				if (y_index == patches.length || y != patches[y_index][0].patch_position.y) {
					/* fill the patches in this row with empty pixels */
					const int64_t patch_offset_y = y - bottom_left_corner.y;
					for (unsigned int a = 0; a <= top_right_corner.x - bottom_left_corner.x; a++) {
						pixel& p = scent_map_texture_data[patch_offset_y * texture_width + a];
						p.a = 255;
					}

					const int64_t offset_y = patch_offset_y * patch_size;
					for (unsigned int b = 0; b < patch_size; b++) {
						for (unsigned int a = 0; a < texture_width_cells; a++) {
							position texture_position = position(a, b + offset_y);
							pixel& p = scent_map_texture_data[texture_position.y * texture_width + texture_position.x];
							p.r = 255; p.g = 255; p.b = 255;
						}
					}
					continue;
				}
				const array<patch_state>& row = patches[y_index++];

				unsigned int x_index = 0;
				for (int64_t x = bottom_left_corner.x; x <= top_right_corner.x; x++) {
					const position patch_offset = position(x, y) - bottom_left_corner;
					const position offset = patch_offset * patch_size;
					if (x_index == row.length || x != row[x_index].patch_position.x) {
						/* fill this patch with empty pixels */
						pixel& p = scent_map_texture_data[patch_offset.y * texture_width + patch_offset.x];
						p.a = 255;

						for (unsigned int b = 0; b < patch_size; b++) {
							for (unsigned int a = 0; a < patch_size; a++) {
								position texture_position = position(a, b) + offset;
								pixel& p = scent_map_texture_data[texture_position.y * texture_width + texture_position.x];
								p.r = 255; p.g = 255; p.b = 255;
							}
						}
						continue;
					}
					const patch_state& patch = row[x_index++];

					pixel& p = scent_map_texture_data[patch_offset.y * texture_width + patch_offset.x];
					if (patch.fixed) {
						p.a = 178;
					} else {
						p.a = 229;
					}

					for (unsigned int b = 0; b < patch_size; b++) {
						for (unsigned int a = 0; a < patch_size; a++) {
							position texture_position = position(a, b) + offset;
							pixel& current_pixel = scent_map_texture_data[texture_position.y * texture_width + texture_position.x];

							float* cell_scent = patch.scent + ((a*patch_size + b)*scent_dimension);
							scent_to_color(cell_scent, current_pixel);
						}
					}

					/* iterate over all items in this patch, creating a quad
					   for each (we use a triangle list so we need two triangles) */
					for (unsigned int i = 0; i < patch.item_count; i++) {
						const item& it = patch.items[i];
						const item_properties& item_props = item_types[it.item_type];
						item_vertices[item_vertex_count].position[0] = it.location.x + 0.5f - 0.4f;
						item_vertices[item_vertex_count].position[1] = it.location.y + 0.5f - 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j] + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j];
						}
						item_vertices[item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[item_vertex_count].position[0] = it.location.x + 0.5f - 0.4f;
						item_vertices[item_vertex_count].position[1] = it.location.y + 0.5f + 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j] + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j];
						}
						item_vertices[item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 1.0f;

						item_vertices[item_vertex_count].position[0] = it.location.x + 0.5f + 0.4f;
						item_vertices[item_vertex_count].position[1] = it.location.y + 0.5f - 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j] + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j];
						}
						item_vertices[item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[item_vertex_count].position[0] = it.location.x + 0.5f + 0.4f;
						item_vertices[item_vertex_count].position[1] = it.location.y + 0.5f + 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j] + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j];
						}
						item_vertices[item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 1.0f;

						item_vertices[item_vertex_count].position[0] = it.location.x + 0.5f + 0.4f;
						item_vertices[item_vertex_count].position[1] = it.location.y + 0.5f - 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j] + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j];
						}
						item_vertices[item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[item_vertex_count].position[0] = it.location.x + 0.5f - 0.4f;
						item_vertices[item_vertex_count].position[1] = it.location.y + 0.5f + 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j] + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = item_props.color[j];
						}
						item_vertices[item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 1.0f;
					}

					/* iterate over all agents in this patch, creating an oriented triangle for each */
					for (unsigned int i = 0; i < patch.agent_count; i++) {
						float first[2] = {0};
						float second[2] = {0};
						float third[2] = {0};
						get_triangle_coords(patch.agent_directions[i], first, second, third);

						item_vertices[item_vertex_count].position[0] = patch.agent_positions[i].x + 0.5f + first[0];
						item_vertices[item_vertex_count].position[1] = patch.agent_positions[i].y + 0.5f + first[1];
						for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = agent_color[j] + 4.0f;
						item_vertices[item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[item_vertex_count].position[0] = patch.agent_positions[i].x + 0.5f + second[0];
						item_vertices[item_vertex_count].position[1] = patch.agent_positions[i].y + 0.5f + second[1];
						for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = agent_color[j] + 4.0f;
						item_vertices[item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[item_vertex_count].position[0] = patch.agent_positions[i].x + 0.5f + third[0];
						item_vertices[item_vertex_count].position[1] = patch.agent_positions[i].y + 0.5f + third[1];
						for (unsigned int j = 0; j < 3; j++) item_vertices[item_vertex_count].color[j] = agent_color[j] + 4.0f;
						item_vertices[item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[item_vertex_count++].tex_coord[1] = 1.0f;
					}
				}
			}
			renderer.transfer_dynamic_texture_image(scent_map_texture, image_format::R8G8B8A8_UNORM);

			/* position the background quad */
			vertex vertices[] = {
				{{(float) bottom_left_corner.x * patch_size, (float) bottom_left_corner.y * patch_size}, {0.0f, 0.0f}},
				{{(float) bottom_left_corner.x * patch_size, (float) (top_right_corner.y + 1) * patch_size}, {0.0f, (float) texture_height_cells / texture_height}},
				{{(float) (top_right_corner.x + 1) * patch_size, (float) bottom_left_corner.y * patch_size}, {(float) texture_width_cells / texture_width, 0.0f}},
				{{(float) (top_right_corner.x + 1) * patch_size, (float) (top_right_corner.y + 1) * patch_size}, {(float) texture_width_cells / texture_width, (float) texture_height_cells / texture_height}}
			};

			/* transfer all data to GPU */
			renderer.fill_vertex_buffer(scent_quad_buffer, vertices, sizeof(vertex) * 4);
			renderer.transfer_dynamic_vertex_buffer(item_quad_buffer, sizeof(item_vertex) * item_vertex_count);

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
		float camera_pos[] = { camera_position[0], camera_position[1], 2.0f };
		make_identity(uniform_data.mvp.model);
		make_view_matrix(uniform_data.mvp.view, forward, up, camera_pos);
		make_orthographic_projection(uniform_data.mvp.projection,
				-0.5f * (width / pixel_density), 0.5f * (width / pixel_density),
				-0.5f * (height / pixel_density), 0.5f * (height / pixel_density), -100.0f, 100.0f);
		uniform_data.pixel_density = pixel_density;
		uniform_data.patch_size = patch_size;

		auto reset_command_buffers = [&]() {
			cleanup_renderer();
			renderer.delete_dynamic_texture_image(scent_map_texture);

			texture_width = width + 2 * sim.get_config().patch_size;
			texture_height = height + 2 * sim.get_config().patch_size;
			size_t image_size = sizeof(pixel) * texture_width * texture_height;
			if (!renderer.create_dynamic_texture_image(scent_map_texture, image_size, texture_width, texture_height, image_format::R8G8B8A8_UNORM))
				return false;

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

		draw_call<1, 0, 1> draw_scent_map;
		draw_scent_map.pipeline = scent_map_pipeline;
		draw_scent_map.first_vertex = 0;
		draw_scent_map.vertex_count = 4;
		draw_scent_map.vertex_buffers[0] = scent_quad_buffer;
		draw_scent_map.vertex_buffer_offsets[0] = 0;
		draw_scent_map.descriptor_sets[0] = set;

		draw_call<0, 1, 0> draw_items;
		draw_items.pipeline = item_pipeline;
		draw_items.first_vertex = 0;
		draw_items.vertex_count = item_vertex_count;
		draw_items.dynamic_vertex_buffers[0] = item_quad_buffer;
		draw_items.dynamic_vertex_buffer_offsets[0] = 0;

		float clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		if (!renderer.record_command_buffer(cb, fb, clear_color, pass, draw_scent_map, draw_items)) {
			cleanup_renderer();
			return false;
		}

		void* pv_uniform_data = (void*) &uniform_data;
		return renderer.draw_frame(cb, resized, reset_command_buffers, get_window_dimensions, &ub, &pv_uniform_data, 1);
	}

private:
	bool setup_renderer() {
		dynamic_texture_image dynamic_textures[] = { scent_map_texture };
		uint32_t texture_bindings[] = { 1 };
		uint32_t ub_binding = 0;
		descriptor_type pool_types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
		if (!renderer.create_render_pass(pass)) {
			return false;
		} else if (!renderer.create_graphics_pipeline(scent_map_pipeline, pass,
				background_vertex_shader, "main", background_fragment_shader, "main",
				primitive_topology::TRIANGLE_STRIP, false, 1.0f, background_binding, background_shader_attributes, &layout, 1))
		{
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_graphics_pipeline(item_pipeline, pass,
				item_vertex_shader, "main", item_fragment_shader, "main",
				primitive_topology::TRIANGLE_LIST, false, 1.0f, item_binding, item_shader_attributes, &layout, 1))
		{
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_frame_buffer(fb, pass)) {
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_uniform_buffer(ub, sizeof(uniform_buffer_data))) {
			renderer.delete_frame_buffer(fb);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_descriptor_pool(pool, pool_types, 2)) {
			renderer.delete_uniform_buffer(ub);
			renderer.delete_frame_buffer(fb);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_descriptor_set(set, &ub, &ub_binding, 1, nullptr, nullptr, 0, dynamic_textures, texture_bindings, 1, &tex_sampler, layout, pool)
				|| !renderer.create_command_buffer(cb))
		{
			renderer.delete_uniform_buffer(ub);
			renderer.delete_descriptor_pool(pool);
			renderer.delete_frame_buffer(fb);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		}
		return true;
	}

	inline void cleanup_renderer()
	{
		renderer.delete_command_buffer(cb);
		renderer.delete_uniform_buffer(ub);
		renderer.delete_descriptor_set(set);
		renderer.delete_descriptor_pool(pool);
		renderer.delete_frame_buffer(fb);
		renderer.delete_graphics_pipeline(scent_map_pipeline);
		renderer.delete_graphics_pipeline(item_pipeline);
		renderer.delete_render_pass(pass);
	}

	static inline void scent_to_color(const float* cell_scent, pixel& out) {
		float x = max(0.0f, min(1.0f, log(pow(cell_scent[0], 0.4f) + 1.0f) / 0.9f));
		float y = max(0.0f, min(1.0f, log(pow(cell_scent[1], 0.4f) + 1.0f) / 0.9f));
		float z = max(0.0f, min(1.0f, log(pow(cell_scent[2], 0.4f) + 1.0f) / 0.9f));

		out.r = 255 - (uint8_t) (255 * ((y + z) / 2));
		out.g = 255 - (uint8_t) (255 * ((x + z) / 2));
		out.b = 255 - (uint8_t) (255 * ((x + y) / 2));
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

	inline void get_triangle_coords(direction dir, float (&first)[2], float (&second)[2], float(&third)[2])
	{
		switch (dir) {
		case direction::UP:
			first[0] = 0.0f;			first[1] = 0.5f - 0.1f;
			second[0] = 0.43301f;		second[1] = -0.25f - 0.1f;
			third[0] = -0.43301f;		third[1] = -0.25f - 0.1f; return;
		case direction::DOWN:
			first[0] = 0.0f;			first[1] = -0.5f + 0.1f;
			second[0] = -0.43301f;		second[1] = 0.25f + 0.1f;
			third[0] = 0.43301f;		third[1] = 0.25f + 0.1f; return;
		case direction::LEFT:
			first[0] = -0.5f + 0.1f;	first[1] = 0.0f;
			second[0] = 0.25f + 0.1f;	second[1] = 0.43301f;
			third[0] = 0.25f + 0.1f;	third[1] = -0.43301f; return;
		case direction::RIGHT:
			first[0] = 0.5f - 0.1f;		first[1] = 0.0f;
			second[0] = -0.25f - 0.1f;	second[1] = -0.43301f;
			third[0] = -0.25f - 0.1f;	third[1] = 0.43301f; return;
		case direction::COUNT: break;
		}
	}

	template<typename A> friend void on_framebuffer_resize(GLFWwindow*, int, int);
	template<typename A> friend void cursor_position_callback(GLFWwindow*, double, double);
	template<typename A> friend void key_callback(GLFWwindow*, int, int, int, int);
};

} /* namespace jbw */

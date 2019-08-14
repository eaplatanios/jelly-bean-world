#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "vulkan_renderer.h"
#include "../mpi.h"

#include <thread>
#include <condition_variable>

namespace jbw {

using namespace core;
using namespace mirage;

/* forward declaration */
template<typename SimulatorType> class visualizer;

template<typename SimulatorType>
inline void cursor_position_callback(GLFWwindow* window, double x, double y)
{
	visualizer<SimulatorType>* v = (visualizer<SimulatorType>*) glfwGetWindowUserPointer(window);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
		v->left_mouse_button_pressed = false;
		return;
	}

	if (!v->left_mouse_button_pressed) {
		v->left_mouse_button_pressed = true;
	} else {
		v->track_agent_id = 0;
		v->camera_position[0] += (v->last_cursor_x - x) / v->pixel_density;
		v->camera_position[1] -= (v->last_cursor_y - y) / v->pixel_density;
		v->translate_start_position[0] = v->camera_position[0];
		v->translate_start_position[1] = v->camera_position[1];
	}

	v->last_cursor_x = x;
	v->last_cursor_y = y;
}

template<typename SimulatorType>
inline void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	visualizer<SimulatorType>* v = (visualizer<SimulatorType>*) glfwGetWindowUserPointer(window);

	if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_MINUS) {
			if (v->target_pixel_density / 1.3 <= 1 / get_config(v->sim).patch_size) {
				/* TODO: handle the case where the pixel density is smaller than 1 (we segfault currently since the texture for the scent visualization could become too small) */
				fprintf(stderr, "Zoom beyond the point where the pixel density is smaller than 1/patch_size is unsupported.\n");
			} else {
				v->zoom_animation_start_time = milliseconds();
				v->zoom_start_pixel_density = v->pixel_density;
				v->target_pixel_density /= 1.3;
			}
		} else if (key == GLFW_KEY_EQUAL) {
			v->zoom_animation_start_time = milliseconds();
			v->zoom_start_pixel_density = v->pixel_density;
			v->target_pixel_density *= 1.3;
		} else if (key == GLFW_KEY_0) {
			v->tracking_animating = false;
			v->track_agent_id = 0;
		} else if (key == GLFW_KEY_1) {
			v->tracking_animating = false;
			v->track_agent_id = 1;
		} else if (key == GLFW_KEY_2) {
			v->tracking_animating = false;
			v->track_agent_id = 2;
		} else if (key == GLFW_KEY_3) {
			v->tracking_animating = false;
			v->track_agent_id = 3;
		} else if (key == GLFW_KEY_4) {
			v->tracking_animating = false;
			v->track_agent_id = 4;
		} else if (key == GLFW_KEY_5) {
			v->tracking_animating = false;
			v->track_agent_id = 5;
		} else if (key == GLFW_KEY_6) {
			v->tracking_animating = false;
			v->track_agent_id = 6;
		} else if (key == GLFW_KEY_7) {
			v->tracking_animating = false;
			v->track_agent_id = 7;
		} else if (key == GLFW_KEY_8) {
			v->tracking_animating = false;
			v->track_agent_id = 8;
		} else if (key == GLFW_KEY_9) {
			v->tracking_animating = false;
			v->track_agent_id = 9;
		} else if (key == GLFW_KEY_B) {
			v->render_background = !v->render_background;
		}
	}
}

struct visualizer_client_data {
	visualizer<client<visualizer_client_data>>* painter = nullptr;

	std::atomic_bool waiting_for_get_map;
	float get_map_left;
	float get_map_right;
	float get_map_bottom;
	float get_map_top;
	bool get_map_render_background = true;

	status get_map_response = status::OK;
	array<array<patch_state>>* map = nullptr;

	std::atomic_bool waiting_for_get_agent_states;
	uint64_t track_agent_id = 0;

	status get_agent_states_response = status::OK;
	const agent_state* agent_states = nullptr;
	size_t agent_state_count = 0;

	visualizer_client_data() {
		waiting_for_get_map.store(false);
		waiting_for_get_agent_states.store(false);
	}

	visualizer_client_data(const visualizer_client_data& src) :
		get_map_left(src.get_map_left), get_map_right(src.get_map_right),
		get_map_bottom(src.get_map_bottom), get_map_top(src.get_map_top),
		get_map_render_background(src.get_map_render_background),
		get_map_response(src.get_map_response), map(src.map),
		track_agent_id(src.track_agent_id),
		get_agent_states_response(src.get_agent_states_response),
		agent_states(src.agent_states), agent_state_count(src.agent_state_count)
	{
		waiting_for_get_map.store(src.waiting_for_get_map.load());
		waiting_for_get_agent_states.store(src.waiting_for_get_agent_states.load());
	}
};

template <typename SimulatorData>
const simulator_config& get_config(const simulator<SimulatorData>& sim) {
	return sim.get_config();
}

template <typename ClientData>
const simulator_config& get_config(const client<ClientData>& c) {
	return c.config;
}

template<typename SimulatorType>
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
		float patch_size_texels;
	};

	GLFWwindow* window;
	uint32_t width;
	uint32_t height;
	uint32_t texture_width;
	uint32_t texture_height;
	float camera_position[2];
	float pixel_density;
	unsigned int current_patch_size_texels;

	SimulatorType& sim;

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
	size_t item_vertex_count;
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

	/* this is zero if we don't want to track anyone */
	float translate_start_position[2];
	float translate_end_position[2];
	unsigned long long translate_animation_start_time;
	uint64_t track_agent_id;
	bool tracking_animating;

	/* the thread that calls `get_map` so we don't need to
	   do so in `draw_frame`, which keeps everything smooth */
	std::thread map_retriever;
	std::mutex scene_lock;
	std::condition_variable scene_ready_cv;
	std::atomic_bool scene_ready;
	float left_bound, right_bound, bottom_bound, top_bound;
	bool render_background;

public:
	std::atomic_bool running;

	visualizer(SimulatorType& sim,
		uint32_t window_width, uint32_t window_height,
		uint64_t track_agent_id, float pixels_per_cell) :
			width(window_width), height(window_height), sim(sim),
			background_binding(0, sizeof(vertex)), item_binding(0, sizeof(item_vertex)),
			track_agent_id(track_agent_id), tracking_animating(false),
			scene_ready(false), render_background(true), running(true)
	{
		camera_position[0] = 0.5f;
		camera_position[1] = 0.5f;
		translate_end_position[0] = camera_position[0];
		translate_end_position[1] = camera_position[1];
		pixel_density = pixels_per_cell;
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
		glfwSetCursorPosCallback(window, cursor_position_callback<SimulatorType>);
		glfwSetKeyCallback(window, key_callback<SimulatorType>);

		// We need to get the actual framebuffer width and height because HiDPI sometimes scale the
		// actual framebuffer size relative to the window size.
		int framebuffer_width, framebuffer_height;
		glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
		width = (uint32_t) framebuffer_width;
		height = (uint32_t) framebuffer_height;

		uint32_t extension_count = 0;
		const char** required_extensions = glfwGetRequiredInstanceExtensions(&extension_count);
		if (!init(renderer, "JBW Visualizer", 0, "no engine", 0,
				required_extensions, extension_count, device_selector::FIRST_ANY,
				glfw_surface(window), width, height, 2, false, false, true))
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

		texture_width = window_width + 2 * get_config(sim).patch_size;
		texture_height = window_height + 2 * get_config(sim).patch_size;
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

		prepare_scene(sim);

		if (track_agent_id != 0) {
			/* since we're just starting, move camera immediately to target agent */
			translate_start_position[0] = translate_end_position[0];
			translate_start_position[1] = translate_end_position[1];
			camera_position[0] = translate_end_position[0];
			camera_position[1] = translate_end_position[1];
		}

		map_retriever = std::thread([&,this]() {
			run_map_retriever(sim);
		});
	}

	~visualizer() {
		running = false;
		scene_ready_cv.notify_one();
		if (map_retriever.joinable()) {
			try {
				map_retriever.join();
			} catch (...) { }
		}
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

	inline void track_agent(uint64_t agent_id) {
		tracking_animating = false;
		track_agent_id = agent_id;
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

		if (tracking_animating) {
			float animation_t = max(0.0f, min(1.0f, (milliseconds() - translate_animation_start_time) / 300.0f));
			float easing = (animation_t - 1) * (animation_t - 1) * (animation_t - 1) + 1;
			camera_position[0] = easing * translate_end_position[0] + (1.0f - easing) * translate_start_position[0];
			camera_position[1] = easing * translate_end_position[1] + (1.0f - easing) * translate_start_position[1];
		}

		float left = camera_position[0] - 0.5f * (width / pixel_density);
		float right = camera_position[0] + 0.5f * (width / pixel_density);
		float bottom = camera_position[1] - 0.5f * (height / pixel_density);
		float top = camera_position[1] + 0.5f * (height / pixel_density);
		{
			while (running && (left < left_bound || right > right_bound || bottom < bottom_bound || top > top_bound))
				scene_ready = false;
			if (!running) return true;
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
		uniform_data.patch_size_texels = current_patch_size_texels;

		auto reset_command_buffers = [&]() {
			cleanup_renderer();
			renderer.delete_dynamic_texture_image(scent_map_texture);

			texture_width = width + 2 * get_config(sim).patch_size;
			texture_height = height + 2 * get_config(sim).patch_size;
			size_t image_size = sizeof(pixel) * texture_width * texture_height;
			if (!renderer.create_dynamic_texture_image(scent_map_texture, image_size, texture_width, texture_height, image_format::R8G8B8A8_UNORM))
				return false;

			if (!setup_renderer()) return false;

			return prepare_scene(sim);
		};

		auto get_window_dimensions = [&](uint32_t& out_width, uint32_t& out_height) {
			int new_width, new_height;
			glfwGetFramebufferSize(window, &new_width, &new_height);
			out_width = (uint32_t) new_width;
			out_height = (uint32_t) new_height;
			width = out_width;
			height = out_height;
		};

		std::unique_lock<std::mutex> lck(scene_lock);
		void* pv_uniform_data = (void*) &uniform_data;
		bool result = renderer.draw_frame(cb, reset_command_buffers, get_window_dimensions, &ub, &pv_uniform_data, 1);
		scene_ready = false;
		scene_ready_cv.notify_one();
		return result;
	}

private:
	template<bool HasLock>
	bool prepare_scene_helper(
			const array<array<patch_state>>& patches,
			bool render_background_map,
			float left, float right, float bottom, float top)
	{
		const unsigned int texel_cell_length = (unsigned int) ceil(1 / pixel_density);

		size_t new_item_vertex_count = 0;
		pixel* scent_map_texture_data = (pixel*) scent_map_texture.mapped_memory;
		const unsigned int patch_size = get_config(sim).patch_size;
		const unsigned int patch_size_texels = (unsigned int) ceil((float) patch_size / texel_cell_length);
		const unsigned int scent_dimension = get_config(sim).scent_dimension;
		const array<item_properties>& item_types = get_config(sim).item_types;
		const float* agent_color = get_config(sim).agent_color;
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
				if (!HasLock) scene_lock.lock();
				renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
				if (!renderer.create_dynamic_vertex_buffer(item_quad_buffer, new_capacity * sizeof(item_vertex))) {
					fprintf(stderr, "visualizer.prepare_scene_helper ERROR: Unable to expand `item_quad_buffer`.\n");
					if (!HasLock) scene_lock.unlock();
					return false;
				}
				if (!HasLock) scene_lock.unlock();
			}

			unsigned int texture_width_cells = (unsigned int) (top_right_corner.x - bottom_left_corner.x + 1) * patch_size_texels;
			unsigned int texture_height_cells = (unsigned int) (top_right_corner.y - bottom_left_corner.y + 1) * patch_size_texels;

			unsigned int y_index = 0;
			item_vertex* item_vertices = (item_vertex*) item_quad_buffer.mapped_memory;
			for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y++) {
				if (y_index == patches.length || y != patches[y_index][0].patch_position.y) {
					/* fill the patches in this row with empty pixels */
					if (render_background_map) {
						const int64_t patch_offset_y = y - bottom_left_corner.y;
						for (unsigned int a = 0; a <= top_right_corner.x - bottom_left_corner.x; a++) {
							pixel& p = scent_map_texture_data[patch_offset_y * texture_width + a];
							p.a = 255;
						}

						const int64_t offset_y = patch_offset_y * patch_size_texels;
						for (unsigned int b = 0; b < patch_size_texels; b++) {
							for (unsigned int a = 0; a < texture_width_cells; a++) {
								position texture_position = position(a, b + offset_y);
								pixel& p = scent_map_texture_data[texture_position.y * texture_width + texture_position.x];
								p.r = 0; p.g = 0; p.b = 0;
							}
						}
					}
					continue;
				}
				const array<patch_state>& row = patches[y_index++];

				unsigned int x_index = 0;
				for (int64_t x = bottom_left_corner.x; x <= top_right_corner.x; x++) {
					const position patch_offset = position(x, y) - bottom_left_corner;
					const position offset = patch_offset * patch_size_texels;
					if (x_index == row.length || x != row[x_index].patch_position.x) {
						/* fill this patch with empty pixels */
						if (render_background_map) {
							pixel& p = scent_map_texture_data[patch_offset.y * texture_width + patch_offset.x];
							p.a = 255;

							for (unsigned int b = 0; b < patch_size_texels; b++) {
								for (unsigned int a = 0; a < patch_size_texels; a++) {
									position texture_position = position(a, b) + offset;
									pixel& p = scent_map_texture_data[texture_position.y * texture_width + texture_position.x];
									p.r = 0; p.g = 0; p.b = 0;
								}
							}
						}
						continue;
					}
					const patch_state& patch = row[x_index++];

					if (render_background_map) {
						pixel& p = scent_map_texture_data[patch_offset.y * texture_width + patch_offset.x];
						p.a = 229;

						for (unsigned int b = 0; b < patch_size_texels; b++) {
							for (unsigned int a = 0; a < patch_size_texels; a++) {
								/* first average the scent across the cells in this texel */
								float average_scent[3] = { 0 };
								unsigned int cell_count = 0;
								for (unsigned int a_inner = 0; a_inner < texel_cell_length; a_inner++) {
									if (a*texel_cell_length + a_inner == patch_size) break;
									for (unsigned int b_inner = 0; b_inner < texel_cell_length; b_inner++) {
										if (b*texel_cell_length + b_inner == patch_size) break;
										float* cell_scent = patch.scent + (((a*texel_cell_length + a_inner)*patch_size + b*texel_cell_length + b_inner)*scent_dimension);
										average_scent[0] += cell_scent[0];
										average_scent[1] += cell_scent[1];
										average_scent[2] += cell_scent[2];
										cell_count++;
									}
								}

								average_scent[0] /= cell_count;
								average_scent[1] /= cell_count;
								average_scent[2] /= cell_count;

								position texture_position = position(a, b) + offset;
								pixel& current_pixel = scent_map_texture_data[texture_position.y * texture_width + texture_position.x];
								scent_to_color(average_scent, current_pixel, patch.fixed);
							}
						}
					}

					auto patch_is_fixed = patch.fixed;
					auto process_item_color = [patch_is_fixed] (float color) {
						if (patch_is_fixed) {
							return color;
						} else {
							return 0.8f * color;
						}
					};

					/* iterate over all items in this patch, creating a quad
					   for each (we use a triangle list so we need two triangles) */
					for (unsigned int i = 0; i < patch.item_count; i++) {
						const item& it = patch.items[i];
						const item_properties& item_props = item_types[it.item_type];
						item_vertices[new_item_vertex_count].position[0] = it.location.x + 0.5f - 0.4f;
						item_vertices[new_item_vertex_count].position[1] = it.location.y + 0.5f - 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]) + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]);
						}
						item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[new_item_vertex_count].position[0] = it.location.x + 0.5f - 0.4f;
						item_vertices[new_item_vertex_count].position[1] = it.location.y + 0.5f + 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]) + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]);
						}
						item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;

						item_vertices[new_item_vertex_count].position[0] = it.location.x + 0.5f + 0.4f;
						item_vertices[new_item_vertex_count].position[1] = it.location.y + 0.5f - 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]) + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]);
						}
						item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[new_item_vertex_count].position[0] = it.location.x + 0.5f + 0.4f;
						item_vertices[new_item_vertex_count].position[1] = it.location.y + 0.5f + 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]) + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]);
						}
						item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;

						item_vertices[new_item_vertex_count].position[0] = it.location.x + 0.5f + 0.4f;
						item_vertices[new_item_vertex_count].position[1] = it.location.y + 0.5f - 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]) + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]);
						}
						item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[new_item_vertex_count].position[0] = it.location.x + 0.5f - 0.4f;
						item_vertices[new_item_vertex_count].position[1] = it.location.y + 0.5f + 0.4f;
						if (item_props.blocks_movement) {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]) + 2.0f;
						} else {
							for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = process_item_color(item_props.color[j]);
						}
						item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;
					}

					/* iterate over all agents in this patch, creating an oriented triangle for each */
					for (unsigned int i = 0; i < patch.agent_count; i++) {
						float first[2] = {0};
						float second[2] = {0};
						float third[2] = {0};
						get_triangle_coords(patch.agent_directions[i], first, second, third);

						item_vertices[new_item_vertex_count].position[0] = patch.agent_positions[i].x + 0.5f + first[0];
						item_vertices[new_item_vertex_count].position[1] = patch.agent_positions[i].y + 0.5f + first[1];
						for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = agent_color[j] + 4.0f;
						item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[new_item_vertex_count].position[0] = patch.agent_positions[i].x + 0.5f + second[0];
						item_vertices[new_item_vertex_count].position[1] = patch.agent_positions[i].y + 0.5f + second[1];
						for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = agent_color[j] + 4.0f;
						item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

						item_vertices[new_item_vertex_count].position[0] = patch.agent_positions[i].x + 0.5f + third[0];
						item_vertices[new_item_vertex_count].position[1] = patch.agent_positions[i].y + 0.5f + third[1];
						for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = agent_color[j] + 4.0f;
						item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
						item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;
					}
				}
			}

			/* position the background quad */
			vertex vertices[] = {
				{{(float) bottom_left_corner.x * patch_size, (float) bottom_left_corner.y * patch_size}, {0.0f, 0.0f}},
				{{(float) bottom_left_corner.x * patch_size, (float) (top_right_corner.y + 1) * patch_size}, {0.0f, (float) texture_height_cells / texture_height}},
				{{(float) (top_right_corner.x + 1) * patch_size, (float) bottom_left_corner.y * patch_size}, {(float) texture_width_cells / texture_width, 0.0f}},
				{{(float) (top_right_corner.x + 1) * patch_size, (float) (top_right_corner.y + 1) * patch_size}, {(float) texture_width_cells / texture_width, (float) texture_height_cells / texture_height}}
			};

			/* transfer all data to GPU */
			if (!HasLock) scene_lock.lock();
			item_vertex_count = new_item_vertex_count;
			current_patch_size_texels = patch_size_texels;
			renderer.transfer_dynamic_vertex_buffer(item_quad_buffer, sizeof(item_vertex) * item_vertex_count);
			if (render_background_map) {
				renderer.transfer_dynamic_texture_image(scent_map_texture, image_format::R8G8B8A8_UNORM);
				renderer.fill_vertex_buffer(scent_quad_buffer, vertices, sizeof(vertex) * 4);
			}

		} else {
			/* no patches are visible, so move the quad outside the view */
			vertex vertices[] = {
				{{top + 10.0f, top + 10.0f}, {1.0f, 0.0f}},
				{{top + 10.0f, top + 10.0f}, {1.0f, 1.0f}},
				{{top + 10.0f, top + 10.0f}, {0.0f, 1.0f}},
				{{top + 10.0f, top + 10.0f}, {0.0f, 0.0f}}
			};

			if (!HasLock) scene_lock.lock();
			item_vertex_count = new_item_vertex_count;
			if (render_background_map)
				renderer.fill_vertex_buffer(scent_quad_buffer, vertices, sizeof(vertex) * 4);
		}

		left_bound = floor(left / patch_size) * patch_size;
		right_bound = ceil(right / patch_size) * patch_size;
		bottom_bound = floor(bottom / patch_size) * patch_size;
		top_bound = ceil(top / patch_size) * patch_size;

		draw_call<1, 0, 1> draw_scent_map;
		draw_scent_map.pipeline = scent_map_pipeline;
		draw_scent_map.first_vertex = 0;
		draw_scent_map.vertex_count = 4;
		draw_scent_map.vertex_buffers[0] = scent_quad_buffer;
		draw_scent_map.vertex_buffer_offsets[0] = 0;
		draw_scent_map.descriptor_sets[0] = set;

		draw_call<0, 1, 1> draw_items;
		draw_items.pipeline = item_pipeline;
		draw_items.first_vertex = 0;
		draw_items.vertex_count = item_vertex_count;
		draw_items.dynamic_vertex_buffers[0] = item_quad_buffer;
		draw_items.dynamic_vertex_buffer_offsets[0] = 0;
		draw_items.descriptor_sets[0] = set;

		float clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		if (render_background_map) {
			if (!renderer.record_command_buffer(cb, fb, clear_color, pass, draw_scent_map, draw_items)) {
				cleanup_renderer();
				if (!HasLock) scene_lock.unlock();
				return false;
			}
		} else {
			if (!renderer.record_command_buffer(cb, fb, clear_color, pass, draw_items)) {
				cleanup_renderer();
				if (!HasLock) scene_lock.unlock();
				return false;
			}
		}
		if (!HasLock) {
			scene_ready_cv.notify_one();
			scene_ready = true;
			scene_lock.unlock();
		}
		return true;
	}

	template<bool HasLock>
	inline bool prepare_scene_helper(array<array<patch_state>>& patches)
	{
		float left = camera_position[0] - 0.5f * (width / pixel_density) - 0.01f;
		float right = camera_position[0] + 0.5f * (width / pixel_density) + 0.01f;
		float bottom = camera_position[1] - 0.5f * (height / pixel_density) - 0.01f;
		float top = camera_position[1] + 0.5f * (height / pixel_density) + 0.01f;

		bool render_background_map = render_background;
		if (render_background_map) {
			if (sim.template get_map<true>({(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, patches) != status::OK) {
				fprintf(stderr, "visualizer.prepare_scene_helper ERROR: Unable to get map from simulator.\n");
				return false;
			}
		} else {
			if (sim.template get_map<false>({(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, patches) != status::OK) {
				fprintf(stderr, "visualizer.prepare_scene_helper ERROR: Unable to get map from simulator.\n");
				return false;
			}
		}

		if (track_agent_id != 0) {
			agent_state* agent;
			sim.get_agent_states(&agent, &track_agent_id, 1);
			if (agent != nullptr) {
				float new_target_position[] = {agent->current_position.x + 0.5f, agent->current_position.y + 0.5f};
				agent->lock.unlock();

				if (new_target_position[0] != translate_end_position[0] || new_target_position[1] != translate_end_position[1]) {
					translate_start_position[0] = camera_position[0];
					translate_start_position[1] = camera_position[1];
					translate_end_position[0] = new_target_position[0];
					translate_end_position[1] = new_target_position[1];
					tracking_animating = false;
				}
				if (!tracking_animating) {
					translate_animation_start_time = milliseconds();
					tracking_animating = true;
				}
			} else {
				fprintf(stderr, "Agent with ID %" PRIu64 " does not exist in the simulation.\n", track_agent_id);
				track_agent_id = 0;
			}
		} else {
			tracking_animating = false;
		}

		return prepare_scene_helper<HasLock>(patches, render_background_map, left, right, bottom, top);
	}

	template<typename SimulatorData>
	inline void run_map_retriever(simulator<SimulatorData>& sim)
	{
		array<array<patch_state>> patches(64);
		while (running) {
			/* wait until the renderer thread has finished drawing the previous scene */
			while (running && scene_ready) { }
			if (!running) break;

			prepare_scene_helper<false>(patches);

			for (array<patch_state>& row : patches) {
				for (patch_state& patch : row) free(patch);
				free(row);
			}
			patches.clear();
		}
	}

	template<typename SimulatorData>
	bool prepare_scene(simulator<SimulatorData>& sim)
	{
		array<array<patch_state>> patches(64);
		bool result = prepare_scene_helper<true>(patches);
		for (array<patch_state>& row : patches) {
			for (patch_state& patch : row) free(patch);
			free(row);
		}
		return result;
	}

	inline bool send_mpi_requests(client<visualizer_client_data>& sim)
	{
		float left = camera_position[0] - 0.5f * (width / pixel_density) - 0.01f;
		float right = camera_position[0] + 0.5f * (width / pixel_density) + 0.01f;
		float bottom = camera_position[1] - 0.5f * (height / pixel_density) - 0.01f;
		float top = camera_position[1] + 0.5f * (height / pixel_density) + 0.01f;

		/* send new `get_map` and `get_agent_states` messages to the server */
		sim.data.waiting_for_get_map = true;
		sim.data.get_map_left = left;
		sim.data.get_map_right = right;
		sim.data.get_map_bottom = bottom;
		sim.data.get_map_top = top;
		sim.data.get_map_render_background = render_background;
		if (!send_get_map(sim, {(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, sim.data.get_map_render_background)) {
			fprintf(stderr, "visualizer.send_mpi_requests ERROR: Unable to send `get_map` message to server.\n");
			sim.data.waiting_for_get_map = false;
			return false;
		}
		sim.data.track_agent_id = track_agent_id;
		if (sim.data.track_agent_id != 0) {
			sim.data.waiting_for_get_agent_states = true;
			if (!send_get_agent_states(sim, &sim.data.track_agent_id, 1)) {
				fprintf(stderr, "visualizer.send_mpi_requests ERROR: Unable to send `get_agent_states` message to server.\n");
				sim.data.waiting_for_get_agent_states = false;
				return false;
			}
		} else {
			sim.data.waiting_for_get_agent_states = false;
			sim.data.get_agent_states_response = status::OK;
		}
		return true;
	}

	template<bool HasLock>
	inline void process_mpi_response(visualizer_client_data& response)
	{
		if (response.get_agent_states_response == status::OK && response.track_agent_id != 0) {
			if (response.agent_state_count > 0) {
				float new_target_position[] = {response.agent_states[0].current_position.x + 0.5f, response.agent_states[0].current_position.y + 0.5f};

				if (new_target_position[0] != translate_end_position[0] || new_target_position[1] != translate_end_position[1]) {
					translate_start_position[0] = camera_position[0];
					translate_start_position[1] = camera_position[1];
					translate_end_position[0] = new_target_position[0];
					translate_end_position[1] = new_target_position[1];
					tracking_animating = false;
				}
				if (!tracking_animating) {
					translate_animation_start_time = milliseconds();
					tracking_animating = true;
				}
			} else {
				fprintf(stderr, "Agent with ID %" PRIu64 " does not exist in the simulation.\n", response.track_agent_id);
				track_agent_id = 0;
			}
		} else {
			tracking_animating = false;
		}

		if (response.get_map_response == status::OK) {
			prepare_scene_helper<HasLock>(*response.map, response.get_map_render_background,
				response.get_map_left, response.get_map_right, response.get_map_bottom, response.get_map_top);

			for (array<patch_state>& row : *response.map) {
				for (patch_state& patch : row) free(patch);
				free(row);
			}
			free(*response.map);
			free(response.map);
		}
	}

	inline bool process_mpi_status(client<visualizer_client_data>& sim)
	{
		if (sim.data.get_map_response == status::PERMISSION_ERROR) {
			fprintf(stderr, "ERROR: We don't have permission to call `get_map` on the server.\n");
			running = false;
			remove_client(sim);
			return false;
		} else if (sim.data.get_map_response != status::OK) {
			fprintf(stderr, "visualizer.process_mpi_status ERROR: `get_map` failed.\n");
			running = false;
			remove_client(sim);
			return false;
		}

		if (sim.data.get_agent_states_response == status::PERMISSION_ERROR) {
			fprintf(stderr, "ERROR: We don't have permission to call `get_agent_states` on the server. We cannot track agents.\n");
			track_agent_id = 0;
			sim.data.track_agent_id = 0;
		} else if (sim.data.get_agent_states_response != status::INVALID_AGENT_ID
				&& sim.data.get_agent_states_response != status::OK)
		{
			fprintf(stderr, "visualizer.process_mpi_status ERROR: `get_agent_states` failed.\n");
		}
		return true;
	}

	inline void run_map_retriever(client<visualizer_client_data>& sim)
	{
		while (running) {
			/* wait until we get responses from the server and the renderer thread has finished drawing the previous scene */
			while (running && sim.client_running && (scene_ready || sim.data.waiting_for_get_map || sim.data.waiting_for_get_agent_states)) { }
			if (!running || !sim.client_running) break;

			if (!process_mpi_status(sim))
				return;

			/* copy the response so we can send the next MPI requests */
			visualizer_client_data response = sim.data;

			if (!send_mpi_requests(sim))
				continue;

			process_mpi_response<false>(response);
		}
	}

	bool prepare_scene(client<visualizer_client_data>& sim)
	{
		/* wait for any existing MPI requests to finish */
		while (sim.client_running && (sim.data.waiting_for_get_map || sim.data.waiting_for_get_agent_states)) { }
		if (!sim.client_running) return false;

		sim.data.painter = this;
		if (!send_mpi_requests(sim))
			return false;

		/* wait until we get responses from the server and the renderer thread has finished drawing the previous scene */
		while (sim.client_running && (sim.data.waiting_for_get_map || sim.data.waiting_for_get_agent_states)) { }
		if (!sim.client_running) return false;

		if (!process_mpi_status(sim))
			return false;

		process_mpi_response<true>(sim.data);

		return send_mpi_requests(sim);
	}

	bool setup_renderer() {
		dynamic_texture_image dynamic_textures[] = { scent_map_texture };
		uint32_t texture_bindings[] = { 1 };
		uint32_t ub_binding = 0;
		descriptor_type pool_types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
		if (!renderer.create_render_pass(pass)) {
			return false;
		} else if (!renderer.create_graphics_pipeline(scent_map_pipeline, pass,
				background_vertex_shader, "main", background_fragment_shader, "main",
				primitive_topology::TRIANGLE_STRIP, false, 1.0f,
				background_binding, background_shader_attributes, &layout, 1))
		{
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_graphics_pipeline(item_pipeline, pass,
				item_vertex_shader, "main", item_fragment_shader, "main",
				primitive_topology::TRIANGLE_LIST, false, 1.0f,
				item_binding, item_shader_attributes, &layout, 1))
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

	static inline void scent_to_color(const float* cell_scent, pixel& out, bool is_patch_fixed) {
		float x = max(0.0f, min(1.0f, log(pow(cell_scent[0], 0.4f) + 1.0f) / 0.9f));
		float y = max(0.0f, min(1.0f, log(pow(cell_scent[1], 0.4f) + 1.0f) / 0.9f));
		float z = max(0.0f, min(1.0f, log(pow(cell_scent[2], 0.4f) + 1.0f) / 0.9f));

		float r = 255 * (1 - (y + z) / 2);
		float g = 255 * (1 - (x + z) / 2);
		float b = 255 * (1 - (x + y) / 2);

		if (is_patch_fixed) {
			out.r = r;
			out.g = g;
			out.b = b;
		} else {
			float black_alpha = 0.2;
			out.r = (uint8_t) ((1 - black_alpha) * r);
			out.g = (uint8_t) ((1 - black_alpha) * g);
			out.b = (uint8_t) ((1 - black_alpha) * b);
		}
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

	template<typename A> friend void cursor_position_callback(GLFWwindow*, double, double);
	template<typename A> friend void key_callback(GLFWwindow*, int, int, int, int);
	friend void on_lost_connection(client<visualizer_client_data>&);
};

void on_add_agent(
		client<visualizer_client_data>& c, uint64_t agent_id,
		status response, const agent_state& state)
{
	fprintf(stderr, "WARNING: `on_add_agent` should not be called.\n");
}

void on_remove_agent(client<visualizer_client_data>& c,
		uint64_t agent_id, status response)
{
	fprintf(stderr, "WARNING: `on_remove_agent` should not be called.\n");
}

void on_move(client<visualizer_client_data>& c, uint64_t agent_id, status response)
{
	fprintf(stderr, "WARNING: `on_move` should not be called.\n");
}

void on_turn(client<visualizer_client_data>& c, uint64_t agent_id, status response)
{
	fprintf(stderr, "WARNING: `on_turn` should not be called.\n");
}

void on_do_nothing(client<visualizer_client_data>& c, uint64_t agent_id, status response)
{
	fprintf(stderr, "WARNING: `on_do_nothing` should not be called.\n");
}

void on_get_map(client<visualizer_client_data>& c,
		status response, array<array<patch_state>>* map)
{
	c.data.map = map;
	c.data.get_map_response = response;
	c.data.waiting_for_get_map = false;
}

void on_get_agent_ids(
		client<visualizer_client_data>& c, status response,
		const uint64_t* agent_ids, size_t count)
{
	fprintf(stderr, "WARNING: `on_get_agent_ids` should not be called.\n");
}

void on_get_agent_states(
		client<visualizer_client_data>& c, status response,
		const uint64_t* agent_ids,
		const agent_state* agent_states, size_t count)
{
	c.data.get_agent_states_response = response;
	c.data.agent_states = agent_states;
	c.data.agent_state_count = count;
	c.data.waiting_for_get_agent_states = false;
}

void on_set_active(client<visualizer_client_data>& c, uint64_t agent_id, status response)
{
	fprintf(stderr, "WARNING: `on_set_active` should not be called.\n");
}

void on_is_active(client<visualizer_client_data>& c, uint64_t agent_id, status response, bool active)
{
	fprintf(stderr, "WARNING: `on_is_active` should not be called.\n");
}

inline void on_step(client<visualizer_client_data>& c,
		status response,
		const array<uint64_t>& agent_ids,
		const agent_state* agent_state_array)
{ }

void on_lost_connection(client<visualizer_client_data>& c) {
	fprintf(stderr, "Lost connection to the server.\n");
	c.client_running = false;
	c.data.painter->running = false;
	c.data.painter->scene_ready_cv.notify_all();
}

} /* namespace jbw */

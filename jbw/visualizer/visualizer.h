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

#include "../mpi.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "vulkan_renderer.h"

#include <thread>
#include <condition_variable>
#include <core/lex.h>

#if defined(RECORD)
extern uint64_t simulation_time;
#endif

namespace jbw {

using namespace core;
using namespace mirage;

/* forward declarations */
unsigned long long milliseconds();
template<typename SimulatorType> class visualizer;
#if defined(RECORD)
bool record_key_press(char key);
bool record_step();
bool record_collected_items(const unsigned int*, unsigned int);
void increment_frame_number();
void write_to_log();
#endif

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
		v->track_agent(0);
		v->camera_position[0] += (float) (v->last_cursor_x - x) / v->pixel_density;
		v->camera_position[1] -= (float) (v->last_cursor_y - y) / v->pixel_density;
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
			if (v->target_pixel_density / 1.3f <= 1 / get_config(v->sim).patch_size) {
				/* TODO: handle the case where the pixel density is smaller than 1 (we segfault currently since the texture for the scent visualization could become too small) */
				fprintf(stderr, "Zoom beyond the point where the pixel density is smaller than 1/patch_size is unsupported.\n");
			} else {
				v->zoom_animation_start_time = jbw::milliseconds();
				v->zoom_start_pixel_density = v->pixel_density;
				v->target_pixel_density /= 1.3f;
#if defined(RECORD)
				record_key_press('-');
#endif
			}
		} else if (key == GLFW_KEY_EQUAL) {
			v->zoom_animation_start_time = jbw::milliseconds();
			v->zoom_start_pixel_density = v->pixel_density;
			v->target_pixel_density *= 1.3f;
#if defined(RECORD)
			record_key_press('+');
#endif
		} else if (key == GLFW_KEY_0) {
			v->track_agent(0);
		} else if (key == GLFW_KEY_1) {
			v->track_agent(1);
		} else if (key == GLFW_KEY_2) {
			v->track_agent(2);
		} else if (key == GLFW_KEY_3) {
			v->track_agent(3);
		} else if (key == GLFW_KEY_4) {
			v->track_agent(4);
		} else if (key == GLFW_KEY_5) {
			v->track_agent(5);
		} else if (key == GLFW_KEY_6) {
			v->track_agent(6);
		} else if (key == GLFW_KEY_7) {
			v->track_agent(7);
		} else if (key == GLFW_KEY_8) {
			v->track_agent(8);
		} else if (key == GLFW_KEY_9) {
			v->track_agent(9);
		} else if (key == GLFW_KEY_B) {
			v->render_background = !v->render_background;
#if defined(RECORD)
			record_key_press('b');
#endif
		} else if (key == GLFW_KEY_V) {
			v->render_agent_visual_field = !v->render_agent_visual_field;
#if defined(RECORD)
			record_key_press('v');
#endif
		} else if (key == GLFW_KEY_P) {
			std::unique_lock<std::mutex> lock(v->agent_path_lock);
			if (v->track_agent_id == 0) {
				fprintf(stderr, "An agent must be tracked to draw its movement path.\n");
			} else {
				v->render_agent_path = !v->render_agent_path;
				v->agent_path.clear();
				v->agent_path_cv.notify_one();
			}
		} else if (key == GLFW_KEY_S) {
			v->screenshot_next_frame = true;
		} else if (key == GLFW_KEY_LEFT_BRACKET) {
			v->semaphore_signal_period *= 2;
			fprintf(stderr, "The maximum simulation steps per second is now %f.\n", 1000.0f / v->semaphore_signal_period);
#if defined(RECORD)
			record_key_press('[');
#endif
		} else if (key == GLFW_KEY_RIGHT_BRACKET) {
			v->semaphore_signal_period = max(1ull, v->semaphore_signal_period / 2);
			fprintf(stderr, "The maximum simulation steps per second is now %f.\n", 1000.0f / v->semaphore_signal_period);
#if defined(RECORD)
			record_key_press(']');
#endif
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
	bool get_map_render_background;
	bool get_map_render_agent_path;
	float pixel_density;

	status get_map_response = status::OK;
	array<array<patch_state>>* map = nullptr;

	std::atomic_bool waiting_for_get_agent_states;
	uint64_t track_agent_id = 0;

	status get_agent_states_response = status::OK;
	const agent_state* agent_states = nullptr;
	size_t agent_state_count = 0;
	bool render_visual_field;

	std::atomic_bool waiting_for_semaphore_op;
	uint64_t semaphore_id;
	status semaphore_op_response = status::OK;

	visualizer_client_data() {
		waiting_for_get_map.store(false);
		waiting_for_get_agent_states.store(false);
	}

	visualizer_client_data(const visualizer_client_data& src) :
		get_map_left(src.get_map_left), get_map_right(src.get_map_right),
		get_map_bottom(src.get_map_bottom), get_map_top(src.get_map_top),
		get_map_render_background(src.get_map_render_background),
		get_map_render_agent_path(src.get_map_render_agent_path),
		pixel_density(src.pixel_density), get_map_response(src.get_map_response),
		map(src.map), track_agent_id(src.track_agent_id),
		get_agent_states_response(src.get_agent_states_response),
		agent_states(src.agent_states), agent_state_count(src.agent_state_count),
		render_visual_field(src.render_visual_field), semaphore_id(src.semaphore_id),
		semaphore_op_response(src.semaphore_op_response)
	{
		waiting_for_get_map.store(src.waiting_for_get_map.load());
		waiting_for_get_agent_states.store(src.waiting_for_get_agent_states.load());
		waiting_for_semaphore_op.store(src.waiting_for_semaphore_op.load());
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
		unsigned int tex_index;
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

	struct alignas(16) vec3 {
		float x, y, z;
	};

	struct uniform_buffer_data {
		float model[16];
		float view[16];
		float projection[16];
		float pixel_density;
		uint32_t patch_size_texels;
		vec3 agent_color;
	};

	GLFWwindow* window;
	uint32_t width;
	uint32_t height;
	uint32_t texture_width;
	uint32_t texture_height;
	float camera_position[2];
	float pixel_density;
	uint32_t current_patch_size_texels;

	SimulatorType& sim;
	uint64_t semaphore;
	unsigned long long semaphore_signal_time;
	unsigned long long semaphore_signal_period;
	std::thread semaphore_signaler;
	std::mutex mpi_lock;

	vulkan_renderer renderer;
	shader background_vertex_shader, background_fragment_shader;
	shader item_vertex_shader, item_fragment_shader;
	shader visual_field_fragment_shader;
	render_pass pass;
	graphics_pipeline scent_map_pipeline, item_pipeline, visual_field_pipeline;
	frame_buffer fb;
	command_buffer cb;
	descriptor_set_layout layout;
	descriptor_pool pool;
	descriptor_set ds;
	uniform_buffer ub;
	dynamic_texture_image scent_map_texture, visual_field_texture;
	sampler tex_sampler;
	vertex_buffer scent_quad_buffer;
	dynamic_vertex_buffer item_quad_buffer;
	uint32_t item_vertex_count;
	uint32_t item_quad_buffer_capacity;
	uniform_buffer_data uniform_data;
	binding_description background_binding;
	attribute_descriptions<3> background_shader_attributes;
	binding_description item_binding;
	attribute_descriptions<3> item_shader_attributes;

	bool left_mouse_button_pressed;
	double last_cursor_x, last_cursor_y;

	float zoom_start_pixel_density;
	float target_pixel_density;
	unsigned long long zoom_animation_start_time;

	/* this is zero if we don't want to track anyone */
	uint64_t track_agent_id;
	float translate_start_position[2];
	float translate_end_position[2];
	unsigned long long translate_animation_start_time;
	bool tracking_animating;

	/* the thread that calls `get_map` so we don't need to
	   do so in `draw_frame`, which keeps everything smooth */
	std::thread map_retriever;
	std::mutex scene_lock;
	std::condition_variable scene_ready_cv;
	std::atomic_bool scene_ready;
	float left_bound, right_bound, bottom_bound, top_bound;
	bool render_background;
	bool render_agent_visual_field;

	/* list of vertices of the tracked agent's movement path */
	array<position> agent_path;
	std::mutex agent_path_lock;
	std::condition_variable agent_path_cv;
	bool agent_position_recorded;
	bool render_agent_path;

	/* for screenshots */
	bool screenshot_next_frame;

public:
	std::atomic_bool running;

	visualizer(SimulatorType& sim,
		uint32_t window_width, uint32_t window_height,
		uint64_t track_agent_id, float pixels_per_cell,
		bool draw_scent_map, bool draw_visual_field,
		bool draw_path, float max_steps_per_second,
		bool fullscreen) :
			width(window_width), height(window_height), sim(sim),
			semaphore(0), semaphore_signal_time(0),
			background_binding(0, sizeof(vertex)), item_binding(0, sizeof(item_vertex)),
			track_agent_id(track_agent_id), tracking_animating(false),
			scene_ready(false), render_background(draw_scent_map),
			render_agent_visual_field(draw_visual_field),
			agent_path(64), agent_position_recorded(false),
			render_agent_path(draw_path), screenshot_next_frame(false),
			running(true)
	{
		semaphore_signal_period = (unsigned long long) round(1000.0f / max_steps_per_second);

		camera_position[0] = 0.5f;
		camera_position[1] = 0.5f;
		translate_end_position[0] = camera_position[0];
		translate_end_position[1] = camera_position[1];
		pixel_density = pixels_per_cell;
		target_pixel_density = pixel_density;
		zoom_start_pixel_density = pixel_density;
		zoom_animation_start_time = jbw::milliseconds();
		uniform_data = {{0}};
		make_identity(uniform_data.model);

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
		window = glfwCreateWindow(window_width, window_height, "JBW Visualizer", fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetCursorPosCallback(window, cursor_position_callback<SimulatorType>);
		glfwSetKeyCallback(window, key_callback<SimulatorType>);

		/* We need to get the actual framebuffer width and height because HiDPI sometimes scale the
		   actual framebuffer size relative to the window size. */
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
		background_shader_attributes.set<2>(0, 2, attribute_type::UINT, offsetof(vertex, tex_index));

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

		fragment_shader_src = read_file<true>("visual_field_fragment_shader.spv", fragment_shader_size);
		if (fragment_shader_src == nullptr) {
			free(fragment_shader_src);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to load fragment shader from file.");
		}

		if (!renderer.create_shader(visual_field_fragment_shader, fragment_shader_src, fragment_shader_size)) {
			free(fragment_shader_src);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create fragment shader.");
		}
		free(fragment_shader_src);

		if (!renderer.create_vertex_buffer(scent_quad_buffer, sizeof(vertex) * 8)) {
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
			renderer.delete_shader(visual_field_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to vertex buffer for scent textured quad.");
		}

		uint32_t binding_indices[] = { 0, 1 };
		descriptor_type types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
		uint32_t descriptor_counts[] = { 1, 2 };
		shader_stage visibilities[] = { shader_stage::ALL, shader_stage::FRAGMENT };
		if (!renderer.create_descriptor_set_layout(layout, binding_indices, types, descriptor_counts, visibilities, 2)) {
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			renderer.delete_shader(visual_field_fragment_shader);
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
			renderer.delete_shader(visual_field_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create `scent_map_texture`.");
		}

		unsigned int vision_range = get_config(sim).vision_range;
		if (!renderer.create_dynamic_texture_image(
			visual_field_texture,
			/* image_size */ sizeof(pixel) * (2 * vision_range + 1) * (2 * vision_range + 1),
			/* texture_width */ 2 * vision_range + 1,
			/* texture_height */ 2 * vision_range + 1,
			image_format::R8G8B8A8_UNORM)
		) {
			renderer.delete_dynamic_texture_image(scent_map_texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			renderer.delete_shader(visual_field_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create `scent_map_texture`.");
		}

		if (!renderer.create_sampler(tex_sampler, filter::NEAREST, filter::NEAREST,
				sampler_address_mode::CLAMP_TO_EDGE, sampler_address_mode::CLAMP_TO_EDGE,
				sampler_address_mode::CLAMP_TO_EDGE, false, 1.0f))
		{
			renderer.delete_dynamic_texture_image(visual_field_texture);
			renderer.delete_dynamic_texture_image(scent_map_texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			renderer.delete_shader(visual_field_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to initialize texture sampler.");
		}

		if (!setup_renderer()) {
			renderer.delete_sampler(tex_sampler);
			renderer.delete_dynamic_texture_image(visual_field_texture);
			renderer.delete_dynamic_texture_image(scent_map_texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			renderer.delete_shader(visual_field_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to initialize rendering pipeline.");
		}

		if (!create_semaphore(sim)) {
			cleanup_renderer();
			renderer.delete_sampler(tex_sampler);
			renderer.delete_dynamic_texture_image(visual_field_texture);
			renderer.delete_dynamic_texture_image(scent_map_texture);
			renderer.delete_descriptor_set_layout(layout);
			renderer.delete_vertex_buffer(scent_quad_buffer);
			renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
			renderer.delete_shader(item_vertex_shader);
			renderer.delete_shader(item_fragment_shader);
			renderer.delete_shader(background_vertex_shader);
			renderer.delete_shader(background_fragment_shader);
			renderer.delete_shader(visual_field_fragment_shader);
			glfwDestroyWindow(window); glfwTerminate();
			throw new std::runtime_error("visualizer ERROR: Failed to create simulator semaphore.");
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

		semaphore_signaler = std::thread([&]() {
			while (running) {
				if (render_agent_path && track_agent_id != 0) {
					/* make sure we don't miss any simulation time steps if we're tracking the agent path */
					std::unique_lock<std::mutex> lock(agent_path_lock);
					while (running && render_agent_path && track_agent_id != 0 && !agent_position_recorded)
						agent_path_cv.wait(lock);
					agent_position_recorded = false;
					lock.unlock();
				}

				signal_semaphore(sim);

				semaphore_signal_time = jbw::milliseconds();
				unsigned long long remaining_time = semaphore_signal_period;
				while (true) {
					std::this_thread::sleep_for(std::chrono::milliseconds(min(remaining_time, 100ull)));
					if (!running) return;
					unsigned long long current_time = jbw::milliseconds();
					if (current_time > semaphore_signal_time + semaphore_signal_period) break;
					remaining_time = semaphore_signal_time + semaphore_signal_period - current_time;
				}
			}
		});
	}

	~visualizer() {
		running = false;
		scene_ready_cv.notify_one();
		agent_path_cv.notify_one();
		if (map_retriever.joinable()) {
			try {
				map_retriever.join();
			} catch (...) { }
		}
		if (semaphore_signaler.joinable()) {
			try {
				semaphore_signaler.join();
			} catch (...) { }
		}
		delete_semaphore(sim);
		renderer.wait_until_idle();
		renderer.delete_sampler(tex_sampler);
		renderer.delete_dynamic_texture_image(scent_map_texture);
		renderer.delete_dynamic_texture_image(visual_field_texture);
		renderer.delete_descriptor_set_layout(layout);
		renderer.delete_vertex_buffer(scent_quad_buffer);
		renderer.delete_dynamic_vertex_buffer(item_quad_buffer);
		cleanup_renderer();
		renderer.delete_shader(item_vertex_shader);
		renderer.delete_shader(item_fragment_shader);
		renderer.delete_shader(background_vertex_shader);
		renderer.delete_shader(background_fragment_shader);
		renderer.delete_shader(visual_field_fragment_shader);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	inline void track_agent(uint64_t agent_id) {
		tracking_animating = false;
		if (track_agent_id != agent_id) {
			std::unique_lock<std::mutex> lock(agent_path_lock);
			agent_path_cv.notify_one();
			agent_path.clear();
			track_agent_id = agent_id;
			if (track_agent_id == 0)
				render_agent_path = false;
		}
	}

	inline bool add_to_agent_path(position agent_position) {
		if (agent_path.length != 0 && agent_path.last() == agent_position)
			return true;
		if (agent_path.length <= 1)
			return agent_path.add(agent_position);

		/* check if the previous point is redundant */
		const position& prev = agent_path[agent_path.length - 1];
		const position& prev_prev = agent_path[agent_path.length - 2];
		if ((prev_prev.x == prev.x && prev.x == agent_position.x && prev_prev.y <= prev.y && prev.y <= agent_position.y)
		 || (prev_prev.x == prev.x && prev.x == agent_position.x && prev_prev.y >= prev.y && prev.y >= agent_position.y)
		 || (prev_prev.y == prev.y && prev.y == agent_position.y && prev_prev.x <= prev.x && prev.x <= agent_position.x)
		 || (prev_prev.y == prev.y && prev.y == agent_position.y && prev_prev.x >= prev.x && prev.x >= agent_position.x))
		{
			agent_path[agent_path.length - 1] = agent_position;
			return true;
		} else {
			return agent_path.add(agent_position);
		}
	}

	inline bool is_window_closed() {
		return (bool) glfwWindowShouldClose(window);
	}

	inline bool draw_frame()
	{
		glfwPollEvents();

		/* compute `pixel_density` according to the zoom animation */
		float animation_t = max(0.0f, min(1.0f, (jbw::milliseconds() - zoom_animation_start_time) / 300.0f));
		float easing = animation_t * (2 - animation_t);
		pixel_density = easing * target_pixel_density + (1.0f - easing) * zoom_start_pixel_density;

		if (tracking_animating) {
			float animation_t = max(0.0f, min(1.0f, (jbw::milliseconds() - translate_animation_start_time) / 300.0f));
			float easing = (animation_t - 1) * (animation_t - 1) * (animation_t - 1) + 1;
			camera_position[0] = easing * translate_end_position[0] + (1.0f - easing) * translate_start_position[0];
			camera_position[1] = easing * translate_end_position[1] + (1.0f - easing) * translate_start_position[1];
		}

		float left = camera_position[0] - 0.5f * (width / pixel_density);
		float right = camera_position[0] + 0.5f * (width / pixel_density);
		float bottom = camera_position[1] - 0.5f * (height / pixel_density);
		float top = camera_position[1] + 0.5f * (height / pixel_density);
		while (running && (left < left_bound || right > right_bound || bottom < bottom_bound || top > top_bound))
			scene_ready = false;
		if (!running) return true;

		/* construct the model view matrix */
		float up[] = { 0.0f, 1.0f, 0.0f };
		float forward[] = { 0.0f, 0.0f, -1.0f };
		float camera_pos[] = { camera_position[0], camera_position[1], 2.0f };
		make_identity(uniform_data.model);
		make_view_matrix(uniform_data.view, forward, up, camera_pos);
		make_orthographic_projection(uniform_data.projection,
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

		while (!scene_lock.try_lock()) { }
		void* pv_uniform_data = (void*) &uniform_data;
		bool result = renderer.draw_frame(cb, reset_command_buffers, get_window_dimensions, &ub, &pv_uniform_data, 1);
		scene_ready = false;
		scene_ready_cv.notify_one();
		scene_lock.unlock();
		return result;
	}

	template<bool HasLock>
	struct vulkan_glfw_backend
	{
		pixel* scent_map_texture_data;
		pixel* visual_field_texture_data;
		unsigned int num_patches_x, num_patches_y;
		unsigned int texture_width_cells;
		unsigned int texture_height_cells;
		uint32_t new_item_vertex_count;
		visualizer<SimulatorType>& v;

		vulkan_glfw_backend(visualizer<SimulatorType>& v) : v(v) { }

		inline unsigned int get_texel_cell_length(float pixel_density) {
			return (unsigned int) ceil(1 / pixel_density);
		}

		inline void begin_drawing(unsigned int x_num_patches, unsigned int y_num_patches, unsigned int patch_size_texels) {
			scent_map_texture_data = (pixel*) v.scent_map_texture.mapped_memory;
			visual_field_texture_data = (pixel*) v.visual_field_texture.mapped_memory;
			num_patches_x = x_num_patches;
			num_patches_y = y_num_patches;
			texture_width_cells = num_patches_x * patch_size_texels;
			texture_height_cells = num_patches_y * patch_size_texels;
			new_item_vertex_count = 0;
		}

		inline bool finish_drawing(
				float left, float right, float bottom, float top,
				float background_grid_bottom_left_x, float background_grid_bottom_left_y,
				float background_grid_top_right_x, float background_grid_top_right_y,
				float visual_field_grid_bottom_left_x, float visual_field_grid_bottom_left_y,
				float visual_field_grid_top_right_x, float visual_field_grid_top_right_y,
				float agent_color_r, float agent_color_g, float agent_color_b,
				unsigned int patch_size_texels, unsigned int patch_size,
				bool draw_visual_field_grid)
		{
			/* position the background quad */
			vertex vertices[] = {
				{{background_grid_bottom_left_x, background_grid_bottom_left_y}, {0.0f, 0.0f}, 0},
				{{background_grid_bottom_left_x, background_grid_top_right_y}, {0.0f, (float) texture_height_cells / v.texture_height}, 0},
				{{background_grid_top_right_x, background_grid_bottom_left_y}, {(float) texture_width_cells / v.texture_width, 0.0f}, 0},
				{{background_grid_top_right_x, background_grid_top_right_y}, {(float) texture_width_cells / v.texture_width, (float) texture_height_cells / v.texture_height}, 0},
				{{visual_field_grid_bottom_left_x, visual_field_grid_bottom_left_y}, {0.0f, 0.0f}, 1},
				{{visual_field_grid_bottom_left_x, visual_field_grid_top_right_y}, {0.0f, 1.0f}, 1},
				{{visual_field_grid_top_right_x, visual_field_grid_bottom_left_y}, {1.0f, 0.0f}, 1},
				{{visual_field_grid_top_right_x, visual_field_grid_top_right_y}, {1.0f, 1.0f}, 1}
			};

			/* transfer all data to GPU */
			if (!HasLock) while (!v.scene_lock.try_lock()) { }

			v.uniform_data.agent_color.x = agent_color_r;
			v.uniform_data.agent_color.y = agent_color_g;
			v.uniform_data.agent_color.z = agent_color_b;

			v.item_vertex_count = new_item_vertex_count;
			v.current_patch_size_texels = patch_size_texels;
			v.renderer.transfer_dynamic_vertex_buffer(v.item_quad_buffer, sizeof(item_vertex) * v.item_vertex_count);
			v.renderer.transfer_dynamic_texture_image(v.scent_map_texture, image_format::R8G8B8A8_UNORM);
			v.renderer.transfer_dynamic_texture_image(v.visual_field_texture, image_format::R8G8B8A8_UNORM);
			v.renderer.fill_vertex_buffer(v.scent_quad_buffer, vertices, sizeof(vertex) * 8);

			return record_draw_calls(left, right, bottom, top, patch_size, draw_visual_field_grid);
		}

		inline bool draw_nothing(float left, float right, float bottom, float top, unsigned int patch_size, bool draw_visual_field_grid)
		{
			/* no patches are visible, so move the quad outside the view */
			vertex vertices[] = {
				{{top + 10.0f, top + 10.0f}, {1.0f, 0.0f}, 0},
				{{top + 10.0f, top + 10.0f}, {1.0f, 1.0f}, 0},
				{{top + 10.0f, top + 10.0f}, {0.0f, 1.0f}, 0},
				{{top + 10.0f, top + 10.0f}, {0.0f, 0.0f}, 0},
				{{top + 10.0f, top + 10.0f}, {1.0f, 0.0f}, 1},
				{{top + 10.0f, top + 10.0f}, {1.0f, 1.0f}, 1},
				{{top + 10.0f, top + 10.0f}, {0.0f, 1.0f}, 1},
				{{top + 10.0f, top + 10.0f}, {0.0f, 0.0f}, 1}
			};

			if (!HasLock) while (!v.scene_lock.try_lock()) { }
			v.item_vertex_count = new_item_vertex_count;
			v.renderer.fill_vertex_buffer(v.scent_quad_buffer, vertices, sizeof(vertex) * 8);

			return record_draw_calls(left, right, bottom, top, patch_size, draw_visual_field_grid);
		}

		inline bool record_draw_calls(float left, float right, float bottom, float top, unsigned int patch_size, bool draw_visual_field_grid)
		{
			v.left_bound = floor(left / patch_size) * patch_size;
			v.right_bound = ceil(right / patch_size) * patch_size;
			v.bottom_bound = floor(bottom / patch_size) * patch_size;
			v.top_bound = ceil(top / patch_size) * patch_size;

			draw_call<1, 0, 1> draw_scent_map;
			draw_scent_map.pipeline = v.scent_map_pipeline;
			draw_scent_map.first_vertex = 0;
			draw_scent_map.vertex_count = 4;
			draw_scent_map.vertex_buffers[0] = v.scent_quad_buffer;
			draw_scent_map.vertex_buffer_offsets[0] = 0;
			draw_scent_map.descriptor_sets[0] = v.ds;

			draw_call<0, 1, 1> draw_items;
			draw_items.pipeline = v.item_pipeline;
			draw_items.first_vertex = 0;
			draw_items.vertex_count = v.item_vertex_count;
			draw_items.dynamic_vertex_buffers[0] = v.item_quad_buffer;
			draw_items.dynamic_vertex_buffer_offsets[0] = 0;
			draw_items.descriptor_sets[0] = v.ds;

			if (draw_visual_field_grid) {
				draw_call<1, 0, 1> draw_visual_field;
				draw_visual_field.pipeline = v.visual_field_pipeline;
				draw_visual_field.first_vertex = 4;
				draw_visual_field.vertex_count = 4;
				draw_visual_field.vertex_buffers[0] = v.scent_quad_buffer;
				draw_visual_field.vertex_buffer_offsets[0] = 0;
				draw_visual_field.descriptor_sets[0] = v.ds;

				float clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				if (!v.renderer.record_command_buffer(
					v.cb, v.fb, clear_color, v.pass,
					draw_scent_map, draw_visual_field, draw_items)
				) {
					v.cleanup_renderer();
					if (!HasLock) v.scene_lock.unlock();
					return false;
				}
			} else {
				float clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				if (!v.renderer.record_command_buffer(
					v.cb, v.fb, clear_color, v.pass,
					draw_scent_map, draw_items)
				) {
					v.cleanup_renderer();
					if (!HasLock) v.scene_lock.unlock();
					return false;
				}
			}

			if (!HasLock) {
				v.scene_ready_cv.notify_one();
				v.scene_ready = true;
				v.scene_lock.unlock();
			}
			return true;
		}

		inline bool ensure_capacity(unsigned int requested_triangle_count, unsigned int requested_quad_or_circle_count)
		{
			unsigned int requested_item_vertices = 3 * requested_triangle_count + 4 * requested_quad_or_circle_count;
			if (requested_item_vertices > v.item_quad_buffer_capacity) {
				uint32_t new_capacity = 2 * v.item_quad_buffer_capacity;
				while (requested_item_vertices > new_capacity)
					new_capacity *= 2;
				if (!HasLock) while (!v.scene_lock.try_lock()) { }
				v.renderer.wait_until_idle();
				v.renderer.delete_dynamic_vertex_buffer(v.item_quad_buffer);
				if (!v.renderer.create_dynamic_vertex_buffer(v.item_quad_buffer, new_capacity * sizeof(item_vertex))) {
					fprintf(stderr, "visualizer.vulkan_glfw_backend.ensure_capacity ERROR: Unable to expand `item_quad_buffer`.\n");
					if (!HasLock) v.scene_lock.unlock();
					return false;
				}
				v.item_quad_buffer_capacity = new_capacity;
				if (!HasLock) v.scene_lock.unlock();
			}
			return true;
		}

		inline void add_triangle(
				float v1_x, float v1_y,
				float v2_x, float v2_y,
				float v3_x, float v3_y,
				const float color[3])
		{
			item_vertex* item_vertices = (item_vertex*) v.item_quad_buffer.mapped_memory;
			item_vertices[new_item_vertex_count].position[0] = v1_x;
			item_vertices[new_item_vertex_count].position[1] = v1_y;
			for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = color[j] + 4.0f;
			item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

			item_vertices[new_item_vertex_count].position[0] = v2_x;
			item_vertices[new_item_vertex_count].position[1] = v2_y;
			for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = color[j] + 4.0f;
			item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

			item_vertices[new_item_vertex_count].position[0] = v3_x;
			item_vertices[new_item_vertex_count].position[1] = v3_y;
			for (unsigned int j = 0; j < 3; j++) item_vertices[new_item_vertex_count].color[j] = color[j] + 4.0f;
			item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;
		}

		inline void add_rect(
				float bottom_left_x, float bottom_left_y,
				float top_right_x, float top_right_y,
				const float color[3])
		{
			item_vertex* item_vertices = (item_vertex*) v.item_quad_buffer.mapped_memory;
			item_vertices[new_item_vertex_count].position[0] = bottom_left_x;
			item_vertices[new_item_vertex_count].position[1] = bottom_left_y;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i] + 2.0f;
			new_item_vertex_count++;

			item_vertices[new_item_vertex_count].position[0] = bottom_left_x;
			item_vertices[new_item_vertex_count].position[1] = top_right_y;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i] + 2.0f;
			new_item_vertex_count++;

			item_vertices[new_item_vertex_count].position[0] = top_right_x;
			item_vertices[new_item_vertex_count].position[1] = top_right_y;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i] + 2.0f;
			new_item_vertex_count++;

			item_vertices[new_item_vertex_count].position[0] = top_right_x;
			item_vertices[new_item_vertex_count].position[1] = top_right_y;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i] + 2.0f;
			new_item_vertex_count++;

			item_vertices[new_item_vertex_count].position[0] = top_right_x;
			item_vertices[new_item_vertex_count].position[1] = bottom_left_y;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i] + 2.0f;
			new_item_vertex_count++;

			item_vertices[new_item_vertex_count].position[0] = bottom_left_x;
			item_vertices[new_item_vertex_count].position[1] = bottom_left_y;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i] + 2.0f;
			new_item_vertex_count++;
		}

		inline void add_circle(
				float center_x, float center_y, float radius, const float color[3])
		{
			item_vertex* item_vertices = (item_vertex*) v.item_quad_buffer.mapped_memory;
			item_vertices[new_item_vertex_count].position[0] = center_x - radius;
			item_vertices[new_item_vertex_count].position[1] = center_y - radius;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i];
			item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

			item_vertices[new_item_vertex_count].position[0] = center_x - radius;
			item_vertices[new_item_vertex_count].position[1] = center_y + radius;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i];
			item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;

			item_vertices[new_item_vertex_count].position[0] = center_x + radius;
			item_vertices[new_item_vertex_count].position[1] = center_y - radius;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i];
			item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

			item_vertices[new_item_vertex_count].position[0] = center_x + radius;
			item_vertices[new_item_vertex_count].position[1] = center_y + radius;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i];
			item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;

			item_vertices[new_item_vertex_count].position[0] = center_x + radius;
			item_vertices[new_item_vertex_count].position[1] = center_y - radius;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i];
			item_vertices[new_item_vertex_count].tex_coord[0] = 1.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 0.0f;

			item_vertices[new_item_vertex_count].position[0] = center_x - radius;
			item_vertices[new_item_vertex_count].position[1] = center_y + radius;
			for (unsigned int i = 0; i < 3; i++) item_vertices[new_item_vertex_count].color[i] = color[i];
			item_vertices[new_item_vertex_count].tex_coord[0] = 0.0f;
			item_vertices[new_item_vertex_count++].tex_coord[1] = 1.0f;
		}

		inline void set_background_patch_opacity(uint64_t x, uint64_t y, uint8_t alpha) {
			pixel& p = scent_map_texture_data[y * v.texture_width + x];
			p.a = alpha;
		}

		inline void set_background_patch_row_opacity(uint64_t y, uint8_t alpha) {
			for (unsigned int a = 0; a < num_patches_x; a++) {
				pixel& p = scent_map_texture_data[y * v.texture_width + a];
				p.a = alpha;
			}
		}

		inline void fill_background_patch(
				uint64_t x, uint64_t y,
				uint8_t r, uint8_t g, uint8_t b,
				unsigned int patch_size_texels)
		{
			for (unsigned int j = 0; j < patch_size_texels; j++) {
				for (unsigned int i = 0; i < patch_size_texels; i++) {
					position texture_position = position(x + i, y + j);
					pixel& p = scent_map_texture_data[texture_position.y * v.texture_width + texture_position.x];
					p.r = r; p.g = g; p.b = b;
				}
			}
		}

		inline void fill_background_patch_row(uint64_t y,
				uint8_t r, uint8_t g, uint8_t b,
				unsigned int patch_size_texels)
		{
			for (unsigned int j = 0; j < patch_size_texels; j++) {
				for (unsigned int i = 0; i < texture_width_cells; i++) {
					position texture_position = position(i, y + j);
					pixel& p = scent_map_texture_data[texture_position.y * v.texture_width + texture_position.x];
					p.r = r; p.g = g; p.b = b;
				}
			}
		}

		inline void set_background_cell_color(uint64_t x, uint64_t y, const pixel& color) {
			scent_map_texture_data[y * v.texture_width + x].r = color.r;
			scent_map_texture_data[y * v.texture_width + x].g = color.g;
			scent_map_texture_data[y * v.texture_width + x].b = color.b;
		}

		inline void set_visual_field_cell_color(unsigned int x, unsigned int y, unsigned int visual_field_grid_size, const pixel& color) {
			visual_field_texture_data[x * visual_field_grid_size + y] = color;
		}
	};

	template<typename Stream>
	struct svg_backend
	{
		struct vertex { float x, y; };

		template<unsigned int N>
		struct polygon {
			vertex vertices[N];
			uint8_t color[3];
		};

		typedef polygon<3> triangle;

		struct rectangle {
			vertex bottom_left;
			vertex top_right;
			uint8_t color[3];
		};

		struct circle {
			vertex center;
			float radius;
			uint8_t color[3];
		};

		struct line {
			vertex first;
			vertex second;
		};

		Stream& out;
		uint32_t width;
		uint32_t height;
		array<triangle> triangles;
		array<rectangle> rectangles;
		array<circle> circles;

		unsigned int num_patches_x;
		unsigned int num_patches_y;
		array<rectangle> background_grid_cells;
		array<pair<rectangle, uint8_t>> visual_field_grid_cells;

		uint8_t* background_patch_opacities;

		constexpr static pixel BACKGROUND_COLOR = {0, 0, 0, 1};

		svg_backend(Stream& out, uint32_t width, uint32_t height) :
				out(out), width(width), height(height),
				triangles(64), rectangles(4096), circles(1024),
				background_grid_cells(4096),
				visual_field_grid_cells(256),
				background_patch_opacities(nullptr)
		{ }

		~svg_backend() {
			if (background_patch_opacities != nullptr)
				free(background_patch_opacities);
		}

		inline constexpr unsigned int get_texel_cell_length(float pixel_density) const {
			return 1;
		}

		inline void begin_drawing(unsigned int x_num_patches, unsigned int y_num_patches, unsigned int patch_size_texels) {
			triangles.clear();
			rectangles.clear();
			circles.clear();
			background_grid_cells.clear();
			visual_field_grid_cells.clear();
			num_patches_x = x_num_patches;
			num_patches_y = y_num_patches;
			background_grid_cells.ensure_capacity(x_num_patches * y_num_patches * patch_size_texels * patch_size_texels);
			background_patch_opacities = (uint8_t*) malloc(sizeof(uint8_t) * x_num_patches * y_num_patches);
		}

		inline bool finish_drawing(
				float left, float right, float bottom, float top,
				float background_grid_bottom_left_x, float background_grid_bottom_left_y,
				float background_grid_top_right_x, float background_grid_top_right_y,
				float visual_field_grid_bottom_left_x, float visual_field_grid_bottom_left_y,
				float visual_field_grid_top_right_x, float visual_field_grid_top_right_y,
				float agent_color_r, float agent_color_g, float agent_color_b,
				unsigned int patch_size_texels, unsigned int patch_size,
				bool draw_visual_field_grid)
		{
			bool success = (fprintf(out,
					"<svg width=\"%u\" height=\"%u\" viewBox=\"%.2f %.2f %.2f %.2f\" xmlns=\"http://www.w3.org/2000/svg\">",
					width, height, left, bottom, right - left, top - bottom) > 0);
			success &= (fprintf(out, "<g transform=\"translate(0,%.2f) scale(1,-1)\">", top + bottom) > 0);
			success &= (fprintf(out,
					"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"#%02x%02x%02x\"/>",
					left, bottom, right - left, top - bottom,
					BACKGROUND_COLOR.r, BACKGROUND_COLOR.g, BACKGROUND_COLOR.b) > 0);

			/* draw the background "heatmap" */
			constexpr float grid_line_width = 0.05f;
			success &= (fprintf(out, "<g stroke-width=\"%.2f\">", grid_line_width) > 0);
			for (const rectangle& r : background_grid_cells) {
				/* get the opacity value for this patch */
				unsigned int patch_x = r.bottom_left.x / patch_size;
				unsigned int patch_y = r.bottom_left.y / patch_size;
				float opacity = (float) background_patch_opacities[patch_y * num_patches_x + patch_x] / 255;
				uint8_t border_r = (uint8_t) round(r.color[0] * opacity);
				uint8_t border_g = (uint8_t) round(r.color[1] * opacity);
				uint8_t border_b = (uint8_t) round(r.color[2] * opacity);

				float bottom_left_x = max(r.bottom_left.x + background_grid_bottom_left_x - 0.01f, background_grid_bottom_left_x);
				float bottom_left_y = max(r.bottom_left.y + background_grid_bottom_left_y - 0.01f, background_grid_bottom_left_y);
				float x = bottom_left_x; float y = bottom_left_y;
				float w = min(r.top_right.x - r.bottom_left.x + 0.02f, background_grid_top_right_x - bottom_left_x);
				float h = min(r.top_right.y - r.bottom_left.y + 0.02f, background_grid_top_right_y - bottom_left_y);
				if (x > right || y > top || x + w < left || y + h < bottom)
					continue;
				success &= (fprintf(out,
						"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"#%02x%02x%02x\"/>",
						x, y, w, h, border_r, border_g, border_b) > 0);
			} for (const rectangle& r : background_grid_cells) {
				/* get the opacity value for this patch */
				unsigned int patch_x = r.bottom_left.x / patch_size;
				unsigned int patch_y = r.bottom_left.y / patch_size;
				float opacity = (float) background_patch_opacities[patch_y * num_patches_x + patch_x] / 255;
				uint8_t border_r = (uint8_t) round(r.color[0] * opacity);
				uint8_t border_g = (uint8_t) round(r.color[1] * opacity);
				uint8_t border_b = (uint8_t) round(r.color[2] * opacity);

				float bottom_left_x = r.bottom_left.x + background_grid_bottom_left_x;
				float bottom_left_y = r.bottom_left.y + background_grid_bottom_left_y;
				float x = bottom_left_x + grid_line_width / 2;
				float y = bottom_left_y + grid_line_width / 2;
				float w = r.top_right.x - r.bottom_left.x - grid_line_width;
				float h = r.top_right.y - r.bottom_left.y - grid_line_width;
				if (x > right || y > top || x + w < left || y + h < bottom)
					continue;
				success &= (fprintf(out,
						"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" stroke=\"#%02x%02x%02x\" fill=\"#%02x%02x%02x\"/>",
						x, y, w, h, border_r, border_g, border_b, r.color[0], r.color[1], r.color[2]) > 0);
			}
			success &= (fprintf(out, "</g>") > 0);

			if (draw_visual_field_grid) {
				/* draw the visual field "heatmap" */
				constexpr float vision_grid_line_width = 0.1f;
				success &= (fprintf(out, "<g stroke-width=\"%.2f\">", vision_grid_line_width) > 0);
				for (const pair<rectangle, uint8_t>& r : visual_field_grid_cells) {
					/* get the opacity value for this cell */
					float opacity = (float) r.value / 255;
					uint8_t border_r = (uint8_t) round(r.key.color[0] * opacity);
					uint8_t border_g = (uint8_t) round(r.key.color[1] * opacity);
					uint8_t border_b = (uint8_t) round(r.key.color[2] * opacity);

					float bottom_left_x = max(r.key.bottom_left.x + visual_field_grid_bottom_left_x - 0.01f, visual_field_grid_bottom_left_x);
					float bottom_left_y = max(r.key.bottom_left.y + visual_field_grid_bottom_left_y - 0.01f, visual_field_grid_bottom_left_y);
					float x = bottom_left_x; float y = bottom_left_y;
					float w = min(r.key.top_right.x - r.key.bottom_left.x + 0.02f, visual_field_grid_top_right_x - bottom_left_x);
					float h = min(r.key.top_right.y - r.key.bottom_left.y + 0.02f, visual_field_grid_top_right_y - bottom_left_y);
					if (x > right || y > top || x + w < left || y + h < bottom)
						continue;
					success &= (fprintf(out,
							"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"#%02x%02x%02x\"/>",
							x, y, w, h, border_r, border_g, border_b) > 0);
				} for (const pair<rectangle, uint8_t>& r : visual_field_grid_cells) {
					/* get the opacity value for this cell */
					float opacity = (float) r.value / 255;
					uint8_t border_r = (uint8_t) round(r.key.color[0] * opacity);
					uint8_t border_g = (uint8_t) round(r.key.color[1] * opacity);
					uint8_t border_b = (uint8_t) round(r.key.color[2] * opacity);

					float bottom_left_x = r.key.bottom_left.x + visual_field_grid_bottom_left_x;
					float bottom_left_y = r.key.bottom_left.y + visual_field_grid_bottom_left_y;
					float x = bottom_left_x + vision_grid_line_width / 2;
					float y = bottom_left_y + vision_grid_line_width / 2;
					float w = r.key.top_right.x - r.key.bottom_left.x - vision_grid_line_width;
					float h = r.key.top_right.y - r.key.bottom_left.y - vision_grid_line_width;
					if (x > right || y > top || x + w < left || y + h < bottom)
						continue;
					success &= (fprintf(out,
							"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" stroke=\"#%02x%02x%02x\" fill=\"#%02x%02x%02x\"/>",
							x, y, w, h, border_r, border_g, border_b, r.key.color[0], r.key.color[1], r.key.color[2]) > 0);
				}
				success &= (fprintf(out, "</g>") > 0);

				/* draw the visual field border */
				const float border_width = 0.01f * (visual_field_grid_top_right_x - visual_field_grid_bottom_left_x);
				success &= (fprintf(out,
						"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" stroke=\"#%02x%02x%02x\" fill=\"none\" stroke-width=\"%.2f\"/>",
						visual_field_grid_bottom_left_x + border_width / 2, visual_field_grid_bottom_left_y + border_width / 2,
						visual_field_grid_top_right_x - visual_field_grid_bottom_left_x - border_width,
						visual_field_grid_top_right_y - visual_field_grid_bottom_left_y - border_width,
						(uint8_t) (agent_color_r * 255), (uint8_t) (agent_color_g * 255), (uint8_t) (agent_color_b * 255), border_width) > 0);
			}

			/* draw the polygons and circles (i.e. items and agents) */
			for (const rectangle& r : rectangles) {
				if (r.bottom_left.x > right || r.bottom_left.y > top || r.top_right.x < left || r.top_right.y < bottom)
					continue;
				success &= (fprintf(out,
						"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"#%02x%02x%02x\"/>",
						r.bottom_left.x, r.bottom_left.y, r.top_right.x - r.bottom_left.x, r.top_right.y - r.bottom_left.y,
						r.color[0], r.color[1], r.color[2]) > 0);
			} for (const circle& c : circles) {
				if (c.center.x - c.radius > right || c.center.y - c.radius > top || c.center.x + c.radius < left || c.center.y + c.radius < bottom)
					continue;
				success &= (fprintf(out,
						"<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" fill=\"#%02x%02x%02x\"/>",
						c.center.x, c.center.y, c.radius,
						c.color[0], c.color[1], c.color[2]) > 0);
			} for (const triangle& t : triangles) {
				if ((t.vertices[0].x >= left && t.vertices[0].x <= right && t.vertices[0].y >= bottom && t.vertices[0].y <= top)
				 || (t.vertices[1].x >= left && t.vertices[1].x <= right && t.vertices[1].y >= bottom && t.vertices[1].y <= top)
				 || (t.vertices[2].x >= left && t.vertices[2].x <= right && t.vertices[2].y >= bottom && t.vertices[2].y <= top))
				{
					success &= (fprintf(out,
							"<polygon points=\"%.2f,%.2f %.2f,%.2f %.2f,%.2f\" fill=\"#%02x%02x%02x\"/>",
							t.vertices[0].x, t.vertices[0].y, t.vertices[1].x, t.vertices[1].y, t.vertices[2].x, t.vertices[2].y,
							t.color[0], t.color[1], t.color[2]) > 0);
				}
			}
			success &= (fprintf(out, "</g></svg>") > 0);
			return success;
		}

		inline bool draw_nothing(float left, float right, float bottom, float top, unsigned int patch_size, bool draw_visual_field_grid)
		{
			/* just draw a black background */
			bool success = (fprintf(out,
					"<svg width=\"%u\" height=\"%u\" viewBox=\"%.2f %.2f %.2f %.2f\" xmlns=\"http://www.w3.org/2000/svg\">",
					width, height, left, bottom, right - left, top - bottom) > 0);
			success &= (fprintf(out, "<g transform=\"translate(0,%.2f) scale(1,-1)\">", top + bottom) > 0);
			success &= (fprintf(out,
					"<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"#%02x%02x%02x\"/></g></svg>",
					left, bottom, right - left, top - bottom,
					BACKGROUND_COLOR.r, BACKGROUND_COLOR.g, BACKGROUND_COLOR.b) > 0);
			return success;
		}

		inline bool ensure_capacity(unsigned int requested_triangle_count, unsigned int requested_quad_or_circle_count)
		{
			return triangles.ensure_capacity(requested_triangle_count)
				&& rectangles.ensure_capacity(requested_quad_or_circle_count)
				&& circles.ensure_capacity(requested_quad_or_circle_count);
		}

		inline void add_triangle(
				float v1_x, float v1_y,
				float v2_x, float v2_y,
				float v3_x, float v3_y,
				const float color[3])
		{
			triangle& new_triangle = triangles[triangles.length];
			new_triangle.vertices[0] = {v1_x, v1_y};
			new_triangle.vertices[1] = {v2_x, v2_y};
			new_triangle.vertices[2] = {v3_x, v3_y};
			for (unsigned int i = 0; i < 3; i++) new_triangle.color[i] = (uint8_t) (color[i] * 255);
			triangles.length++;
		}

		inline void add_rect(
				float bottom_left_x, float bottom_left_y,
				float top_right_x, float top_right_y,
				const float color[3])
		{
			rectangle& new_rect = rectangles[rectangles.length];
			new_rect.bottom_left.x = bottom_left_x;
			new_rect.bottom_left.y = bottom_left_y;
			new_rect.top_right.x = top_right_x;
			new_rect.top_right.y = top_right_y;
			for (unsigned int i = 0; i < 3; i++) new_rect.color[i] = (uint8_t) (color[i] * 255);
			rectangles.length++;
		}

		inline void add_circle(
				float center_x, float center_y, float radius, const float color[3])
		{
			circle& new_circle = circles[circles.length];
			new_circle.center = {center_x, center_y};
			new_circle.radius = radius;
			for (unsigned int i = 0; i < 3; i++) new_circle.color[i] = (uint8_t) (color[i] * 255);
			circles.length++;
		}

		inline void set_background_patch_opacity(uint64_t x, uint64_t y, uint8_t alpha) {
			background_patch_opacities[y * num_patches_x + x] = alpha;
		}

		inline void set_background_patch_row_opacity(uint64_t y, uint8_t alpha) {
			for (unsigned int x = 0; x < num_patches_x; x++)
				background_patch_opacities[y * num_patches_x + x] = alpha;
		}

		inline void fill_background_patch(
				uint64_t x, uint64_t y,
				uint8_t r, uint8_t g, uint8_t b,
				unsigned int patch_size_texels)
		{
			if (BACKGROUND_COLOR.r == r && BACKGROUND_COLOR.g == g && BACKGROUND_COLOR.b == b)
				return;
			for (unsigned int x_curr = x; x_curr < x + patch_size_texels; x_curr++)
				for (unsigned int y_curr = y; y_curr < y + patch_size_texels; y_curr++)
					set_background_cell_color(x_curr, y_curr, {r, g, b, 0});
		}

		inline void fill_background_patch_row(uint64_t y,
				uint8_t r, uint8_t g, uint8_t b,
				unsigned int patch_size_texels)
		{
			if (BACKGROUND_COLOR.r == r && BACKGROUND_COLOR.g == g && BACKGROUND_COLOR.b == b)
				return;
			for (unsigned int x = 0; x < num_patches_x * patch_size_texels; x++)
				for (unsigned int y_curr = y; y_curr < y + patch_size_texels; y_curr++)
					set_background_cell_color(x, y_curr, {r, g, b, 0});
		}

		inline void set_background_cell_color(uint64_t x, uint64_t y, const pixel& color) {
			rectangle& cell = background_grid_cells[background_grid_cells.length];
			cell.bottom_left.x = x;
			cell.bottom_left.y = y;
			cell.top_right.x = x + 1;
			cell.top_right.y = y + 1;
			cell.color[0] = color.r;
			cell.color[1] = color.g;
			cell.color[2] = color.b;
			background_grid_cells.length++;
		}

		inline void set_visual_field_cell_color(unsigned int x, unsigned int y, unsigned int visual_field_grid_size, const pixel& color) {
			rectangle cell;
			cell.bottom_left.x = y;
			cell.bottom_left.y = x;
			cell.top_right.x = y + 1;
			cell.top_right.y = x + 1;
			cell.color[0] = color.r;
			cell.color[1] = color.g;
			cell.color[2] = color.b;
			visual_field_grid_cells.add(make_pair(cell, color.a));
		}
	};

private:
	template<typename RenderBackend>
	bool prepare_scene_helper(
			const array<array<patch_state>>& patches,
			position agent_position,
			direction agent_direction,
			const float* agent_visual_field,
			bool render_background_map,
			unsigned int agent_path_length,
			float left, float right,
			float bottom, float top,
			float pixel_density,
			RenderBackend& backend)
	{
		const unsigned int texel_cell_length = backend.get_texel_cell_length(pixel_density);

		const unsigned int patch_size = get_config(sim).patch_size;
		const unsigned int patch_size_texels = (unsigned int) ceil((float) patch_size / texel_cell_length);
		const unsigned int vision_range = get_config(sim).vision_range;
		const unsigned int color_dimension = get_config(sim).color_dimension;
		const unsigned int scent_dimension = get_config(sim).scent_dimension;
		const array<item_properties>& item_types = get_config(sim).item_types;
		const float* agent_color = get_config(sim).agent_color;
		if (patches.length > 0) {
			/* find position of the bottom-left corner and the top-right corner, and compute the max scent */
			float max_scent = 0.0f;
			size_t required_item_quads = 0, required_item_triangles = 0;
			position bottom_left_corner(INT64_MAX, INT64_MAX), top_right_corner(INT64_MIN, INT64_MIN);
			bottom_left_corner.y = patches[0][0].patch_position.y;
			top_right_corner.y = patches.last().last().patch_position.y;
			for (const array<patch_state>& row : patches) {
				bottom_left_corner.x = min(bottom_left_corner.x, row[0].patch_position.x);
				top_right_corner.x = max(top_right_corner.x, row.last().patch_position.x);
				for (const patch_state& patch : row) {
					required_item_quads += patch.item_count;
					required_item_triangles += patch.agent_count;
					if (render_background_map) {
						for (unsigned int a = 0; a < patch_size; a++) {
							for (unsigned int b = 0; b < patch_size; b++) {
								float* cell_scent = patch.scent + ((a*patch_size + b)*scent_dimension);
								max_scent = max(cell_scent[0], max(cell_scent[1], max(cell_scent[2], max_scent)));
							}
						}
					}
				}
			}

			if (agent_path_length > 1)
				required_item_quads += (agent_path_length - 1);

			backend.begin_drawing(top_right_corner.x - bottom_left_corner.x + 1, top_right_corner.y - bottom_left_corner.y + 1, patch_size_texels);
			if (!backend.ensure_capacity(required_item_triangles, required_item_quads))
				return false;

			if (agent_path_length > 1) {
				for (unsigned int i = 1; i < agent_path_length; i++) {
					position first = agent_path[i - 1];
					position second = agent_path[i];

					if (second.x < first.x || second.y < first.y)
						swap(first, second);

					float bottom_left_x, bottom_left_y, top_right_x, top_right_y;
					if (second.x > first.x) {
						constexpr position forward = {1, 0};
						constexpr position right = {0, -1};
						bottom_left_x = first.x + 0.5f + 0.13f * (right.x - forward.x);
						bottom_left_y = first.y + 0.5f + 0.13f * (right.y - forward.y);
						top_right_x = second.x + 0.5f + 0.13f * (-right.x + forward.x);
						top_right_y = second.y + 0.5f + 0.13f * (-right.y + forward.y);
					} else {
						constexpr position forward = {0, 1};
						constexpr position right = {1, 0};
						bottom_left_x = first.x + 0.5f + 0.13f * (-right.x - forward.x);
						bottom_left_y = first.y + 0.5f + 0.13f * (-right.y - forward.y);
						top_right_x = second.x + 0.5f + 0.13f * (right.x + forward.x);
						top_right_y = second.y + 0.5f + 0.13f * (right.y + forward.y);
					}

					static constexpr float path_color[] = { 1.0f, 0.0f, 0.0f };
					backend.add_rect(bottom_left_x, bottom_left_y, top_right_x, top_right_y, path_color);
				}
			}

			unsigned int y_index = 0;
			for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y++) {
				if (y_index == patches.length || y != patches[y_index][0].patch_position.y) {
					/* fill the patches in this row with empty pixels */
					const int64_t patch_offset_y = y - bottom_left_corner.y;
					backend.set_background_patch_row_opacity(patch_offset_y, 255);

					const int64_t offset_y = patch_offset_y * patch_size_texels;
					backend.fill_background_patch_row(offset_y, 0, 0, 0, patch_size_texels);
					continue;
				}
				const array<patch_state>& row = patches[y_index++];

				unsigned int x_index = 0;
				for (int64_t x = bottom_left_corner.x; x <= top_right_corner.x; x++) {
					const position patch_offset = position(x, y) - bottom_left_corner;
					const position offset = patch_offset * patch_size_texels;
					if (x_index == row.length || x != row[x_index].patch_position.x) {
						/* fill this patch with empty pixels */
						backend.set_background_patch_opacity(patch_offset.x, patch_offset.y, 255);
						backend.fill_background_patch(offset.x, offset.y, 0, 0, 0, patch_size_texels);
						continue;
					}

					const patch_state& patch = row[x_index++];
					backend.set_background_patch_opacity(patch_offset.x, patch_offset.y, 240);

					if (!render_background_map) {
						/* fill this patch with blank pixels */
						uint8_t blank = (patch.fixed ? 255 : 204);
						backend.fill_background_patch(offset.x, offset.y, blank, blank, blank, patch_size_texels);
					} else {
						/* fill this patch with values from the scent map */
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
								pixel current_pixel;
								scent_to_color(average_scent, current_pixel, patch.fixed, max_scent);
								backend.set_background_cell_color(texture_position.x, texture_position.y, current_pixel);
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
						if (agent_visual_field != nullptr) {
							const position relative_position = it.location - agent_position;
							if (abs(relative_position.x) <= vision_range && abs(relative_position.y) <= vision_range) {
								continue;
							}
						}
						const item_properties& item_props = item_types[it.item_type];
						float item_color[3];
						for (unsigned int i = 0; i < 3; i++) item_color[i] = process_item_color(item_props.color[i]);
						if (item_props.blocks_movement) {
							backend.add_rect(
									it.location.x + 0.5f - 0.4f, it.location.y + 0.5f - 0.4f,
									it.location.x + 0.5f + 0.4f, it.location.y + 0.5f + 0.4f, item_color);
						} else {
							backend.add_circle(it.location.x + 0.5f, it.location.y + 0.5f, 0.4f, item_color);
						}
					}

					/* iterate over all agents in this patch, creating an oriented triangle for each */
					for (unsigned int i = 0; i < patch.agent_count; i++) {
						float first[2] = {0};
						float second[2] = {0};
						float third[2] = {0};
						get_triangle_coords(patch.agent_directions[i], first, second, third);
						float color[3];
						for (unsigned int j = 0; j < 3; j++) color[j] =  agent_color[j];
						backend.add_triangle(
								patch.agent_positions[i].x + 0.5f + first[0], patch.agent_positions[i].y + 0.5f + first[1],
								patch.agent_positions[i].x + 0.5f + second[0], patch.agent_positions[i].y + 0.5f + second[1],
								patch.agent_positions[i].x + 0.5f + third[0], patch.agent_positions[i].y + 0.5f + third[1], color);
					}
				}
			}

			if (agent_visual_field != nullptr) {
				const unsigned int V = 2 * vision_range + 1;
				for (unsigned int i = 0; i < V; i++) {
					for (unsigned int j = 0; j < V; j++) {
						int index = 0;
						switch (agent_direction) {
						case direction::UP: index = j * V + i; break;
						case direction::DOWN: index = (V - j - 1) * V + V - i - 1; break;
						case direction::LEFT: index = i * V + V - j - 1; break;
						case direction::RIGHT: index = (V - i - 1) * V + j; break;
						case direction::COUNT: break;
						}
						pixel p;
						vision_to_color(agent_visual_field + (index * color_dimension), p);
						p.a = 240;
						backend.set_visual_field_cell_color(i, j, V, p);
					}
				}
			}

			if (!backend.finish_drawing(left, right, bottom, top,
					(float) bottom_left_corner.x * patch_size, (float) bottom_left_corner.y * patch_size,
					(float) (top_right_corner.x + 1) * patch_size, (float) (top_right_corner.y + 1) * patch_size,
					(float) (agent_position.x - vision_range), (float) (agent_position.y - vision_range),
					(float) (agent_position.x + vision_range + 1), (float) (agent_position.y + vision_range + 1),
					agent_color[0], agent_color[1], agent_color[2], patch_size_texels, patch_size, agent_visual_field != nullptr))
			{
				return false;
			}

		} else {
			backend.begin_drawing(0, 0, patch_size_texels);
			if (!backend.draw_nothing(left, right, bottom, top, patch_size, agent_visual_field != nullptr))
				return false;
		}
		return true;
	}

	inline FILE* open_next_available_screenshot_file()
	{
		array<string> existing_files(64);
		if (!get_files_in_directory(existing_files, ".")) {
			fprintf(stderr, "ERROR: Unable to access current directory.\n");
			return nullptr;
		}

		constexpr const char prefix[] = "screenshot";
		constexpr unsigned int prefix_length = array_length(prefix) - 1;
		constexpr const char suffix[] = ".svg";
		constexpr unsigned int suffix_length = array_length(suffix) - 1;

		array<unsigned int> existing_screenshot_ids(16);
		for (const string& filename : existing_files) {
			if (filename.length < max(prefix_length, suffix_length)) continue;
			bool affix_matches = true;
			for (unsigned int i = 0; affix_matches && i < prefix_length; i++)
				if (prefix[i] != filename[i]) affix_matches = false;
			if (!affix_matches) continue;

			for (unsigned int i = 0; affix_matches && i < suffix_length; i++)
				if (suffix[i] != filename[filename.length - suffix_length + i]) affix_matches = false;
			if (!affix_matches) continue;

			unsigned int id;
			if (!parse_uint(string(filename.data + prefix_length, filename.length - suffix_length - prefix_length), id))
				continue;

			if (!existing_screenshot_ids.add(id)) {
				for (string& s : existing_files) free(s);
				return nullptr;
			}
		}
		for (string& s : existing_files) free(s);

		if (existing_screenshot_ids.length > 1)
			sort(existing_screenshot_ids);

		for (unsigned int i = 0; true; i++) {
			if (i >= existing_screenshot_ids.length || existing_screenshot_ids[i] != i) {
				int size = snprintf(nullptr, 0, "%s%u%s", prefix, i, suffix);
				char* filename = (char*) malloc(sizeof(char) * (size + 1));
				if (filename == nullptr) {
					fprintf(stderr, "open_next_available_screenshot_file ERROR: Out of memory.\n");
					return nullptr;
				}
				snprintf(filename, size + 1, "%s%u%s", prefix, i, suffix);

				FILE* out = fopen(filename, "w");
				if (out == nullptr) {
					fprintf(stderr, "ERROR: Unable to open file '%s' for writing.\n", filename);
					return nullptr;
				}
				fprintf(stderr, "Saving screenshot to '%s'.\n", filename);
				return out;
			}
		}
	}

	template<bool HasLock>
	inline bool prepare_scene_helper(array<array<patch_state>>& patches)
	{
		float left = camera_position[0] - 0.5f * (width / pixel_density) - 0.01f;
		float right = camera_position[0] + 0.5f * (width / pixel_density) + 0.01f;
		float bottom = camera_position[1] - 0.5f * (height / pixel_density) - 0.01f;
		float top = camera_position[1] + 0.5f * (height / pixel_density) + 0.01f;
		float current_pixel_density = pixel_density;

		bool render_background_map = render_background;
		if (render_background_map) {
			if (sim.template get_map<true, false>({(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, patches) != status::OK) {
				fprintf(stderr, "visualizer.prepare_scene_helper ERROR: Unable to get map from simulator.\n");
				return false;
			}
		} else {
			if (sim.template get_map<false, false>({(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, patches) != status::OK) {
				fprintf(stderr, "visualizer.prepare_scene_helper ERROR: Unable to get map from simulator.\n");
				return false;
			}
		}

		position agent_position = {0, 0};
		direction agent_direction = direction::UP;
		float* agent_visual_field = nullptr;
		bool render_visual_field = render_agent_visual_field;
		bool render_path = render_agent_path;
		unsigned int render_path_length = 0;
		if (track_agent_id != 0) {
			agent_state* agent;
			sim.get_agent_states(&agent, &track_agent_id, 1);
			if (agent != nullptr) {
				agent_position = agent->current_position;
				agent_direction = agent->current_direction;
				unsigned int color_dimension = sim.get_config().color_dimension;
				unsigned int vision_range = sim.get_config().vision_range;
				unsigned int visual_field_size = sizeof(float) * (2 * vision_range + 1) * (2 * vision_range + 1) * color_dimension;
				if (render_visual_field) {
					agent_visual_field = (float*) malloc(visual_field_size);
					memcpy(agent_visual_field, agent->current_vision, visual_field_size);
				}
				float new_target_position[] = {agent->current_position.x + 0.5f, agent->current_position.y + 0.5f};
#if defined(RECORD)
				record_collected_items(agent->collected_items, sim.get_config().item_types.length);
#endif
				agent->lock.unlock();

				if (render_path) {
					agent_path_lock.lock();
					add_to_agent_path(agent_position);
					agent_position_recorded = true;
					render_path_length = agent_path.length;
					agent_path_cv.notify_one();
					agent_path_lock.unlock();
				}

				if (new_target_position[0] != translate_end_position[0] || new_target_position[1] != translate_end_position[1]) {
					float animation_t = max(0.0f, min(1.0f, (jbw::milliseconds() - translate_animation_start_time) / 300.0f));
					float easing = (animation_t - 1) * (animation_t - 1) * (animation_t - 1) + 1;
					translate_start_position[0] = easing * translate_end_position[0] + (1.0f - easing) * translate_start_position[0];
					translate_start_position[1] = easing * translate_end_position[1] + (1.0f - easing) * translate_start_position[1];
					translate_end_position[0] = new_target_position[0];
					translate_end_position[1] = new_target_position[1];
					tracking_animating = false;
				}
				if (!tracking_animating) {
					translate_animation_start_time = jbw::milliseconds();
					tracking_animating = true;
				}
			} else {
				fprintf(stderr, "Agent with ID %" PRIu64 " does not exist in the simulation.\n", track_agent_id);
				track_agent(0);
			}
		} else {
			tracking_animating = false;
		}

		vulkan_glfw_backend<HasLock> backend(*this); /* TODO: maybe this should be a class variable */
		bool result = prepare_scene_helper(
			patches, agent_position, agent_direction,
			agent_visual_field, render_background_map,
			render_path_length, left, right, bottom, top,
			current_pixel_density, backend);
#if defined(RECORD)
		fprintf(stderr, "simulation_time: %lu\n", simulation_time);
		if (simulation_time >= 0)
			screenshot_next_frame = true;
		increment_frame_number();
#endif
		if (screenshot_next_frame) {
			/* pick next available filename to avoid overwrites */
			FILE* out = open_next_available_screenshot_file();
			if (out != nullptr) {
				svg_backend<FILE*> svg(out, width, height);
				prepare_scene_helper(
					patches, agent_position, agent_direction,
					agent_visual_field, render_background_map,
					render_path_length, left, right, bottom, top,
					current_pixel_density, svg);
				fflush(out); fclose(out);
			}
#if defined(RECORD)
			write_to_log();
#else
			screenshot_next_frame = false;
#endif
		}
		if (agent_visual_field != nullptr) { free(agent_visual_field); }
		return result;
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
		sim.data.get_map_render_agent_path = render_agent_path;
		sim.data.pixel_density = pixel_density;
		while (!mpi_lock.try_lock()) { }
		if (!send_get_map(sim, {(int64_t) left, (int64_t) bottom}, {(int64_t) ceil(right), (int64_t) ceil(top)}, sim.data.get_map_render_background, false)) {
			mpi_lock.unlock();
			fprintf(stderr, "visualizer.send_mpi_requests ERROR: Unable to send `get_map` message to server.\n");
			sim.data.waiting_for_get_map = false;
			return false;
		}
		sim.data.track_agent_id = track_agent_id;
		if (sim.data.track_agent_id != 0) {
			sim.data.waiting_for_get_agent_states = true;
			sim.data.render_visual_field = render_agent_visual_field;
			if (!send_get_agent_states(sim, &sim.data.track_agent_id, 1)) {
				mpi_lock.unlock();
				fprintf(stderr, "visualizer.send_mpi_requests ERROR: Unable to send `get_agent_states` message to server.\n");
				sim.data.waiting_for_get_agent_states = false;
				return false;
			}
			mpi_lock.unlock();
		} else {
			mpi_lock.unlock();
			sim.data.waiting_for_get_agent_states = false;
			sim.data.get_agent_states_response = status::OK;
		}
		return true;
	}

	template<bool HasLock>
	inline void process_mpi_response(visualizer_client_data& response)
	{
		position agent_position = {0, 0};
		direction agent_direction = direction::UP;
		float* agent_visual_field = nullptr;
		unsigned int render_path_length = 0;
		if (response.get_agent_states_response == status::OK && response.track_agent_id != 0) {
			if (response.agent_state_count > 0) {
				agent_position = response.agent_states[0].current_position;
				agent_direction = response.agent_states[0].current_direction;
				if (response.render_visual_field)
					agent_visual_field = response.agent_states[0].current_vision;
#if defined(RECORD)
				record_collected_items(response.agent_states[0].collected_items, sim.config.item_types.length);
#endif

				if (response.get_map_render_agent_path) {
					agent_path_lock.lock();
					add_to_agent_path(agent_position);
					agent_position_recorded = true;
					render_path_length = agent_path.length;
					agent_path_cv.notify_one();
					agent_path_lock.unlock();
				}

				float new_target_position[] = {response.agent_states[0].current_position.x + 0.5f, response.agent_states[0].current_position.y + 0.5f};

				if (new_target_position[0] != translate_end_position[0] || new_target_position[1] != translate_end_position[1]) {
					float animation_t = max(0.0f, min(1.0f, (jbw::milliseconds() - translate_animation_start_time) / 300.0f));
					float easing = (animation_t - 1) * (animation_t - 1) * (animation_t - 1) + 1;
					translate_start_position[0] = easing * translate_end_position[0] + (1.0f - easing) * translate_start_position[0];
					translate_start_position[1] = easing * translate_end_position[1] + (1.0f - easing) * translate_start_position[1];
					translate_end_position[0] = new_target_position[0];
					translate_end_position[1] = new_target_position[1];
					tracking_animating = false;
				}
				if (!tracking_animating) {
					translate_animation_start_time = jbw::milliseconds();
					tracking_animating = true;
				}
			} else {
				fprintf(stderr, "Agent with ID %" PRIu64 " does not exist in the simulation.\n", response.track_agent_id);
				track_agent(0);
			}
		} else {
			tracking_animating = false;
		}

		if (response.get_map_response == status::OK) {
			vulkan_glfw_backend<HasLock> backend(*this); /* TODO: maybe this should be a class variable */
			prepare_scene_helper(
				*response.map, agent_position, agent_direction, agent_visual_field,
				response.get_map_render_background, render_path_length,
				response.get_map_left, response.get_map_right,
				response.get_map_bottom, response.get_map_top,
				response.pixel_density, backend);

#if defined(RECORD)
			fprintf(stderr, "simulation_time: %lu\n", simulation_time);
			if (simulation_time >= 0)
				screenshot_next_frame = true;
			increment_frame_number();
#endif
			if (screenshot_next_frame) {
				/* pick next available filename to avoid overwrites */
				FILE* out = open_next_available_screenshot_file();
				if (out != nullptr) {
					svg_backend<FILE*> svg(out, width, height);
					prepare_scene_helper(
						*response.map, agent_position, agent_direction, agent_visual_field,
						response.get_map_render_background, render_path_length,
						response.get_map_left, response.get_map_right,
						response.get_map_bottom, response.get_map_top,
						response.pixel_density, svg);
					fflush(out); fclose(out);
				}
#if defined(RECORD)
				write_to_log();
#else
				screenshot_next_frame = false;
#endif
			}

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
			track_agent(0);
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

	template<typename SimulatorData>
	bool create_semaphore(simulator<SimulatorData>& sim) {
		if (sim.add_semaphore(semaphore) != status::OK) {
			fprintf(stderr, "visualizer.create_semaphore ERROR: Unable to add simulator semaphore.\n");
			return false;
		} else if (sim.signal_semaphore(semaphore) != status::OK) {
			fprintf(stderr, "visualizer.create_semaphore ERROR: Unable to signal simulator semaphore.\n");
			return false;
		}
		return true;
	}

	template<typename SimulatorData>
	void delete_semaphore(simulator<SimulatorData>& sim) {
		if (sim.remove_semaphore(semaphore) != status::OK)
			fprintf(stderr, "visualizer.delete_semaphore ERROR: Unable to remove simulator semaphore.\n");
	}

	template<typename SimulatorData>
	void signal_semaphore(simulator<SimulatorData>& sim) {
		status result = sim.signal_semaphore(semaphore);
		if (result != status::OK && result != status::SEMAPHORE_ALREADY_SIGNALED)
			fprintf(stderr, "visualizer.signal_semaphore ERROR: Unable to signal simulator semaphore.\n");
	}

	bool create_semaphore(client<visualizer_client_data>& sim)
	{
		sim.data.waiting_for_semaphore_op = true;
		mpi_lock.lock();
		if (!send_add_semaphore(sim)) {
			mpi_lock.unlock();
			fprintf(stderr, "visualizer.create_semaphore ERROR: Unable to send `add_semaphore` to server.\n");
			return false;
		}
		mpi_lock.unlock();

		/* wait for `add_semaphore` response */
		while (sim.client_running && sim.data.waiting_for_semaphore_op) { }
		if (!sim.client_running) return false;

		if (sim.data.semaphore_op_response != status::OK
		 && sim.data.semaphore_op_response != status::SEMAPHORE_ALREADY_SIGNALED)
		{
			fprintf(stderr, "visualizer.create_semaphore ERROR: `add_semaphore` failed.\n");
			return false;
		}
		semaphore = sim.data.semaphore_id;
		return true;
	}

	void delete_semaphore(client<visualizer_client_data>& sim) {
		if (sim.client_running) return;
		sim.data.waiting_for_semaphore_op = true;
		mpi_lock.lock();
		if (!send_remove_semaphore(sim, semaphore)) {
			mpi_lock.unlock();
			fprintf(stderr, "visualizer.delete_semaphore ERROR: Unable to send `remove_semaphore` to server.\n");
			return;
		}
		mpi_lock.unlock();

		/* wait for `add_semaphore` response */
		while (sim.client_running && sim.data.waiting_for_semaphore_op) { }
	}

	void signal_semaphore(client<visualizer_client_data>& sim) {
		while (sim.client_running && sim.data.waiting_for_semaphore_op) { }
		if (!sim.client_running) return;

		if (sim.data.semaphore_op_response != status::OK
		 && sim.data.semaphore_op_response != status::SEMAPHORE_ALREADY_SIGNALED)
		{
			fprintf(stderr, "visualizer.signal_semaphore ERROR: `signal_semaphore` failed.\n");
			sim.data.semaphore_op_response = status::OK;
			return;
		}

		sim.data.waiting_for_semaphore_op = true;
		while (!mpi_lock.try_lock()) { }
		if (!send_signal_semaphore(sim, semaphore)) {
			mpi_lock.unlock();
			fprintf(stderr, "visualizer.signal_semaphore ERROR: Unable to send `signal_semaphore` to server.\n");
			return;
		}
		mpi_lock.unlock();
	}

	bool setup_renderer() {
		dynamic_texture_image textures[] = { scent_map_texture, visual_field_texture };
		descriptor_type pool_types[] = { descriptor_type::UNIFORM_BUFFER, descriptor_type::COMBINED_IMAGE_SAMPLER };
		uint32_t descriptor_counts[] = { 1, 2 };
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
		} else if (!renderer.create_graphics_pipeline(visual_field_pipeline, pass,
				background_vertex_shader, "main", visual_field_fragment_shader, "main",
				primitive_topology::TRIANGLE_STRIP, false, 1.0f,
				background_binding, background_shader_attributes, &layout, 1)) {
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_render_pass(pass);
		} else if (!renderer.create_frame_buffer(fb, pass)) {
			renderer.delete_graphics_pipeline(visual_field_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_uniform_buffer(ub, sizeof(uniform_buffer_data))) {
			renderer.delete_frame_buffer(fb);
			renderer.delete_graphics_pipeline(visual_field_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_descriptor_pool(pool, pool_types, descriptor_counts, 2)) {
			renderer.delete_uniform_buffer(ub);
			renderer.delete_frame_buffer(fb);
			renderer.delete_graphics_pipeline(visual_field_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		} else if (!renderer.create_descriptor_set(ds, &ub, 0, 1,
					nullptr, 0, textures, 2, 1, &tex_sampler, layout, pool)
				|| !renderer.create_command_buffer(cb)) 
		{
			renderer.delete_descriptor_pool(pool);
			renderer.delete_uniform_buffer(ub);
			renderer.delete_frame_buffer(fb);
			renderer.delete_graphics_pipeline(visual_field_pipeline);
			renderer.delete_graphics_pipeline(item_pipeline);
			renderer.delete_graphics_pipeline(scent_map_pipeline);
			renderer.delete_render_pass(pass);
			return false;
		}
		return true;
	}

	inline void cleanup_renderer()
	{
		renderer.delete_command_buffer(cb);
		renderer.delete_uniform_buffer(ub);
		renderer.delete_descriptor_set(ds);
		renderer.delete_descriptor_pool(pool);
		renderer.delete_frame_buffer(fb);
		renderer.delete_graphics_pipeline(visual_field_pipeline);
		renderer.delete_graphics_pipeline(item_pipeline);
		renderer.delete_graphics_pipeline(scent_map_pipeline);
		renderer.delete_render_pass(pass);
	}

	static inline float gamma_correction(const float channel_value) {
		float corrected_value;
		if (channel_value <= 0.0031308f) {
			corrected_value = 12.92f * channel_value;
		} else {
			corrected_value = 1.055f * pow(channel_value, 1.0f / 2.4f) - 0.055f;
		}
		return max(0.0f, min(1.0f, corrected_value));
	}

	static inline void invert_scent_color_brightness(
			const float x, const float y, const float z,
			float& r, float& g, float& b
	) {
		float m = max(x, max(y, z));
		r = min(1.0f, x + 1.0f - m);
		g = min(1.0f, y + 1.0f - m);
		b = min(1.0f, z + 1.0f - m);

		r = gamma_correction(r);
		g = gamma_correction(g);
		b = gamma_correction(b);
	}

	static inline void invert_vision_color_brightness(
		const float x, const float y, const float z,
		float& r, float& g, float& b
	) {
		/* Convert from RGB to HSL. */
		float min_c = min(x, min(y, z));
		float max_c = max(x, max(y, z));
		float delta = max_c - min_c;
		float h = 0;
		float s = 0;
		float l = (max_c + min_c) / 2.0f;
		if (delta != 0) {
			if (l < 0.5f) {
				s = delta / (max_c + min_c);
			} else {
				s = delta / (2.0f - max_c - min_c);
			}
			if (x == max_c) {
				h = (y - z) / delta;
			} else if (y == max_c) {
				h = 2.0f + (z - x) / delta;
			} else if (z == max_c) {
				h = 4.0f + (x - y) / delta;
			}
		}

		/* Adjust hue and lightness. */
		h /= 6.0f;
		l = 1.0f - l;

		/* Convert from HSL to RGB. */
		auto color_calc = [](float c, const float t1, const float t2) {
			if (c < 0) c += 1.0f;
			if (c > 1) c -= 1.0f;
			if (6.0f * c < 1.0f) return t1 + (t2 - t1) * 6.0f * c;
			if (2.0f * c < 1.0f) return t2;
			if (3.0f * c < 2.0f) return t1 + (t2 - t1) * (2.0f / 3.0f - c) * 6.0f;
			return t1;
		};

		if (s == 0.0f) {
			r = l;
			g = l;
			b = l;
		} else {
			float t2 = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
			float t1 = 2.0f * l - t2;
			r = color_calc(h + 1.0f / 3.0f, t1,  t2);
			g = color_calc(h, t1,  t2);
			b = color_calc(h - 1.0f / 3.0f, t1,  t2);
		}

		r = gamma_correction(r);
		g = gamma_correction(g);
		b = gamma_correction(b);
	}

	static inline void scent_to_color(
		const float* cell_scent,
		pixel& out, bool is_patch_fixed,
		const float max_scent
	) {
		const float scent_x = cell_scent[0];
		const float scent_y = cell_scent[1];
		const float scent_z = cell_scent[2];
		float x = max(0.0f, min(1.0f, pow(1.1f * scent_x / max_scent, 0.22f)));
		float y = max(0.0f, min(1.0f, pow(1.1f * scent_y / max_scent, 0.22f)));
		float z = max(0.0f, min(1.0f, pow(1.1f * scent_z / max_scent, 0.22f)));

		float r, g, b;
		invert_scent_color_brightness(x, y, z, r, g, b);

		if (is_patch_fixed) {
			out.r = (uint8_t) (255 * r);
			out.g = (uint8_t) (255 * g);
			out.b = (uint8_t) (255 * b);
		} else {
			constexpr float black_alpha = 0.2f;
			out.r = (uint8_t) (255 * ((1 - black_alpha) * r));
			out.g = (uint8_t) (255 * ((1 - black_alpha) * g));
			out.b = (uint8_t) (255 * ((1 - black_alpha) * b));
		}
	}

	static inline void vision_to_color(const float* cell_vision, pixel& out) {
		float r, g, b;
		invert_vision_color_brightness(cell_vision[0], cell_vision[1], cell_vision[2], r, g, b);
		out.r = (uint8_t) (255 * r);
		out.g = (uint8_t) (255 * g);
		out.b = (uint8_t) (255 * b);
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
	friend void on_step(client<visualizer_client_data>&,
		status, const array<uint64_t>&, const agent_state*);
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

void on_add_semaphore(client<visualizer_client_data>& c,
		uint64_t semaphore_id, status response)
{
	c.data.semaphore_op_response = response;
	c.data.semaphore_id = semaphore_id;
	c.data.waiting_for_semaphore_op = false;
}

void on_remove_semaphore(client<visualizer_client_data>& c,
		uint64_t semaphore_id, status response)
{
	c.data.semaphore_op_response = response;
	c.data.semaphore_id = semaphore_id;
	c.data.waiting_for_semaphore_op = false;
}

void on_signal_semaphore(client<visualizer_client_data>& c,
		uint64_t semaphore_id, status response)
{
	c.data.semaphore_op_response = response;
	c.data.semaphore_id = semaphore_id;
	c.data.waiting_for_semaphore_op = false;
}

void on_get_semaphores(client<visualizer_client_data>& c, status response,
		uint64_t* semaphore_ids, bool* signaled, size_t semaphore_count)
{
	fprintf(stderr, "WARNING: `on_get_semaphores` should not be called.\n");
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

inline void on_step(
		client<visualizer_client_data>& c,
		status response,
		const array<uint64_t>& agent_ids,
		const agent_state* agent_state_array)
{
#if defined(RECORD)
	record_step();
#endif
}

void on_lost_connection(client<visualizer_client_data>& c) {
	fprintf(stderr, "Lost connection to the server.\n");
	c.client_running = false;
	c.data.painter->running = false;
	c.data.painter->scene_ready_cv.notify_all();
}

} /* namespace jbw */

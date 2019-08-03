#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>
#include <stdexcept>
#include <limits.h>
#include <new>

namespace mirage {

inline size_t min(size_t a, size_t b) {
	return (a < b) ? a : b;
}

inline size_t max(size_t a, size_t b) {
	return (a > b) ? a : b;
}

enum class device_selector {
	FIRST_DISCRETE_GPU,
	FIRST_INTEGRATED_GPU,
	FIRST_VIRTUAL_GPU,
	FIRST_CPU,
	FIRST_OTHER,
	FIRST_ANY
};

enum class primitive_topology {
	POINT_LIST = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    LINE_LIST = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    LINE_STRIP = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
    TRIANGLE_LIST = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    TRIANGLE_STRIP = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    TRIANGLE_FAN = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    LINE_LIST_WITH_ADJACENCY = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
    LINE_STRIP_WITH_ADJACENCY = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
    TRIANGLE_LIST_WITH_ADJACENCY = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
    TRIANGLE_STRIP_WITH_ADJACENCY = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY
};

enum class attribute_type {
	HALF = VK_FORMAT_R16_SFLOAT,
	HALF2 = VK_FORMAT_R16G16_SFLOAT,
	HALF3 = VK_FORMAT_R16G16B16_SFLOAT,
	HALF4 = VK_FORMAT_R16G16B16A16_SFLOAT,
	FLOAT = VK_FORMAT_R32_SFLOAT,
	FLOAT2 = VK_FORMAT_R32G32_SFLOAT,
	FLOAT3 = VK_FORMAT_R32G32B32_SFLOAT,
	FLOAT4 = VK_FORMAT_R32G32B32A32_SFLOAT,
	DOUBLE = VK_FORMAT_R64_SFLOAT,
	DOUBLE2 = VK_FORMAT_R64G64_SFLOAT,
	DOUBLE3 = VK_FORMAT_R64G64B64_SFLOAT,
	DOUBLE4 = VK_FORMAT_R64G64B64A64_SFLOAT,
	INT = VK_FORMAT_R32_SINT,
	INT2 = VK_FORMAT_R32G32_SINT,
	INT3 = VK_FORMAT_R32G32B32_SINT,
	INT4 = VK_FORMAT_R32G32B32A32_SINT,
	UINT = VK_FORMAT_R32_UINT,
	UINT2 = VK_FORMAT_R32G32_UINT,
	UINT3 = VK_FORMAT_R32G32B32_UINT,
	UINT4 = VK_FORMAT_R32G32B32A32_UINT
};

enum class descriptor_type {
	COMBINED_IMAGE_SAMPLER = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	UNIFORM_BUFFER = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
};

enum class shader_stage : uint32_t {
	VERTEX = VK_SHADER_STAGE_VERTEX_BIT,
	FRAGMENT = VK_SHADER_STAGE_FRAGMENT_BIT,
	GEOMETRY = VK_SHADER_STAGE_GEOMETRY_BIT
};

enum class filter {
	NEAREST = VK_FILTER_NEAREST,
	LINEAR = VK_FILTER_LINEAR
};

enum class sampler_address_mode {
	REPEAT = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	MIRRORED_REPEAT = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
	CLAMP_TO_EDGE = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
	MIRROR_CLAMP_TO_EDGE = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
	CLAMP_TO_BORDER = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
};

class shader {
	VkShaderModule module;

	friend class vulkan_renderer;
};

class graphics_pipeline {
	VkRenderPass render_pass;
	VkPipelineLayout layout;
	VkPipeline pipeline;

	friend class vulkan_renderer;
};

class frame_buffer {
	VkFramebuffer* swap_chain_framebuffers;

	friend class vulkan_renderer;
};

class command_buffer {
	VkCommandBuffer* command_buffers;

	friend class vulkan_renderer;
};

class binding_description {
	VkVertexInputBindingDescription description;

public:
	binding_description(uint32_t binding, uint32_t stride) {
		description.binding = binding;
		description.stride = stride;
		description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	}

	friend class vulkan_renderer;
};

template<size_t Count>
class attribute_descriptions {
	VkVertexInputAttributeDescription descriptions[Count];

public:
	template<size_t Index>
	inline void set(
			uint32_t binding, uint32_t location,
			attribute_type type, uint32_t offset)
	{
		descriptions[Index].binding = binding;
		descriptions[Index].location = location;
		descriptions[Index].format = (VkFormat) type;
		descriptions[Index].offset = offset;
	}

	friend class vulkan_renderer;
};

class vertex_buffer {
	VkBuffer buffer;
	VkDeviceMemory memory;

	friend class vulkan_renderer;
};

class uniform_buffer {
	VkBuffer* buffers;
	VkDeviceMemory* memories;
	uint64_t size;

	friend class vulkan_renderer;

public:
	static constexpr descriptor_type DescriptorType = descriptor_type::UNIFORM_BUFFER;
};

class descriptor_set_layout {
	VkDescriptorSetLayout layout;

	friend class vulkan_renderer;
};

class descriptor_pool {
	VkDescriptorPool pool;

	friend class vulkan_renderer;
};

class descriptor_set {
	VkDescriptorSet* sets;

	friend class vulkan_renderer;
};

class texture_image {
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;

	friend class vulkan_renderer;
};

class dynamic_texture_image {
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	VkBuffer staging_buffer;
	VkDeviceMemory staging_buffer_memory;
	uint32_t width;
	uint32_t height;

public:
	void* mapped_memory;

	friend class vulkan_renderer;
};

class sampler {
	VkSampler sampler;

	friend class vulkan_renderer;
};


/* TODO: fix detection for Win32 */
#if defined(HND)
#include <vulkan/vulkan_win32.h>

struct win32_surface {
	HWND handle;
};
#endif


/* TODO: fix detection for Wayland */
#if defined(wl_display)
#include <vulkan/vulkan_wayland.h>

struct wayland_surface {
	wl_display* display;
	wl_surface* surface;
};
#endif


#if MAC_OS_X_VERSION_MAX_ALLOWED >= 101100
#include <vulkan/vulkan_macos.h>

struct macos_surface {
	const void* view;

	macos_surface(const void* view) : view(view) { }
};
#endif


/* TODO: fix detection for XCB */
#if defined(xcb_connection_t)
#include <vulkan/vulkan_xcb.h>

struct xcb_surface {
	xcb_connection_t* connection;
	xcb_window_t window;

	xcb_surface(xcb_connection_t* connection, xcb_window_t window) : connection(connection), window(window) { }
};
#endif


/* TODO: fix detection for X11 */
#if defined(Display) && defined(Window)
#include <vulkan/vulkan_xlib.h>

struct x11_surface {
	Display* display;
	Window window;

	x11_surface(Display* display, Window window) : display(display), window(window) { }
};
#endif


#if defined(_glfw3_h_)
struct glfw_surface {
	GLFWwindow* window;

	glfw_surface(GLFWwindow* window) : window(window) { }
};
#endif


class vulkan_renderer
{
	VkInstance instance;
	VkSurfaceKHR surface;
	VkPhysicalDevice physical_device;
	VkDevice logical_device;
	VkQueue queue;
	VkCommandPool command_pool;

	VkSwapchainKHR swap_chain;
	VkImage* swap_chain_images;
	uint32_t swap_chain_image_count;
	VkFormat swap_chain_image_format;
	VkExtent2D swap_chain_extent;

	VkImageView* swap_chain_image_views;

	VkSemaphore* image_available_semaphores;
	VkSemaphore* render_finished_semaphores;
	VkFence* in_flight_fences;
	unsigned int max_frames_in_flight;
	unsigned int current_frame = 0;

public:
	/* NOTE: this constructor does not initialize this `vulkan_renderer` */
	vulkan_renderer() { }

	template<typename SurfaceType>
	vulkan_renderer(const char* application_name, uint32_t application_version,
			const char* engine_name, uint32_t engine_version,
			const char* const* enabled_extensions, uint32_t extension_count,
			device_selector device_selection, const SurfaceType& window,
			uint32_t window_width, uint32_t window_height,
			unsigned int max_frames_in_flight,
			bool require_anisotropic_filtering) :
		max_frames_in_flight(max_frames_in_flight)
	{
		if (!init_helper(application_name, application_version, engine_name, engine_version,
					enabled_extensions, extension_count, device_selection, window,
					window_width, window_height, require_anisotropic_filtering))
		{
			throw new std::runtime_error("Failed to initialize vulkan_renderer.");
		}
	}

	/* the renderer is not copyable */
	vulkan_renderer(const vulkan_renderer& src) = delete;

	~vulkan_renderer() {
		vkDeviceWaitIdle(logical_device);
		for (unsigned int i = 0; i < max_frames_in_flight; i++) {
			vkDestroySemaphore(logical_device, render_finished_semaphores[i], nullptr);
			vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
			vkDestroyFence(logical_device, in_flight_fences[i], nullptr);
		}
		free(render_finished_semaphores); free(image_available_semaphores); free(in_flight_fences);
		vkDestroyCommandPool(logical_device, command_pool, nullptr);
		free_swap_chain();
		vkDestroyDevice(logical_device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
	}

	inline void wait_until_idle() {
		vkDeviceWaitIdle(logical_device);
	}

	inline bool create_shader(shader& out,
			const char* data, size_t length)
	{
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = length;
		createInfo.pCode = reinterpret_cast<const uint32_t*>(data);
		return (vkCreateShaderModule(logical_device, &createInfo, nullptr, &out.module) == VK_SUCCESS);
	}

	inline void delete_shader(shader& shader) {
		vkDestroyShaderModule(logical_device, shader.module, nullptr);
	}

	inline bool create_graphics_pipeline(graphics_pipeline& pipeline,
			shader& vertex_shader, const char* vertex_shader_entry_point,
			shader& fragment_shader, const char* fragment_shader_entry_point,
			primitive_topology topology)
	{
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertex_shader.module;
		vertShaderStageInfo.pName = vertex_shader_entry_point;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragment_shader.module;
		fragShaderStageInfo.pName = fragment_shader_entry_point;

		VkPipelineShaderStageCreateInfo shaders[] = {vertShaderStageInfo, fragShaderStageInfo};
		return create_graphics_pipeline(pipeline, shaders, topology);
	}

	template<size_t N>
	inline bool create_graphics_pipeline(graphics_pipeline& pipeline,
			shader& vertex_shader, const char* vertex_shader_entry_point,
			shader& fragment_shader, const char* fragment_shader_entry_point,
			primitive_topology topology, const binding_description& binding,
			const attribute_descriptions<N>& attributes)
	{
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertex_shader.module;
		vertShaderStageInfo.pName = vertex_shader_entry_point;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragment_shader.module;
		fragShaderStageInfo.pName = fragment_shader_entry_point;

		VkPipelineShaderStageCreateInfo shaders[] = {vertShaderStageInfo, fragShaderStageInfo};
		return create_graphics_pipeline(pipeline, shaders, topology, binding.description, attributes.descriptions);
	}

	template<size_t N>
	inline bool create_graphics_pipeline(graphics_pipeline& pipeline,
			shader& vertex_shader, const char* vertex_shader_entry_point,
			shader& fragment_shader, const char* fragment_shader_entry_point,
			primitive_topology topology, const binding_description& binding,
			const attribute_descriptions<N>& attributes,
			const descriptor_set_layout* layouts, uint32_t layout_count)
	{
		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertex_shader.module;
		vertShaderStageInfo.pName = vertex_shader_entry_point;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragment_shader.module;
		fragShaderStageInfo.pName = fragment_shader_entry_point;

		VkPipelineShaderStageCreateInfo shaders[] = {vertShaderStageInfo, fragShaderStageInfo};
		return create_graphics_pipeline(pipeline, shaders, topology, binding.description, attributes.descriptions, layouts, layout_count);
	}

	inline void delete_graphics_pipeline(graphics_pipeline& pipeline) {
		vkDestroyPipeline(logical_device, pipeline.pipeline, nullptr);
		vkDestroyPipelineLayout(logical_device, pipeline.layout, nullptr);
		vkDestroyRenderPass(logical_device, pipeline.render_pass, nullptr);
	}

	inline bool create_frame_buffer(frame_buffer& buffer, const graphics_pipeline& pipeline) {
		buffer.swap_chain_framebuffers = (VkFramebuffer*) malloc(sizeof(VkFramebuffer) * swap_chain_image_count);
		if (buffer.swap_chain_framebuffers == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_frame_buffer ERROR: Out of memory.\n");
			return false;
		}

		for (uint32_t i = 0; i < swap_chain_image_count; i++) {
			VkImageView attachments[] = { swap_chain_image_views[i] };

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = pipeline.render_pass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = swap_chain_extent.width;
			framebufferInfo.height = swap_chain_extent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(logical_device, &framebufferInfo, nullptr, &buffer.swap_chain_framebuffers[i]) != VK_SUCCESS) {
				fprintf(stderr, "vulkan_renderer.create_frame_buffer ERROR: Failed to create framebuffer.\n");
				for (uint32_t j = 0; j < i; j++)
					vkDestroyFramebuffer(logical_device, buffer.swap_chain_framebuffers[j], nullptr);
				free(buffer.swap_chain_framebuffers); return false;
			}
		}
		return true;
	}

	inline void delete_frame_buffer(frame_buffer& buffer) {
		for (uint32_t j = 0; j < swap_chain_image_count; j++)
			vkDestroyFramebuffer(logical_device, buffer.swap_chain_framebuffers[j], nullptr);
		free(buffer.swap_chain_framebuffers);
	}

	inline bool create_command_buffer(command_buffer& buffer) {
		buffer.command_buffers = (VkCommandBuffer*) malloc(sizeof(VkCommandBuffer) * swap_chain_image_count);
		if (buffer.command_buffers == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_command_buffer ERROR: Out of memory.\n");
			return false;
		}

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = command_pool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = swap_chain_image_count;

		if (vkAllocateCommandBuffers(logical_device, &allocInfo, buffer.command_buffers) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_command_buffer ERROR: Failed to allocate command buffers.\n");
			free(buffer.command_buffers); return false;
		}
		return true;
	}

	inline void delete_command_buffer(command_buffer& buffer) {
		vkFreeCommandBuffers(logical_device, command_pool, swap_chain_image_count, buffer.command_buffers);
		free(buffer.command_buffers);
	}

	inline bool record_command_buffer(
			command_buffer& cb, frame_buffer& fb,
			graphics_pipeline& pipeline, float (&clear_color)[4],
			uint32_t vertex_count, uint32_t first_vertex)
	{
		for (uint32_t i = 0; i < swap_chain_image_count; i++)
		{
			if (!begin_render_pass(cb.command_buffers[i], fb.swap_chain_framebuffers[i], pipeline, clear_color))
				return false;

			vkCmdBindPipeline(cb.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
			vkCmdDraw(cb.command_buffers[i], vertex_count, 1, first_vertex, 0);

			if (!end_render_pass(cb.command_buffers[i]))
				return false;
		}
		return true;
	}

	template<size_t N>
	inline bool record_command_buffer(
			command_buffer& cb, frame_buffer& fb,
			graphics_pipeline& pipeline, float (&clear_color)[4],
			uint32_t vertex_count, uint32_t first_vertex,
			const vertex_buffer (&vertex_buffers)[N],
			const uint64_t (&offsets)[N])
	{
		VkBuffer vbs[N];
		for (unsigned int j = 0; j < N; j++)
			vbs[j] = vertex_buffers[j].buffer;
		for (uint32_t i = 0; i < swap_chain_image_count; i++)
		{
			if (!begin_render_pass(cb.command_buffers[i], fb.swap_chain_framebuffers[i], pipeline, clear_color))
				return false;

			vkCmdBindPipeline(cb.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
			vkCmdBindVertexBuffers(cb.command_buffers[i], 0, N, vbs, offsets);
			vkCmdDraw(cb.command_buffers[i], vertex_count, 1, first_vertex, 0);

			if (!end_render_pass(cb.command_buffers[i]))
				return false;
		}
		return true;
	}

	template<size_t N>
	inline bool record_command_buffer(
			command_buffer& cb, frame_buffer& fb,
			graphics_pipeline& pipeline, float (&clear_color)[4],
			uint32_t vertex_count, uint32_t first_vertex,
			const vertex_buffer (&vertex_buffers)[N],
			const uint64_t (&offsets)[N],
			const descriptor_set* descriptor_sets,
			uint32_t descriptor_set_count)
	{
		VkDescriptorSet* sets = (VkDescriptorSet*) malloc(max((size_t) 1, sizeof(VkDescriptorSet) * descriptor_set_count));
		if (sets == nullptr) {
			fprintf(stderr, "vulkan_renderer.record_command_buffer ERROR: Insufficient memory for VkDescriptorSet array.\n");
			return false;
		}

		VkBuffer vbs[N];
		for (unsigned int j = 0; j < N; j++)
			vbs[j] = vertex_buffers[j].buffer;
		for (uint32_t i = 0; i < swap_chain_image_count; i++)
		{
			if (!begin_render_pass(cb.command_buffers[i], fb.swap_chain_framebuffers[i], pipeline, clear_color)) {
				free(sets); return false;
			}

			for (uint32_t j = 0; j < descriptor_set_count; j++)
				sets[j] = descriptor_sets[j].sets[i];

			vkCmdBindPipeline(cb.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
			vkCmdBindVertexBuffers(cb.command_buffers[i], 0, N, vbs, offsets);
			vkCmdBindDescriptorSets(cb.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, descriptor_set_count, sets, 0, nullptr);
			vkCmdDraw(cb.command_buffers[i], vertex_count, 1, first_vertex, 0);

			if (!end_render_pass(cb.command_buffers[i])) {
				free(sets); return false;
			}
		}
		free(sets);
		return true;
	}

	inline bool create_vertex_buffer(vertex_buffer& vb, uint64_t size_in_bytes)
	{
		return create_buffer(vb.buffer, vb.memory,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, size_in_bytes);
	}

	inline void delete_vertex_buffer(vertex_buffer& vb) {
		vkDestroyBuffer(logical_device, vb.buffer, nullptr);
		vkFreeMemory(logical_device, vb.memory, nullptr);
	}

	inline bool fill_vertex_buffer(vertex_buffer& vb, void* src_data, uint64_t size_in_bytes)
	{
		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;
		if (!create_buffer(staging_buffer, staging_buffer_memory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size_in_bytes))
		{
			fprintf(stderr, "vulkan_renderer.fill_vertex_buffer ERROR: Unable to create staging buffer.\n");
			return false;
		}

		void* data;
		vkMapMemory(logical_device, staging_buffer_memory, 0, size_in_bytes, 0, &data);
		memcpy(data, src_data, size_in_bytes);
		vkUnmapMemory(logical_device, staging_buffer_memory);

		copy_buffer(staging_buffer, vb.buffer, size_in_bytes);

		vkDestroyBuffer(logical_device, staging_buffer, nullptr);
		vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
		return true;
	}

	inline bool create_uniform_buffer(uniform_buffer& ub, uint64_t size_in_bytes)
	{
		ub.buffers = (VkBuffer*) malloc(sizeof(VkBuffer) * swap_chain_image_count);
		ub.memories = (VkDeviceMemory*) malloc(sizeof(VkDeviceMemory) * swap_chain_image_count);
		ub.size = size_in_bytes;
		if (ub.buffers == nullptr || ub.memories == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_uniform_buffer ERROR: Out of memory.\n");
			if (ub.buffers != nullptr) free(ub.buffers);
			return false;
		}

		for (uint32_t i = 0; i < swap_chain_image_count; i++) {
			if (!create_buffer(ub.buffers[i], ub.memories[i], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size_in_bytes))
			{
				for (uint32_t j = 0; j < i; j++) {
					vkDestroyBuffer(logical_device, ub.buffers[j], nullptr);
					vkFreeMemory(logical_device, ub.memories[j], nullptr);
				}
				free(ub.buffers); free(ub.memories);
				return false;
			}
		}
		return true;
	}

	inline void delete_uniform_buffer(uniform_buffer& ub) {
		for (uint32_t j = 0; j < swap_chain_image_count; j++) {
			vkDestroyBuffer(logical_device, ub.buffers[j], nullptr);
			vkFreeMemory(logical_device, ub.memories[j], nullptr);
		}
		free(ub.buffers); free(ub.memories);
	}

	inline bool create_descriptor_set_layout(descriptor_set_layout& layout,
			uint32_t* bindings, descriptor_type* types, uint32_t* descriptor_counts,
			shader_stage* stage_visibilities, uint32_t binding_count)
	{
		VkDescriptorSetLayoutBinding* uboLayoutBindings = (VkDescriptorSetLayoutBinding*) calloc(binding_count, sizeof(VkDescriptorSetLayoutBinding));
		if (uboLayoutBindings == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_set_layout ERROR: Out of memory.\n");
			return false;
		}
		for (uint32_t i = 0; i < binding_count; i++) {
			uboLayoutBindings[i].binding = bindings[i];
			uboLayoutBindings[i].descriptorType = (VkDescriptorType) types[i];
			uboLayoutBindings[i].descriptorCount = descriptor_counts[i];
			uboLayoutBindings[i].stageFlags = (VkShaderStageFlags) stage_visibilities[i];
			uboLayoutBindings[i].pImmutableSamplers = nullptr;
		}

		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = binding_count;
		layoutInfo.pBindings = uboLayoutBindings;

		if (vkCreateDescriptorSetLayout(logical_device, &layoutInfo, nullptr, &layout.layout) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_set_layout ERROR: Failed to create descriptor set layout.\n");
			free(uboLayoutBindings); return false;
		}
		free(uboLayoutBindings);
		return true;
	}

	inline void delete_descriptor_set_layout(descriptor_set_layout& layout) {
		vkDestroyDescriptorSetLayout(logical_device, layout.layout, nullptr);
	}

	inline bool create_descriptor_pool(descriptor_pool& pool,
			const descriptor_type* descriptor_types, uint32_t pool_count)
	{
		VkDescriptorPoolSize* pool_sizes = (VkDescriptorPoolSize*) calloc(pool_count, sizeof(VkDescriptorPoolSize));
		if (pool_sizes == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_pool ERROR: Out of memory.\n");
			return false;
		}

		for (uint32_t i = 0; i < pool_count; i++) {
			pool_sizes[i].type = (VkDescriptorType) descriptor_types[i];
			pool_sizes[i].descriptorCount = swap_chain_image_count;
		}

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = pool_count;
		poolInfo.pPoolSizes = pool_sizes;
		poolInfo.maxSets = swap_chain_image_count;
		if (vkCreateDescriptorPool(logical_device, &poolInfo, nullptr, &pool.pool) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_set ERROR: Failed to create descriptor pool.\n");
			free(pool_sizes); return false;
		}
		free(pool_sizes);
		return true;
	}

	inline void delete_descriptor_pool(descriptor_pool& pool) {
		vkDestroyDescriptorPool(logical_device, pool.pool, nullptr);
	}

	inline bool create_descriptor_set(
			descriptor_set& ds, const uniform_buffer* uniform_buffers,
			uint32_t* uniform_buffer_bindings, size_t uniform_buffer_count,
			const texture_image* texture_images, uint32_t* texture_image_bindings, size_t texture_image_count,
			const dynamic_texture_image* dyn_texture_images, uint32_t* dyn_texture_image_bindings, size_t dyn_texture_image_count,
			const sampler& sampler, const descriptor_set_layout& layout, const descriptor_pool& pool)
	{
		ds.sets = (VkDescriptorSet*) malloc(sizeof(VkDescriptorSet) * swap_chain_image_count);
		if (ds.sets == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_set ERROR: Insufficient memory for `descriptor_set.sets`.\n");
			return false;
		}

		VkDescriptorSetLayout* layouts = (VkDescriptorSetLayout*) malloc(sizeof(VkDescriptorSetLayout) * swap_chain_image_count);
		if (layouts == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_set ERROR: Insufficient memory for `layouts`.\n");
			free(ds.sets); return false;
		}
		for (uint32_t i = 0; i < swap_chain_image_count; i++)
			layouts[i] = layout.layout;

		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = pool.pool;
		allocInfo.descriptorSetCount = swap_chain_image_count;
		allocInfo.pSetLayouts = layouts;
		if (vkAllocateDescriptorSets(logical_device, &allocInfo, ds.sets) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_set ERROR: Failed to allocate descriptor sets.\n");
			free(layouts); free(ds.sets); return false;
		}
		free(layouts);

		VkDescriptorBufferInfo* buffer_info_array = (VkDescriptorBufferInfo*) calloc(max(1, uniform_buffer_count), sizeof(VkDescriptorBufferInfo));
		VkDescriptorImageInfo* image_info_array = (VkDescriptorImageInfo*) calloc(max(1, texture_image_count + dyn_texture_image_count), sizeof(VkDescriptorImageInfo));
		VkWriteDescriptorSet* descriptorWrites = (VkWriteDescriptorSet*) calloc(uniform_buffer_count + texture_image_count + dyn_texture_image_count, sizeof(VkWriteDescriptorSet));
		if (buffer_info_array == nullptr || image_info_array == nullptr || descriptorWrites == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_descriptor_set ERROR: Insufficient memory for `descriptorWrites`.\n");
			if (buffer_info_array != nullptr) free(buffer_info_array);
			if (image_info_array != nullptr) free(image_info_array);
			free(ds.sets); return false;
		}
		for (size_t i = 0; i < swap_chain_image_count; i++) {
			for (size_t j = 0; j < uniform_buffer_count; j++) {
				buffer_info_array[j].buffer = uniform_buffers[j].buffers[i];
				buffer_info_array[j].offset = 0;
				buffer_info_array[j].range = uniform_buffers[j].size;

				descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[j].dstSet = ds.sets[i];
				descriptorWrites[j].dstBinding = uniform_buffer_bindings[j];
				descriptorWrites[j].dstArrayElement = 0;
				descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[j].descriptorCount = 1;
				descriptorWrites[j].pBufferInfo = &buffer_info_array[j];
			}
			for (size_t j = 0; j < texture_image_count; j++) {
				image_info_array[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				image_info_array[j].imageView = texture_images[j].view;
				image_info_array[j].sampler = sampler.sampler;

				descriptorWrites[uniform_buffer_count + j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[uniform_buffer_count + j].dstSet = ds.sets[i];
				descriptorWrites[uniform_buffer_count + j].dstBinding = texture_image_bindings[j];
				descriptorWrites[uniform_buffer_count + j].dstArrayElement = 0;
				descriptorWrites[uniform_buffer_count + j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[uniform_buffer_count + j].descriptorCount = 1;
				descriptorWrites[uniform_buffer_count + j].pImageInfo = &image_info_array[j];
			}
			for (size_t j = 0; j < dyn_texture_image_count; j++) {
				image_info_array[texture_image_count + j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				image_info_array[texture_image_count + j].imageView = dyn_texture_images[j].view;
				image_info_array[texture_image_count + j].sampler = sampler.sampler;

				descriptorWrites[uniform_buffer_count + texture_image_count + j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[uniform_buffer_count + texture_image_count + j].dstSet = ds.sets[i];
				descriptorWrites[uniform_buffer_count + texture_image_count + j].dstBinding = dyn_texture_image_bindings[j];
				descriptorWrites[uniform_buffer_count + texture_image_count + j].dstArrayElement = 0;
				descriptorWrites[uniform_buffer_count + texture_image_count + j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[uniform_buffer_count + texture_image_count + j].descriptorCount = 1;
				descriptorWrites[uniform_buffer_count + texture_image_count + j].pImageInfo = &image_info_array[texture_image_count + j];
			}

			vkUpdateDescriptorSets(logical_device, uniform_buffer_count + texture_image_count + dyn_texture_image_count, descriptorWrites, 0, nullptr);
		}
		free(buffer_info_array);
		free(image_info_array);
		free(descriptorWrites);
		return true;
	}

	inline bool create_texture_image(texture_image& image,
			const void* pixels, uint64_t image_size_in_bytes,
			uint32_t image_width, uint32_t image_height)
	{
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		if (!create_buffer(stagingBuffer, stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, image_size_in_bytes))
		{
			fprintf(stderr, "vulkan_renderer.create_texture_image ERROR: Unable to create staging buffer.\n");
			return false;
		}

		void* data;
		vkMapMemory(logical_device, stagingBufferMemory, 0, image_size_in_bytes, 0, &data);
		memcpy(data, pixels, (size_t) image_size_in_bytes);
		vkUnmapMemory(logical_device, stagingBufferMemory);

		if (!create_image(image_width, image_height, VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image.image, image.memory))
		{
			vkDestroyBuffer(logical_device, stagingBuffer, nullptr);
			vkFreeMemory(logical_device, stagingBufferMemory, nullptr);
			return false;
		}

		transition_image_layout<VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL>(image.image, VK_FORMAT_R8G8B8A8_UNORM);
		copy_buffer_to_image(stagingBuffer, image.image, image_width, image_height);
		transition_image_layout<VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL>(image.image, VK_FORMAT_R8G8B8A8_UNORM);

		vkDestroyBuffer(logical_device, stagingBuffer, nullptr);
		vkFreeMemory(logical_device, stagingBufferMemory, nullptr);

		if (!create_image_view(image.view, image.image, VK_FORMAT_R8G8B8A8_UNORM)) {
			vkDestroyImage(logical_device, image.image, nullptr);
			vkFreeMemory(logical_device, image.memory, nullptr);
			return false;
		}
		return true;
	}

	inline void delete_texture_image(texture_image& image) {
		vkDestroyImageView(logical_device, image.view, nullptr);
		vkDestroyImage(logical_device, image.image, nullptr);
		vkFreeMemory(logical_device, image.memory, nullptr);
	}

	inline bool create_dynamic_texture_image(
			dynamic_texture_image& image,
			uint64_t image_size_in_bytes,
			uint32_t image_width, uint32_t image_height)
	{
		if (!create_buffer(image.staging_buffer, image.staging_buffer_memory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, image_size_in_bytes))
		{
			fprintf(stderr, "vulkan_renderer.create_texture_image ERROR: Unable to create staging buffer.\n");
			return false;
		}

		vkMapMemory(logical_device, image.staging_buffer_memory, 0, image_size_in_bytes, 0, &image.mapped_memory);

		if (!create_image(image_width, image_height, VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image.image, image.memory))
		{
			vkDestroyBuffer(logical_device, image.staging_buffer, nullptr);
			vkFreeMemory(logical_device, image.staging_buffer_memory, nullptr);
			return false;
		}

		if (!create_image_view(image.view, image.image, VK_FORMAT_R8G8B8A8_UNORM)) {
			vkDestroyImage(logical_device, image.image, nullptr);
			vkFreeMemory(logical_device, image.memory, nullptr);
			return false;
		}
		image.width = image_width;
		image.height = image_height;
		return true;
	}

	inline void transfer_dynamic_texture_image(dynamic_texture_image& image)
	{
		transition_image_layout<VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL>(image.image, VK_FORMAT_R8G8B8A8_UNORM);
		copy_buffer_to_image(image.staging_buffer, image.image, image.width, image.height);
		transition_image_layout<VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL>(image.image, VK_FORMAT_R8G8B8A8_UNORM);
	}

	inline void delete_dynamic_texture_image(dynamic_texture_image& image) {
		vkDestroyImageView(logical_device, image.view, nullptr);
		vkDestroyImage(logical_device, image.image, nullptr);
		vkFreeMemory(logical_device, image.memory, nullptr);
	}

	inline bool create_sampler(sampler& s,
			filter mag_filter, filter min_filter,
			sampler_address_mode address_mode_u,
			sampler_address_mode address_mode_v,
			sampler_address_mode address_mode_w,
			bool enable_anisotropic_filtering,
			float max_anisotropic_filtering_samples)
	{
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = (VkFilter) mag_filter;
		samplerInfo.minFilter = (VkFilter) min_filter;
		samplerInfo.addressModeU = (VkSamplerAddressMode) address_mode_u;
		samplerInfo.addressModeV = (VkSamplerAddressMode) address_mode_v;
		samplerInfo.addressModeW = (VkSamplerAddressMode) address_mode_w;
		samplerInfo.anisotropyEnable = (VkBool32) enable_anisotropic_filtering;
		samplerInfo.maxAnisotropy = max_anisotropic_filtering_samples;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(logical_device, &samplerInfo, nullptr, &s.sampler) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_sampler ERROR: Failed to create sampler.\n");
			return false;
		}
		return true;
	}

	inline void delete_sampler(sampler& s) {
		vkDestroySampler(logical_device, s.sampler, nullptr);
	}

	template<typename ResetCommandBuffersFunction, typename GetWindowDimensionsFunction>
	inline bool draw_frame(command_buffer& cb, bool& resized,
			ResetCommandBuffersFunction reset_command_buffers,
			GetWindowDimensionsFunction get_window_dimensions)
	{
		vkWaitForFences(logical_device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(logical_device, swap_chain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || resized) {
			/* free and reinitialize the swap chain */
			resized = false;
			return reset_swap_chain(reset_command_buffers, get_window_dimensions);
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			fprintf(stderr, "vulkan_renderer.draw_frame ERROR: Failed to acquire next frame.\n");
			return false;
		}

		return present_frame(cb, image_index, resized, reset_command_buffers, get_window_dimensions);
	}

	template<typename ResetCommandBuffersFunction, typename GetWindowDimensionsFunction>
	inline bool draw_frame(command_buffer& cb, bool& resized,
			ResetCommandBuffersFunction reset_command_buffers,
			GetWindowDimensionsFunction get_window_dimensions,
			uniform_buffer* uniform_buffers,
			const void* const* uniform_buffer_data,
			unsigned int uniform_buffer_count)
	{
		vkWaitForFences(logical_device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(logical_device, swap_chain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || resized) {
			/* free and reinitialize the swap chain */
			resized = false;
			return reset_swap_chain(reset_command_buffers, get_window_dimensions);
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			fprintf(stderr, "vulkan_renderer.draw_frame ERROR: Failed to acquire next frame.\n");
			return false;
		}

		for (size_t i = 0; i < uniform_buffer_count; i++) {
			void* data;
			vkMapMemory(logical_device, uniform_buffers[i].memories[image_index], 0, uniform_buffers[i].size, 0, &data);
			memcpy(data, uniform_buffer_data[i], uniform_buffers[i].size);
			vkUnmapMemory(logical_device, uniform_buffers[i].memories[image_index]);
		}

		return present_frame(cb, image_index, resized, reset_command_buffers, get_window_dimensions);
	}

private:
	template<typename SurfaceType>
	inline bool init_helper(const char* application_name, uint32_t application_version,
			const char* engine_name, uint32_t engine_version,
			const char* const* enabled_extensions, uint32_t extension_count,
			device_selector device_selection, const SurfaceType& window,
			uint32_t window_width, uint32_t window_height,
			bool require_anisotropic_filtering)
	{
		VkApplicationInfo appInfo = { };
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = application_name;
		appInfo.applicationVersion = application_version;
		appInfo.pEngineName = engine_name;
		appInfo.engineVersion = engine_version;
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo = { };
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = extension_count;
		createInfo.ppEnabledExtensionNames = enabled_extensions;
		createInfo.enabledLayerCount = 0;

		/* get supported layers */
#if !defined(NDEBUG)
		constexpr const char* requested_layers[] = { "VK_LAYER_KHRONOS_validation" };
		constexpr unsigned int requested_layer_count = 1;
#else
		constexpr const char* requested_layers[] = { "" };
		constexpr unsigned int requested_layer_count = 0;
#endif

		if (requested_layer_count > 0) {
			uint32_t supported_layer_count;
			vkEnumerateInstanceLayerProperties(&supported_layer_count, nullptr);

			VkLayerProperties* supported_layers = (VkLayerProperties*) malloc(max(1, sizeof(VkLayerProperties) * supported_layer_count));
			if (supported_layers == nullptr)
				throw std::bad_alloc();
			vkEnumerateInstanceLayerProperties(&supported_layer_count, supported_layers);

			for (unsigned int i = 0; i < requested_layer_count; i++) {
				bool found_layer = false;
				for (uint32_t j = 0; j < supported_layer_count; j++) {
					if (strcmp(requested_layers[i], supported_layers[j].layerName) == 0) {
						found_layer = true;
						break;
					}
				}
				if (!found_layer) {
					fprintf(stderr, "vulkan_renderer.init_helper ERROR: Layer '%s' is not supported.\n", requested_layers[i]);
					return false;
				}
			}
			free(supported_layers);
		}

		createInfo.ppEnabledLayerNames = requested_layers;
		createInfo.enabledLayerCount = requested_layer_count;

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to create instance.\n");
			return false;
		}


		/* create the window surface */
		if (create_window_surface(window) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to create window surface.\n");
			vkDestroyInstance(instance, nullptr);
			return false;
		}


		/* select the physical device (i.e. GPU) */
		constexpr const char* requested_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
		constexpr unsigned int requested_extension_count = 1;

		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

		if (device_count == 0) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to find devices with Vulkan support.\n");
			vkDestroySurfaceKHR(instance, surface, nullptr); vkDestroyInstance(instance, nullptr);
			return false;
		}

		VkPhysicalDevice* devices = (VkPhysicalDevice*) malloc(max((size_t) 1, sizeof(VkPhysicalDevice) * device_count));
		if (devices == nullptr) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Insufficient memory for `devices`.\n");
			vkDestroySurfaceKHR(instance, surface, nullptr); vkDestroyInstance(instance, nullptr);
			return false;
		}
		vkEnumeratePhysicalDevices(instance, &device_count, devices);

		uint32_t queue_index = 0; swap_chain_details swap_chain_info;
		physical_device = VK_NULL_HANDLE;
		for (uint32_t i = 0; i < device_count; i++) {
			if (is_device_suitable(devices[i], swap_chain_info, device_selection, requested_extensions, require_anisotropic_filtering, queue_index)) {
				physical_device = devices[i]; break;
			}
		}
		free(devices);

		if (physical_device == VK_NULL_HANDLE) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Unable to find supported device.\n");
			vkDestroySurfaceKHR(instance, surface, nullptr); vkDestroyInstance(instance, nullptr);
			return false;
		}


		/* setup the queue and create the logical device */
		float queue_priority = 1.0f;
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queue_index;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queue_priority;

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.samplerAnisotropy = (VkBool32) require_anisotropic_filtering;

		VkDeviceCreateInfo device_create_info = {};
		device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		device_create_info.pQueueCreateInfos = &queueCreateInfo;
		device_create_info.queueCreateInfoCount = 1;
		device_create_info.pEnabledFeatures = &deviceFeatures;
		device_create_info.enabledExtensionCount = requested_extension_count;
		device_create_info.ppEnabledExtensionNames = requested_extensions;
		device_create_info.enabledLayerCount = requested_layer_count;
		device_create_info.ppEnabledLayerNames = requested_layers;

		if (vkCreateDevice(physical_device, &device_create_info, nullptr, &logical_device) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to create logical device.\n");
			swap_chain_info.free_helper();
			vkDestroySurfaceKHR(instance, surface, nullptr); vkDestroyInstance(instance, nullptr);
			return false;
		}
		vkGetDeviceQueue(logical_device, queue_index, 0, &queue);


		/* setup the swap chain */
		if (!init_swap_chain(swap_chain_info, window_width, window_height)) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to initialize swap chain.\n");
			swap_chain_info.free_helper(); vkDestroyDevice(logical_device, nullptr);
			vkDestroySurfaceKHR(instance, surface, nullptr); vkDestroyInstance(instance, nullptr);
			return false;
		}
		swap_chain_info.free_helper();


		/* create the command pool */
		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queue_index;
		poolInfo.flags = 0;
		if (vkCreateCommandPool(logical_device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to create command pool.\n");
			free_swap_chain();
			vkDestroyDevice(logical_device, nullptr);
			vkDestroySurfaceKHR(instance, surface, nullptr);
			vkDestroyInstance(instance, nullptr);
			return false;
		}


		/* create semaphores and fences */
		image_available_semaphores = (VkSemaphore*) malloc(sizeof(VkSemaphore) * max_frames_in_flight);
		render_finished_semaphores = (VkSemaphore*) malloc(sizeof(VkSemaphore) * max_frames_in_flight);
		in_flight_fences = (VkFence*) malloc(sizeof(VkFence) * max_frames_in_flight);
		if (image_available_semaphores == nullptr || render_finished_semaphores == nullptr || in_flight_fences == nullptr) {
			fprintf(stderr, "vulkan_renderer.init_helper ERROR: Insufficient memory for `in_flight_fences`.\n");
			if (image_available_semaphores != nullptr) free(image_available_semaphores);
			if (render_finished_semaphores != nullptr) free(render_finished_semaphores);
			vkDestroyCommandPool(logical_device, command_pool, nullptr);
			free_swap_chain();
			vkDestroyDevice(logical_device, nullptr);
			vkDestroySurfaceKHR(instance, surface, nullptr);
			vkDestroyInstance(instance, nullptr);
			return false;
		}
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		for (unsigned int i = 0; i < max_frames_in_flight; i++) {
			if (vkCreateSemaphore(logical_device, &semaphoreInfo, nullptr, &image_available_semaphores[i]) != VK_SUCCESS) {
				fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to create `image_available_semaphores`.\n");
				for (unsigned int j = 0; j < i; j++) {
					vkDestroySemaphore(logical_device, image_available_semaphores[j], nullptr);
					vkDestroySemaphore(logical_device, render_finished_semaphores[j], nullptr);
					vkDestroyFence(logical_device, in_flight_fences[j], nullptr);
				}
				free(image_available_semaphores); free(render_finished_semaphores); free(in_flight_fences);
				vkDestroyCommandPool(logical_device, command_pool, nullptr);
				free_swap_chain();
				vkDestroyDevice(logical_device, nullptr);
				vkDestroySurfaceKHR(instance, surface, nullptr);
				vkDestroyInstance(instance, nullptr);
				return false;
			} if (vkCreateSemaphore(logical_device, &semaphoreInfo, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS) {
				fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to create `render_finished_semaphores`.\n");
				for (unsigned int j = 0; j < i; j++) {
					vkDestroySemaphore(logical_device, image_available_semaphores[j], nullptr);
					vkDestroySemaphore(logical_device, render_finished_semaphores[j], nullptr);
					vkDestroyFence(logical_device, in_flight_fences[j], nullptr);
				}
				vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
				free(image_available_semaphores); free(render_finished_semaphores); free(in_flight_fences);
				vkDestroyCommandPool(logical_device, command_pool, nullptr);
				free_swap_chain();
				vkDestroyDevice(logical_device, nullptr);
				vkDestroySurfaceKHR(instance, surface, nullptr);
				vkDestroyInstance(instance, nullptr);
				return false;
			} if (vkCreateFence(logical_device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
				fprintf(stderr, "vulkan_renderer.init_helper ERROR: Failed to create `in_flight_fences`.\n");
				for (unsigned int j = 0; j < i; j++) {
					vkDestroySemaphore(logical_device, image_available_semaphores[j], nullptr);
					vkDestroySemaphore(logical_device, render_finished_semaphores[j], nullptr);
					vkDestroyFence(logical_device, in_flight_fences[j], nullptr);
				}
				vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
				vkDestroySemaphore(logical_device, render_finished_semaphores[i], nullptr);
				free(image_available_semaphores); free(render_finished_semaphores); free(in_flight_fences);
				vkDestroyCommandPool(logical_device, command_pool, nullptr);
				free_swap_chain();
				vkDestroyDevice(logical_device, nullptr);
				vkDestroySurfaceKHR(instance, surface, nullptr);
				vkDestroyInstance(instance, nullptr);
				return false;
			}
		}

		return true;
	}

	template<typename ResetCommandBuffersFunction, typename GetWindowDimensionsFunction>
	inline bool present_frame(
			command_buffer& cb, uint32_t image_index, bool& resized,
			ResetCommandBuffersFunction reset_command_buffers,
			GetWindowDimensionsFunction get_window_dimensions)
	{
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkSemaphore waitSemaphores[] = {image_available_semaphores[current_frame]};
		VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cb.command_buffers[image_index];
		VkSemaphore signalSemaphores[] = {render_finished_semaphores[current_frame]};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(logical_device, 1, &in_flight_fences[current_frame]);

		if (vkQueueSubmit(queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.present_frame ERROR: Failed to submit draw command buffer.\n");
			return false;
		}

		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = {swap_chain};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &image_index;
		presentInfo.pResults = nullptr;
		VkResult result = vkQueuePresentKHR(queue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || resized) {
			resized = false;
			return reset_swap_chain(reset_command_buffers, get_window_dimensions);
		}

		current_frame = (current_frame + 1) % max_frames_in_flight;
		return true;
	}

	inline bool begin_render_pass(
			VkCommandBuffer cb, VkFramebuffer fb,
			graphics_pipeline& pipeline, float (&clear_color)[4])
	{
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(cb, &beginInfo) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.begin_render_pass ERROR: Failed to begin recording command buffer.\n");
			return false;
		}

		VkRenderPassBeginInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_info.renderPass = pipeline.render_pass;
		render_pass_info.framebuffer = fb;
		render_pass_info.renderArea.offset = {0, 0};
		render_pass_info.renderArea.extent = swap_chain_extent;

		VkClearValue clearColor;
		clearColor.color.float32[0] = clear_color[0];
		clearColor.color.float32[1] = clear_color[1];
		clearColor.color.float32[2] = clear_color[2];
		clearColor.color.float32[3] = clear_color[3];
		render_pass_info.clearValueCount = 1;
		render_pass_info.pClearValues = &clearColor;

		vkCmdBeginRenderPass(cb, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
		return true;
	}

	inline bool end_render_pass(VkCommandBuffer cb) {
		vkCmdEndRenderPass(cb);
		if (vkEndCommandBuffer(cb) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.end_render_pass ERROR: Failed to record command buffer.\n");
			return false;
		}
		return true;
	}

	inline VkCommandBuffer begin_one_time_command_buffer() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = command_pool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer command_buffer;
		vkAllocateCommandBuffers(logical_device, &allocInfo, &command_buffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(command_buffer, &beginInfo);

		return command_buffer;
	}

	inline void end_one_time_command_buffer(VkCommandBuffer command_buffer) {
		vkEndCommandBuffer(command_buffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &command_buffer;

		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue);

		vkFreeCommandBuffers(logical_device, command_pool, 1, &command_buffer);
	}

	inline bool create_image(uint32_t width, uint32_t height,
			VkFormat format, VkImageTiling tiling,
			VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
			VkImage& image, VkDeviceMemory& imageMemory)
	{
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(logical_device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_image ERROR: Failed to create image.\n");
			return false;
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(logical_device, image, &memRequirements);
		uint32_t memory_type_index;
		if (!find_memory_type(memory_type_index, memRequirements.memoryTypeBits, properties)) {
			fprintf(stderr, "vulkan_renderer.create_image ERROR: Failed to find suitable memory type.\n");
			vkDestroyImage(logical_device, image, nullptr);
			return false;
		}

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = memory_type_index;

		if (vkAllocateMemory(logical_device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_image ERROR: Failed to allocate image memory.\n");
			vkDestroyImage(logical_device, image, nullptr);
			return false;
		}

		vkBindImageMemory(logical_device, image, imageMemory, 0);
		return true;
	}

	template<VkImageLayout OldLayout, VkImageLayout NewLayout>
	inline void transition_image_layout(
			VkImage image, VkFormat format)
	{
		static_assert(
			(OldLayout == VK_IMAGE_LAYOUT_UNDEFINED && NewLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		 || (OldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && NewLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
		 	"Unsupported image layout transition");

		VkCommandBuffer command_buffer = begin_one_time_command_buffer();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = OldLayout;
		barrier.newLayout = NewLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;
		if (OldLayout == VK_IMAGE_LAYOUT_UNDEFINED && NewLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		} else if (OldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && NewLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}

		vkCmdPipelineBarrier(command_buffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		end_one_time_command_buffer(command_buffer);
	}

	void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer command_buffer = begin_one_time_command_buffer();
		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		end_one_time_command_buffer(command_buffer);
	}

	inline bool create_image_view(VkImageView& view, VkImage image, VkFormat format)
	{
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(logical_device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_image_view ERROR: Failed to create image view.\n");
			return false;
		}
		return true;
	}

	inline bool create_buffer(
			VkBuffer& buffer, VkDeviceMemory& memory,
			VkBufferUsageFlags usage,
			VkMemoryPropertyFlags memory_properties,
			uint64_t size_in_bytes)
	{
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size_in_bytes;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(logical_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_buffer ERROR: Failed to create buffer.\n");
			return false;
		}

		VkMemoryRequirements memRequirements; uint32_t memory_type;
		vkGetBufferMemoryRequirements(logical_device, buffer, &memRequirements);
		if (!find_memory_type(memory_type, memRequirements.memoryTypeBits, memory_properties)) {
			fprintf(stderr, "vulkan_renderer.create_buffer ERROR: Failed to find suitable memory type.\n");
			vkDestroyBuffer(logical_device, buffer, nullptr);
			return false;
		}

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = memory_type;
		if (vkAllocateMemory(logical_device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_buffer ERROR: Failed to allocate buffer memory.\n");
			vkDestroyBuffer(logical_device, buffer, nullptr);
			return false;
		}

		vkBindBufferMemory(logical_device, buffer, memory, 0);
		return true;
	}

	inline void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, uint64_t size_in_bytes) {
		VkCommandBuffer command_buffer = begin_one_time_command_buffer();
		VkBufferCopy copyRegion = {};
		copyRegion.size = size_in_bytes;
		vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copyRegion);
		end_one_time_command_buffer(command_buffer);
	}

	inline bool create_pipeline_layout(
			graphics_pipeline& pipeline)
	{
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(logical_device, &pipelineLayoutInfo, nullptr, &pipeline.layout) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_pipeline_layout ERROR: Failed to create pipeline layout.\n");
			return false;
		}
		return true;
	}

	inline bool create_pipeline_layout(
			graphics_pipeline& pipeline,
			const descriptor_set_layout* layouts,
			uint32_t layout_count)
	{
		VkDescriptorSetLayout* layout_array = (VkDescriptorSetLayout*) malloc(sizeof(VkDescriptorSetLayout) * layout_count);
		if (layout_array == nullptr) {
			fprintf(stderr, "vulkan_renderer.create_pipeline_layout ERROR: Out of memory.\n");
			return false;
		}
		for (uint32_t i = 0; i < layout_count; i++)
			layout_array[i] = layouts[i].layout;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = layout_count;
		pipelineLayoutInfo.pSetLayouts = layout_array;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(logical_device, &pipelineLayoutInfo, nullptr, &pipeline.layout) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_pipeline_layout ERROR: Failed to create pipeline layout.\n");
			free(layout_array); return false;
		}
		free(layout_array);
		return true;
	}

	template<size_t N>
	inline bool create_graphics_pipeline(graphics_pipeline& pipeline,
			VkPipelineShaderStageCreateInfo (&shaders)[N],
			primitive_topology topology)
	{
		if (!create_pipeline_layout(pipeline)) return false;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr;
		if (!create_graphics_pipeline(pipeline, shaders, topology, vertexInputInfo)) {
			vkDestroyPipelineLayout(logical_device, pipeline.layout, nullptr);
			return false;
		}
		return true;
	}

	template<size_t N, size_t M>
	inline bool create_graphics_pipeline(graphics_pipeline& pipeline,
			VkPipelineShaderStageCreateInfo (&shaders)[N],
			primitive_topology topology,
			const VkVertexInputBindingDescription binding,
			const VkVertexInputAttributeDescription (&attributes)[M])
	{
		if (!create_pipeline_layout(pipeline)) return false;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &binding;
		vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t) M;
		vertexInputInfo.pVertexAttributeDescriptions = attributes;
		if (!create_graphics_pipeline(pipeline, shaders, topology, vertexInputInfo)) {
			vkDestroyPipelineLayout(logical_device, pipeline.layout, nullptr);
			return false;
		}
		return true;
	}

	template<size_t N, size_t M>
	inline bool create_graphics_pipeline(graphics_pipeline& pipeline,
			VkPipelineShaderStageCreateInfo (&shaders)[N],
			primitive_topology topology,
			const VkVertexInputBindingDescription binding,
			const VkVertexInputAttributeDescription (&attributes)[M],
			const descriptor_set_layout* layouts,
			uint32_t layout_count)
	{
		if (!create_pipeline_layout(pipeline, layouts, layout_count)) return false;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &binding;
		vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t) M;
		vertexInputInfo.pVertexAttributeDescriptions = attributes;
		if (!create_graphics_pipeline(pipeline, shaders, topology, vertexInputInfo)) {
			vkDestroyPipelineLayout(logical_device, pipeline.layout, nullptr);
			return false;
		}
		return true;
	}

	/* NOTE: this function assumes `pipeline.layout` is initialized */
	template<size_t N>
	bool create_graphics_pipeline(graphics_pipeline& pipeline,
			VkPipelineShaderStageCreateInfo (&shaders)[N],
			primitive_topology topology,
			VkPipelineVertexInputStateCreateInfo vertexInputInfo)
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = (VkPrimitiveTopology) topology;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float) swap_chain_extent.width;
		viewport.height = (float) swap_chain_extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = {0, 0};
		scissor.extent = swap_chain_extent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swap_chain_image_format;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;
		if (vkCreateRenderPass(logical_device, &renderPassInfo, nullptr, &pipeline.render_pass) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_render_pipeline ERROR: Failed to create render pass.\n");
			return false;
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = N;
		pipelineInfo.pStages = shaders;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr;
		pipelineInfo.layout = pipeline.layout;
		pipelineInfo.renderPass = pipeline.render_pass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;
		if (vkCreateGraphicsPipelines(logical_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.pipeline) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.create_render_pipeline ERROR: Failed to create graphics pipeline.\n");
			vkDestroyRenderPass(logical_device, pipeline.render_pass, nullptr);
			return false;
		}
		return true;
	}

	bool find_memory_type(uint32_t& out, uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memory_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

		for (out = 0; out < memory_properties.memoryTypeCount; out++) {
			if ((typeFilter & (1 << out)) && (memory_properties.memoryTypes[out].propertyFlags & properties) == properties) {
				return true;
			}
		}
		return false;
	}

	struct swap_chain_details {
		VkSurfaceCapabilitiesKHR capabilities;
		VkSurfaceFormatKHR* formats;
		uint32_t format_count;
		VkPresentModeKHR* presentation_modes;
		uint32_t presentation_mode_count;

		inline void free_helper() {
			::free(formats);
			::free(presentation_modes);
		}
	};

	template<typename ResetCommandBuffersFunction, typename GetWindowDimensionsFunction>
	inline bool reset_swap_chain(
			ResetCommandBuffersFunction reset_command_buffers,
			GetWindowDimensionsFunction get_window_dimensions)
	{
		vkDeviceWaitIdle(logical_device);
		free_swap_chain();

		swap_chain_details swap_chain_info;
		if (!get_swap_chain_info(physical_device, swap_chain_info)) {
			fprintf(stderr, "vulkan_renderer.reset_swap_chain ERROR: No supported swap chain.\n");
			return false;
		}
		uint32_t new_width, new_height;
		get_window_dimensions(new_width, new_height);
		if (!init_swap_chain(swap_chain_info, new_width, new_height)) {
			swap_chain_info.free_helper();
			fprintf(stderr, "vulkan_renderer.reset_swap_chain ERROR: Failed to reinitialize swap chain.\n");
			return false;
		}
		swap_chain_info.free_helper();
		if (!reset_command_buffers()) {
			fprintf(stderr, "vulkan_renderer.reset_swap_chain ERROR: Failed to reinitialize command buffers.\n");
			return false;
		}
		return true;
	}

	inline bool init_swap_chain(const swap_chain_details& swap_chain_info, uint32_t window_width, uint32_t window_height)
	{
		VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swap_chain_info.formats, swap_chain_info.format_count);
		VkPresentModeKHR presentation_mode = choose_swap_presentation_mode(swap_chain_info.presentation_modes, swap_chain_info.presentation_mode_count);
		VkExtent2D extent = choose_swap_extent(swap_chain_info.capabilities, window_width, window_height);

		uint32_t image_count = swap_chain_info.capabilities.minImageCount + 1;
		if (swap_chain_info.capabilities.maxImageCount > 0 && image_count > swap_chain_info.capabilities.maxImageCount)
			image_count = swap_chain_info.capabilities.maxImageCount;

		VkSwapchainCreateInfoKHR swap_chain_create_info = {};
		swap_chain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swap_chain_create_info.surface = surface;
		swap_chain_create_info.minImageCount = image_count;
		swap_chain_create_info.imageFormat = surface_format.format;
		swap_chain_create_info.imageColorSpace = surface_format.colorSpace;
		swap_chain_create_info.imageExtent = extent;
		swap_chain_create_info.imageArrayLayers = 1;
		swap_chain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		/* NOTE: we assume the graphics queue and presentation queues are the same */
		swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		swap_chain_create_info.preTransform = swap_chain_info.capabilities.currentTransform;
		swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		swap_chain_create_info.presentMode = presentation_mode;
		swap_chain_create_info.clipped = VK_TRUE;
		swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(logical_device, &swap_chain_create_info, nullptr, &swap_chain) != VK_SUCCESS) {
			fprintf(stderr, "vulkan_renderer.init_swap_chain ERROR: Failed to create swap chain.\n");
			vkDestroyDevice(logical_device, nullptr);
			vkDestroySurfaceKHR(instance, surface, nullptr);
			vkDestroyInstance(instance, nullptr);
			return false;
		}


		/* retrieve the swap chain images */
		vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, nullptr);
		swap_chain_images = (VkImage*) malloc(max((size_t) 1, sizeof(VkImage) * image_count));
		if (swap_chain_images == nullptr) {
			fprintf(stderr, "vulkan_renderer.init_swap_chain ERROR: Insufficient memory for `swap_chain_images`.\n");
			vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);
			vkDestroyDevice(logical_device, nullptr);
			vkDestroySurfaceKHR(instance, surface, nullptr);
			vkDestroyInstance(instance, nullptr);
			return false;
		}
		vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, swap_chain_images);
		swap_chain_image_count = image_count;
		swap_chain_image_format = surface_format.format;
		swap_chain_extent = extent;


		/* create the image views */
		swap_chain_image_views = (VkImageView*) malloc(max((size_t) 1, sizeof(VkImageView) * image_count));
		if (swap_chain_image_views == nullptr) {
			fprintf(stderr, "vulkan_renderer.init_swap_chain ERROR: Insufficient memory for `swap_chain_image_views`.\n");
			free(swap_chain_images);
			vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);
			vkDestroyDevice(logical_device, nullptr);
			vkDestroySurfaceKHR(instance, surface, nullptr);
			vkDestroyInstance(instance, nullptr);
			return false;
		}
		for (uint32_t i = 0; i < image_count; i++) {
			if (!create_image_view(swap_chain_image_views[i], swap_chain_images[i], swap_chain_image_format)) {
				fprintf(stderr, "vulkan_renderer.init_swap_chain ERROR: Failed to create image views.\n");
				for (uint32_t j = 0; j < i; j++)
					vkDestroyImageView(logical_device, swap_chain_image_views[j], nullptr);
				free(swap_chain_images);
				vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);
				vkDestroyDevice(logical_device, nullptr);
				vkDestroySurfaceKHR(instance, surface, nullptr);
				vkDestroyInstance(instance, nullptr);
				return false;
			}
		}

		return true;
	}

	inline void free_swap_chain() {
		for (uint32_t i = 0; i < swap_chain_image_count; i++)
			vkDestroyImageView(logical_device, swap_chain_image_views[i], nullptr);
		free(swap_chain_images);
		vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);
	}

	inline bool is_device_type_suitable(VkPhysicalDeviceType type, device_selector device_selection) {
		switch (device_selection) {
		case device_selector::FIRST_ANY: return true;
		case device_selector::FIRST_DISCRETE_GPU:
			return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
		case device_selector::FIRST_INTEGRATED_GPU:
			return type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
		case device_selector::FIRST_VIRTUAL_GPU:
			return type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
		case device_selector::FIRST_CPU:
			return type == VK_PHYSICAL_DEVICE_TYPE_CPU;
		case device_selector::FIRST_OTHER:
			return type == VK_PHYSICAL_DEVICE_TYPE_OTHER;
		}
		fprintf(stderr, "vulkan_renderer.is_device_type_suitable ERROR: Unrecognized device_selector.\n");
		return false;
	}

	template<size_t N>
	inline bool is_device_suitable(
			VkPhysicalDevice device,
			swap_chain_details& swap_chain,
			device_selector device_selection,
			const char* const (&requested_extensions)[N],
			bool require_anisotropic_filtering,
			uint32_t& queue_index)
	{
		/* check that the device properties are suitable */
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(device, &device_properties);
		if (!is_device_type_suitable(device_properties.deviceType, device_selection))
			return false;

		/* check that the available queues are suitable */
		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

		VkQueueFamilyProperties* queue_families = (VkQueueFamilyProperties*) malloc(max((size_t) 1, sizeof(VkQueueFamilyProperties) * queue_family_count));
		if (queue_families == nullptr) {
			fprintf(stderr, "vulkan_renderer.is_device_suitable ERROR: Insufficient memory for `queue_families`.\n");
			return false;
		}
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families);

		queue_index = queue_family_count;
		for (uint32_t i = 0; i < queue_family_count; i++) {
			VkBool32 supports_presentation = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &supports_presentation);
			if (supports_presentation && queue_families[i].queueCount > 0 && (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				queue_index = i;
				break;
			}
		}
		free(queue_families);

		if (queue_index == queue_family_count)
			/* no suitable queue can be found for this device */
			return false;

		/* check that the requested extensions are supported */
		uint32_t extension_count;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

		VkExtensionProperties* available_extensions = (VkExtensionProperties*) malloc(max((size_t) 1, sizeof(VkExtensionProperties) * extension_count));
		if (available_extensions == nullptr) {
			fprintf(stderr, "vulkan_renderer.is_device_suitable ERROR: Insufficient memory for `available_extensions`.\n");
			return false;
		}
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions);
		for (size_t i = 0; i < N; i++) {
			bool found_extension = false;
			for (uint32_t j = 0; j < extension_count; j++) {
				if (strcmp(requested_extensions[i], available_extensions[j].extensionName) == 0) {
					found_extension = true;
					break;
				}
			}
			if (!found_extension) {
				free(available_extensions);
				return false;
			}
		}
		free(available_extensions);

		if (require_anisotropic_filtering) {
			VkPhysicalDeviceFeatures supportedFeatures;
			vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
			if (supportedFeatures.samplerAnisotropy == VK_FALSE)
				return false;
		}

		/* check that swap chain support is adequate */
		return get_swap_chain_info(device, swap_chain);
	}

/* TODO: fix detection for Win32 */
#if defined(HWND)
	VkResult create_window_surface(win32_surface window)
	{
		vkCreateWin32SurfaceKHR sci;
		PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR =
				(PFN_vkCreateWin32SurfaceKHR) vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");
		if (!vkCreateWin32SurfaceKHR) {
			fprintf(stderr, "vulkan_renderer.create_window_surface ERROR: Vulkan missing 'VK_KHR_win32_surface' extension.\n");
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}

		memset(&sci, 0, sizeof(sci));
		sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
		sci.hinstance = GetModuleHandle(NULL);
		sci.hwnd = window.handle;

		return vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface);
	}
#endif

/* TODO: fix detection for Wayland */
#if defined(wl_display)
	VkResult create_window_surface(wayland_surface wayland_surface)
	{
		VkWaylandSurfaceCreateInfoKHR sci;
		PFN_vkCreateWaylandSurfaceKHR vkCreateWaylandSurfaceKHR =
				(PFN_vkCreateWaylandSurfaceKHR) vkGetInstanceProcAddr(instance, "vkCreateWaylandSurfaceKHR");
		if (!vkCreateWaylandSurfaceKHR) {
			fprintf(stderr, "vulkan_renderer.create_window_surface ERROR: Vulkan missing 'VK_KHR_wayland_surface' extension.\n");
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}

 		memset(&sci, 0, sizeof(sci));
		sci.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
		sci.display = wayland_surface.display;
		sci.surface = wayland_surface.surface;

		return vkCreateWaylandSurfaceKHR(instance, &sci, nullptr, &surface);
	}
#endif

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 101100
	VkResult create_window_surface(macos_surface window)
	{
		VkMacOSSurfaceCreateInfoMVK sci;
		PFN_vkCreateMacOSSurfaceMVK vkCreateMacOSSurfaceMVK =
				(PFN_vkCreateMacOSSurfaceMVK) vkGetInstanceProcAddr(instance, "vkCreateMacOSSurfaceMVK");
		if (!vkCreateMacOSSurfaceMVK) {
			fprintf(stderr, "vulkan_renderer.create_window_surface ERROR: Vulkan missing 'VK_MVK_macos_surface' extension.\n");
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}

		memset(&sci, 0, sizeof(sci));
		sci.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
		sci.pView = window.view;

		return vkCreateMacOSSurfaceMVK(instance, &sci, nullptr, &surface);
	}
#endif

/* TODO: fix detection for XCB */
#if defined(xcb_connection_t)
	VkResult create_window_surface(xcb_surface xcb_surface)
	{
		VkXcbSurfaceCreateInfoKHR sci;
		PFN_vkCreateXcbSurfaceKHR vkCreateXcbSurfaceKHR =
				(PFN_vkCreateXcbSurfaceKHR) vkGetInstanceProcAddr(instance, "vkCreateXcbSurfaceKHR");
		if (!vkCreateXcbSurfaceKHR) {
			fprintf(stderr, "vulkan_renderer.create_window_surface ERROR: Vulkan missing 'VK_KHR_xcb_surface' extension.\n");
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}

		memset(&sci, 0, sizeof(sci));
		sci.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
		sci.connection = xcb_surface.connection;
		sci.window = xcb_surface.window;

		return vkCreateXcbSurfaceKHR(instance, &sci, nullptr, &surface);
	}
#endif

/* TODO: fix detection for X11 */
#if defined(Display) && defined(Window)
	VkResult create_window_surface(x11_surface x11_window)
	{
		VkXlibSurfaceCreateInfoKHR sci;
		PFN_vkCreateXlibSurfaceKHR vkCreateXlibSurfaceKHR =
				(PFN_vkCreateXlibSurfaceKHR) vkGetInstanceProcAddr(instance, "vkCreateXlibSurfaceKHR");
		if (!vkCreateXlibSurfaceKHR) {
			fprintf(stderr, "vulkan_renderer.create_window_surface ERROR: Vulkan missing 'VK_KHR_xlib_surface' extension.\n");
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}

		memset(&sci, 0, sizeof(sci));
		sci.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
		sci.dpy = x11_window.display;
		sci.window = x11_window.window;

		return vkCreateXlibSurfaceKHR(instance, &sci, nullptr, &surface);
	}
#endif

#if defined(_glfw3_h_)
	inline VkResult create_window_surface(glfw_surface glfw_window) {
		return glfwCreateWindowSurface(instance, glfw_window.window, nullptr, &surface);
	}
#endif

	inline bool get_swap_chain_info(VkPhysicalDevice physical_device, swap_chain_details& swap_chain) {
		if (!query_swap_chain_support(swap_chain, physical_device)) return false;
		if (swap_chain.format_count == 0 || swap_chain.presentation_mode_count == 0) {
			swap_chain.free_helper(); return false;
		}
		return true;
	}

	bool query_swap_chain_support(swap_chain_details& details, VkPhysicalDevice device)
	{
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &details.format_count, nullptr);
		details.formats = (VkSurfaceFormatKHR*) malloc(max((size_t) 1, sizeof(VkSurfaceFormatKHR) * details.format_count));
		if (details.formats == nullptr) {
			fprintf(stderr, "vulkan_renderer.query_swap_chain_support ERROR: Insufficient memory for `details.formats`.\n");
			return false;
		}
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &details.format_count, details.formats);

		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &details.presentation_mode_count, nullptr);
		details.presentation_modes = (VkPresentModeKHR*) malloc(max((size_t) 1, sizeof(VkPresentModeKHR) * details.presentation_mode_count));
		if (details.presentation_modes == nullptr) {
			fprintf(stderr, "vulkan_renderer.query_swap_chain_support ERROR: Insufficient memory for `details.presentation_modes`.\n");
			free(details.formats); return false;
		}
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &details.presentation_mode_count, details.presentation_modes);

		return true;
	}

	VkSurfaceFormatKHR choose_swap_surface_format(const VkSurfaceFormatKHR* formats, uint32_t format_count) {
		for (uint32_t i = 0; i < format_count; i++) {
			if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM
			 && formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return formats[i];
			}
		}

		return formats[0];
	}

	VkPresentModeKHR choose_swap_presentation_mode(const VkPresentModeKHR* presentation_modes, uint32_t presentation_mode_count) {
		for (uint32_t i = 0; i < presentation_mode_count; i++) {
			if (presentation_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
				return presentation_modes[i];
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D choose_swap_extent(
			const VkSurfaceCapabilitiesKHR& capabilities,
			uint32_t window_width, uint32_t window_height)
	{
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		} else {
			VkExtent2D actualExtent = {window_width, window_height};

			actualExtent.width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, actualExtent.height));
			return actualExtent;
		}
	}

	template<typename A> friend bool init(vulkan_renderer&,
			const char*, uint32_t, const char*, uint32_t, const char* const*, uint32_t,
			device_selector, const A&, uint32_t, uint32_t, unsigned int, bool);
};

template<typename SurfaceType>
inline bool init(vulkan_renderer& renderer,
		const char* application_name, uint32_t application_version,
		const char* engine_name, uint32_t engine_version,
		const char* const* enabled_extensions, uint32_t extension_count,
		device_selector device_selection, const SurfaceType& window,
		uint32_t window_width, uint32_t window_height,
		unsigned int max_frames_in_flight,
		bool require_anisotropic_filtering)
{
	renderer.max_frames_in_flight = max_frames_in_flight;
	return renderer.init_helper(application_name, application_version,
			engine_name, engine_version, enabled_extensions, extension_count,
			device_selection, window, window_width, window_height,
			require_anisotropic_filtering);
}

} /* namespace mirage */

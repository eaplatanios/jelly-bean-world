#include "visualizer.h"
#include <thread>
#include <condition_variable>
#include <core/lex.h>

#if defined(WIN32)
#include <windows.h>
#else
#include <signal.h>
#endif

using namespace jbw;

bool connected_to_server = false;
client<visualizer_client_data> sim;

#if defined(WIN32)
BOOL WINAPI signal_handler(DWORD sig_num) 
{
    if (sig_num == CTRL_C_EVENT || sig_num ==  CTRL_CLOSE_EVENT
	 || sig_num ==  CTRL_LOGOFF_EVENT || sig_num ==  CTRL_SHUTDOWN_EVENT) 
    {
		if (connected_to_server)
			remove_client(sim);
        return TRUE;
    }
    return FALSE;
}
#else
void signal_handler(int sig_num) {
	if (connected_to_server)
		remove_client(sim);
	exit(EXIT_SUCCESS);
}
#endif

bool parse_address(
	const char* arg, bool& fail,
	char*& address, const char*& port)
{
	if (arg[0] == '-' && arg[1] == '-')
		return false;

	unsigned int colon_index = 0;
	while (arg[colon_index] != ':' && arg[colon_index] != '\0')
		colon_index++;

	if (arg[colon_index] == '\0') {
		fprintf(stderr, "ERROR: The server address must be of the form <address>:<port>.\n");
		fail = true; return true;
	}

	address = (char*) malloc(sizeof(char) * (colon_index + 1));
	if (address == nullptr) {
		fprintf(stderr, "parse_address ERROR: Out of memory.\n");
		fail = true; return true;
	}

	for (unsigned int i = 0; i < colon_index; i++)
		address[i] = arg[i];
	address[colon_index] = '\0';

	port = arg + colon_index + 1;
	return true;
}

inline bool parse_option(const char* arg,
		bool& fail, const char* to_match)
{
	return (strcmp(arg, to_match) == 0);
}

inline bool parse_option(
		const char* arg, bool& fail,
		const char* to_match, uint64_t& out)
{
	unsigned int length = strlen(to_match);
	if (strncmp(arg, to_match, length) != 0)
		return false;
	const char* option = arg + length;

	unsigned long long value;
	if (!parse_ulonglong(string(option), value)) {
		fprintf(stderr, "ERROR: Unable to parse option '%s'.\n", arg);
		fail = true; return true;
	}
	out = (uint64_t) value;
	return true;
}

template<typename Stream>
void print_usage(Stream& out) {
	fprintf(out, "Usage: jbw_visualizer <address>:<port> [options]\n"
		"Connects to the JBW server at the given address visualizes the simulated environment.\n"
		"\n"
		"Available options:\n"
		"  --track=ID   Starts tracking the agent with the given ID.\n"
		"  --local      Starts a simulation locally, rather than connecting to a server.\n"
		"  --help       Prints this usage text.\n");
}

template<typename Stream>
void print_controls(Stream& out) {
	fprintf(out, "\nControls:\n"
		"Click and drag with left mouse button to move camera.\n"
		"  +/= key: Zoom in.\n"
		"  -/_ key: Zoom out.\n"
		"  b key:   Toggle drawing of the background grid and scent map.\n"
		"  0 key:   Disable agent tracking.\n"
		"  1 key:   Track agent with ID 1.\n"
		"  2 key:   Track agent with ID 2.\n"
		"  3 key:   Track agent with ID 3.\n"
		"  4 key:   Track agent with ID 4.\n"
		"  5 key:   Track agent with ID 5.\n"
		"  6 key:   Track agent with ID 6.\n"
		"  7 key:   Track agent with ID 7.\n"
		"  8 key:   Track agent with ID 8.\n"
		"  9 key:   Track agent with ID 9.\n");
}

inline void set_interaction_args(
		item_properties* item_types, unsigned int first_item_type,
		unsigned int second_item_type, interaction_function interaction,
		std::initializer_list<float> args)
{
	item_types[first_item_type].interaction_fns[second_item_type].fn = interaction;
	item_types[first_item_type].interaction_fns[second_item_type].arg_count = (unsigned int) args.size();
	item_types[first_item_type].interaction_fns[second_item_type].args = (float*) malloc(max((size_t) 1, sizeof(float) * args.size()));

	unsigned int counter = 0;
	for (auto i = args.begin(); i != args.end(); i++)
		item_types[first_item_type].interaction_fns[second_item_type].args[counter++] = *i;
}

inline void on_step(const simulator<empty_data>* sim,
		const hash_map<uint64_t, agent_state*>& agents, uint64_t time)
{ }

bool run_locally(uint64_t track_agent_id)
{
	simulator_config config;
	config.max_steps_per_movement = 1;
	config.scent_dimension = 3;
	config.color_dimension = 3;
	config.vision_range = 5;
	for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
		config.allowed_movement_directions[i] = action_policy::ALLOWED;
	for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
		config.allowed_rotations[i] = action_policy::ALLOWED;
	config.no_op_allowed = false;
	config.patch_size = 32;
	config.mcmc_iterations = 4000;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.agent_color[2] = 1.0f;
	config.collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
	config.decay_param = 0.4f;
	config.diffusion_param = 0.14f;
	config.deleted_item_lifetime = 2000;

	/* configure item types */
	unsigned int item_type_count = 4;
	config.item_types.ensure_capacity(item_type_count);
	config.item_types[0].name = "banana";
	config.item_types[0].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[0].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[0].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[0].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[0].scent[1] = 1.0f;
	config.item_types[0].color[1] = 1.0f;
	config.item_types[0].required_item_counts[0] = 1;
	config.item_types[0].blocks_movement = false;
	config.item_types[1].name = "onion";
	config.item_types[1].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[1].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[1].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].scent[0] = 1.0f;
	config.item_types[1].color[0] = 1.0f;
	config.item_types[1].required_item_counts[1] = 1;
	config.item_types[1].blocks_movement = false;
	config.item_types[2].name = "jellybean";
	config.item_types[2].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[2].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[2].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].scent[2] = 1.0f;
	config.item_types[2].color[2] = 1.0f;
	config.item_types[2].blocks_movement = false;
	config.item_types[3].name = "wall";
	config.item_types[3].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[3].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[3].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[3].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[3].color[0] = 0.5f;
	config.item_types[3].color[1] = 0.5f;
	config.item_types[3].color[2] = 0.5f;
	config.item_types[3].required_item_counts[3] = 1;
	config.item_types[3].blocks_movement = true;
	config.item_types.length = item_type_count;

	config.item_types[0].intensity_fn.fn = constant_intensity_fn;
	config.item_types[0].intensity_fn.arg_count = 1;
	config.item_types[0].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[0].intensity_fn.args[0] = -5.3f;
	config.item_types[0].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[1].intensity_fn.fn = constant_intensity_fn;
	config.item_types[1].intensity_fn.arg_count = 1;
	config.item_types[1].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[1].intensity_fn.args[0] = -5.0f;
	config.item_types[1].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[2].intensity_fn.fn = constant_intensity_fn;
	config.item_types[2].intensity_fn.arg_count = 1;
	config.item_types[2].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[2].intensity_fn.args[0] = -5.3f;
	config.item_types[2].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[3].intensity_fn.fn = constant_intensity_fn;
	config.item_types[3].intensity_fn.arg_count = 1;
	config.item_types[3].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[3].intensity_fn.args[0] = 0.0f;
	config.item_types[3].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);

	set_interaction_args(config.item_types.data, 0, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(config.item_types.data, 0, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(config.item_types.data, 0, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(config.item_types.data, 0, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 0, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(config.item_types.data, 1, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 2, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(config.item_types.data, 1, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 2, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(config.item_types.data, 2, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(config.item_types.data, 2, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(config.item_types.data, 2, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 0, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 2, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 3, cross_interaction_fn, {10.0f, 15.0f, 20.0f, -200.0f, -20.0f, 1.0f});

	simulator<empty_data> sim(config, empty_data());

	uint64_t agent_id; agent_state* agent;
	if (sim.add_agent(agent_id, agent) != status::OK) {
		fprintf(stderr, "run_locally ERROR: Unable to add new agent.\n");
		return false;
	}

	unsigned int move_count = 0;
	bool simulation_running = true;
	std::mutex lock;
	std::condition_variable cv;
	unsigned int max_steps_per_frame = 1;
	unsigned int steps_in_current_frame = 0;
	std::thread simulation_worker = std::thread([&]() {
		for (unsigned int t = 0; simulation_running && t < 1000000; t++)
		{
			if (sim.move(agent_id, (rand() % 2 == 0) ? direction::UP : direction::RIGHT, 1) != status::OK) {
				fprintf(stderr, "run_locally ERROR: Unable to move agent.\n");
				break;
			}
			move_count++;
			steps_in_current_frame++;

			std::unique_lock<std::mutex> lck(lock);
			while (simulation_running && steps_in_current_frame == max_steps_per_frame) cv.wait(lck);
		}
	});

	timer stopwatch;
	unsigned long long elapsed = 0;
	unsigned int frame_count = 0;
	visualizer<simulator<empty_data>> visualizer(sim, 800, 800, track_agent_id);
	print_controls(stdout); fflush(stdout);
	while (simulation_running) {
		if (visualizer.is_window_closed())
			break;

		visualizer.draw_frame();
		frame_count++;

		lock.lock();
		steps_in_current_frame = 0;
		cv.notify_one();
		lock.unlock();

		if (stopwatch.milliseconds() >= 1000) {
			elapsed += stopwatch.milliseconds();
			printf("Completed %u moves: %lf simulation steps per second. (%lf fps)\n", move_count, ((double) sim.time / elapsed) * 1000, ((double) frame_count / elapsed) * 1000);
			stopwatch.start();
		}
	}
	elapsed += stopwatch.milliseconds();
	printf("Completed %u moves: %lf simulation steps per second. (%lf fps)\n", move_count, ((double) sim.time / elapsed) * 1000, ((double) frame_count / elapsed) * 1000);
	simulation_running = false;
	cv.notify_one();
	if (simulation_worker.joinable()) {
		try {
			simulation_worker.join();
		} catch (...) { }
	}
	return true;
}

bool run(
		const char* server_address,
		const char* server_port,
		uint64_t track_agent_id)
{
	uint64_t client_id;
	uint64_t simulation_time = connect_client(sim, server_address, server_port, client_id);
	if (simulation_time == UINT64_MAX) {
		fprintf(stderr, "ERROR: Unable to connect to '%s:%s'.\n", server_address, server_port);
		return false;
	}
	connected_to_server = true;

	visualizer<client<visualizer_client_data>> visualizer(sim, 800, 800, track_agent_id);
	print_controls(stdout); fflush(stdout);
	while (sim.client_running) {
		if (visualizer.is_window_closed())
			break;

		visualizer.draw_frame();
	}

	return remove_client(sim);
}

int main(int argc, const char** argv)
{
#if defined(WIN32)
	SetConsoleCtrlHandler(signal_handler, TRUE);
#else
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
#endif

	if (argc <= 1) {
		fprintf(stderr, "Not enough arguments.\n");
		print_usage(stderr);
		return EXIT_FAILURE;
	}

	uint64_t track_agent_id = 1;
	char* server_address = nullptr;
	const char* server_port = nullptr;
	bool local = false;

	/* parse command-line arguments */
	bool fail = false;
	for (int i = 1; i < argc && !fail; i++) {
		if (parse_address(argv[i], fail, server_address, server_port)) continue;
		if (parse_option(argv[i], fail, "--track=", track_agent_id)) continue;
		if (parse_option(argv[i], fail, "--local")) { local = true; continue; }
		if (parse_option(argv[i], fail, "--help")) {
			print_usage(stdout);
			fflush(stdout);
			if (server_address != nullptr)
				free(server_address);
			return EXIT_SUCCESS;
		}
	}

	if (fail) {
		if (server_address != nullptr)
			free(server_address);
		return EXIT_FAILURE;
	}

	if (local) {
		if (server_address != nullptr) {
			free(server_address);
			server_address = nullptr;
		}
		if (!run_locally(track_agent_id))
			return EXIT_FAILURE;
	} else {
		if (server_address == nullptr || server_port == nullptr) {
			fprintf(stderr, "ERROR: Address of JBW server not provided.\n");
			return EXIT_FAILURE;
		}

		if (!run(server_address, server_port, track_agent_id)) {
			free(server_address);
			return EXIT_FAILURE;
		}
	}

	if (server_address != nullptr)
		free(server_address);
	return EXIT_SUCCESS;
}
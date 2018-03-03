
#define _USE_MATH_DEFINES
#include "simulator.h"
#include "mpi.h"

#include <core/timer.h>
#include <cmath>
#include <thread>
#include <condition_variable>
#include <signal.h>

using namespace core;
using namespace nel;

inline void set_interaction_args(float* args, unsigned int item_type_count,
		unsigned int first_item_type, unsigned int second_item_type,
		float first_cutoff, float second_cutoff, float first_value, float second_value)
{
	args[4 * (first_item_type * item_type_count + second_item_type) + 1] = first_cutoff;
	args[4 * (first_item_type * item_type_count + second_item_type) + 2] = second_cutoff;
	args[4 * (first_item_type * item_type_count + second_item_type) + 3] = first_value;
	args[4 * (first_item_type * item_type_count + second_item_type) + 4] = second_value;
}

enum class movement_pattern {
	RADIAL,
	BACK_AND_FORTH
};

constexpr unsigned int agent_count = 1;
constexpr unsigned int max_time = 10000;
constexpr movement_conflict_policy collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
constexpr movement_pattern move_pattern = movement_pattern::RADIAL;
unsigned int sim_time = 0;
bool agent_direction[agent_count];
bool waiting_for_server[agent_count];
std::condition_variable conditions[agent_count];
std::mutex locks[agent_count];
std::mutex print_lock;
FILE* out = stderr;
async_server server;

//#define MULTITHREADED
//#define USE_MPI
//#define TEST_SERIALIZATION
//#define TEST_SERVER_CONNECTION_LOSS
//#define TEST_CLIENT_CONNECTION_LOSS

struct empty_data {
	static inline void free(empty_data& data) { }
};

constexpr bool init(empty_data& data, const empty_data& src) { return true; }

inline direction next_direction(position agent_position, double theta) {
	if (theta == M_PI) {
		return direction::UP;
	} else if (theta == 3 * M_PI / 2) {
		return direction::DOWN;
	} else if ((theta >= 0 && theta < M_PI)
			|| (theta > 3 * M_PI / 2 && theta < 2 * M_PI))
	{
		double slope = tan(theta);
		if (slope * (agent_position.x + 0.5) > agent_position.y + 0.5) return direction::UP;
		else if (slope * (agent_position.x + 0.5) < agent_position.y - 0.5) return direction::DOWN;
		else return direction::RIGHT;
	} else {
		double slope = tan(theta);
		if (slope * (agent_position.x - 0.5) > agent_position.y + 0.5) return direction::UP;
		else if (slope * (agent_position.x - 0.5) < agent_position.y - 0.5) return direction::DOWN;
		else return direction::LEFT;
	}
}

inline direction next_direction(position agent_position,
		int64_t min_x, int64_t max_x, bool& reverse)
{
	if (!reverse && agent_position.x >= max_x) {
		reverse = true;
		return direction::LEFT;
	} else if (reverse && agent_position.x <= min_x) {
		reverse = false;
		return direction::RIGHT;
	} else if (!reverse) {
		return direction::RIGHT;
	} else {
		return direction::LEFT;
	}
}

inline bool try_move(
		simulator<empty_data>& sim,
		unsigned int i, bool& reverse)
{
	position current_position = sim.get_position((uint64_t) i);

	direction dir;
	switch (move_pattern) {
	case movement_pattern::RADIAL:
		dir = next_direction(current_position, (2 * M_PI * i) / agent_count); break;
	case movement_pattern::BACK_AND_FORTH:
		dir = next_direction(current_position, -10 * (int64_t) agent_count, 10 * agent_count, reverse); break;
	}

	if (!sim.move((uint64_t) i, dir, 1)) {
		print_lock.lock();
		print("ERROR: Unable to move agent ", out);
		print(i, out); print(" from ", out);
		print(current_position, out);
		print(" in direction ", out);
		print(dir, out); print(".\n", out);
		print_lock.unlock();
		return false;
	}
	return true;
}

void run_agent(simulator<empty_data>& sim, unsigned int id,
		std::atomic_uint& move_count, bool& simulation_running)
{
	while (simulation_running) {
		waiting_for_server[id] = true;
		if (try_move(sim, id, agent_direction[id])) {
			move_count++;

			std::unique_lock<std::mutex> lck(locks[id]);
			while (waiting_for_server[id] && simulation_running)
				conditions[id].wait(lck);
			lck.unlock();
		}
	}
}

void on_step(const simulator<empty_data>* sim, empty_data& data, uint64_t time)
{
	sim_time++;
#if defined(USE_MPI)
	if (!send_step_response(server)) {
		print_lock.lock();
		fprintf(out, "on_step ERROR: send_step_response failed.\n");
		print_lock.unlock();
	}
#elif defined(MULTITHREADED)
	for (unsigned int i = 0; i < agent_count; i++) {
		std::unique_lock<std::mutex> lck(locks[i]);
		waiting_for_server[i] = false;
		conditions[i].notify_one();
	}
#endif
}

bool add_agents(simulator<empty_data>& sim)
{
	for (unsigned int i = 0; i < agent_count; i++) {
		uint64_t agent_id = sim.add_agent();
		if (agent_id != i) {
			fprintf(out, "add_agents ERROR: Unable to add new agent.\n");
			return false;
		}
		agent_direction[i] = (i <= agent_count / 2);

		/* advance time by one to avoid collision at (0,0) */
		for (unsigned int j = 0; j <= i; j++)
			try_move(sim, j, agent_direction[j]);
	}
	return true;
}

bool test_singlethreaded(const simulator_config& config)
{
	simulator<empty_data>& sim = *((simulator<empty_data>*) alloca(sizeof(simulator<empty_data>)));
	if (!init(sim, config, empty_data())) {
		fprintf(stderr, "ERROR: Unable to initialize simulator.\n");
		return false;
	}

	if (!add_agents(sim)) {
		free(sim); return false;
	}

	timer stopwatch;
	std::atomic_uint move_count(0);
	unsigned long long elapsed = 0;
	for (unsigned int t = 0; t < max_time; t++) {
#if defined(TEST_SERIALIZATION)
		if (t % 1000 == 0) {
			char filename[1024];
			sprintf(filename, "simulator_state%u", t);
			FILE* file = fopen(filename, "wb");
			fixed_width_stream<FILE*> out(file);
			if (!write(sim, out))
				fprintf(stderr, "ERROR: write failed.\n");
			fclose(file);

			/* end the simulation and restart it by reading from file */
			free(sim);
			file = fopen(filename, "rb");
			fixed_width_stream<FILE*> in(file);
			if (!read(sim, in, empty_data())) {
				fprintf(stderr, "ERROR: read failed.\n");
				free(sim); return false;
			}
			fclose(file);
		}
#endif

		for (unsigned int j = 0; j < agent_count; j++)
			try_move(sim, j, agent_direction[j]);
		move_count += agent_count;
		if (stopwatch.milliseconds() >= 1000) {
			elapsed += stopwatch.milliseconds();
			fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
			stopwatch.start();
		}
	}
	elapsed += stopwatch.milliseconds();
	fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
	free(sim);
	return true;
}

bool test_multithreaded(const simulator_config& config)
{
	simulator<empty_data> sim(config, empty_data());

	if (!add_agents(sim))
		return false;

	std::atomic_uint move_count(0);
	bool simulation_running = true;
	std::thread clients[agent_count];
	for (unsigned int i = 0; i < agent_count; i++)
		clients[i] = std::thread([&,i]() { run_agent(sim, i, move_count, simulation_running); });

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (sim_time < max_time) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		elapsed += stopwatch.milliseconds();
		fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	simulation_running = false;
	for (unsigned int i = 0; i < agent_count; i++) {
		conditions[i].notify_one();
		if (!clients[i].joinable()) continue;
		try {
			clients[i].join();
		} catch (...) { }
	}
	return true;
}

struct client_data {
	unsigned int index;
	uint64_t agent_id;
	const float* perception;
	const unsigned int* items;
	const hash_map<position, patch_state>* map;

	bool move_result, waiting_for_step;
	position pos;
};

void on_add_agent(client<client_data>& c, uint64_t agent_id) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.agent_id = agent_id;
	conditions[id].notify_one();
}

void on_move(client<client_data>& c, uint64_t agent_id, bool request_success) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.move_result = request_success;
	conditions[id].notify_one();
}

void on_get_position(client<client_data>& c, uint64_t agent_id, const position& pos) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.pos = pos;
	conditions[id].notify_one();
}

void on_get_scent(client<client_data>& c, uint64_t agent_id, const float* scent) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.perception = scent;
	conditions[id].notify_one();
}

void on_get_vision(client<client_data>& c, uint64_t agent_id, const float* vision) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.perception = vision;
	conditions[id].notify_one();
}

void on_get_collected_items(client<client_data>& c, uint64_t agent_id, const unsigned int* items) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.items = items;
	conditions[id].notify_one();
}

void on_get_map(client<client_data>& c, const hash_map<position, patch_state>* map) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.map = map;
	conditions[id].notify_one();
}

void on_step(client<client_data>& c) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	c.data.waiting_for_step = false;
	conditions[id].notify_one();
}

void on_lost_connection(client<client_data>& c) {
	print_lock.lock();
	fprintf(out, "Client %u lost connection to server.\n", c.data.index);
	print_lock.unlock();
	c.client_running = false;
	conditions[c.data.index].notify_one();
}

inline void wait_for_server(std::condition_variable& cv,
		std::mutex& lock, bool& waiting_for_server, bool& client_running)
{
	std::unique_lock<std::mutex> lck(lock);
	while (waiting_for_server && client_running) cv.wait(lck);
}

inline bool mpi_try_move(
		client<client_data>& c, unsigned int i, bool& reverse)
{
	/* get current position */
	waiting_for_server[i] = true;
	if (!send_get_position(c, c.data.agent_id)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send get_position request.\n");
		print_lock.unlock();
		return false;
	}
	wait_for_server(conditions[i], locks[i], waiting_for_server[i], c.client_running);
	if (!c.client_running) return true;

	direction dir;
	switch (move_pattern) {
	case movement_pattern::RADIAL:
		dir = next_direction(c.data.pos, (2 * M_PI * i) / agent_count); break;
	case movement_pattern::BACK_AND_FORTH:
		dir = next_direction(c.data.pos, -10 * (int64_t) agent_count, 10 * agent_count, reverse); break;
	}

	/* send move request */
	waiting_for_server[i] = true;
	if (!send_move(c, c.data.agent_id, dir, 1)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send move request.\n");
		print_lock.unlock();
		return false;
	}
	wait_for_server(conditions[i], locks[i], waiting_for_server[i], c.client_running);
	if (!c.client_running) return true;

	if (!c.data.move_result) {
		print_lock.lock();
		print("ERROR: Unable to move agent ", out);
		print(i, out); print(" from ", out);
		print(c.data.pos, out);
		print(" in direction ", out);
		print(dir, out); print(".\n", out);
		print_lock.unlock();
		return false;
	}
	return true;
}

void run_mpi_agent(unsigned int id,
		client<client_data>* clients,
		std::atomic_uint& move_count)
{
	while (clients[id].client_running) {
		clients[id].data.waiting_for_step = true;
		if (mpi_try_move(clients[id], id, agent_direction[id])) {
			move_count++;
			wait_for_server(conditions[id], locks[id], clients[id].data.waiting_for_step, clients[id].client_running);
		}
	}
}

void cleanup_mpi(client<client_data>* clients,
		unsigned int length = agent_count)
{
	for (unsigned int i = 0; i < length; i++)
		stop_client(clients[i]);
	stop_server(server);
}

bool test_mpi(const simulator_config& config)
{
	simulator<empty_data> sim(config, empty_data());
	if (!init_server(server, sim, 54353, 16, 4)) {
		fprintf(out, "ERROR: init_server returned false.\n");
		return false;
	}

	/* below is client-side code */
	client<client_data> clients[agent_count];
	for (unsigned int i = 0; i < agent_count; i++) {
		clients[i].data.index = i;
		if (!init_client(clients[i], "localhost", "54353")) {
			fprintf(out, "ERROR: Unable to initialize client %u.\n", i);
			cleanup_mpi(clients, i); return false;
		}

		/* each client adds one agent to the simulation */
		waiting_for_server[i] = true;
		if (!send_add_agent(clients[i])) {
			fprintf(out, "ERROR: Unable to send add_agent request.\n");
			cleanup_mpi(clients, i); return false;
		}
		/* wait for response from server */
		wait_for_server(conditions[i], locks[i], waiting_for_server[i], clients[i].client_running);

		if (clients[i].data.agent_id == UINT64_MAX) {
			fprintf(out, "ERROR: Server returned failure for add_agent request.\n");
			cleanup_mpi(clients, i); return false;
		}

		/* advance time by one to avoid collision at (0,0) */
		for (unsigned int j = 0; j <= i; j++) {
			clients[j].data.waiting_for_step = true;
			if (!mpi_try_move(clients[j], j, agent_direction[j])) {
				cleanup_mpi(clients, i); return false;
			}
		}
		for (unsigned int j = 0; j <= i; j++)
			wait_for_server(conditions[j], locks[j], clients[j].data.waiting_for_step, clients[j].client_running);
	}

	std::atomic_uint move_count(0);
	std::thread client_threads[agent_count];
	for (unsigned int i = 0; i < agent_count; i++)
		client_threads[i] = std::thread([&,i]() { run_mpi_agent(i, clients, move_count); });

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (server.state != server_state::STOPPING && sim_time < max_time)
	{
		if (sim_time > max_time / 2) {
#if defined(TEST_SERVER_CONNECTION_LOSS)
			/* try closing all TCP sockets */
			close(server.server_socket);
			for (socket_type& client : server.client_connections)
				close(client);
#elif defined(TEST_CLIENT_CONNECTION_LOSS)
			if (server.client_connections.size == agent_count) {
				/* try closing half of the client TCP sockets */
				unsigned int index = 0;
				for (socket_type& client : server.client_connections) {
					close(client); index++;
					if (index > agent_count / 2) break;
				}
			}
#endif
		}

		std::this_thread::sleep_for(std::chrono::seconds(1));
		elapsed += stopwatch.milliseconds();
		fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	for (unsigned int i = 0; i < agent_count; i++) {
		clients[i].client_running = false;
		conditions[i].notify_one();
		try {
			client_threads[i].join();
		} catch (...) { }
	}
	cleanup_mpi(clients);
	return true;
}

int main(int argc, const char** argv)
{
#if !defined(_WIN32)
	signal(SIGPIPE, SIG_IGN);
#endif
	simulator_config config;
	config.max_steps_per_movement = 1;
	config.scent_dimension = 3;
	config.color_dimension = 3;
	config.vision_range = 10;
	config.patch_size = 32;
	config.gibbs_iterations = 10;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.agent_color[2] = 1.0f;
	config.collision_policy = collision_policy;
	config.decay_param = 0.5f;
	config.diffusion_param = 0.12f;
	config.deleted_item_lifetime = 2000;

	/* configure item types */
	config.item_types.ensure_capacity(3);
	config.item_types[0].name = "banana";
	config.item_types[0].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[0].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[0].scent[1] = 1.0f;
	config.item_types[0].color[1] = 1.0f;
	config.item_types[0].automatically_collected = false;
	config.item_types[1].name = "onion";
	config.item_types[1].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[1].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[1].scent[0] = 1.0f;
	config.item_types[1].color[0] = 1.0f;
	config.item_types[1].automatically_collected = false;
	config.item_types[2].name = "jellybean";
	config.item_types[2].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[2].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[2].scent[2] = 1.0f;
	config.item_types[2].color[2] = 1.0f;
	config.item_types[2].automatically_collected = true;
	config.item_types.length = 3;

	config.intensity_fn_arg_count = (unsigned int) config.item_types.length;
	config.interaction_fn_arg_count = (unsigned int) (4 * config.item_types.length * config.item_types.length + 1);
	config.intensity_fn = constant_intensity_fn;
	config.interaction_fn = piecewise_box_interaction_fn;
	config.intensity_fn_args = (float*) malloc(sizeof(float) * config.intensity_fn_arg_count);
	config.interaction_fn_args = (float*) malloc(sizeof(float) * config.interaction_fn_arg_count);
	config.intensity_fn_args[0] = -5.0f;
	config.intensity_fn_args[1] = -5.4f;
	config.intensity_fn_args[2] = -5.0f;
	config.interaction_fn_args[0] = (float) config.item_types.length;
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 0, 0, 10.0f, 200.0f, 0.0f, -6.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 0, 1, 200.0f, 0.0f, -6.0f, -6.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 0, 2, 10.0f, 200.0f, 2.0f, -100.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 1, 0, 0.0f, 0.0f, 0.0f, 0.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 1, 1, 0.0f, 0.0f, 0.0f, 0.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 1, 2, 200.0f, 0.0f, -100.0f, -100.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 2, 0, 10.0f, 200.0f, 2.0f, -100.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 2, 1, 200.0f, 0.0f, -100.0f, -100.0f);
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 2, 2, 10.0f, 200.0f, 0.0f, -6.0f);

#if defined(USE_MPI)
	test_mpi(config);
#elif defined(MULTITHREADED)
	test_multithreaded(config);
#else
	test_singlethreaded(config);
#endif
}

#include "mpi.h"

using namespace nel;

struct test_server {
	std::thread server_thread;
	socket_type server_socket;
	bool server_running;
};

void process_test_server_message(socket_type& server) {
	unsigned int i;
	if (!read(i, server)) {
		fprintf(stderr, "Server failed to read message.\n");
		return;
	}
	fprintf(stderr, "Server received message: %u.\n", i);
}

bool init_server(test_server& new_server, uint16_t server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	std::condition_variable cv; std::mutex lock;
	auto dispatch = [&]() {
		run_server(new_server.server_socket, server_port, connection_queue_capacity,
				worker_count, new_server.server_running, cv, process_test_server_message);
	};
	new_server.server_running = true;
	new_server.server_thread = std::thread(dispatch);

	std::unique_lock<std::mutex> lck(lock);
	cv.wait(lck);
	if (!new_server.server_running) {
		new_server.server_thread.join();
		return false;
	}
	return true;
}

void stop_server(test_server& server) {
	server.server_running = false;
	write(0, server.server_socket);
	server.server_thread.join();
}

bool init_client(socket_type& new_client,
		const char* server_address, const char* server_port)
{
	auto process_connection = [&](socket_type& connection) {
		new_client = connection;
	};
	return run_client(server_address, server_port, process_connection);
}

void stop_client(client& c) {
	shutdown(c.connection, 2);
	c.client_running = false;
	c.response_listener.join();
}

int main(int argc, const char** argv) {
	test_server new_server;
	bool success = init_server(new_server, 52342, 16, 8);
	fprintf(stderr, "init_server returned %s.\n", success ? "true" : "false");

	socket_type new_client;
	success = init_client(new_client, "localhost", "52342");
	fprintf(stderr, "init_client returned %s.\n", success ? "true" : "false");

	//for (int i = 0; i < 10; i++)
		write(2, new_client);
	shutdown(new_client, 2);

	std::this_thread::sleep_for(std::chrono::milliseconds(1));
	stop_server(new_server);
	return EXIT_SUCCESS;
}

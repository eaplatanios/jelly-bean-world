#include "mpi.h"

#include <inttypes.h>

using namespace nel;
using namespace core;

/* a lock to synchronize printing */
std::mutex lock;
FILE* out = stderr;

struct test_server {
	std::thread server_thread;
	socket_type server_socket;
	bool server_running;
};

void process_test_server_message(socket_type& server) {
	bool is_string;
	lock.lock();
	if (!read(is_string, server)) {
		fprintf(stderr, "Server failed to read is_string.\n");
		lock.unlock(); return;
	}
	if (is_string) {
		string s;
		if (!read(s, server)) {
			fprintf(stderr, "Server failed to read string.\n");
			lock.unlock(); return;
		}
		fprintf(out, "Server received message: \"");
		print(s, out); fprintf(out, "\".\n");
	} else {
		int64_t i;
		if (!read(i, server)) {
			fprintf(stderr, "Server failed to read int64_t.\n");
			lock.unlock(); return;
		}
		fprintf(out, "Server received message: %" PRId64 ".\n", i);
	}
	lock.unlock();
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
	//write(0, server.server_socket);
	closesocket(server.server_socket.handle);
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
	shutdown(c.connection.handle, 2);
	c.client_running = false;
	c.response_listener.join();
}

void test_client_send(socket_type& client, int64_t i) {
	if (!write(false, client)
	 || !write(i, client))
		 fprintf(stderr, "test_client_send ERROR: Failed to send int64_t to server.\n");
}

void test_client_send(socket_type& client, const string& s) {
	if (!write(true, client)
	 || !write(s, client))
		 fprintf(stderr, "test_client_send ERROR: Failed to send string to server.\n");
}

void test_network() {
	test_server new_server;
	bool success = init_server(new_server, 52342, 16, 8);
	fprintf(out, "init_server returned %s.\n", success ? "true" : "false");
	if (!success) return;

	unsigned int client_count = 10;
	std::thread* client_threads = new std::thread[client_count];
	std::atomic_uint counter(0);
	auto dispatch = [&counter]() {
		unsigned int thread_id = counter++;
		socket_type client;
		bool success = init_client(client, "localhost", "52342");
		lock.lock();
		fprintf(out, "[client %u] init_client returned %s.\n", thread_id, success ? "true" : "false");
		lock.unlock();
		if (!success) return;

		char message[1024];
		snprintf(message, 1024, "Hello from client %u!", thread_id);
		test_client_send(client, string(message));

		std::this_thread::sleep_for(std::chrono::milliseconds(500));

		for (int64_t i = 0; i < 10; i++)
			test_client_send(client, thread_id * 10 + i);
		shutdown(client.handle, 2);
	};
	for (unsigned int i = 0; i < client_count; i++)
		client_threads[i] = std::thread(dispatch);
	for (unsigned int i = 0; i < client_count; i++)
		client_threads[i].join();
	delete[] client_threads;

	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	stop_server(new_server);
}

int main(int argc, const char** argv) {
	test_network();
	fflush(out);
	return EXIT_SUCCESS;
}

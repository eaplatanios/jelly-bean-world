#include "mpi.h"

using namespace nel;

int main(int argc, const char** argv) {
	async_server new_server;
	simulator& sim = *((simulator*) alloca(sizeof(simulator)));
	bool success = init_server(new_server, sim, "127.0.0.1", "52342", 16, 8);
	fprintf(stderr, "init_server returned %s.\n", success ? "true" : "false");

	stop_server(new_server);
	return EXIT_SUCCESS;
}

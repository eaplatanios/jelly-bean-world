#ifndef JBW_NETWORK_H_
#define JBW_NETWORK_H_

#include <core/array.h>
#include <core/map.h>
#include <stdio.h>
#include <thread>
#include <condition_variable>

#if defined(_WIN32) /* on Windows */
#include <winsock2.h>
#include <ws2tcpip.h>
typedef int socklen_t;

#elif __APPLE__ /* on Mac */
#include <sys/event.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <netinet/tcp.h>

#else /* on Linux */
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <netinet/tcp.h>
#endif


#define EVENT_QUEUE_CAPACITY 1024


/** A structure with no contents. */
struct empty_data {
	static inline void move(const empty_data& src, const empty_data& dst) { }
	static inline void free(const empty_data& data) { }
};

constexpr bool init(const empty_data& data) { return true; }
constexpr bool init(const empty_data& data, const empty_data& src) { return true; }


namespace jbw {

using namespace core;

struct socket_type {
#if defined(_WIN32)
	SOCKET handle;

	static constexpr SOCKET EMPTY_SOCKET = INVALID_SOCKET;

	socket_type() { }
	socket_type(const SOCKET& src) : handle(src) { }

	inline bool is_valid() const {
		return handle != INVALID_SOCKET;
	}
#else
	int handle;

	static constexpr int EMPTY_SOCKET = -1;

	socket_type() { }
	socket_type(int src) : handle(src) { }

	inline bool is_valid() const {
		return handle >= 0;
	}
#endif

	inline bool operator == (const socket_type& other) const {
		return handle == other.handle;
	}

	inline bool operator != (const socket_type& other) const {
		return handle != other.handle;
	}

	static inline bool is_empty(const socket_type& key) {
		return key.handle == EMPTY_SOCKET;
	}

	static inline void set_empty(socket_type& key) {
		key.handle = EMPTY_SOCKET;
	}

	static inline unsigned int hash(const socket_type& key) {
		return default_hash(key.handle);
	}

	static inline void move(const socket_type& src, socket_type& dst) {
		dst.handle = src.handle;
	}
};

inline void close(socket_type& socket) {
#if defined(_WIN32)
	closesocket(socket.handle);
#else
	shutdown(socket.handle, 2);
#endif
}

void* alloc_socket_keys(size_t n, size_t element_size) {
	socket_type* keys = (socket_type*) malloc(sizeof(socket_type) * n);
	if (keys == NULL) return NULL;
	for (unsigned int i = 0; i < n; i++)
		socket_type::set_empty(keys[i]);
	return (void*) keys;
}

inline void listener_error(const char* message) {
#if defined(_WIN32)
	errno = (int) GetLastError();
#endif
	perror(message);
}

struct socket_listener {
#if defined(_WIN32) /* on Windows */
	HANDLE listener;
	WSABUF buffer_wrapper;
	char buffer[4];

	inline bool add_server_socket(socket_type& socket) const { return true; }

	inline bool add_client_socket(socket_type& socket) {
		if (CreateIoCompletionPort((HANDLE) socket.handle, listener, (ULONG_PTR) socket.handle, 0) == NULL) {
			listener_error("socket_listener.add_client_socket ERROR: Failed to listen to socket");
			return false;
		}
		return true;
	}

	inline bool update_socket(socket_type& socket) {
		OVERLAPPED* overlapped = (OVERLAPPED*) calloc(1, sizeof(OVERLAPPED));
		DWORD bytes_received = 0;
		DWORD flags = MSG_PEEK;
		int result = WSARecv(socket.handle, &buffer_wrapper, 1, &bytes_received, &flags, overlapped, NULL);
		if (result == SOCKET_ERROR) {
			int error = WSAGetLastError();
			if (error == WSAECONNABORTED) {
				/* the server is shutting down */
				return false;
			} else if (error != WSA_IO_PENDING) {
				errno = error;
				perror("socket_listener.update_socket ERROR: Unable to begin receiving data from client");
				shutdown(socket.handle, 2); return false;
			}
		}
		return true;
	}

	inline bool remove_socket(socket_type& socket) const { return true; }

	template<typename AcceptedConnectionCallback, typename... CallbackArgs>
	inline bool accept(socket_type& server_socket, AcceptedConnectionCallback callback, CallbackArgs&&... callback_args)
	{
		/* listen for a new connection on the server socket */
		sockaddr_storage client_address;
		socklen_t address_size = sizeof(client_address);
		socket_type connection = ::accept(server_socket.handle, (sockaddr*) &client_address, &address_size);
		if (!connection.is_valid()) {
			int error = WSAGetLastError();
			if (error == WSAEINTR) {
				/* the server is shutting down */
				return true;
			} else {
				errno = error;
				perror("socket_listener.accept ERROR: Error establishing connection with client");
				return false;
			}
		}

		if (!add_client_socket(connection)) {
			shutdown(connection.handle, 2); return false;
		}

		OVERLAPPED* overlapped = (OVERLAPPED*) calloc(1, sizeof(OVERLAPPED));
		DWORD bytes_received = 0;
		DWORD flags = MSG_PEEK;
		int result = WSARecv(connection.handle, &buffer_wrapper, 1, &bytes_received, &flags, overlapped, NULL);
		if (result == SOCKET_ERROR) {
			int error = WSAGetLastError();
			if (error == WSAECONNABORTED) {
				/* the server is shutting down */
				return false;
			} else if (error != WSA_IO_PENDING) {
				errno = error;
				perror("socket_listener.accept ERROR: Unable to begin receiving data from client");
				shutdown(connection.handle, 2); return false;
			}
		}

		callback(connection, std::forward<CallbackArgs>(callback_args)...);
		return true;
	}

	template<typename IsRunningFunction>
	inline bool listen(socket_type& connection, IsRunningFunction is_running) {
		ULONG_PTR completion_key = NULL;
		DWORD bytes_transferred;
		OVERLAPPED* overlapped;
		bool result = GetQueuedCompletionStatus(listener,
				&bytes_transferred, &completion_key, &overlapped, INFINITE);
		core::free(overlapped);
		if (!is_running()) {
			return true;
		} else if (completion_key == NULL) {
			listener_error("socket_listener.listen ERROR: Error waiting for IO completion packet");
			return false;
		} else if (!result) {
			DWORD error = GetLastError();
			if (error != ERROR_NETNAME_DELETED) {
				listener_error("socket_listener.listen ERROR: Error waiting for IO completion packet");
				return false;
			}
			/* the client closed the connection */
		}
		connection = (SOCKET) completion_key;
		return true;
	}

	static inline void free(socket_listener& listener, unsigned int thread_count) {
		for (unsigned int i = 0; i < thread_count; i++)
			PostQueuedCompletionStatus(listener.listener, 0, (DWORD)NULL, NULL);
		CloseHandle(listener.listener);
	}

#elif defined(__APPLE__) /* on Mac */
	int listener;
	struct kevent events[EVENT_QUEUE_CAPACITY];
	array<socket_type> event_queue;
	std::condition_variable cv;
	std::mutex event_queue_lock;

	socket_listener() : event_queue(EVENT_QUEUE_CAPACITY) { }

	template<bool ServerSocket>
	inline bool add_socket(socket_type& socket,
		const char* error_message = "socket_listener.add_socket ERROR: Failed to listen to socket")
	{
		struct kevent new_event;
		EV_SET(&new_event, socket.handle, EVFILT_READ, EV_ADD | (!ServerSocket ? EV_ONESHOT : 0), 0, 0, NULL);
		if (kevent(listener, &new_event, 1, NULL, 0, NULL) == -1) {
			if (errno == EBADF)
				return true; /* server is shutting down */
			listener_error(error_message);
			return false;
		}
		return true;
	}

	inline bool add_server_socket(socket_type& socket) { return add_socket<true>(socket); }
	inline bool add_client_socket(socket_type& socket) { return add_socket<false>(socket); }

	inline bool update_socket(socket_type& socket) {
		return add_socket<false>(socket, "socket_listener.update_socket ERROR: Failed to modify listen event");
	}

	inline bool remove_socket(socket_type& socket) const {
		return true;
	}

	template<typename AcceptedConnectionCallback, typename... CallbackArgs>
	inline bool accept(socket_type& server_socket, AcceptedConnectionCallback callback, CallbackArgs&&... callback_args) {
		int event_count = kevent(listener, NULL, 0, events, EVENT_QUEUE_CAPACITY, NULL);
		if (event_count == -1) {
			listener_error("socket_listener.accept ERROR: Error listening for incoming network activity");
			return false;
		}

		for (int i = 0; i < event_count; i++) {
			socket_type socket = (int) events[i].ident;

			if (socket == server_socket) {
				/* there's a new connection on the server socket */
				sockaddr_storage client_address;
				socklen_t address_size = sizeof(client_address);
				socket_type connection = ::accept(server_socket.handle, (sockaddr*) &client_address, &address_size);
				if (!connection.is_valid()) {
					if (errno == EINVAL)
						return true; /* the server is shutting down */
					perror("socket_listener.accept ERROR: Error establishing connection with client");
					return false;
				}

				if (!add_client_socket(connection)) {
					shutdown(connection.handle, 2); continue;
				}

				callback(connection, std::forward<CallbackArgs>(callback_args)...);
			} else {
				/* there is an event on a client connection */
				std::unique_lock<std::mutex> lck(event_queue_lock);
				event_queue.add(socket);
				cv.notify_one();
			}
		}
		return true;
	}

	template<typename IsRunningFunction>
	inline bool listen(socket_type& connection, IsRunningFunction is_running) {
		std::unique_lock<std::mutex> lck(event_queue_lock);
		while (event_queue.length == 0 && is_running())
			cv.wait(lck);
		if (is_running())
			connection = event_queue.pop();
		return true;
	}

	static inline void free(socket_listener& listener, unsigned int thread_count) {
		listener.cv.notify_all();
		::close(listener.listener);
	}

#else /* on Linux */
	int listener;
	epoll_event events[EVENT_QUEUE_CAPACITY];
	array<socket_type> event_queue;
	std::condition_variable cv;
	std::mutex event_queue_lock;

	socket_listener() : event_queue(EVENT_QUEUE_CAPACITY) { }

	template<bool ServerSocket>
	inline bool add_socket(socket_type& socket) {
		epoll_event new_event = {};
		new_event.events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLRDHUP | (!ServerSocket ? EPOLLONESHOT : 0);
		new_event.data.fd = socket.handle;
		if (epoll_ctl(listener, EPOLL_CTL_ADD, socket.handle, &new_event) == -1) {
			listener_error("socket_listener.add_socket ERROR: Failed to listen to socket");
			return false;
		}
		return true;
	}

	inline bool add_server_socket(socket_type& socket) { return add_socket<true>(socket); }
	inline bool add_client_socket(socket_type& socket) { return add_socket<false>(socket); }

	inline bool update_socket(socket_type& socket) {
		epoll_event new_event = {};
		new_event.events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLRDHUP | EPOLLONESHOT;
		new_event.data.fd = socket.handle;
		if (epoll_ctl(listener, EPOLL_CTL_MOD, socket.handle, &new_event) == -1) {
			if (errno == EBADF)
				return true; /* server is shutting down */
			listener_error("socket_listener.update_socket ERROR: Failed to modify listen event");
			shutdown(socket.handle, 2); return false;
		}
		return true;
	}

	inline bool remove_socket(socket_type& socket) {
		if (epoll_ctl(listener, EPOLL_CTL_DEL, socket.handle, NULL) == -1) {
			listener_error("socket_listener.remove_socket ERROR: Failed to remove listen event");
			return false;
		}
		return true;
	}

	template<typename AcceptedConnectionCallback, typename... CallbackArgs>
	inline bool accept(socket_type& server_socket, AcceptedConnectionCallback callback, CallbackArgs&&... callback_args) {
		int event_count = epoll_wait(listener, events, EVENT_QUEUE_CAPACITY, -1);
		if (event_count == -1) {
			listener_error("socket_listener.accept ERROR: Error listening for incoming network activity");
			return false;
		}

		for (int i = 0; i < event_count; i++) {
			socket_type socket = events[i].data.fd;

			if (socket == server_socket) {
				/* there's a new connection on the server socket */
				sockaddr_storage client_address;
				socklen_t address_size = sizeof(client_address);
				socket_type connection = ::accept(server_socket.handle, (sockaddr*) &client_address, &address_size);
				if (!connection.is_valid()) {
					if (errno == EINVAL)
						return true; /* the server is shutting down */
					perror("socket_listener.accept ERROR: Error establishing connection with client");
					return false;
				}

				if (!add_client_socket(connection)) {
					shutdown(connection.handle, 2); continue;
				}

				callback(connection, std::forward<CallbackArgs>(callback_args)...);
			} else {
				/* there is an event on a client connection */
				std::unique_lock<std::mutex> lck(event_queue_lock);
				event_queue.add(socket);
				cv.notify_one();
			}
		}
		return true;
	}

	template<typename IsRunningFunction>
	inline bool listen(socket_type& connection, IsRunningFunction is_running) {
		std::unique_lock<std::mutex> lck(event_queue_lock);
		while (event_queue.length == 0 && is_running())
			cv.wait(lck);
		if (is_running())
			connection = event_queue.pop();
		return true;
	}

	static inline void free(socket_listener& listener, unsigned int thread_count) {
		listener.cv.notify_all();
		::close(listener.listener);
	}
#endif
};

inline bool init(socket_listener& listener) {
#if defined(_WIN32)
	listener.buffer_wrapper.buf = listener.buffer;
	listener.buffer_wrapper.len = 4;
	listener.listener = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
	bool success = (listener.listener != NULL);
#elif defined(__APPLE__)
	listener.listener = kqueue();
	bool success = (listener.listener != -1);
#else
	listener.listener = epoll_create1(0);
	bool success = (listener.listener != -1);
#endif
	if (!success) {
		listener_error("init ERROR: Unable to initialize socket listener");
		return false;
	}
	return true;
}

/**
 * Reads `sizeof(T)` bytes from `in` and writes them to the memory referenced
 * by `value`. This function does not perform endianness transformations.
 * \param in a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T& value, socket_type& in) {
	return (recv(in.handle, (char*) &value, sizeof(T), MSG_WAITALL) > 0);
}

/**
 * Reads `length` elements from `in` and writes them to the native array
 * `values`. This function does not perform endianness transformations.
 *
 * **NOTE:** `length` should be non-zero. Otherwise, `recv` may wait
 * indefinitely for incoming data.
 *
 * \param in a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T* values, socket_type& in, unsigned int length)
{
#if !defined(NDEBUG)
	if (length == 0) {
		fprintf(stderr, "read WARNING: 'length' is zero.\n");
		return true;
	}
#endif
	return (recv(in.handle, (char*) values, sizeof(T) * length, MSG_WAITALL) > 0);
}

inline void network_error(const char* message) {
#if defined(_WIN32)
	errno = WSAGetLastError();
#endif
	perror(message);
}

enum class server_state {
	STOPPING = 0,
	STARTING = 1,
	STARTED = 2
};

template<typename ConnectionData, typename ProcessMessageCallback, typename... CallbackArgs>
void run_worker(socket_listener& listener, hash_map<socket_type, ConnectionData>& connections,
		std::mutex& connection_set_lock, server_state& state,
		ProcessMessageCallback process_message, CallbackArgs&&... callback_args)
{
	while (state != server_state::STOPPING) {
		socket_type connection;
		if (!listener.listen(connection, [&]() { return state != server_state::STOPPING; }))
			continue;
		if (state == server_state::STOPPING) return;

		uint8_t next;
		if (recv(connection.handle, (char*)&next, sizeof(next), MSG_PEEK) <= 0) {
			/* the other end of the socket was closed by the client */
			listener.remove_socket(connection);
			connection_set_lock.lock();
			bool contains; unsigned int index;
			free(connections.get(connection, contains, index));
			connections.remove_at(index);
			connection_set_lock.unlock();
			shutdown(connection.handle, 2);
		} else {
			/* there is a data waiting to be read, so read it */
			process_message(connection, connections, std::forward<CallbackArgs>(callback_args)...);

			/* continue listening on this socket */
			if (!listener.update_socket(connection)) {
				connection_set_lock.lock();
				bool contains; unsigned int index;
				free(connections.get(connection, contains, index));
				connections.remove_at(index);
				connection_set_lock.unlock();
				shutdown(connection.handle, 2);
			}
		}
	}
}

template<typename ConnectionData, typename ProcessMessageCallback>
inline std::thread start_worker(socket_listener& listener,
		hash_map<socket_type, ConnectionData>& connections,
		std::mutex& connection_set_lock, server_state& state,
		ProcessMessageCallback process_message)
{
	return std::thread(run_worker<ConnectionData, ProcessMessageCallback>,
			std::ref(listener), std::ref(connections), std::ref(connection_set_lock),
			std::ref(state), process_message);
}

template<typename ConnectionData, typename ProcessMessageCallback, typename... CallbackArgs>
inline std::thread start_worker(socket_listener& listener,
		hash_map<socket_type, ConnectionData>& connections,
		std::mutex& connection_set_lock, server_state& state,
		ProcessMessageCallback process_message, CallbackArgs&&... callback_args)
{
	return std::thread(run_worker<ConnectionData, ProcessMessageCallback, CallbackArgs...>,
			std::ref(listener), std::ref(connections), std::ref(connection_set_lock),
			std::ref(state), process_message, std::ref(std::forward<CallbackArgs>(callback_args)...));
}

template<typename ConnectionData, typename NewConnectionCallback, typename... CallbackArgs>
inline void accept_connection(socket_type& connection,
		hash_map<socket_type, ConnectionData>& connections,
		std::mutex& connection_set_lock,
		NewConnectionCallback new_connection_callback,
		CallbackArgs&&... callback_args)
{
	connection_set_lock.lock();
	bool contains; unsigned int index;
	connections.check_size();
	ConnectionData& data = connections.get(connection, contains, index);
	connections.table.keys[index] = connection;
	connections.table.size++;
	init(data);
	new_connection_callback(connection, data, std::forward<CallbackArgs>(callback_args)...);
	connection_set_lock.unlock();
}

template<bool Success>
inline void cleanup_server(server_state& state,
		std::condition_variable& init_cv, std::mutex& init_lock)
{
#if defined(_WIN32)
	WSACleanup();
#endif
	if (!Success) {
		std::unique_lock<std::mutex> lock(init_lock);
		state = server_state::STOPPING;
		init_cv.notify_all();
	}
}

template<bool Success>
inline void cleanup_server(server_state& state,
		std::condition_variable& init_cv,
		std::mutex& init_lock, socket_type& sock)
{
	shutdown(sock.handle, 2);
	cleanup_server<Success>(state, init_cv, init_lock);
}

template<typename ConnectionData = empty_data, typename ProcessMessageCallback,
	typename NewConnectionCallback, typename... CallbackArgs>
bool run_server(socket_type& sock, uint16_t server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count,
		server_state& state, std::condition_variable& init_cv, std::mutex& init_lock,
		hash_map<socket_type, ConnectionData>& connections, std::mutex& connection_set_lock,
		ProcessMessageCallback process_message, NewConnectionCallback new_connection_callback,
		CallbackArgs&&... callback_args)
{
#if defined(_WIN32)
	WSADATA wsa_state;
	if (WSAStartup(MAKEWORD(2,2), &wsa_state) != NO_ERROR) {
		fprintf(stderr, "run_server ERROR: Unable to initialize WinSock.\n");
		std::unique_lock<std::mutex> lock(init_lock);
		state = server_state::STOPPING; init_cv.notify_all(); return false;
	}
	sock = WSASocket(AF_INET6, SOCK_STREAM, 0, NULL, 0, WSA_FLAG_OVERLAPPED);
#else
	sock = socket(AF_INET6, SOCK_STREAM, 0);
#endif

	if (!sock.is_valid()) {
		network_error("run_server ERROR: Unable to open socket");
		cleanup_server<false>(state, init_cv, init_lock); return false;
	}

	int yes = 1;
	if (setsockopt(sock.handle, SOL_SOCKET, SO_REUSEADDR, (const char*) &yes, sizeof(yes)) != 0
	 || setsockopt(sock.handle, IPPROTO_TCP, TCP_NODELAY, (const char*) &yes, sizeof(yes)) != 0) {
		network_error("run_server ERROR: Unable to set socket options");
		cleanup_server<false>(state, init_cv, init_lock, sock); return false;
	}

	sockaddr_in6 server_addr;
	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin6_family = AF_INET6;
	server_addr.sin6_port = htons(server_port);
	server_addr.sin6_addr = in6addr_any;
	if (bind(sock.handle, (sockaddr*) &server_addr, sizeof(server_addr)) != 0) {
		network_error("run_server ERROR: Unable to bind to socket");
		cleanup_server<false>(state, init_cv, init_lock, sock); return false;
	}

	if (listen(sock.handle, connection_queue_capacity) != 0) {
		network_error("run_server ERROR: Unable to listen to socket");
		cleanup_server<false>(state, init_cv, init_lock, sock); return false;
	}

	socket_listener listener;
	if (!init(listener)) {
		network_error("run_server ERROR: Failed to initialize socket listener");
		cleanup_server<false>(state, init_cv, init_lock, sock); return false;
	}

	if (!listener.add_server_socket(sock)) {
		core::free(listener, worker_count);
		cleanup_server<false>(state, init_cv, init_lock, sock);
		return false;
	}

	/* make the thread pool */
	std::thread* workers = new std::thread[worker_count];
	for (unsigned int i = 0; i < worker_count; i++)
		workers[i] = start_worker(listener, connections, connection_set_lock, state, process_message, std::forward<CallbackArgs>(callback_args)...);

	/* notify that the server has successfully started */
	std::unique_lock<std::mutex> lock(init_lock);
	state = server_state::STARTED;
	init_cv.notify_all();
	lock.unlock();

	/* the main loop */
	while (state != server_state::STOPPING) {
		listener.accept(sock, accept_connection<ConnectionData, NewConnectionCallback, CallbackArgs...>,
				connections, connection_set_lock, new_connection_callback, std::forward<CallbackArgs>(callback_args)...);
	}

	core::free(listener, worker_count);
	for (unsigned int i = 0; i < worker_count; i++) {
		if (!workers[i].joinable()) continue;
		try {
			workers[i].join();
		} catch (...) { }
	}
	for (auto connection : connections)
		shutdown(connection.key.handle, 2);
	cleanup_server<true>(state, init_cv, init_lock, sock);
	delete[] workers;
	return true;
}

template<typename ProcessConnectionCallback>
bool run_client(
		const char* server_address, const char* server_port,
		ProcessConnectionCallback process_connection)
{
	addrinfo hints;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	addrinfo* addresses;
	int result = getaddrinfo(server_address, server_port, &hints, &addresses);
	if (result != 0 || addresses == NULL) {
#if defined(_WIN32)
		network_error("run_client ERROR: Unable to resolve address");
#else
		fprintf(stderr, "run_client ERROR: Unable to resolve address. %s\n", gai_strerror(result));
#endif
		return false;
	}

	socket_type sock;
	addrinfo* entry = addresses;
	for (; entry != NULL; entry = entry->ai_next) {
		sock = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
		if (!sock.is_valid()) {
			network_error("run_client ERROR: Unable to open socket");
			continue;
		}

		int yes = 1;
		if (setsockopt(sock.handle, IPPROTO_TCP, TCP_NODELAY, (const char*) &yes, sizeof(yes)) != 0) {
			network_error("run_client ERROR: Unable to set socket options");
			shutdown(sock.handle, 2); continue;
		}

		if (connect(sock.handle, entry->ai_addr, (socklen_t) entry->ai_addrlen) != 0) {
			network_error("run_client ERROR: Unable to connect");
			shutdown(sock.handle, 2); continue;
		}
		break;
	}
	freeaddrinfo(addresses);

	if (entry == NULL) {
		fprintf(stderr, "run_client ERROR: Unable to find server at %s:%s.\n", server_address, server_port);
		return false;
	}

	process_connection(sock);
	return true;
}

} /* namespace jbw */

#endif /* JBW_NETWORK_H_ */

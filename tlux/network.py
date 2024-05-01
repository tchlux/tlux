import logging
import pickle
import socket
import threading
from queue import Queue
from typing import Any, Callable, Optional


# Use pickle to serialize data.
def PICKLE_DUMPS(o: Any) -> bytes:
    return pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)


# Use pickle to load serialized data.
# WARNING: Generally unsafe.
def PICKLE_LOADS(o: bytes) -> Any:
    return pickle.loads(o)


# Get the IP address of this host, otherwise return the host name.
def get_ip(socket_type):
    try:
        # Get the fully qualified local hostname
        hostname = socket.getfqdn()
        if socket_type == socket.AF_INET:
            # Resolve the hostname to an IP address
            ip = socket.gethostbyname(hostname)
        elif socket_type == socket.AF_INET6:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET6)
            ip = addr_info[0][4][0]  # Retrieves the first IPv6 address
        else:
            raise ValueError(f"network.NetworkQueue Unexpected socket type {repr(socket_type)}")
    except Exception as e:
        # Otherwise just use the host name.
        ip = socket.gethostname()
    logging.info(f"network.NetworkQueue local IP {repr(ip)}")
    return ip


# This class provides the functionality of a "Queue" that works across multiple hosts using network connections.
class NetworkQueue:
    def __init__(
            self,
            host: Optional[str] = None,
            port: Optional[int] = None,
            listen_host: Optional[str] = None,
            listen_port: Optional[int] = 0,
            socket_type = socket.AF_INET,
            serialize: Callable[[Any],bytes] = PICKLE_DUMPS,
            deserialize: Callable[[bytes],Any] = PICKLE_LOADS,
    ):
        # Create a holder for clients, queues that contains incoming data from client connections.
        self.active = True
        self.exit_event = threading.Event()
        self.clients = {}
        self.queues = {}
        self.arrivals = Queue()
        self.popped = []
        self.socket_type = socket_type
        self.serialize = serialize
        self.deserialize = deserialize

        # Set the host name and port of the other NetworkQueue.
        self.host = host
        self.port = port

        # If we want to listen for outside connections, then do that.
        if (listen_port is not None):
            self.listener_socket = socket.socket(self.socket_type, socket.SOCK_STREAM)
            self.listener_host = listen_host or get_ip(self.socket_type)
            self.listener_socket.bind((self.listener_host, listen_port))
            self.listener_port = self.listener_socket.getsockname()[1]  # Retrieves the automatically assigned port if 0.
            self.listener_socket.listen()
            # Start a thread that listens for incoming connections.
            self.listener_thread = threading.Thread(target=self._accept_connections, daemon=True)
            self.listener_thread.start()
            logging.info(f"network.NetworkQueue listening at {self.listener_host}:{self.listener_port}")

        # Connect to the other NetworkQueue.
        if ((host is not None) and (port is not None)):
            logging.info(f"network.NetworkQueue establishing connection with {host}:{port}")
            host_socket = socket.socket(self.socket_type, socket.SOCK_STREAM)
            host_socket.connect((host, port))
            self.clients["host"] = (
                host_socket,
                threading.Thread(target=self._client_listener, args=(host_socket, "host"), daemon=True)
            )
            # Start the dedicated host communications listener.
            self.clients["host"][1].start()
            logging.info(f"network.NetworkQueue connected to host {host}:{port}")


    # Listen for connections from other NetworkQueue objects.
    def _accept_connections(self):
        while (not self.exit_event.is_set()):
            try:
                client_socket, addr = self.listener_socket.accept()
                if self.exit_event.is_set():
                    client_socket.close()
                    break
                client_id = addr[1]  # Using the client's port as a simple ID
                # Store the socket and the communication thread for this client.
                self.clients[client_id] = (
                    client_socket,
                    threading.Thread(target=self._client_listener, args=(client_socket, client_id), daemon=True)
                )
                # Start a dedicated client communications listener.
                self.clients[client_id][1].start()
                logging.info(f"network.NetworkQueue connected to client {repr(client_id)} at {addr}")
            except socket.error:
                break


    # Establishes a dedicated Queue for a client and listens for communications.
    def _client_listener(self, client_socket: socket.socket, client_id: int):
        q = Queue()
        self.queues[client_id] = q
        while (not self.exit_event.is_set()):
            try:
                # First listen for an integer that specifies the size of the data.
                size_data = client_socket.recv(8)
                if (size_data == b''):
                    logging.info(f"NetowrkQueue received data b'' from client {repr(client_id)} indicating a closed connection.")
                    break
                # Then decode that integer listen for the object that follows.
                obj_size = int.from_bytes(size_data, "big")
                logging.info(f"network.NetworkQueue object sized {obj_size} incoming from client {repr(client_id)}")
                # Now iteratively receive all chunks of data from the client.
                data_received = b''
                while len(data_received) < obj_size:
                    chunk = client_socket.recv(obj_size - len(data_received))
                    if not chunk:
                        logging.error(f"network.NetworkQueue got size {obj_size} with incomplete data from client {repr(client_id)}, connection closed prematurely.")
                        break
                    data_received += chunk
                # Put the received data into the client's designated queue stored locally.
                q.put(self.deserialize(data_received))
                # Indicate the arrival of data from this client in the parent arrivals queue.
                self.arrivals.put(client_id)
                logging.info(f"network.NetworkQueue data of size {obj_size} successfully arrived from client {repr(client_id)}")
            # Socket errors trigger an immediate break.
            except socket.error:
                logging.error(f"network.NetworkQueue encountered socket error with client {repr(client_id)}")
                break
            # In case of any errors, 
            except Exception as e:
                logging.error(f"network.NetworkQueue encountered error with client {repr(client_id)}: {e}")
                break
        # Close the connection.
        client_socket.close()
        # Remove this client from the tracked queues and clients.
        self.queues.pop(client_id, None)
        self.clients.pop(client_id, None)
        logging.info(f"network.NetworkQueue Connection to client {client_id} closed")


    # Send an item to clients (or a specific client if provided).
    def put(self, item, client_id=None):
        serialized_item = self.serialize(item)
        total_size = len(serialized_item)
        size_bytes = total_size.to_bytes(8, "big")

        # Send data in a loop until all is sent
        def send_all(socket, data):
            total_sent = 0
            while total_sent < len(data):
                sent = socket.send(data[total_sent:])
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                total_sent += sent

        # Sending to a specific client or all clients
        if client_id in self.clients:
            client_socket, _ = self.clients[client_id]
            try:
                send_all(client_socket, size_bytes + serialized_item)
            except Exception as e:
                logging.error(f"network.NetworkQueue failed to send item to client {repr(client_id)}: {e}")
            logging.info(f"network.NetworkQueue sent {8 + total_size} bytes to {repr(client_id)}")
        else:
            for client_id, (client_socket, _) in self.clients.items():
                try:
                    send_all(client_socket, size_bytes + serialized_item)
                except Exception as e:
                    logging.error(f"network.NetworkQueue failed to send item to client {repr(client_id)}: {e}")
                logging.info(f"network.NetworkQueue sent to {repr(client_id)}")


    # Get an item from clients (or a specific client if provided).
    def get(self, client_id=None):
        if (client_id in self.queues):
            q = self.queues[client_id]
            item = q.get()
            self.popped.append(client_id)
            return item
        else:
            client_id = self.arrivals.get()
            while (client_id in self.popped):
                self.popped.remove(client_id)
                client_id = self.arrivals.get()
            return self.queues[client_id].get()


    # Close this Queue once it is done being used.
    def close(self):
        self.exit_event.set()
        for (client_socket, client_thread) in self.clients.values():
            client_socket.close()
        if hasattr(self, 'server_socket'):
            self.listener_socket.close()
        logging.info("network.NetworkQueue shutdown completed")


    # Deletion of this class should trigger everything to be closed.
    def __del__(self):
        self.close()


    # Produce a string summarizing this NetworkQueue.
    def __str__(self):
        host = f""
        if self.host is not None:
            host = f"host-name={repr(self.host)} host-port={repr(self.port)} "
        listener = f""
        if hasattr(self, "listener_port"):
            listener = f"listen-name={repr(self.listener_host)} listen-port={self.listener_port} "
        clients = f""
        if len(self.clients) > 0:
            clients = f"client-count={len(self.clients)}"
        return f"<NetworkQueue {listener}{host}{clients}>"

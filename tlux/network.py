import logging
import pickle
import socket
import threading
import time
from queue import Queue
from typing import Any, Callable, Optional

SERIALIZER = pickle
ONE_MB = 1048576

# Use pickle to serialize data.
def DUMPS(o: Any) -> bytes:
    return SERIALIZER.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)


# Use pickle to load serialized data.
# WARNING: Generally unsafe.
def LOADS(o: bytes) -> Any:
    return SERIALIZER.loads(o)


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
    except Exception:
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
            chunk_size: int = 8*1024, # 8KB
            recv_buffer: int = 8*2048, # 16KB
            send_buffer: int = 8*1024 + 12, # 8KB + TCP header
            egress_limit_mb_sec: int = -1, # 100 MB/s
            serialize: Callable[[Any],bytes] = DUMPS,
            deserialize: Callable[[bytes],Any] = LOADS,
    ):
        # Create a holder for clients, queues that contains incoming data from client connections.
        self.active = True
        self.exit_event = threading.Event()
        self.clients = {}
        self.queues = {}
        self.arrivals = Queue()
        self.popped = []
        self.socket_type = socket_type
        self.chunk_size = chunk_size
        self.recv_buffer = recv_buffer
        self.send_buffer = send_buffer
        self.egress_limit_mb_sec = egress_limit_mb_sec
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
            self._establish_host_connection()


    # Set up a socket for communication as part of the NetworkQueue.
    def _setup_socket(self, s):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buffer)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer)


    # Establish a connection to the 'host' NetworkQueue.
    def _establish_host_connection(self):
        logging.info(f"network.NetworkQueue establishing connection with 'host' {self.host}:{self.port}")
        host_socket = socket.socket(self.socket_type, socket.SOCK_STREAM)
        self._setup_socket(host_socket)
        host_socket.connect((self.host, self.port))
        self.clients["host"] = (
            host_socket,
            threading.Thread(target=self._client_listener, args=(host_socket, "host"), daemon=True)
        )
        # Start the dedicated host communications listener.
        self.clients["host"][1].start()
        logging.info(f"network.NetworkQueue connected to host {self.host}:{self.port}")


    # Listen for connections from other NetworkQueue objects.
    def _accept_connections(self):
        while (not self.exit_event.is_set()):
            try:
                client_socket, addr = self.listener_socket.accept()
                if self.exit_event.is_set():
                    client_socket.close()
                    break
                self._setup_socket(client_socket)
                client_id = f"{addr[0]}:{addr[1]}"  # Using the client's IP:port as a simple ID
                # Store the socket and the communication thread for this client.
                self.clients[client_id] = (
                    client_socket,
                    threading.Thread(target=self._client_listener, args=(client_socket, client_id), daemon=True)
                )
                # Start a dedicated client communications listener.
                self.clients[client_id][1].start()
                logging.info(f"network.NetworkQueue connected to client {repr(client_id)} at {addr}")
            except socket.error as e:
                logging.error(f"network.NetworkQueue encountered accepter socket error: {e}")
                break


    # Establishes a dedicated Queue for a client and listens for communications.
    def _client_listener(self, client_socket: socket.socket, client_id: str):
        q = Queue()
        self.queues[client_id] = q
        while (not self.exit_event.is_set()):
            try:
                # First listen for an integer that specifies the size of the data.
                size_data = client_socket.recv(8)
                if (size_data == b''):
                    logging.info(f"NetworkQueue received data b'' from client {repr(client_id)} indicating a closed connection.")
                    break
                # TODO: Have a loop to ensure "size_data" is 8 bytes long.
                # Then decode that integer listen for the object that follows.
                obj_size = int.from_bytes(size_data, "big")
                logging.info(f"network.NetworkQueue object sized {obj_size} incoming from client {repr(client_id)}")
                # Now iteratively receive all chunks of data from the client.
                data_received = 0
                chunks = []
                while data_received < obj_size:
                    chunk = client_socket.recv(min(self.chunk_size, obj_size - data_received))
                    if not chunk:
                        logging.error(f"network.NetworkQueue got size {obj_size} with incomplete data from client {repr(client_id)}, connection closed prematurely.")
                        break
                    # logging.info(f"network.NetworkQueue received {len(chunk)} bytes from client {repr(client_id)}")
                    chunks.append(chunk)
                    data_received += len(chunk)
                # Put the received data into the client's designated queue stored locally.
                q.put(self.deserialize(b''.join(chunks)))
                # Indicate the arrival of data from this client in the parent arrivals queue.
                self.arrivals.put(client_id)
                logging.info(f"network.NetworkQueue data of size {obj_size} successfully arrived from client {repr(client_id)}")
            # Socket errors trigger an immediate break.
            except socket.error as e:
                logging.error(f"network.NetworkQueue encountered listener socket error with client {repr(client_id)}: {e}")
                break
            # In case of any errors,
            except Exception as e:
                logging.error(f"network.NetworkQueue encountered error with client {repr(client_id)}: {e}")
                break
        # Close the connection.
        client_socket.close()
        if (client_id == "host"):
            logging.warning("network.NetworkQueue host connection failed, attempting to reconnect..")
            self._establish_host_connection()
        else:
            # Remove this client from the tracked queues and clients.
            self.queues.pop(client_id, None)
            self.clients.pop(client_id, None)
            logging.info(f"network.NetworkQueue Connection to client {client_id} closed")


    # Send an item to clients (or a specific client if provided).
    def put(self, item, client_id=None, retries=5, retry_wait_sec=1.0):
        serialized_item = self.serialize(item)
        total_size = len(serialized_item)
        size_bytes = total_size.to_bytes(8, "big")

        # Send data in a loop until all is sent
        def send_all(sock, data, chunk_size=self.chunk_size):
            start = time.time()
            total_sent = 0
            while total_sent < len(data):
                sent = sock.send(data[total_sent:total_sent+chunk_size])
                if sent == 0:
                    raise RuntimeError("network.NetworkQueue socket connection broken")
                # logging.info(f"network.NetworkQueue sent {sent} bytes to client {repr(client_id)}")
                total_sent += sent
                # Enforce an upper limit on egress rate per chunk by sleeping if needed.
                if 0 < self.egress_limit_mb_sec < float('inf'):
                    end = time.time()
                    sent_mb = sent / ONE_MB
                    egress_rate_mb_sec = sent_mb / (end - start)
                    if egress_rate_mb_sec > self.egress_limit_mb_sec:
                        wait_time_sec = sent_mb / self.egress_limit_mb_sec - (end - start)
                        time.sleep(wait_time_sec)
                        if total_sent <= 5*self.chunk_size:
                            logging.info(f"network.NetworkQueue sleeping for {wait_time_sec} seconds to enforce egress rate limit of {self.egress_limit_mb_sec:.2f} MB/s after observing {egress_rate_mb_sec:.2f} MB/s transmission.")
                    start = time.time()

        # Sending to a specific client or all clients
        if client_id in self.clients:
            clients = [(client_id, self.clients[client_id])]
        else:
            clients = list(self.clients.items())
        # Send the bytes
        logging.info(f"network.NetworkQueue sending object sized {total_size} to {'all ' if len(self.clients) > 1 else ''}{len(self.clients)} client{'s' if len(self.clients) != 1 else ''}..")
        for client_id, (client_socket, _) in clients:
            try:
                # Then send all of the data.
                send_all(client_socket, size_bytes + serialized_item)
            except Exception as e:
                logging.error(f"network.NetworkQueue failed to send item to client {repr(client_id)} ({retries} retries remaining): {e}")
                if (retries > 0):
                    time.sleep(retry_wait_sec)
                    self.put(item, client_id=client_id, retries=retries-1, retry_wait_sec=2*retry_wait_sec)
                else:
                    raise e
            logging.info(f"network.NetworkQueue sent {8 + total_size} bytes to {repr(client_id)}")


    # Get an item from clients (or a specific client if provided).
    def get(self, client_id=None, return_client=False, **get_kwargs):
        if (client_id in self.queues):
            q = self.queues[client_id]
            item = q.get(**get_kwargs)
            self.popped.append(client_id)
        else:
            client_id = self.arrivals.get(**get_kwargs)
            while (client_id in self.popped):
                self.popped.remove(client_id)
                client_id = self.arrivals.get(**get_kwargs)
            item = self.queues[client_id].get(**get_kwargs)
        # Return either both the item and client, or just the item.
        if return_client:
            return item, client_id
        else:
            return item


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
        host = ""
        if self.host is not None:
            host = f"host={repr(self.host)}, port={repr(self.port)}, "
        listener = ""
        if hasattr(self, "listener_port"):
            listener = f"listen_host={repr(self.listener_host)}, listen_port={self.listener_port}, "
        clients = ""
        if len(self.clients) > 0:
            clients = f"client_count={len(self.clients)}, "
        socket_type = str(repr(self.socket_type)).split(":")[0].split(".")[1]
        socket_type = f"socket_type=socket.{socket_type}"
        return f"NetworkQueue({(listener+host+clients+socket_type)})"

# Set logging to be verbose.
import logging
logging.basicConfig(level=logging.INFO)

# Running the NetworkQueue test.
if __name__ == "__main__":
    # Import the NetworkQueue object.
    from tlux.network import NetworkQueue

    # Offer a "listen-and-repeat" testing mode for across-network tests.
    import sys
    if "listen-and-repeat" in sys.argv:
        queue = NetworkQueue(listen_port=64512)
        result = queue.get()
        while (result is not None):
            queue.put(result)
            result = queue.get()
        exit()

    # Offer a "ping-and-show" testing mode for across-network tests.
    elif "ping-and-show" in sys.argv:
        host = sys.argv[sys.argv.index("ping-and-show") + 1]
        port = int(sys.argv[sys.argv.index("ping-and-show") + 2])
        if (len(sys.argv) > sys.argv.index("ping-and-show") + 3):
            message = sys.argv[sys.argv.index("ping-and-show") + 3]
        else:
            message = "Hello WORLD!"
        queue = NetworkQueue(host=host, port=port, listen_port=None)
        queue.put(message)
        result = queue.get()
        print(result, flush=True)
        exit()

    # Otherwise run a local test.

    # Start server
    server = NetworkQueue()

    # Start a client (that does not listen for incoming connections).
    client = NetworkQueue(host=server.listener_host, port=server.listener_port, listen_port=None)

    # Put something into the server queue.
    print("", flush=True)
    print("Sending from server to client..", flush=True)
    phrase = "hello world!"
    server.put(phrase)
    result = client.get()
    assert (phrase == result), f"Expected phrase retrieved {repr(result)} to be same as submitted {repr(phrase)}."
    print(result, flush=True)
    print()

    # Try sending the opposite direction.
    print("", flush=True)
    print("Sending from client to server..", flush=True)
    phrase = "world hello!"
    client.put(phrase)
    result = server.get()
    assert (phrase == result), f"Expected phrase retrieved {repr(result)} to be same as submitted {repr(phrase)}."
    print(result, flush=True)
    print()

    # Try sending a large object.
    LARGE_OBJECT_TEST = True
    if LARGE_OBJECT_TEST:
        print("", flush=True)
        print("Sending large object from server to client..", flush=True)
        large_object = bytearray(200 * 2**20)
        server.put(large_object)
        result = client.get()
        assert (large_object == result), f"Expected phrase retrieved {len(result)} to be same as submitted {len(large_object)}."

    print("", flush=True)
    print("server: ", server, flush=True)
    print("client: ", client, flush=True)

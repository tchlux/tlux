import socket
from typing import Optional
from tlux.decorators import auto_cli
from tlux.network import NetworkQueue


# Listen with a NetworkQueue and echo things transmitted back to any attached clients.
#
# Arguments:
#   host (str): The string name of the host where this process will listen.
#   port - The integer port number to communicate with this process on.
#   greeting -- The string that is printed when this process launches.
#   
@auto_cli
def echo(host: Optional[str] = None, port: int = 0, greeting: str = "Hello!"):
    print("greeting: ", greeting, flush=True)

    # Create a queue.
    queue = NetworkQueue(listen_host=host, listen_port=port) # , socket_type=socket.AF_INET6)
    print(queue, flush=True)

    # Listen and echo.
    result = queue.get()
    while (result is not None):
        queue.put(result)
        result = queue.get()


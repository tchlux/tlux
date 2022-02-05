import sys

# Function for printing the help message.
def print_help_message():
    print("""

tlux -- A package.

### Python

    > import tlux

  Descriptions of the contents follow.

### Command line

    $ python -m tlux <args>

  Run the 'tlux' package.

    """)

# Provide help.
if len(sys.argv) >= 0:
    print_help_message()

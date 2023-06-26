import sys

# Function for printing the help message.
def print_help_message():
    print("""
tlux -- A package aligned with the research compendium of T.C.H.Lux containing a all necessary code to reproduce the work.

### Python

    > import tlux
    > help(tlux)

  Descriptions of the contents follow.

### Command line

    $ python -m tlux [--clean] [--build] [-h] [--help]

  Run the 'tlux' package. When '--clean' is provided, all built directories and compiled
  codes are moved into a stated temporary directory. When '--build' is provided, all
  internal compiled modules are build (using 'fmodpy'). When '-h' or '--help' is provided,
  this message is printed.
    """)

# Clean the build.
if ("--clean" in sys.argv):
    sys.argv.remove("--clean")
    from tlux.setup import clean_all
    clean_all()

# Clean the build.
if ("--build" in sys.argv):
    sys.argv.remove("--build")
    from tlux.setup import build_all
    build_all()

# Provide help.
if (len(sys.argv) > 1) or ("-h" in sys.argv) or ("--help" in sys.argv):
    print_help_message()

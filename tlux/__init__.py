# Get the version number from the setup file
import os

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ABOUT_DIR = os.path.join(DIRECTORY, "about")
VERSION_FILE = os.path.join(ABOUT_DIR, "version.txt")
if (os.path.exists(VERSION_FILE)):
    with open(VERSION_FILE) as f:
        __version__ = f.read().strip()
else:
    __version__ = "unknown"

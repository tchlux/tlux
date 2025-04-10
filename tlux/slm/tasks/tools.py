# tools.py
# 
# This file defines all available tools that can be used by the worker to operate on
# and modify their environment.

import os
import re
import subprocess
import sys

from utilities import extract_code, remove_wrappers, setup_venv
from tlux.slm import logged_server_chat_complete as chat_complete

# Sandbox directory
sandbox_dir = "sandbox_dir"
os.makedirs(sandbox_dir, exist_ok=True)

# Virtual environment directory
venv_dir = "sandbox_venv"

# ----------------------------------------------------------------------------------------
# Private helpers for the tools.

# Read tools and extract documentation comments and function
# definitions.
def _get_tools(file_path: str = __file__, include_source: bool = False) -> list[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()
    # Read the lines, looking for public functions, comments that precede them, and source.
    lines = source.splitlines()
    in_function = False
    extracted = []
    comment_block = []
    function_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            comment_block.append(line)
        elif stripped == '':
            comment_block = []  # Reset comment block on blank line
        elif stripped.startswith('def ') and (not stripped.startswith('def _')) and (not line.startswith(' ')):  # Top-level public def
            if in_function:
                extracted.append('\n'.join(function_lines))
                function_lines = []
            in_function = True
            function_lines.extend(comment_block)
            function_lines.append(line)
            comment_block = []
        elif in_function:
            if not line.startswith(' '):
                # End of function
                extracted.append('\n'.join(function_lines))
                function_lines = []
                in_function = False
                if stripped.startswith('#'):
                    comment_block = [line]
                else:
                    comment_block = []
            elif include_source:
                function_lines.append(line)
        else:
            comment_block = []
    if in_function:
        extracted.append('\n'.join(function_lines))
    # Add ellipses for function bodies if the source is not included.
    if not include_source:
        for i in range(len(extracted)):
            extracted[i] += "\n    ..."
    # Return the list of function strings.
    return extracted

# ----------------------------------------------------------------------------------------


# Write content to a specified file. If the file already exists, an
# error will be raised. If the specified directories do not exist,
# they will be created.
#
# Arguments:
#   filename - A string specifying the path to the file relative to
#              the root directory.
#   content - A string containing the text to write to the file (e.g.,
#             Python code, test cases, or any text data).
#
# Returns:
#   A string with a confirmation message indicating whether the write
#   operation was successful.
#
def create_file(filename: str, content: str) -> str:
    filename = remove_wrappers(filename)
    full_path = os.path.join(sandbox_dir, filename)
    if os.path.exists(full_path):
        return f"Error: File `{filename}` already exists."
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        return f"File `{filename}` successfully written with {len(content)} bytes from `content`."
    except Exception as e:
        return f"Error writing file `{filename}`: {str(e)}"


# Alter the contents of an existing file to meet specified changes.
#
# Arguments:
#   filename - A string specifying the path to an existing file
#              relative to the root directory.
#   changes - A plain text description of the changes that should be
#             made to the contents of the existing file.
#
# Returns:
#   A string with a confirmation message indicating whether the
#   operation was successful.
#
def modify_file(filename: str, changes: str) -> str:
    filename = remove_wrappers(filename)
    full_path = os.path.join(sandbox_dir, filename) 
    # Actual function.
    if not os.path.exists(full_path):
        return f"Error: '{filename}' does not exist."
    with open(full_path) as f:
        file_contents = f.read()
    # Chat 
    response, _ = chat_complete(messages=[
        f"The next message will contain the contents of '{filename}'. Further instruction for changes will follow.",
        "Understood.",
        "```\n" + file_contents.replace("```", "``") + "\n```",
        "Received. Ready for instructions.",
        "In my next message, I will specify the changes. I want you to respond in plain text by first explaining how you plan to implement the changes to the file and what parts need to be updated and what parts can remain the same. After specifying your plans, produce a code block that contains the entirety of the new content for the file.",
        "Understood. Please specify what changes you would like.",
        changes,
    ])
    new_code = extract_code(response)
    if len(new_code) == 0:
        return "Error: Changes failed and resulted in an empty code file."
    with open(full_path, "w") as f:
        f.write(new_code)
    return f"Successfully updated '{filename}' contents with specified changes."


# Execute a specified Python file in a virtual environment with the
# newest Python 3 available.  Only Python files (.py) can be executed.
#
# Arguments:
#   filename - A string specifying the path to the Python file
#              relative to the root directory.
#
# Returns:
#   A string containing the output of the execution, including any
#   error messages, or an error message if the file does not exist or
#   cannot be executed.
#
def run_file(filename: str) -> str:
    filename = remove_wrappers(filename)
    if not filename.endswith('.py'):
        return f"Error: Only Python files (.py) can be executed. Provided filename '{filename}' is invalid."
    full_path = os.path.join(sandbox_dir, filename)
    if not os.path.exists(full_path):
        return f"Error: File '{filename}' not found."
    setup_venv(venv_dir)
    python_exe = os.path.join(venv_dir, "bin", "python")
    try:
        result = subprocess.run([python_exe, full_path], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Output: {result.stdout}"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error executing file '{filename}': {str(e)}"


# Read the contents of a specified file.
#
# Arguments:
#   filename - A string specifying the path to the file relative to
#              the root directory.
#
# Returns:
#   A list of strings, where each string is a chunk of the file
#   content. For small files, this may be a list with a single string
#   containing the entire content. For large files, it will be split
#   into multiple chunks. If the file does not exist, a string with an
#   error message is returned (e.g., "Error: File not found").
#
def read_file(filename: str) -> str:
    filename = remove_wrappers(filename)
    full_path = os.path.join(sandbox_dir, filename)
    if not os.path.exists(full_path):
        return f"Error: File '{filename}' not found"
    try:
        with open(full_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"


# List all files in the root directory and its subdirectories.
#
# No Arguments.
#
# Returns:
#   A list of strings, where each string is a file path relative to the root
#   directory.
#
def see_all_files() -> list[str]:
    file_list = []
    for root, dirs, files in os.walk(sandbox_dir):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), sandbox_dir)
            file_list.append(relative_path)
    return file_list


# Install a specified Python module in the virtual environment.
#
# Arguments:
#   dependency - A string specifying the name of the Python module to
#                install (e.g., "numpy").
#
# Returns:
#   A string with a confirmation message indicating whether the
#   installation was successful (e.g., "Dependency 'numpy' installed
#   successfully").
#
def install_dependency(dependency: str) -> str:
    dependency = remove_wrappers(dependency)
    setup_venv(venv_dir)
    pip_exe = os.path.join(venv_dir, "bin", "pip")
    try:
        result = subprocess.run([pip_exe, "install", dependency], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Dependency '{dependency}' installed successfully"
        else:
            return f"Error installing dependency '{dependency}': {result.stderr}"
    except Exception as e:
        return f"Error installing dependency '{dependency}': {str(e)}"

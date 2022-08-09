'''--------------------------------------------------------------------
                            regex.py
 
 A fast regular expression matching code for Python, built on top of
 the 'regex.c' library. The main function provided here is:

   match(regex, string) -> (start, end) or None or RegexError,

 Documentation for 'regex.c' follows.

''' 


#  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
#                         Python development
# 
# 
#   write `fmatch` function to go with the existing
#   `match`, `matcha`, and `fmatcha` functions.
# 
#   add parallelism into the C library code, including a file walk,
#   to offer the full `frex` stack within the C code.
# 
#   fix all documentation to align with the current structure
# 
#   write a C function for translating from classic regular
#   expressions into fast regular expressions
# 
#   default regex library has a `finditer` command that iterates over
#   matches and streams bytes to the regular experssion library from
#   a buffer, can that be done here?
#   ^^ write a smatch and sfmatch that do 'stream' style matching,
#      being given a start index and number of bytes to parse,
#      maintaining a global state that is used for repeated calls to
#      the same C function
# 
#   create a Python test case that would reveal any memory leak in the
#   returned allocatable arrays from the C library
# 
#   look into building c extensions automatically on any platform,
#   some compilation customization will need to be done for windows
#   ^^ should I support something I don't plan on using myself?
# 
#   write a "summary" of the algorithm in HTML with embedded SVG files
#   and a theoretic description
# 
#   build some other code that uses "match" and make sure it works as
#   would be expected, measure performance to see what's slow
# 
#   use Pool.map instead of the homegrown solution, remove dependency
#   on the parallel map code <- the Pool.map solution is slower, so
#   maybe it's best to use the custom matching code. Otherwise, it
#   might be best to just use `os.fork` and manually create all of
#   the multiprocessing environment. Then this code will definitely
#   not work on Windows.
# 
#   convert things that are not bytes nor strings into strings using
#   the "str" function inside of `match`?
#
#   make the test code automatically run when the code is compiled,
#   checking for correct compilation
# 
#   regex line based mode from the command line
# 
#   regex invert match from the command line (default to activate line based mode)
# 
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
#                            C development
# 
#   The token following a ".*" can get excluded from the final match,
#    because the 'start index' of that token is updated before the
#    token after it is matched and the regex is completely discovered.
#    Use a queue to store the tokens instead of stack (track start and
#    end instead of just count. Special tokens should be given top
#    priority, then character sets.
# 
#   If the "start" index hasn't changed, don't add another match (for
#   `matcha` type operations, causes too many overlapping matches).
# 
#  Make this a pure C executable, place all Python functionality into
#   this file when compiled (without DEBUG).
# 
#  Swap out DEBUG for DEVELOP.
# 
#  Refactor into functions:
#    match -- find first regex match in a string
#    matcha -- find all regex matches in a string
#    matchs -- stream string matches by making repeated calls to same function
#    fmatch -- find the first regex match in a file
#    fmatcha -- find all regex matches in a file
#    fmatchs -- stream file matches by making repeated calls to same function
#    imatch -- find the first match from standard input and exit
#    imatcha -- find all matches from standard input
#    imatchs -- stream matches from standard input
#    frex -- file regular experession search utility, like `grep` but better
#            parallel (num cores) or serial
#            recurse into directories or not
#            provide path names or path patterns
#            optional redirect from standard in input
#            translate (most) standard regexes or give error message
#            parallel `fmatchs` (`imatchs`) that prints output to
#              standard output, or specified file object
# 
#  Write tests for `fmatch`.
# 
#  Rescan comments and update all documenation throughout.
# 
#  Identify ways to reduce size of code within reason, should some
#   things be abstracted as functions to save repetition?
# 
#  Write `randre` that generates a random string that matches the
#   given regular expression, tile the space of matches so that
#   all the shortest matches come first, then longer ones
#  
#  Find way to pass in arbitrary "next token" function to the match
#   function, that way `match` and `fmatch` can all go through one
#   function whether given a character array or a file
# 
#  ___________________________________________________________________


PRINT_PREFIX = "# "
PRINT_FILE_SEPARATORS = False
HALF_MAX_PREVIEW_WIDTH = 20

# Import ctypes for loading the underlying C regex library.
import os, ctypes

# --------------------------------------------------------------------
#                 Darwin (macOS) / Linux (Ubuntu) import
_clib_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libregex.so")
# Import or compile the C file.
try:
    clib = ctypes.CDLL(_clib_bin)
except:
    from tlux.setup import build_regex
    clib = build_regex()
# --------------------------------------------------------------------


# Exception to raise when errors are reported by the regex library.
class RegexError(Exception): pass


# Given a regular expression in a Unix-like format, translate it to a
# regular experssion that is (roughly) equivalent in the language of
# the `regex.c` library.
def translate_regex(regex, case_sensitive=True):
    # Do substitutions that make the underlying regex implementation
    # behave more like common existing regex packages.
    if (len(regex) > 0):
        # Add a ".*" to the front of the regex if the beginning of the
        # string was not explicitly desired in the match pattern.
        if (regex[0] == "^"): regex = regex[1:]
        elif ((len(regex) < 2) or (regex[0] != ".") or (regex[1] != '*')): 
            regex = ".*" + regex
        # Add a "{.}" to the end of the regex if the end of the string
        # was explicitly requested in the pattern.
        if (regex[-1] == "$"): regex = regex[:-1] + "{.}"
    # Replace all alphebetical characters with token sets that include
    # all cases of that character.
    if (not case_sensitive):
        i = 0
        in_literal = False
        while (i < len(regex)):
            # Check for the beginning of a token set.
            if ((regex[i] == '[') and (not in_literal)):
                in_literal = True
                contains = set()
                missing = set()
            # Handle a currently-active token set.
            elif (in_literal):
                # Check for the end of this token set.
                if (regex[i] == ']'):
                    in_literal = False
                    # Add all the missing characters to this token set.
                    missing = ''.join(sorted(missing.difference(contains)))
                    regex = regex[:i] + missing + regex[i:]
                    # Increment i appropriately.
                    i += len(missing)-1
                # Add the paired-case if we see a cased character.
                elif (regex[i].isalpha()):
                    contains.add(regex[i])
                    if (regex[i].islower()):
                        missing.add(regex[i].upper())
                    else:
                        missing.add(regex[i].lower())
            # Otherwise replace this single character with a token set.
            elif (regex[i].isalpha()):
                if (regex[i].islower()):
                    token_set = f'[{regex[i]}{regex[i].upper()}]'
                else:
                    token_set = f'[{regex[i]}{regex[i].lower()}]'
                regex = regex[:i] + token_set + regex[i+1:]
                i += len(token_set)-1
            # Increment to the next token.
            i += 1
    # Return the now-prepared regular expression.
    return regex


# Translate 'start' and 'end' values that are returned by the `regex.c`
# library, raising appropriate errors as defined by the library.
def _translate_return_values(regex, start, end):
    # Return appropriately (handling error flags).
    if (start >= 0): return (start, end)
    elif (start < 0):
        if (end == 0): return None # no match found
        elif (end == -1): return (0, 0) # empty regular expression
        elif ((start == -1) and (end == -5)): return None  # empty string
        elif (end < -1): # error code provided by C
            err = f"Invalid regular expression (code {-end})"
            if (start < -1):
                start -= sum(1 for c in str(regex,"utf-8") if c in "\n\t\r\0")
                err += f", error at position {-start-1}.\n"
                err += f"  {str([regex])[1:-1]}\n"
                err += f"  {(-start)*' '}^"
            else: err += "."
            raise(RegexError(err))


# Find a match for the given regex in string.
# 
#   match(regex, string) -> (start, end) or None or RegexError,
# 
# where "regex" is a string defining a regular expression and "string"
# is a string to be searched against. The function will return None
# if there is no match found. If a match is found, a tuple with
# integers "start" (inclusive index) and "end" (exclusive index) will
# be returned. If there is a problem with the regular expression,
# a RegexError will be raised.
#
# Some substitutions are made before passing the regular expressions
# "regex" to the "match" function in 'regex.c'. These substitutions
# include:
#
#  - If "^" is at the beginning of "regex", it will be removed (because
#    the underlying library implicitly assumes beginning-of-string.
#  - If no "^" is placed at the beginning of "regex", then ".*" will
#    be appened to the beginning of "regex" to behave like other 
#    regular expression libraries.
#  - If "$" is the last character of "regex", it will be substituted
#    with "{.}", the appropriate pattern for end-of-string matches.
# 
def match(regex, string, **translate_kwargs):
    # Translate the regular expression to expected syntax.
    regex = translate_regex(regex, **translate_kwargs)
    # Call the C utillity.
    #   initialize memory storage for the start and end of a match
    start = ctypes.c_int()
    end = ctypes.c_int()
    #   convert strings into character arrays
    if (type(regex) == str): regex = regex.encode("utf-8")
    if (type(string) == str): string = string.encode("utf-8")
    c_regex = ctypes.c_char_p(regex);
    c_string = ctypes.c_char_p(string);
    #   execute the C function
    clib.match(c_regex, c_string, ctypes.byref(start), ctypes.byref(end))
    del(c_regex, c_string, string)
    # Return the values from the C library (translating them appropriately)
    return _translate_return_values(regex, start.value, end.value)

# Get all matches for a regex.
def matcha(regex, string, **translate_kwargs):
    # Use a C utility to scal the file for all matches.
    import ctypes
    # Translate the regular expression to expected syntax.
    regex = translate_regex(regex, **translate_kwargs)
    # Call the C utillity.
    #   initialize memory storage for the start and end of a match
    n = ctypes.c_int()
    starts = ctypes.c_void_p()
    ends = ctypes.c_void_p()
    #   convert strings into character arrays
    if (type(regex) == str): regex = regex.encode("utf-8")
    if (type(string) == str): string = string.encode("utf-8")
    c_regex = ctypes.c_char_p(regex);
    c_string = ctypes.c_char_p(string);
    #   execute the C function
    clib.matcha(c_regex, c_string, ctypes.byref(n),
                ctypes.byref(starts), ctypes.byref(ends))
    # Get the number of matches.
    n = n.value
    # Return the values from the C library (translating them appropriately)
    if (n == 0): return [], []
    elif (n < 0):
        if (n == -2): raise(TypeError("`matcha` must be provided with a nonempty string."))
        elif (n == -1):
            starts = (ctypes.c_int*1).from_address(starts.value)
            ends = (ctypes.c_int*1).from_address(ends.value)
            result = _translate_return_values(regex, starts[0], ends[0])
            # ^^ this might raise an error, otherwise it's a no match
            if (result == None):    return [], []
            elif (result == (0,0)): raise(RegexError("`matcha` requires nonempty regular expression."))
            else:                   raise(NotImplementedError)
    else:
        starts = (ctypes.c_int*n).from_address(starts.value)
        ends = (ctypes.c_int*n).from_address(ends.value)
    return list(starts), list(ends)


# Given a path to a file, search for all nonoverlapping matches in the
# file and return the count of number of matches and a summary string.
def fmatcha(path, regex, ascii_ratio=0.7, **translate_kwargs):
    # Make sure the file exists.
    if (not os.path.exists(path)): return path, 0, ""
    # Use a C utility to scal the file for all matches.
    import ctypes
    # Translate the regular expression to expected syntax.
    regex = translate_regex(regex, **translate_kwargs)
    # Call the C utillity.
    #   initialize memory storage for the start and end of a match
    n = ctypes.c_int()
    starts = ctypes.c_void_p()
    ends = ctypes.c_void_p()
    lines = ctypes.c_void_p()
    min_ascii_ratio = ctypes.c_float(ascii_ratio)
    #   convert strings into character arrays
    if (type(regex) == str): regex = regex.encode("utf-8")
    if (type(path) == str): path = path.encode("utf-8")
    c_regex = ctypes.c_char_p(regex);
    c_path = ctypes.c_char_p(path);
    #   execute the C function
    clib.fmatcha(c_regex, c_path, ctypes.byref(n),
                 ctypes.byref(starts), ctypes.byref(ends),
                 ctypes.byref(lines), min_ascii_ratio)
    n = n.value
    path = str(path, 'utf-8')
    # Return the values from the C library (translating them appropriately)
    if (n == 0):
        return path, 0, f"  no matches in '{path}'"
    elif (n < 0):
        if (n == -3):
            return path, 0, f"  binary file skipped at '{path}'"
        if (n == -2):
            raise(OSError(f"Failed to open file '{path}'."))
        elif (n == -1):
            starts = (ctypes.c_int*1).from_address(starts.value)[0]
            ends = (ctypes.c_int*1).from_address(ends.value)[0]
            result = _translate_return_values(regex, starts, ends)
            # ^^ this might raise an error, otherwise it's a no match
            if (result == None): return path, 0, f"  no matches in '{path}'"
            elif (result == (0,0)):
                raise(RegexError("`fmatcha` requires nonempty regular expression."))
            else:
                raise(NotImplementedError)
    else:
        # cd ~/Git/Old/VarSys/ ; frex "poetry"
        # cd ~/Git/regex ; frex -s "he" "[.]py"

        if PRINT_FILE_SEPARATORS: summary = f"{'-'*70}\n{' '*(32-len(path)//2)}{path}"
        else: summary = ""
        starts = (ctypes.c_int*n).from_address(starts.value)
        ends = (ctypes.c_int*n).from_address(ends.value)
        lines = (ctypes.c_int*n).from_address(lines.value)
        with open(path, "rb") as f:
            for i in range(n):
                f.seek(max(0,starts[i]-HALF_MAX_PREVIEW_WIDTH), 0)
                bytes_to_read = (ends[i] - starts[i]) + 2*HALF_MAX_PREVIEW_WIDTH
                if (bytes_to_read < 1):
                    print("ERROR:")
                    print("list(starts): ",list(starts))
                    print("list(ends):   ",list(ends))
                    print("list(lines):  ",list(lines))
                    print("i: ",i)
                    print("bytes_to_read: ",bytes_to_read)
                    raise(Exception())
                match_line_string = str(f.read(bytes_to_read))[1:]
                if PRINT_FILE_SEPARATORS: summary += f"\n{lines[i]}: {match_line_string}"
                else: summary += f"\n{PRINT_PREFIX}{path}:{lines[i]:<5d} {match_line_string}"
        return path, n, summary


# Do a fast regular expression search over files that match a given
# pattern. Find all nonoverlapping matches in the files and print
# all matching patterns, their files, and their locations.
def frex(regex, *path_patterns, curdir=".", recursive=True, 
         parallel=True, chunksize=10, **translate_kwargs):
    # Get all candidate paths that *might* be searched.
    if recursive:
        candidate_paths = os.walk(curdir)
    else:
        candidate_paths = ((curdir,None,[p]) for p in os.listdir(curdir)
                           if os.path.isfile(os.path.join(curdir,p)))
    # Determine searchable paths from candidate paths.
    paths = []
    for (dirname, _, file_paths) in candidate_paths:
        for fname in file_paths:
            path = os.path.join(dirname, fname)
            for pattern in path_patterns:
                if (match(pattern, path, **translate_kwargs) is not None): break
            else: continue
            paths.append(path)
    # Perform the search over all the candidate paths (in parallel).
    if parallel:
        from functools import partial
        from multiprocessing import Pool, cpu_count
        pool = Pool(cpu_count())
        translate_kwargs["regex"] = regex
        p_fmatcha = partial(fmatcha, **translate_kwargs)
        match_iterator = pool.imap_unordered(p_fmatcha, paths, chunksize=chunksize)
    else:
        match_iterator = (fmatcha(p, regex, **translate_kwargs) for p in paths)
    # Cycle over all matches and print the summaries.
    matches = {}
    for p,n,s in match_iterator:
        matches[p] = n
        if (n > 0): print(s)
    # Clean up parallel Pool object.
    if parallel:
        pool.close()
        pool.terminate()
    # Return the dictionary containing all paths and matches.
    return matches


# Main for when this is executed as a program.
def main():
    import sys
    # Extract the "not recursive" optional flag if it exists.
    if ("-n" in sys.argv):
        recursive = False
        sys.argv.remove("-n")
    else: recursive = True
    # Extract the "case sensitive" optional flag if it exists.
    if ("-c" in sys.argv):
        case_sensitive = True
        sys.argv.remove("-c")
    else: case_sensitive = False
    # Extract the "serial" optional flag if it exists.
    if ("-s" in sys.argv):
        serial = True
        sys.argv.remove("-s")
    else: serial = False
    # Check for proper usage.
    if (len(sys.argv) < 2):
        print(f'''
ERROR: Only {len(sys.argv)} command line argument{'s' if len(sys.argv) > 1 else ''} provided.

Expected call to look like:
  python3 -m regex [-n] [-c] [-s] "<search-pattern>" ["<path-pattern-1>"] ["<path-pattern-2>"] [...]

"-n" is provided if the call to `frex` should NOT recursively
search all files in the directory tree from the current directory.
If no path patterns are given, all files are searched.
Documentation for this module follows.

"-c" is provided if the given regular expression is case sensitive,
meaning the letters should be matched with the exact same case.

"-s" is provided if the search should be run serially (not in parallel).

See `python -c "import regex; help(regex)"` for more detailed 
documentation including the regular expression language specification.
''')
        exit()
    regex = sys.argv[1]
    print("Given regex:",str([regex])[1:-1])
    print("Using regex:",str([translate_regex(regex,case_sensitive)])[1:-1])
    path_patterns = sys.argv[2:]
    # Set the default path pattern to match all paths.
    if (len(path_patterns) == 0): path_patterns = [""]
    curdir = os.path.abspath(os.path.curdir)
    # Do a fast regular expression search.
    matches = frex(regex, *path_patterns, curdir, recursive=recursive,
                   case_sensitive=case_sensitive, parallel=(not serial))
    total_matches = sum(matches.values())
    if (total_matches > 0):
        print(f"\n found {total_matches} match{'es' if total_matches > 0 else ''} across {len(matches)} files")
    else:
        print(f"\n no matches found across {len(matches)} files")


# Append the documentation from "regex.c" to the documentation for this module.
with open(os.path.join(os.path.dirname(__file__), "regex.c")) as _f:
    _source_file = _f.read()
    _start, _end = match("^/.*\n{/}", _source_file)
    __doc__ += _source_file[_start+3:_end].replace("\n//", "\n")
    del(_source_file, _start, _end)
del(_f)


# When using "from regex import *", only get these variables:
__all__ = ["RegexError", "match", "frex", "match", "matcha", "main"]


if __name__ == "__main__":
    import cProfile as profile
    profile.run("main()", sort='time') # sort by total time
    # profile.run("main()", sort='cumtime') # sort by cumulative time

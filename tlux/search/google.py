import os
from tlux.decorators import cache

# 
# You can get a search engine ID (first line) and Google Search API key from
#   https://developers.google.com/custom-search/v1/overview#prerequisites
# 
# 
import os
MAX_CACHE_FILES = 100
DIR = os.path.dirname(__file__)
GOOGLE_SEARCH_API_ENGINE_AND_KEY_LOCATION = os.path.expanduser("~/.google_api_key")
with open(GOOGLE_SEARCH_API_ENGINE_AND_KEY_LOCATION) as f:
    GOOGLE_SEARCH_ENGINE_ID, GOOGLE_SEARCH_API_KEY = f.read().strip().split("\n")


# Get the content at a URL.
@cache(max_files=MAX_CACHE_FILES, cache_dir=".cache/urls")
def get_url(url, method=None, script_path=os.path.join(DIR,"safari_get_url.scpt")):
    # Get the page content with a Safari script.
    if (method == "safari"):
        import subprocess
        content = subprocess.check_output(f"osascript '{script_path}' '{url}'", shell=True).decode().strip()
    # The default method is to just use `urllib` to read the page.
    else:
        import urllib.request
        with urllib.request.urlopen(url) as response:
            content = response.read()
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                pass
    # Return the content string.
    return content


# Function for loading an image from a URL.
@cache(max_files=MAX_CACHE_FILES, cache_dir=".cache/images")
def get_image(image_url, save_dir="/tmp/downloaded_images", method="curl", unique_name=True, load=False, default_extension=""):
    import os
    import urllib.parse
    # Make sure the save directory exists.
    os.makedirs(save_dir, exist_ok=True)
    parsed_url = urllib.parse.urlparse(image_url)
    # Generate a path for the downloaded file.
    if (unique_name):
        # Parse the URL to extract the file name
        file_name = os.path.basename(parsed_url.path)
        file_name, file_extension = os.path.splitext(file_name)
        if (file_extension == ""):
            file_extension = default_extension
        # Replace unsafe characters with underscores or another safe character
        safe_file_name = ''.join(char if (char.isalnum() or char in {"_","-"}) else '_' for char in file_name)
        # Add a number to the file name if that name is taken to make it unique.
        if (os.path.exists(os.path.join(save_dir,safe_file_name+file_extension))):
            safe_file_name = safe_file_name + "_{n}"
            n = 1
            while (os.path.exists(os.path.join(save_dir,safe_file_name.format(n=n)+file_extension))):
                n += 1
            safe_file_name = safe_file_name.format(n=n)
        # Insert the directory into the file name.
        file_path = os.path.join(save_dir, safe_file_name + file_extension)
    else:
        file_extension = os.path.splitext(os.path.basename(parsed_url.path))[1]
        file_path = os.path.join(save_dir, "image" + file_extension)
    print(f"Saving image to '{file_path}'", flush=True)
    # Download and save the image, open it, and return.
    if (method == "curl"):
        import subprocess
        command = "curl -s -o '{local_path}' '{url}'".format(
            local_path=file_path,
            url=image_url,
        )
        print("COMMAND:", command, flush=True)
        result = subprocess.check_output(command, shell=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(image_url, file_path)
    # Now ready to be returned.
    if (load):
        from PIL import Image
        result = Image.open(file_path)
    else:
        result = file_path
    # Complete.
    return result


# Given a string, search for it on Google and return the result as a dictionary.
# If an image search is desired instead of a text search, pass "images=True".
# 
@cache(max_files=MAX_CACHE_FILES, cache_dir=".cache/search_results")
def search(query, images=False, n=10, start=1, **params):
    num_results = n  # Rename for local readability.
    assert (num_results <= 100), f"At most 100 results can be returned by a search with the Google API."
    # Libraries needed to execute the search.
    import urllib.request
    import urllib.parse
    import json
    # Define the parameters
    params = {
        # Refernece example:
        #   https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?apix=true&apix_params=%7B%22c2coff%22%3A%221%22%2C%22cx%22%3A%228591e881e3108483e%22%2C%22hl%22%3A%22en%22%2C%22num%22%3A5%2C%22q%22%3A%22how%20many%20apples%20are%20in%20the%20world%3F%22%7D
        #   https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?apix_params=%7B%22c2coff%22%3A%221%22%2C%22cx%22%3A%228591e881e3108483e%22%2C%22hl%22%3A%22en%22%2C%22num%22%3A3%2C%22q%22%3A%22how%20many%20apples%20are%20in%20the%20world%3F%22%7D
        # 
        "q": query, # The search query
        "start": str(start), # The starting index for the resturned items.
        "num": str(min(10,num_results)), # Number of results to return (up to 10, use "start=11" to get the next 10 up to 100).
        "hl": "en", # English results.
        "c2coff": "1", # Disables simplified traditional chinese search
        "lr": "lang_en", # Restricts results to documents written in a particular language
        # 
        # "exactTerms": "", # Terms that *must* be included
        # "excludeTerms": "", # Word or phrase that should *not* appear
        # "orTerms": "", # Words that at least *one* of these must appear in the document
        # 
        "cx": GOOGLE_SEARCH_ENGINE_ID, # Google custom search engine identifier
        "key": GOOGLE_SEARCH_API_KEY, # Google Search API key
    }
    # Special keyword insertion for image searches to make it mores usable.
    if (images):
        params["searchType"] = "image" # Use this to get image results.
    # Add in any custom parameters.
    params.update(params)
    # Encode the parameters
    encoded_params = urllib.parse.urlencode(params)
    # Construct the URL
    url = f"https://customsearch.googleapis.com/customsearch/v1?{encoded_params}"
    # Make the request
    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    # Handle the response.
    with urllib.request.urlopen(request) as response:
        # When the response is successful, parse the JSON and return it.
        if response.status == 200:
            response_body = response.read()
            parsed_response = json.loads(response_body.decode('utf-8'))
            search_results = parsed_response["items"]
            # If more results are needed, recursively call this function.
            if (num_results > 10):
                search_results += search(
                    query,
                    images=images,
                    n=num_results-10,
                    start=start+10,
                    **params
                )
            # Return the sum of all items.
            return search_results
        # When the response is unsuccessful, provide the status as a result.
        else:
            return [dict(error=response.status)]


# Function to turn an nested object (dict, list) into a nicely formatted string.
def to_str(o, indent=2, gap=0):
    if (type(o) is dict):
        string = " "*gap + "{"
        for k,v in o.items():
            string += "\n" + " "*(gap+indent) + f"'{k}': " + to_str(v, indent=indent, gap=gap+indent).lstrip(" ")
        string += "\n" + " "*gap + "}"
    elif (type(o) is list):
        string = " "*gap + "["
        for v in o:
            string += "\n" + to_str(v, indent=indent, gap=gap+indent)
        string += "\n" + " "*gap + "]"
    else:
        string = " "*gap + repr(o)
    return string


# A simple MCP server based command line interface.
def mcp():
    import sys
    import json
    # Single tool definition reused for initialize/tools.list
    search_tool = {
        "name": "search",
        "title": "Search Internet",
        "description": "Use Google to search the internet only when asked. Good for real time information, citations, or facts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string."
                },
                # "images": {
                #     "type": "boolean",
                #     "description": "If true, perform image search.",
                #     "default": False
                # },
                # "n": {
                #     "type": "integer",
                #     "description": "Number of results requested (max 10).",
                #     "default": 10
                # }
            },
            "required": ["query"]
        },
    }

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            # Ignore garbage input
            continue
        method = msg.get("method")
        req_id = msg.get("id")
        # Notifications (no id) can be safely ignored
        if method is None:
            continue
        # --- initialize ------------------------------------------------------
        if method == "initialize":
            # LM Studio currently uses protocolVersion "2024-11-05" 
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        # We support tools, no extra options needed
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "tlux-search",
                        "version": "0.1.0",
                    },
                },
            })
            continue
        # --- tools/list ------------------------------------------------------
        if method == "tools/list":
            # LM Studio uses this to discover available tools 
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [search_tool],
                },
            })
            continue
        # --- tools/call ------------------------------------------------------
        if method == "tools/call":
            params = msg.get("params", None) or {}
            tool_name = params.get("name")
            args = params.get("arguments") or {}
            if tool_name != "search":
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,  # Method not found
                        "message": f"Unknown tool: {tool_name!r}",
                    },
                })
                continue
            try:
                query = args["query"]
            except KeyError:
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,  # Invalid params
                        "message": "Missing required parameter 'query'.",
                    },
                })
                continue
            images = bool(args.get("images", False))
            n = int(args.get("n", 5))
            start = int(args.get("start", 1))
            try:
                # Run your existing search()
                raw = search(query, images=images, n=n, start=start)
                # Strip to the fields you care about
                to_keep = {"title", "link", "displayLink", "snippet"}
                simplified = []
                for r in raw:
                    if not isinstance(r, dict):
                        continue
                    s = {k: v for k, v in r.items() if k in to_keep}
                    simplified.append(s)
                # MCP tool result: unstructured text, plus the JSON as text
                # (LM Studio will feed this back to the model)
                text_payload = to_str(simplified)
                text_payload = ""
                for i, r in enumerate(simplified, start=1):
                    text_payload += f" <search-result-{i}> \n\n"
                    title = r.get("title", "").replace("\n", " ")
                    text_payload += f"Title: {title} \n"
                    link = r.get("displayLink", "")
                    text_payload += f"Link: {link} \n"
                    snippet = r.get("snippet", "")
                    text_payload += f"Snippet: {snippet} \n"
                    text_payload += f"\n </search-result-{i}> \n\n"
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": text_payload.strip(),
                            }
                        ]
                    },
                })
            except Exception as e:
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32000,  # Server error
                        "message": f"search failed: {e}",
                    },
                })
            continue
        # Ignore other methods (e.g. notifications/initialized, roots/*, etc.)
        # No response is required for unknown notifications; for unknown
        # requests with an id, you *could* send an error, but LM Studio
        # will mostly send the three handled above.


# Enable both: CLI for manual use, MCP for LM Studio (no args).
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Old CLI behavior
        import argparse
        parser = argparse.ArgumentParser(description="Search the Internet with Google")
        parser.add_argument("--query", "-q", type=str, help="The query to search.")
        parser.add_argument("--num_results", "-n", type=int, default=10,
                            help="The number of results to return.")
        parser.add_argument("--images", "-i", action="store_true",
                            help="Return image search results.")
        args = parser.parse_args()

        if args.query:
            results = search(args.query, images=args.images, n=args.num_results)
            to_keep = {"title", "link", "displayLink", "snippet"}
            for r in results:
                for k in list(r):
                    if k not in to_keep:
                        r.pop(k, None)
            print(to_str(results))
    else:
        # No args: act as an MCP stdio server
        mcp()

#!/opt/homebrew/bin/python3

# The following is a dependency for loading the model files and generating output.
# 
#   python3 -m pip install --user llama-cpp-python
# 

# Import the library for loading ".gguf" model files and specifying grammars (constrained outputs).
import argparse
import logging
import json
import os
import requests
import sys
import time


# Disable logs for diskcache.
logging.getLogger("diskcache").setLevel(logging.WARNING)

# Models downloaded with LMStudio will be in a path such as:
#    ~/.cache/lm-studio/models/...

# Set the path to the model.
MODEL_PATH = os.path.expanduser('~/.slm.gguf')
CACHE_PATH = os.path.expanduser('~/.slm.cache')

# CHAT_FORMAT = "llama-3"  # Meta models
# CHAT_FORMAT = "gemma"  # Google models
CHAT_FORMAT = "chatml"  # Devstral and Qwen models

DEFAULT_CTX = 8192
DEFAULT_TEMP = 0.4
DEFAULT_N_THREADS = 8
LM = None
COMPLETION = None

# Log directory for the inspectable chat logs.
LOG_DIR = os.path.expanduser("~/.cache/lm-studio/conversations/Logs/")


# ------------------------------------------------------------------------------------
# Define useful grammars to constrain the output.

try:
    from llama_cpp.llama import Llama, LlamaGrammar
    from llama_cpp.llama_cache import LlamaDiskCache
    # from llama_cpp.llama_cache import LlamaRAMCache

    # Grammar for stricly "yes" / "no" outputs.
    YES_NO = LlamaGrammar.from_string(r'''
    root ::= (([nN] "o") | ([yY] "es"))
    ''', verbose=False)

    # Grammar for a single sentence.
    ONE_SENTENCE_GRAMMAR = LlamaGrammar.from_string(r'''
    root ::= " "? word (" " word)* "."
    word ::= [0-9A-Za-z',()-]+
    ''', verbose=False)

    # Grammar for valid JSON-parseable output (doesn't handle premature stops).
    JSON_ARRAY_GRAMMAR = LlamaGrammar.from_string(r'''# For generating JSON arrays
    root ::= "[" ws ( value ("," ws value)* )? "]"
    ws ::= ([ \t\n] ws)?
    number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
    string ::= "\"" ( [^"\\]  # anything that's not a quote or backslash, OR ...
       | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
      )* "\"" ws
    array  ::= "[" ws (value ("," ws value)* )? "]" ws
    object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)* )? "}" ws
    value ::= object | array | string | number | ("true" | "false" | "null") ws
    ''', verbose=False)
except ModuleNotFoundError:
    pass

# ------------------------------------------------------------------------------------



class suppress_stderr:
    def __enter__(self):
        self.stderr_fileno = sys.stderr.fileno()
        self.saved_stderr = os.dup(self.stderr_fileno)
        self.null_stderr = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null_stderr, self.stderr_fileno)
    def __exit__(self, *args):
        os.dup2(self.saved_stderr, self.stderr_fileno)
        os.close(self.null_stderr)
        os.close(self.saved_stderr)


# Load the model.
def load_lm(model_path=MODEL_PATH, n_ctx=DEFAULT_CTX, embedding=False, verbose=False, n_gpu_layers=-1, chat_format=CHAT_FORMAT, n_threads=DEFAULT_N_THREADS, **llama_kwargs):
    model_path = os.path.expanduser(model_path)
    cache = LlamaDiskCache(CACHE_PATH)
    _load_model = lambda: Llama(
        model_path,
        n_ctx=n_ctx,
        embedding=embedding,
        verbose=verbose,
        n_gpu_layers=n_gpu_layers,
        chat_format=chat_format,
        n_threads=n_threads,
        cache=cache,
        **llama_kwargs
    )
    if verbose:
        model = _load_model()
    else:
        with suppress_stderr():
            model = _load_model()
    # Cache the loaded model globally.
    global LM
    LM = model
    # Return the loaded model.
    return model


# Truncate some text to the specified token length.
def truncate(lm, text, n_ctx=DEFAULT_CTX):
    return lm.detokenize(lm.tokenize(text.encode())[-n_ctx+1:]).decode()


# Get the completion for a prompt. Return it and the stop reason.
#   "min_tokens" is the minimum number of tokens that there will be
#   room for in the response before "n_ctx" is exhausted.
def complete(*, lm=None, prompt="", max_tokens=-1, min_tokens=64, n_ctx=DEFAULT_CTX, temperature=DEFAULT_TEMP, grammar=None, stream=False, **kwargs):
    if (lm is None):
        if (LM is None):
            load_lm()
        lm = LM
    prompt = truncate(lm, prompt, n_ctx=n_ctx-min_tokens)
    # Ensure no "messages" are included.
    kwargs.pop("messages", None)
    # Generate a response.
    response = lm(
        prompt,
        grammar=grammar,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        **kwargs
    )
    # Return a generator of [(text, reason) ...] if streaming is wanted.
    if (stream):
        generator = ((token['choices'][0]['text'], token['choices'][0]['finish_reason']) for token in response)
        return generator
    # Otherwise just return the response.
    else:
        return response['choices'][0]['text'], response['choices'][0]['finish_reason']


# Get the completion for a prompt using chat formatting. Return it and the stop reason.
def chat_complete(*, lm=None, prompt=None, max_tokens=-1, min_tokens=64, n_ctx=DEFAULT_CTX, temperature=DEFAULT_TEMP, grammar=None, stream=False, messages=(), system="", **kwargs):
    if (lm is None):
        if (LM is None):
            load_lm()
        lm = LM
    # If only messages were provided, extract out the prompt as the last one.
    if (prompt is None) and (len(messages) > 0):
        messages, prompt = messages[:-1], messages[-1]
        if type(prompt) is dict:
            assert (prompt["role"] == "user"), "Expected the last message to be from the user."
            prompt = prompt["content"]
    # If a string was given as a grammar, convert it into a llama grammar.
    if (type(grammar) is str):
        if ("root ::=" not in grammar):
            grammar = "root ::= " + grammar
        grammar = LlamaGrammar.from_string(grammar)
    # Ensure that the prompt is sufficiently short.
    prompt = truncate(lm, prompt, n_ctx=n_ctx-min_tokens)
    if len(messages) > 0:
        messages = [
            m if type(m) is dict else dict(role=('user' if (i%2==0) else 'assistant'), content=m)
            for (i, m) in enumerate(messages)
        ]
    # Get the system prompt if provided.
    system = [dict(role='system', content=system)] if (system is not None and (len(system) > 0)) else []
    response = lm.create_chat_completion(
        messages=system + list(messages) + [dict(role='user', content=prompt)],
        grammar=grammar,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        **kwargs
    )
    # Return a generator of [(text, reason) ...] if streaming is wanted.
    if (stream):
        generator = ((token.get('choices',[{}])[0].get('delta',{}).get('content',''), token.get('choices',[{}])[0].get('finish_reason',None)) for token in response)
        return generator
    # Otherwise just return the response.
    else:
        response = response.get('choices', [{}])[0]
        return response.get('message',{}).get('content',''), response.get('finish_reason', None)


# A minimal wrapper to interact with a local LM Studio server via direct HTTP requests.
# 
# Args:
#     prompt (str): The user prompt to send.
#     max_tokens (int): Maximum tokens to generate (-1 for no limit).
#     temperature (float): Sampling temperature (default: 0.1).
#     stream (bool): Whether to stream the response (default: False).
#     messages (list): Previous conversation messages (strings or dicts).
#     system (str): System prompt (default: "").
#     base_url (str): Server base URL (default: "http://127.0.0.1:3544/v1").
#     **kwargs: Additional parameters to pass to the API.
# 
# Returns:
#     tuple: (text, finish_reason) for non-streaming, or generator yielding (token, finish_reason) for streaming.
# 
def server_chat_complete(prompt=None, max_tokens=-1, temperature=0.1, stream=False, messages=(), system="", base_url="http://127.0.0.1:3544/v1", **kwargs):
    # Convert messages to proper format
    if len(messages) > 0:
        messages = [
            m if isinstance(m, dict) else {'role': 'user' if i % 2 == 0 else 'assistant', 'content': m}
            for i, m in enumerate(messages)
        ]
    else:
        messages = []
    # Push the prompt to the back of the messages.
    if prompt is not None:
        messages += [{"role": "user", "content": prompt}]
    # Construct the full conversation.
    full_messages = ([{"role": "system", "content": system}] if system else []) + messages
    
    # Build request body
    data = {
        "model": "default",  # Adjust this based on your LM Studio model
        "messages": full_messages,
        "temperature": temperature,
        "stream": stream,
        **kwargs
    }
    if max_tokens != -1:
        data["max_tokens"] = max_tokens
    
    # Set headers (no Authorization)
    headers = {"Content-Type": "application/json"}
    url = f"{base_url}/chat/completions"
    
    if stream:
        # Streaming request
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        def stream_generator():
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        event_data = line[6:].decode("utf-8")
                        if event_data == "[DONE]":
                            yield "", "stop"
                        else:
                            try:
                                chunk = json.loads(event_data)
                                choice = chunk["choices"][0]
                                token = choice.get("delta", {}).get("content", "")
                                finish_reason = choice.get("finish_reason")
                                yield token, finish_reason
                            except json.JSONDecodeError:
                                continue
        
        return stream_generator()
    else:
        # Non-streaming request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "stop")
        return text, finish_reason


# Chat completion with logging to LM Studio conversations folder (so that it is easy to inspect).
def logged_server_chat_complete(log_dir=LOG_DIR, **kwargs):
    # Ensure logs directory exists
    now = time.strftime("%Y-%m-%d_%H.%M.%S")
    timestamp = len(os.listdir(log_dir))
    # Count all messages.
    message_count = 0
    if "system" in kwargs:
        message_count += 1
    if "messages" in kwargs:
        message_count += len(list(kwargs["messages"]))
    if "prompt" in kwargs:
        message_count += 1
    # Create a chat name.
    chat_log_name = f"{message_count:02d}-messages ({now})"
    # Generate the response
    response, stop_reason = server_chat_complete(**kwargs)
    # Initialize JSON structure
    json_data = {
        "name": chat_log_name,
        "pinned": False,
        "createdAt": timestamp,
        "tokenCount": 0,
        "messages": [],
        "usePerChatPredictionConfig": True,
        "perChatPredictionConfig": {
            "fields": []
        }
    }
    token_count = 0
    # Handle system prompt
    system_prompt = kwargs.get("system", "")
    if system_prompt:
        json_data["perChatPredictionConfig"]["fields"].append({
            "key": "llm.prediction.systemPrompt",
            "value": system_prompt
        })
        token_count += len(system_prompt)    
    # Construct all of the expected message objects.
    messages = []
    if "messages" in kwargs:
        for i, content in enumerate(kwargs["messages"]):
            token_count += len(content)
            # Even numbered (starting at 0) are user messages.
            if (i % 2 == 0):
                message_obj = {
                    "versions": [{
                        "type": "singleStep",
                        "role": "user",
                        "content": [{"type": "text", "text": content}]
                    }],
                    "currentlySelected": 0
                }
            # Odd numbered (starting at 1) are bot messages.
            else:
                # Thinking blocks need to be extracted and look like this.
                if ("<think>" in content) and ("</think>" in content):
                    thoughts = content[
                        content.index("<think>") + len("<think>") :
                        content.index("</think>")
                    ]
                    content = content[content.index("</think>") + len("</think>"):]
                    steps = [{
                        "type": "contentBlock",
                        "stepIdentifier": "",
                        "content": [{"type": "text", "text": thoughts, "tokensCount": len(thoughts)}],
                        "defaultShouldIncludeInContext": True,
                        "shouldIncludeInContext": True,
                        "style": {"type": "thinking", "ended": True, "title": ""},
                        "prefix": "<think>",
                        "suffix": "</think>"
                    }]
                else:
                    steps = []
                # Construct the message object for this.
                message_obj = {
                    "versions": [{
                        "type": "multiStep",
                        "role": "assistant",
                        "steps": steps + [{
                            "type": "contentBlock",
                            "stepIdentifier": "",
                            "content": [{"type": "text", "text": content}],
                            "defaultShouldIncludeInContext": True,
                            "shouldIncludeInContext": True
                        }]
                    }],
                    "currentlySelected": 0
                }
            # Append the message.
            messages.append(message_obj)
    # Add the "prompt" if it was added too.
    if "prompt" in kwargs:
        token_count += len(kwargs["prompt"])
        messages.append({
            "versions": [{
                "type": "singleStep",
                "role": "user",
                "content": [{"type": "text", "text": kwargs["prompt"]}]
            }],
            "currentlySelected": 0
        })

    # Thinking blocks need to be extracted and look like this.
    steps = []
    if ("<think>" in response) and ("</think>" in response):
        thoughts = response[
            response.index("<think>") + len("<think>") :
            response.index("</think>")
        ]
        response = response[response.index("</think>") + len("</think>"):]
        steps = [{
            "type": "contentBlock",
            "stepIdentifier": "",
            "content": [{"type": "text", "text": thoughts, "tokensCount": len(thoughts)}],
            "defaultShouldIncludeInContext": True,
            "shouldIncludeInContext": True,
            "style": {"type": "thinking", "ended": True, "title": ""},
            "prefix": "<think>",
            "suffix": "</think>"
        }]
    # Add the response from the bot as the final message.
    token_count += len(response)
    messages.append({
        "versions": [{
            "type": "multiStep",
            "role": "assistant",
            "steps": steps + [{
                "type": "contentBlock",
                "stepIdentifier": "",
                "content": [{"type": "text", "text": response}],
                "defaultShouldIncludeInContext": True,
                "shouldIncludeInContext": True
            }]
        }],
        "currentlySelected": 0
    })
    # Update the token count.
    json_data["tokenCount"] = token_count
    # Update the messages.
    json_data["messages"] = messages
    # Write JSON log
    chat_log_name = os.path.join(log_dir, f"{timestamp}.conversation.json")
    with open(chat_log_name, "w") as f:
        json.dump(json_data, f, indent=2)
    # Return original results
    return response, stop_reason



if __name__ == '__main__':
    # Add command line arguments.
    parser = argparse.ArgumentParser(description='Complete a prompt with a local language model.')
    parser.add_argument('-e', '--embed', action='store_true', help='Produce an average token (at final state) embedding for the prompt.')
    parser.add_argument('-j', '--json', action='store_true', help='Output in JSON format')
    parser.add_argument('-v', '--chat', action='store_true', help='True if a turn-based conversation should be initiated.')
    parser.add_argument('-q', '--no-echo', action='store_true', help='True if the echo of the provided prompt should be skipped.')
    parser.add_argument('-m', '--min_tokens', type=int, default=1, help='The minimum number of tokens to produce.')
    parser.add_argument('-n', '--max_tokens', type=int, default=-1, help='The maximum number of tokens to produce.')
    parser.add_argument('-c', '--context_size', type=int, default=DEFAULT_CTX, help='The upper limit in prompt size before the head is truncated.')
    parser.add_argument('-t', '--temperature', type=float, default=DEFAULT_TEMP, help='The temperature used in response generation.')
    parser.add_argument('-p', '--prompt', type=str, default="", help='The prompt to pass to the model.')

    # Parse the command line arguments.
    args = parser.parse_args()

    # Extract these parameters from the command line arguments.
    embed: bool = args.embed
    json: bool = args.json
    min_tokens: int = args.min_tokens
    max_tokens: int = args.max_tokens
    context_size: int = args.context_size
    prompt: str = args.prompt
    temperature: float = args.temperature
    echo_prompt: bool = not args.no_echo
    chat: bool = args.chat

    # Local variables for tracking execution (in chat mode).
    response: str = ""
    kwargs = dict(messages=[])

    # Ensure viable context size is set, re-initialize model if larger context is needed.
    context_size = max(context_size, min_tokens)
    
    # Produce only the embedding.
    if embed:
        import numpy as np
        lm = load_lm(n_ctx=context_size, embedding=True)
        print(','.join(map(str, np.asarray(lm.embed(prompt)).mean(axis=0))))
    else:
        lm = load_lm(n_ctx=context_size)

        # First output the prompt itself so the response is in context.
        if chat:
            generate_response = chat_complete
            print(flush=True)
            if (prompt.strip() == ""):
                prompt = "When you are ready, please say 'Hello.'"
            # kwargs["messages"].append(dict(role="system", content="An expert polymath with lots of scientific and technical knowledge and strong communication skills is helping a person solve problems. The expert replies in markdown format. Following are questions and requests written by the person and high quality responses from the expert."))
        else:
            generate_response = complete
            if echo_prompt:
                print(prompt, end='', flush=True)


        # Store messages and prompts in case this is a chat.
        while True:
            # Then generate the completion.
            for (token, stop_reason) in generate_response(
                lm=lm,
                prompt=prompt,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                n_ctx=context_size,
                temperature=temperature,
                stream=True,
                **kwargs,
                # Add the grammar constraint
                **({} if (not json) else dict(grammar=JSON_ARRAY_GRAMMAR)),
            ):
                response += token
                print(token, end='', flush=True)

            # If this a chat, then listen and loop.
            if chat:
                # Get a new message.
                kwargs['messages'].extend([
                    dict(role="user", content=prompt),
                    dict(role="assistant", content=response)
                ])
                print("\n", flush=True)
                try: prompt = input()
                except (KeyboardInterrupt, EOFError): break
                print(flush=True)
                response = ""
            else:
                break

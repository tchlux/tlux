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

from typing import Any, Iterable, Iterator, Optional, Tuple

# Attempt to load a key for calling a remote completions endpoint.
try:
    from local_auth import COMPLETIONS_KEY, COMPLETIONS_URL
except:
    COMPLETIONS_KEY = ""

# Disable logs for diskcache.
logging.getLogger("diskcache").setLevel(logging.WARNING)

# Models downloaded with LMStudio will be in a path such as:
#    ~/.cache/lm-studio/models/...

# Set the path to the model.
MODEL_PATH = os.path.expanduser('~/.slm.gguf')
CACHE_PATH = os.path.expanduser('~/.slm.cache')

# CHAT_FORMAT = "llama-3"  # Meta models
# CHAT_FORMAT = "gemma"  # Google models
# CHAT_FORMAT = "chatml"  # Devstral and Qwen models
CHAT_FORMAT = None

DEFAULT_CTX = 8192
DEFAULT_N_THREADS = 8
LM = None
COMPLETION = None
LOCAL_COMPLETIONS_URL = "http://127.0.0.1:3544/v1/chat/completions"
REMOTE_COMPLETIONS_URL = COMPLETIONS_URL

# Log directory for the inspectable chat logs.
LOG_DIR = os.path.expanduser("~/.cache/lm-studio/conversations/Logs/")

# llama-cpp-python and gpt-oss-20b configuration presets for macOS (M-series, Metal)
# All values are ASCII-safe and follow community-verified + author-endorsed defaults.
BASE_CONFIG = {
    "model_path": "gpt-oss-20b.Q5_K_M.gguf",
    "n_ctx": 0,
    "n_gpu_layers": -1,
    "n_threads": -1,
    "flash_attn": True,
    "seed": 0,
    "max_tokens": -1,
}

PRESETS = {
    "openai": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        # "repeat_last_n": 64,
    },
    "reliable": {
        "temperature": 0.65,
        "top_p": 1.0,
        "top_k": 40,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.05,
        # "repeat_last_n": 128,
    },
    "creative": {
        "temperature": 1.1,
        "top_p": 0.9,
        "top_k": 60,
        "min_p": 0.05,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        # "repeat_last_n": 64,
    },
    "deterministic": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        # "repeat_last_n": 64,
    },
}

DEFAULT_PRESET: Optional[str] = "reliable"


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
def complete(
    *,
    lm: Any = None,
    prompt: str = "",
    max_tokens: int = -1,
    min_tokens: int = 64,
    n_ctx: int = DEFAULT_CTX,
    grammar: Optional[Any] = None,
    stream: bool = True,
    preset: Optional[str] = DEFAULT_PRESET,
    **kwargs: Any,
) -> Iterator[Tuple[str, Optional[str]]]:
    if not stream:
        raise ValueError("Streaming is required.")
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        kwargs = {**PRESETS[preset], **kwargs}
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
        stream=True,
        **kwargs
    )
    return (
        (token["choices"][0].get("text", ""), token["choices"][0].get("finish_reason"))
        for token in response
    )


# Get the completion for a prompt using chat formatting. Return it and the stop reason.
def chat_complete(
    *,
    lm: Any = None,
    prompt: Optional[str] = None,
    max_tokens: int = -1,
    min_tokens: int = 64,
    n_ctx: int = DEFAULT_CTX,
    grammar: Optional[Any] = None,
    stream: bool = True,
    preset: Optional[str] = DEFAULT_PRESET,
    messages: Iterable[Any] = (),
    system: str = "",
    reasoning_stop: str = "<|start|>assistant<|channel|>final<|message|>",
    **kwargs: Any,
) -> Iterator[Tuple[str, Optional[str]]]:
    if not stream:
        raise ValueError("Streaming is required.")
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        kwargs = {**PRESETS[preset], **kwargs}
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
        stream=True,
        **kwargs
    )
    def generator() -> Iterator[Tuple[str, Optional[str]]]:
        buffer = ""
        streaming = False
        for token in response:
            choice = token.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            text = delta.get("content", "")
            stop_reason = choice.get("finish_reason")
            if text:
                buffer += text
                if not streaming:
                    marker_index = buffer.find(reasoning_stop)
                    if marker_index != -1:
                        streaming = True
                        remainder = buffer[marker_index + len(reasoning_stop):]
                        if remainder:
                            yield remainder, None
                        buffer = ""
                else:
                    yield text, None
            if stop_reason:
                if (not streaming) and buffer:
                    streaming = True
                    yield buffer, None
                yield "", stop_reason
                break
    return generator()


# A minimal wrapper to interact with a local LM Studio server via direct HTTP requests.
# 
# Args:
#     prompt (str): The user prompt to send.
#     max_tokens (int): Maximum tokens to generate (-1 for no limit).
#     stream (bool): Must remain True; streaming is always enabled.
#     messages (list): Previous conversation messages (strings or dicts).
#     system (str): System prompt (default: "").
#     url (str): Server completions URL (default: "http://127.0.0.1:1234/v1/chat/completions").
#     **kwargs: Additional parameters to pass to the API.
# 
# Returns:
#     generator yielding (token, finish_reason) tuples.
# 
def server_chat_complete(
    prompt: Optional[str] = None,
    max_tokens: int = -1,
    stream: bool = True,
    messages: Iterable[Any] = (),
    system: str = "",
    model: str = "default",
    url: str = LOCAL_COMPLETIONS_URL,
    **kwargs: Any,
) -> Iterator[Tuple[str, Optional[str]]]:
    if not stream:
        raise ValueError("Streaming is required.")
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
        "model": model,
        "messages": full_messages,
        "stream": stream,
        **kwargs
    }
    if max_tokens != -1:
        data["max_tokens"] = max_tokens
    
    # Set headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {COMPLETIONS_KEY}",
    }
    response = requests.post(url, headers=headers, json=data, stream=True)
    response.raise_for_status()
    def stream_generator() -> Iterator[Tuple[str, Optional[str]]]:
        finish_reason: Optional[str] = None
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                yield "", finish_reason or "stop"
                return
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            token = choice.get("delta", {}).get("content", "")
            if token:
                yield token, None
            finish = choice.get("finish_reason")
            if finish:
                finish_reason = finish
        yield "", finish_reason or "stop"
    return stream_generator()


# Chat completion with logging to LM Studio conversations folder (so that it is easy to inspect).
def logged_server_chat_complete(log_dir: str = LOG_DIR, **kwargs: Any) -> Iterator[Tuple[str, Optional[str]]]:
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
    chat_log_name = f"{message_count:02d}-messages ({now})"
    stream = server_chat_complete(stream=True, **kwargs)
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

    def finalize_log(response_text: str) -> None:
        steps = []
        if ("<think>" in response_text) and ("</think>" in response_text):
            thoughts = response_text[
                response_text.index("<think>") + len("<think>") :
                response_text.index("</think>")
            ]
            response_text = response_text[response_text.index("</think>") + len("</think>"):]
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
        token_count_local = token_count + len(response_text)
        messages.append({
            "versions": [{
                "type": "multiStep",
                "role": "assistant",
                "steps": steps + [{
                    "type": "contentBlock",
                    "stepIdentifier": "",
                    "content": [{"type": "text", "text": response_text}],
                    "defaultShouldIncludeInContext": True,
                    "shouldIncludeInContext": True
                }]
            }],
            "currentlySelected": 0
        })
        json_data["tokenCount"] = token_count_local
        json_data["messages"] = messages
        chat_log_path = os.path.join(log_dir, f"{timestamp}.conversation.json")
        with open(chat_log_path, "w") as f:
            json.dump(json_data, f, indent=2)
    def generator() -> Iterator[Tuple[str, Optional[str]]]:
        collected: list[str] = []
        try:
            for token, reason in stream:
                if token:
                    collected.append(token)
                yield token, reason
        finally:
            finalize_log("".join(collected))
    return generator()



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
    parser.add_argument('-p', '--prompt', type=str, default="", help='The prompt to pass to the model.')
    parser.add_argument('--preset', choices=tuple(PRESETS.keys()), help='Sampling preset to apply.', default=DEFAULT_PRESET)

    # Parse the command line arguments.
    args = parser.parse_args()

    # Extract these parameters from the command line arguments.
    embed: bool = args.embed
    json: bool = args.json
    min_tokens: int = args.min_tokens
    max_tokens: int = args.max_tokens
    context_size: int = args.context_size
    prompt: str = args.prompt
    echo_prompt: bool = not args.no_echo
    chat: bool = args.chat
    preset: Optional[str] = args.preset

    # Local variables for tracking execution (in chat mode).
    response: str = ""
    kwargs = dict(messages=[])
    if preset:
        kwargs["preset"] = preset

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
                prompt = "This is a plain text conversation in a terminal. There is no markdown rendering, so use easy to read and concise repsonses. When you are ready, please say 'Hello.'"
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

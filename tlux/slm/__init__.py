#!/opt/homebrew/bin/python3

# The following is a dependency for loading the model files and generating output.
# 
#   python3 -m pip install --user llama-cpp-python
# 

# Import the library for loading ".gguf" model files and specifying grammars (constrained outputs).
import argparse
import os
from llama_cpp.llama import Llama, LlamaGrammar
from llama_cpp.llama_cache import LlamaDiskCache

# Models downloaded with LMStudio will be in a path such as:
#    ~/.cache/lm-studio/models/...

# Set the path to the model.
MODEL_PATH = os.path.expanduser('~/.slm.gguf')
CACHE_PATH = os.path.expanduser('~/.slm.cache')
CHAT_FORMAT = "llama-3"
DEFAULT_CTX = 1024
DEFAULT_TEMP = 0.1

# ------------------------------------------------------------------------------------
# Define useful grammars to constrain the output.

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

# ------------------------------------------------------------------------------------


# Load the model.
def load_lm(model_path=MODEL_PATH, n_ctx=DEFAULT_CTX, embedding=False, verbose=False, num_gpu_layers=-1, chat_format=CHAT_FORMAT):
    model = Llama(model_path, n_ctx=n_ctx, embedding=embedding, verbose=verbose, num_gpu_layers=num_gpu_layers, chat_format=chat_format)
    # model.set_cache(LlamaDiskCache(CACHE_PATH))  # This appears to have fixed debug print statments, not ready yet.
    return model


# Truncate some text to the specified token length.
def truncate(lm, text, n_ctx=DEFAULT_CTX):
    return lm.detokenize(lm.tokenize(text.encode())[-n_ctx+1:]).decode()


# Get the completion for a prompt. Return it and the stop reason.
#   "min_tokens" is the minimum number of tokens that there will be
#   room for in the response before "n_ctx" is exhausted.
def complete(lm, prompt, max_tokens=-1, min_tokens=64, n_ctx=DEFAULT_CTX, temperature=DEFAULT_TEMP, grammar=None, stream=False, **kwargs):
    prompt = truncate(lm, prompt, n_ctx=n_ctx-min_tokens)
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
def chat_complete(lm, prompt, max_tokens=-1, min_tokens=64, n_ctx=DEFAULT_CTX, temperature=DEFAULT_TEMP, grammar=None, stream=False, messages=(), **kwargs):
    prompt = truncate(lm, prompt, n_ctx=n_ctx-min_tokens)
    response = lm.create_chat_completion(
        messages=list(messages) + [dict(role='user', content=prompt)],
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
    parser.add_argument('prompt', type=str, help='The prompt to pass to the model.')

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
                lm,
                prompt,
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
            else:
                break

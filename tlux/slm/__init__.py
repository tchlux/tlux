#!/opt/homebrew/bin/python3

# The following is a dependency for loading the model files and generating output.
# 
#   python3 -m pip install --user llama-cpp-python
# 

# Import the library for loading ".gguf" model files and specifying grammars (constrained outputs).
import argparse
import os
from llama_cpp.llama import Llama, LlamaGrammar

# Models downloaded with LMStudio will be in a path such as:
#    ~/.cache/lm-studio/models/...

# Set the path to the model.
MODEL_PATH = os.path.expanduser("~/.slm.gguf")

# Load the model.
LLM = Llama(MODEL_PATH, embedding=True, verbose=False, num_gpu_layers=-1)

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


# Truncate some text to the specified token length.
def truncate(text, llm=LLM, n_ctx=512):
    return llm.detokenize(llm.tokenize(text.encode())[-n_ctx+1:]).decode()


# Get the completion for a prompt. Return it and the stop reason.
#   "min_tokens" is the minimum number of tokens that there will be
#   room for in the response before "n_ctx" is exhausted.
def complete(prompt, max_tokens=-1, min_tokens=64, n_ctx=2**12, temperature=0.1, llm=LLM, grammar=None, stream=False, **kwargs):
    prompt = truncate(prompt, n_ctx=n_ctx-min_tokens, llm=llm)
    response = llm(
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


if __name__ == "__main__":
    # Add command line arguments.
    parser = argparse.ArgumentParser(description='Complete a prompt with a local language model.')
    parser.add_argument('-e', '--embed', action='store_true', help='Produce an average token (at final state) embedding for the prompt.')
    parser.add_argument('-j', '--json', action='store_true', help='Output in JSON format')
    parser.add_argument('-m', '--min_tokens', type=int, default=1, help='The minimum number of tokens to produce.')
    parser.add_argument('-n', '--max_tokens', type=int, default=256, help='The maximum number of tokens to produce.')
    parser.add_argument('-c', '--context_size', type=int, default=512, help='The upper limit in prompt size before the head is truncated.')
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

    if embed:
        # Produce only the embedding.
        import numpy as np
        print(",".join(map(str, np.asarray(LLM.embed(prompt)).mean(axis=0))))
    else:
        # First output the prompt itself so the response is in context.
        print(prompt, end="", flush=True)

        # Then generate the completion.
        for (token, stop_reason) in complete(
            prompt,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            n_ctx=context_size,
            stream=True,
            # Add the grammar constraint
            **({} if (not json) else dict(grammar=JSON_ARRAY_GRAMMAR)),
        ):
            print(token, end="", flush=True)

import requests
import json

def server_chat_complete(prompt, max_tokens=-1, temperature=0.1, stream=False, messages=(), system="", base_url="http://127.0.0.1:3544/v1", **kwargs):
    """
    A minimal wrapper to interact with a local LM Studio server via direct HTTP requests.
    
    Args:
        prompt (str): The user prompt to send.
        max_tokens (int): Maximum tokens to generate (-1 for no limit).
        temperature (float): Sampling temperature (default: 0.1).
        stream (bool): Whether to stream the response (default: False).
        messages (list): Previous conversation messages (strings or dicts).
        system (str): System prompt (default: "").
        base_url (str): Server base URL (default: "http://127.0.0.1:3544/v1").
        **kwargs: Additional parameters to pass to the API.
    
    Returns:
        tuple: (text, finish_reason) for non-streaming, or generator yielding (token, finish_reason) for streaming.
    """
    # Convert messages to proper format
    if len(messages) > 0:
        messages = [
            m if isinstance(m, dict) else {'role': 'user' if i % 2 == 0 else 'assistant', 'content': m}
            for i, m in enumerate(messages)
        ]
    else:
        messages = []
    full_messages = ([{"role": "system", "content": system}] if system else []) + messages + [{"role": "user", "content": prompt}]
    
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


# Example usage
if __name__ == "__main__":
    # Non-streaming example
    print(">>> server_chat_complete(prompt='Hello, how are you?', stream=False)")
    text, finish_reason = server_chat_complete(prompt="Hello, how are you?", stream=False)
    print(f"Response: {text}")
    print(f"Finish reason: {finish_reason}")
    
    # Streaming example
    print("\n>>> server_chat_complete(prompt='Tell me a story', stream=True)")
    print("Streaming response: ", end="")
    for token, finish_reason in server_chat_complete(prompt="Tell me a story", stream=True):
        print(token, end="", flush=True)
        if finish_reason:
            print(f"\nFinish reason: {finish_reason}")

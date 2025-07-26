from queue import Queue
from threading import Thread
import time
from tlux.slm import load_lm, chat_complete
from stt import listen
from tts import speak, is_speaking, interrupt

def audio_chat():
    # Configuration parameters
    temperature = 0.5  # Controls randomness in response generation

    # Initialize conversation history with a system message
    system_prompt = """
You are an interactive voice-driven agent. Your responses are converted to audio, and user input is transcribed from speech. Keep your responses short, concise, and conversational to ensure a smooth interaction. When ready, say 'Hello.'
"""
    messages = [dict(role="system", content=system_prompt)]

    # Queue for transcribed text and silence indicators from STT
    transcription_queue = Queue()

    # Callback function for STT to send transcribed text or "..." to the queue
    def callback(text):
        transcription_queue.put(text)

    # Start STT in a background thread
    stt_thread = Thread(target=listen, args=(callback,))
    stt_thread.daemon = True  # Thread stops when main program exits
    stt_thread.start()

    # Main conversation loop
    while True:
        # Accumulate user input until silence ("...") is detected
        user_input = ""
        while True:
            item = transcription_queue.get()  # Blocking call to wait for next item
            if item.strip().endswith("..."):
                break
            user_input += item + "\n"  # Append transcribed text with a space

        # Skip empty input
        user_input = user_input.strip()
        if not user_input:
            continue

        print("User: ", user_input, flush=True)
        # Add user message to conversation history
        messages.append(dict(role="user", content=user_input))

        # Generate a response using the language model
        print("Assistant:", flush=True)
        response = ""
        for response_part, stop_reason in chat_complete(
            temperature=temperature,
            stream=True,
            messages=messages,
        ):
            response += response_part  # Build response incrementally
            print(response_part, end="", flush=True)
        print()

        # Add assistant response to conversation history
        messages.append(dict(role="assistant", content=response))

        # Speak the response
        try:
            speak(response)
        except RuntimeError:
            messages.eppnd(dict(role="system", content="The last message failed to be spoken to the user. They did not hear it."))

        # Monitor for user interruption while speaking
        while is_speaking():
            if not transcription_queue.empty():
                interrupt()  # Stop current speech
                # Clear queue to discard any input during interruption
                while not transcription_queue.empty():
                    transcription_queue.get()
                break
            time.sleep(0.1)  # Avoid busy waiting

if __name__ == "__main__":
    audio_chat()

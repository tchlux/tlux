"""
Interactive voice-driven chat agent
==================================

This program ties the *stt* (speech-to-text) and *tts* (text-to-speech) helper
modules together with a local language model loaded through **tlux.slm** to
create a hands-free conversational assistant.

Major features
--------------
* **Real-time listening** – Uses Whisper + Silero VAD via *stt.listen* running
  in its own daemon thread.  Each recognised speech segment is forwarded to the
  main loop through a *Queue*.
* **Natural turn-taking** – The user indicates the end of an utterance by
  finishing with an explicit literal string ``"..."``.  When this marker is
  observed the assistant formulates a reply; otherwise the partial transcript
  is buffered.
* **Interruptible playback** – If the user begins to speak while a reply is
  being voiced, playback is faded-out and cancelled immediately via
  *tts.interrupt()* so the user can take the floor.
* **Streaming LLM responses** – The model (any *tlux* compatible model) is
  queried through **chat_complete(..., stream=True)**, allowing the text to be
  printed as it is produced while the full response is accumulated for TTS
  rendering.

Run the module directly to start an interactive session:

>>> python -m audio.chat

Press *Ctrl-C* to terminate.
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from queue import Queue, Empty
from typing import List, Dict

from tlux.slm import load_lm, chat_complete  # type: ignore

import stt  # local package
import tts  # local package

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
USER_END_MARKER = "..."            # Marks end of the user's turn
CONTEXT_LEN     = 2048             # Context window given to the language model
MODEL_TEMP      = 0.4              # LLM temperature for a balanced style

# System prompt that governs assistant behaviour.  Keep it concise: replies
# are spoken aloud, so brevity is paramount.
SYSTEM_PROMPT = (
    "You are an audio chat assistant.  The user hears your replies through a "
    "speech synthesiser and speaks back in natural language.  Be *concise*, "
    "friendly, and helpful – aim for a couple of short sentences at most.  "
    "If a clarification is required, ask for it *briefly*.  Respond only after "
    f"the user's message ends with the token \"{USER_END_MARKER}\"."
)

# -----------------------------------------------------------------------------
# Helper classes / functions
# -----------------------------------------------------------------------------

def _listen_in_background(transcript_queue: Queue[str]) -> None:
    """Start *stt.listen* in a daemon thread and forward every transcript line
    into *transcript_queue* so the main thread can react to it."""

    def _callback(text: str) -> None:
        print("User: ", text, flush=True)
        transcript_queue.put(text)

    listener = threading.Thread(
        target=stt.listen, kwargs=dict(callback=_callback), daemon=True, name="stt-listen"
    )
    listener.start()


def _generate_response(dialogue: List[Dict[str, str]]) -> str:
    """Stream a response from the language-model and return the *full* text."""
    response_parts: List[str] = []
    kwargs = dict(messages=dialogue)  # the tlux API expects this mapping
    print("dialogue: ", dialogue, flush=True)

    print("⏳ Thinking…", flush=True)
    for part, _ in chat_complete(
        n_ctx=CONTEXT_LEN,
        temperature=MODEL_TEMP,
        stream=True,
        **kwargs,
    ):
        sys.stdout.write(part)
        sys.stdout.flush()
        response_parts.append(part)
    print()  # newline after the assistant's reply

    return "".join(response_parts).strip()


# -----------------------------------------------------------------------------
# Main conversational loop
# -----------------------------------------------------------------------------

def audio_chat() -> None:  # noqa: C901  (complexity OK for a CLI entry-point)
    """Run the interactive voice chat session."""

    # Ongoing conversation state sent to the model.
    dialogue: List[Dict[str, str]] = [dict(role="system", content=SYSTEM_PROMPT)]
    # Buffer for the *current* user utterance (may arrive in several chunks).
    pending: List[str] = []

    # Queue used by the STT callback to deliver transcripts.
    transcripts: Queue[str] = Queue()
    _listen_in_background(transcripts)

    # Ensure SIGINT causes a clean shutdown.
    shutdown = threading.Event()

    def _sigint_handler(sig, frame):  # noqa: D401  (simple handler)
        shutdown.set()
    signal.signal(signal.SIGINT, _sigint_handler)

    print("🎙  Microphone is live – start speaking!  (Ctrl-C to quit)")

    while not shutdown.is_set():
        try:
            # Wait briefly for new speech.  Timeout allows us to check *shutdown*.
            chunk = transcripts.get(timeout=0.1)
        except Empty:
            continue  # no new audio – loop again

        # Any incoming speech implies the user has grabbed the floor.  If the
        # assistant is mid-sentence we must yield immediately.
        if tts.is_speaking():
            tts.interrupt(wait=False)

        # Accumulate the transcript fragment.
        pending.append(chunk.rstrip())

        # Decide if the user has finished their turn.
        if chunk.rstrip().endswith(USER_END_MARKER):
            # Build the full utterance, strip the marker, and reset the buffer.
            full_user_text = " ".join(pending).rstrip(USER_END_MARKER).strip()
            pending.clear()

            # Record turn for model context.
            dialogue.append({"role": "user", "content": full_user_text})

            # --- Generate the assistant's reply --------------------------------
            response_text = _generate_response(dialogue)
            dialogue.append({"role": "assistant", "content": response_text})

            # --- Speak the reply (non-blocking) --------------------------------
            try:
                tts.speak(response_text)
            except RuntimeError:
                # A previous speak() may still be initialising – in that rare
                # case just wait briefly and try again.
                time.sleep(0.2)
                tts.speak(response_text)

    print("\n👋 Exiting – goodbye!")


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    audio_chat()

# This script provides a text-to-speech (TTS) system using the Kokoro TTS model.
# It handles audio generation, playback, and interruption using threading for concurrency.
# The system allows for speaking text, interrupting the speech, and checking if audio is playing.
#
# Key components:
# - Audio generation: Uses the KPipeline from the kokoro module to generate audio from text.
# - Audio playback: A dedicated thread plays the generated audio using sounddevice.
# - Interruption: Allows interrupting the current speech with a fade-out effect.
#
# The system ensures that only one audio generation runs at a time and provides
# mechanisms to check if audio is currently playing.

import os
import threading
import queue
import numpy as np
import soundfile as sf
import sounddevice as sd
from kokoro import KPipeline

# Global TTS pipeline instance, loaded lazily
TEXT_TO_SPEECH = None

# Voice identifier for the TTS model
VOICE = 'af_heart'

# Sample rate for audio output in Hz
SAMPLE_RATE = 24_000

# Flag indicating whether the playback thread has started
_playback_thread_started = False

# Counter tracking active generator threads (limits to one at a time)
_generator_thread_count = 0

# Queue holding audio data tuples for playback: (audio_array, sample_rate, gap_indices)
_audio_queue = queue.Queue()

# Event to signal an interruption request for audio playback
_interrupt_event = threading.Event()

# Event set when audio playback is active
_speaking_event = threading.Event()

# Event set when no audio is playing (silent state)
_silence_event = threading.Event()
_silence_event.set()  # Initially silent since no audio is playing


# Suppress specific PyTorch warnings to keep output clean
import warnings
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer*",  # RNN UserWarning
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.nn\.utils\.weight_norm",
)


# Load the TTS pipeline if not already initialized
# Returns the loaded TTS pipeline instance
def load_text_to_speech():
    global TEXT_TO_SPEECH
    if TEXT_TO_SPEECH is None:
        # Create a new KPipeline instance with language code 'a' and specified model
        TEXT_TO_SPEECH = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return TEXT_TO_SPEECH


# Identify silent gaps in audio for interruption points
# 
# Parameters:
# - audio: numpy array of audio samples
# - sr: sample rate of the audio in Hz
# - win_ms: window size in milliseconds for silence detection (default: 20)
# - min_gap_ms: minimum silence duration in milliseconds to qualify as a gap (default: 80)
# - thresh: amplitude threshold for silence detection (default: 0.005)
# 
# Returns:
# - List of sample indices marking the end of silent gaps
# 
def _find_gaps(audio, sr, win_ms=20, min_gap_ms=100, thresh=0.005):
    win = int(sr * win_ms / 1000)  # Convert window size from ms to samples
    min_gap = int(sr * min_gap_ms / 1000)  # Convert min gap size from ms to samples
    gaps = []  # List to store end indices of detected gaps
    silent_len = 0  # Track consecutive silent samples
    for i in range(0, len(audio), win):
        # Calculate mean amplitude of current window
        if np.mean(np.abs(audio[i:i + win])) < thresh:
            silent_len += win  # Increment silent length if below threshold
            if silent_len >= min_gap:
                # Record gap end if silence duration meets minimum requirement
                gaps.append(i + win)
        else:
            silent_len = 0  # Reset if window is not silent
    return gaps


# Worker thread to handle audio playback from the queue
def _audio_worker():
    while True:
        # Retrieve audio data, sample rate, and gaps from the queue
        audio, sr, gaps = _audio_queue.get()
        # Convert audio to float32 for compatibility with sounddevice
        audio = audio.astype("float32")
        # Reset interruption flag before playback
        _interrupt_event.clear()
        # Signal that playback is active
        _silence_event.clear()
        _speaking_event.set()
        pos = 0  # Current position in audio array
        try:
            # Initialize audio output stream with given sample rate and single channel
            with sd.OutputStream(samplerate=sr, channels=1, dtype='float32') as stream:
                # Iterate through gaps plus end of audio for chunked playback
                for gap in gaps + [len(audio)]:
                    chunk = audio[pos:gap]  # Extract current audio chunk
                    if _interrupt_event.is_set():
                        # Apply fade-out effect if interrupted
                        chunk *= np.linspace(1.0, 0.0, chunk.size)
                        stream.write(chunk)  # Play faded chunk
                        break
                    else:
                        # Play chunk normally and update position
                        stream.write(chunk)
                        pos = gap
                stream.stop()  # Stop stream after playback completes
        finally:
            # Mark queue task as completed
            _audio_queue.task_done()
            # Signal that playback has stopped
            _speaking_event.clear()
            _silence_event.set()


# Start the audio playback thread if it hasn't been started yet
def _ensure_audio_thread():
    global _playback_thread_started
    if not _playback_thread_started:
        # Launch playback worker as a daemon thread named 'kokoro-playback'
        threading.Thread(target=_audio_worker, name='kokoro-playback', daemon=True).start()
        _playback_thread_started = True  # Mark thread as started


# Worker thread to generate audio from text and queue it for playback
def _generator_worker(text):
    global _generator_thread_count
    try:
        # Load or retrieve the TTS pipeline
        pipe = load_text_to_speech()
        # Generate audio segments for the text using the specified voice
        buf = [audio for _, _, audio in pipe(text, voice=VOICE)]
        if not buf:
            return  # Exit if no audio is generated
        # Combine all audio segments into a single array
        full_audio = np.concatenate(buf)
        # Detect silent gaps in the generated audio
        gaps = _find_gaps(full_audio, SAMPLE_RATE)
        # Queue audio data for playback
        _audio_queue.put((full_audio, SAMPLE_RATE, gaps))
    finally:
        # Decrease generator count regardless of success or failure
        _generator_thread_count -= 1


# Initiate text-to-speech for the given text
# 
# Parameters:
# - text: String to convert to speech
# 
# Raises RuntimeError if another speak operation is in progress
# 
def speak(text):
    # Ensure playback thread is running
    _ensure_audio_thread()
    global _generator_thread_count
    if _generator_thread_count > 0:
        # Prevent multiple simultaneous generators
        raise RuntimeError('speak() already running')
    # Increment generator count to track active generation
    _generator_thread_count += 1
    # Start audio generation in a new daemon thread
    threading.Thread(target=_generator_worker, args=(text,), daemon=True).start()


# Interrupt current audio playback
# 
# Parameters:
# - wait: If True, block until playback stops (default: True)
# 
def interrupt(wait=True):
    # Request playback interruption
    _interrupt_event.set()
    # Remove and mark all pending audio tasks as done
    try:
        while True:
            _audio_queue.get_nowait()
            _audio_queue.task_done()
    except queue.Empty:
        pass  # Exit when queue is empty
    if wait:
        # Wait for playback to stop if requested
        _silence_event.wait()


# Check if audio is currently being played
# 
# Returns:
# - Boolean indicating whether playback is active
# 
def is_speaking():
    return _speaking_event.is_set()


# Wait until speaking starts and then return.
def when_speaking():
    _speaking_event.wait()


# Wait until speaking starts and then return.
def when_done_speaking():
    _silence_event.wait()


# Demonstration script to test TTS functionality
if __name__ == "__main__":
    import time
    import random

    SAMPLE_TEXT = (
        "[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with eighty-two million parameters. "
        "Despite its lightweight architecture, it delivers comparable quality "
        "to larger models while being significantly faster and more cost-efficient. "
        "With Apache-licensed weights, "
        "[Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects. "
    )

    print("Speaking text..", flush=True)
    speak(SAMPLE_TEXT)

    # Wait until audio playback begins
    _speaking_event.wait()

    # Set a wait time before interrupting (overridden for testing)
    wait_time = 10 * random.random()
    # wait_time = 0.50  # Fix for consistent testing
    print("", flush=True)
    print(f"Interrupting in {wait_time:.2f} seconds..", flush=True)
    time.sleep(wait_time)

    # Measure and perform interruption
    interrupt_time = -time.time()
    print()
    print("Interrupting..", flush=True)
    interrupt()
    print("  done.", flush=True)
    interrupt_time += time.time()
    print()
    print("Successfully interrupted in")
    print(f" {interrupt_time:.2f} seconds.")
    time.sleep(2)  # Brief pause before exiting

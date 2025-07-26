#!/usr/bin/env python3

# Realtime transcription with Whisper and Silero VAD.
# 
#  - 16 kHz, 16-bit, mono audio
#  - 32 ms VAD frames (512 samples) as required by Silero
#  - Only voiced segments longer than MIN_SEGMENT_SEC are sent to Whisper
#  - Prints "... silence ..." after SILENCE_ANNOUNCE_SEC without speech
# 

import time
import signal
import threading
from collections import deque
from queue import Queue, Empty
from contextlib import contextmanager

import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

# Configuration settings for audio processing and transcription
# RATE: Audio sample rate in Hz
# FRAME_MS: Duration of each VAD frame in milliseconds
# FRAME_SAMPLES: Number of audio samples per frame (fixed for Silero VAD)
# BYTES_PER_SAMPLE: Bytes per audio sample (int16 format)
# FRAME_BYTES: Total bytes per frame, calculated from samples and bytes per sample
# THRESHOLD: Minimum speech probability for VAD to detect speech
# SEGMENT_PADDING_MS: Silence duration in milliseconds before ending a segment
# PRE_ROLL_MS: Duration of audio to buffer before speech starts, in milliseconds
# MIN_SEGMENT_SEC: Minimum segment length in seconds for Whisper transcription
# SILENCE_ANNOUNCE_SEC: Time in seconds before announcing silence
# WHISPER_MODEL: Name of the Whisper model to use for transcription
# DEVICE: Hardware device for Whisper (cuda if available, otherwise cpu)
RATE                 = 16000          # sample rate (Hz)
FRAME_MS             = 32             # VAD frame size (ms)
FRAME_SAMPLES        = 512            # samples per frame (fixed)
BYTES_PER_SAMPLE     = 2              # int16 mono
FRAME_BYTES          = FRAME_SAMPLES * BYTES_PER_SAMPLE
THRESHOLD            = 0.05           # speech probability threshold
SEGMENT_PADDING_MS   = 300            # trailing silence before flush
PRE_ROLL_MS          = 320            # audio to keep *before* speech start
MIN_SEGMENT_SEC      = 0.30           # min duration Whisper will see
SILENCE_ANNOUNCE_SEC = 2.0            # silence notice timeout
# WHISPER_MODEL        = "base.en"
# WHISPER_MODEL        = "large-v3"
WHISPER_MODEL        = "distil-large-v3"  # alternative model option
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
FINISHED_TRANSCRIBING = threading.Event()


# Context manager for safely handling PyAudio stream lifecycle
@contextmanager
def pyaudio_stream(rate: int, frames_per_buffer: int):
    # Initialize PyAudio instance
    pyaudio_instance = pyaudio.PyAudio()
    # Open a mono, 16-bit audio input stream with specified rate and buffer size
    stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    try:
        # Yield the stream for use within the context
        yield stream
    finally:
        # Ensure stream is stopped and closed, and PyAudio is terminated
        stream.stop_stream()
        stream.close()
        pyaudio_instance.terminate()

# Load the Whisper model for transcription
def load_whisper(model_name: str, device: str):
    # Print loading message with model and device details
    print(f"Loading Whisper '{model_name}' on '{device}'")
    # Initialize and return the Whisper model with int8 computation for efficiency
    return WhisperModel(model_name, device=device, compute_type="int8")

# Transcribe audio segments from a queue in a separate thread
def transcriber(transcription_queue: Queue, stop_event: threading.Event, callback):
    # Load Whisper model once at the start of the thread
    model = load_whisper(WHISPER_MODEL, DEVICE)
    # Continue processing until the stop event is set
    while not stop_event.is_set():
        try:
            # Attempt to retrieve an audio blob from the queue with a short timeout
            blob = transcription_queue.get(timeout=0.1)
        except Empty:
            # Skip to next iteration if queue is empty
            continue
        # Exit thread if a None sentinel is received
        if blob is None:
            break
        # Convert raw audio bytes to normalized float32 array (-1 to 1)
        pcm = np.frombuffer(blob, np.int16).astype(np.float32) / 32768.0
        # Skip transcription if segment is too short for reliable results
        if pcm.size < RATE * MIN_SEGMENT_SEC:
            continue
        # Transcribe audio using Whisper model, assuming English language
        FINISHED_TRANSCRIBING.clear()
        segments, info = model.transcribe(pcm, language="en")
        result = ""
        # Concatenate text from all segments
        for segment in segments:
            result += segment.text
        # Call with timestamp and transcribed text
        callback(f"{time.ctime()} {result.strip()}")
        FINISHED_TRANSCRIBING.set()

# Main listening function to handle audio capture, speech detection, and transcription queuing
def listen(callback=print):
    # Load Silero VAD model for voice activity detection (runs on CPU)
    vad_model = load_silero_vad()
    # Pre-allocate tensor for VAD frame processing
    vad_frame_tensor = torch.empty(1, FRAME_SAMPLES)

    # Initialize data structures
    speech_frames = deque()  # Stores audio frames during speech
    pre_roll_buffer = deque(maxlen=PRE_ROLL_MS // FRAME_MS)  # Buffers frames before speech
    transcription_queue = Queue()  # Queue for passing audio to transcription thread
    stop_event = threading.Event()  # Event to signal thread termination
    last_speech_time = time.monotonic()  # Track time of last speech detection
    silence_notified = False  # Flag to avoid repeated silence messages
    in_speech_segment = False  # Indicates if currently processing a speech segment

    # Start transcription thread as a daemon
    transcription_thread = threading.Thread(
        target=transcriber, args=(transcription_queue, stop_event, callback), daemon=True
    )
    transcription_thread.start()

    try:
        # Define signal handler for graceful Ctrl-C exit
        def on_sigint(sig, frame):
            # Set stop event to terminate loops and threads
            stop_event.set()
        signal.signal(signal.SIGINT, on_sigint)
        # Inform user that the program is ready
        print("Microphone enabled...  press Ctrl-C to stop")
    except:
        pass

    # Open audio stream and process input
    with pyaudio_stream(RATE, FRAME_SAMPLES) as stream:
        # Main loop runs until stop event is triggered
        while not stop_event.is_set():
            # Read one frame of audio data from the stream
            frame_bytes = stream.read(FRAME_SAMPLES, exception_on_overflow=False)

            # Convert audio frame to float32 tensor normalized between -1 and 1
            frame_np = (
                np.frombuffer(frame_bytes, np.int16).astype(np.float32) / 32768.0
            )
            vad_frame_tensor[0] = torch.from_numpy(frame_np)

            # Calculate speech probability using VAD model
            with torch.no_grad():
                prob = vad_model(vad_frame_tensor, RATE).item()
            in_speech = prob > THRESHOLD  # Determine if frame contains speech

            # Store frame in pre-roll buffer for context before speech
            pre_roll_buffer.append(frame_bytes)

            if in_speech:
                # Start of a new speech segment
                if not in_speech_segment:
                    print(" listening..", end="", flush=True)
                    # Add pre-roll frames to provide context
                    speech_frames.extend(pre_roll_buffer)
                    in_speech_segment = True
                # Append current frame to speech segment
                speech_frames.append(frame_bytes)
                last_speech_time = time.monotonic()  # Update last speech time
                silence_notified = False  # Reset silence notification flag
            else:
                if in_speech_segment:
                    # Calculate duration of silence since last speech
                    quiet_ms = (time.monotonic() - last_speech_time) * 1000.0
                    # End segment if silence exceeds padding threshold
                    if quiet_ms > SEGMENT_PADDING_MS:
                        print("done")
                        # Combine frames into a single audio blob
                        segment = b"".join(speech_frames)
                        # Send segment to transcription queue
                        transcription_queue.put(segment)
                        speech_frames.clear()  # Reset for next segment
                        in_speech_segment = False

            # Notify user of prolonged silence if conditions are met
            if (
                not silence_notified
                and time.monotonic() - last_speech_time >= SILENCE_ANNOUNCE_SEC
                and FINISHED_TRANSCRIBING.is_set()
            ):
                callback(f"{time.ctime()} ...")
                silence_notified = True

    # Cleanup: signal transcription thread to exit and wait for it to finish
    transcription_queue.put(None)  # Send sentinel to stop transcriber
    transcription_thread.join(timeout=5)  # Wait up to 5 seconds for thread to end

# Entry point to run the program
if __name__ == "__main__":
    listen()

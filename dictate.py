#!/usr/bin/env python3
"""
DICTATION.SYS v2.0 - Terminal Edition
Simple voice-to-text transcription in your terminal
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import tempfile
import os
import sys
import pyperclip
import threading
import time

# ANSI color codes for that retro green CRT look
GREEN = '\033[92m'
BRIGHT_GREEN = '\033[1;92m'
RED = '\033[91m'
RESET = '\033[0m'
DIM = '\033[2m'

# Global variables
model = None
is_recording = False
recording_data = []
sample_rate = 16000  # Whisper's native sample rate for faster processing
stream = None
recording_start_time = None
indicator_thread = None
stop_indicator = False


def print_banner():
    """Print the retro ASCII banner"""
    banner = f"""{BRIGHT_GREEN}
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  ██████╗ ██╗ ██████╗████████╗ █████╗ ████████╗███████╗  ║
║  ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝  ║
║  ██║  ██║██║██║        ██║   ███████║   ██║   █████╗    ║
║  ██║  ██║██║██║        ██║   ██╔══██║   ██║   ██╔══╝    ║
║  ██████╔╝██║╚██████╗   ██║   ██║  ██║   ██║   ███████╗  ║
║  ╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝  ║
║                                                           ║
║              VOICE-TO-TEXT TRANSCRIPTION SYSTEM          ║
║                        VERSION 2.0                        ║
╚═══════════════════════════════════════════════════════════╝
{RESET}"""
    print(banner)


def print_status(message, status_type="info"):
    """Print a status message with retro styling"""
    if status_type == "info":
        print(f"{GREEN}[●] {message}{RESET}")
    elif status_type == "success":
        print(f"{BRIGHT_GREEN}[✓] {message}{RESET}")
    elif status_type == "error":
        print(f"{RED}[✗] {message}{RESET}")
    elif status_type == "prompt":
        print(f"{BRIGHT_GREEN}> {message}{RESET}", end='')


def load_model():
    """Lazy-load the Whisper model"""
    global model
    if model is None:
        print_status("LOADING WHISPER MODEL...", "info")
        model = whisper.load_model("large")
        print_status("MODEL LOADED", "success")
    return model


def audio_callback(indata, _frames, _time_info, _status):
    """Callback for continuous audio capture"""
    global recording_data, is_recording
    if is_recording:
        recording_data.append(indata.copy())


def recording_indicator():
    """Display animated recording indicator with duration timer"""
    global stop_indicator, recording_start_time

    dots = ['   ', '.  ', '.. ', '...']
    dot_index = 0

    while not stop_indicator:
        elapsed = time.time() - recording_start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        # Clear current line and print indicator
        sys.stdout.write(f'\r{RED}● REC{RESET} {dots[dot_index]} {GREEN}[{minutes:02d}:{seconds:02d}]{RESET} {DIM}Press ENTER to stop{RESET}')
        sys.stdout.flush()

        dot_index = (dot_index + 1) % len(dots)
        time.sleep(0.5)


def start_recording():
    """Start recording audio"""
    global is_recording, recording_data, stream, recording_start_time, indicator_thread, stop_indicator

    recording_data = []
    is_recording = True
    recording_start_time = time.time()
    stop_indicator = False

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()

    # Start the recording indicator in a separate thread
    indicator_thread = threading.Thread(target=recording_indicator, daemon=True)
    indicator_thread.start()


def stop_recording():
    """Stop recording and save to temporary file"""
    global is_recording, recording_data, stream, stop_indicator, indicator_thread

    is_recording = False
    stop_indicator = True

    # Wait for indicator thread to finish
    if indicator_thread and indicator_thread.is_alive():
        indicator_thread.join(timeout=1.0)

    # Print newline to clear the indicator line
    print()

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not recording_data:
        print_status("NO AUDIO RECORDED", "error")
        return None

    # Concatenate all recorded chunks
    recording = np.concatenate(recording_data, axis=0)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file.close()

    sf.write(temp_file.name, recording, sample_rate)
    return temp_file.name


def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper"""
    print_status("TRANSCRIBING AUDIO...", "info")
    model = load_model()
    result = model.transcribe(
        audio_file,
        fp16=False,
        language="en",  # Force English for consistent punctuation
        initial_prompt="Transcribe with proper punctuation and capitalization."
    )
    return result["text"].strip()


def print_transcript(text):
    """Print the transcript in a nice box"""
    lines = text.split('\n')
    max_width = max(len(line) for line in lines) if lines else 0
    max_width = min(max_width, 70)  # Cap at 70 chars

    # Word wrap
    wrapped_lines = []
    for line in lines:
        if len(line) <= max_width:
            wrapped_lines.append(line)
        else:
            words = line.split()
            current_line = ""
            for word in words:
                # Handle words longer than max_width
                if len(word) > max_width:
                    if current_line:
                        wrapped_lines.append(current_line.strip())
                        current_line = ""
                    wrapped_lines.append(word[:max_width])
                    continue

                if len(current_line) + len(word) + 1 <= max_width:
                    current_line += word + " "
                else:
                    wrapped_lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                wrapped_lines.append(current_line.strip())

    # Print box
    print(f"\n{GREEN}{'─' * (max_width + 4)}{RESET}")
    for line in wrapped_lines:
        print(f"{GREEN}│ {RESET}{BRIGHT_GREEN}{line}{RESET}{GREEN}{' ' * (max_width - len(line))} │{RESET}")
    print(f"{GREEN}{'─' * (max_width + 4)}{RESET}\n")


def main():
    """Main program loop"""
    os.system('clear')
    print_banner()

    print(f"{DIM}Press CTRL+C to exit{RESET}\n")

    try:
        while True:
            print_status("Press ENTER to start recording", "prompt")
            input()

            # Start recording
            start_recording()

            # Wait for user to stop
            input()

            # Stop recording
            audio_file = stop_recording()

            if audio_file:
                try:
                    # Transcribe
                    transcription = transcribe_audio(audio_file)

                    # Display result
                    print_status("TRANSCRIPTION COMPLETE", "success")
                    print_transcript(transcription)

                    # Copy to clipboard
                    try:
                        pyperclip.copy(transcription)
                        print_status("Copied to clipboard", "success")
                    except (pyperclip.PyperclipException, OSError):
                        print_status("Manual copy needed (clipboard unavailable)", "info")

                except Exception as e:
                    print_status(f"TRANSCRIPTION ERROR: {str(e)}", "error")

                finally:
                    # Always cleanup temp file, even on error
                    try:
                        os.remove(audio_file)
                    except OSError:
                        pass  # File already deleted or doesn't exist

            print()  # Blank line before next recording

    except KeyboardInterrupt:
        print(f"\n\n{GREEN}SYSTEM SHUTDOWN{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()

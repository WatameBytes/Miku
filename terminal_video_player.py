import os
import sys
import time
import shutil
import threading
import subprocess
import pygame
import cv2
import numpy as np
from PIL import Image
from pynput import keyboard
import io


class TerminalVideoPlayer:
    def __init__(self, video_path, downscale_factor=2, buffer_size=10):
        self.video_path = video_path
        self.audio_path = "temp_audio.wav"
        self.buffer_size = buffer_size

        # ASCII gradient for video rendering
        self.ASCII_CHARS = "@%#*+=-:. "

        # Downscale factor for resolution
        self.downscale_factor = downscale_factor

        self.should_stop = False
        self.volume = 0.5

        # Output buffer for efficient terminal writing
        self.output_buffer = io.StringIO()

        # OpenCV capture with error handling
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            # Set buffer size to reduce thread contention
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            # Disable threading if we encounter issues
            if not self.cap.isOpened():
                print("Retrying with threading disabled...")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_threads=1"
                self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.video_path}")

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0

        except Exception as e:
            raise RuntimeError(f"Error initializing video: {str(e)}")

        # Frame tracking
        self.frame_index = 0

    def extract_audio(self):
        cmd = [
            'ffmpeg', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2',
            self.audio_path, '-y'
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def play_audio(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.audio_path)
        pygame.mixer.music.set_volume(self.volume)
        pygame.mixer.music.play()

    def stop_audio(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    def on_key_press(self, key):
        try:
            if key.char in ['+', '=']:
                self.adjust_volume(0.1)
                print(f"\nVolume: {self.volume:.1f}")
            elif key.char == '-':
                self.adjust_volume(-0.1)
                print(f"\nVolume: {self.volume:.1f}")
            elif key.char == 'q':
                self.should_stop = True
        except AttributeError:
            pass

    def adjust_volume(self, delta):
        self.volume = max(0.0, min(1.0, self.volume + delta))
        pygame.mixer.music.set_volume(self.volume)

    def clear_terminal(self):
        # More efficient terminal clearing using cursor movement
        print('\033[H', end='')

    def get_terminal_size(self):
        size = shutil.get_terminal_size(fallback=(80, 25))
        return size.columns, size.lines

    def frame_to_ascii(self, frame):
        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).convert('L')

        # Resize image
        pil_img = self.resize_to_terminal(pil_img)

        # Use numpy for faster processing
        pixels = np.array(pil_img)
        ascii_indices = (pixels // (256 // len(self.ASCII_CHARS))).clip(0, len(self.ASCII_CHARS) - 1)
        ascii_chars = np.array(list(self.ASCII_CHARS))[ascii_indices]

        # Join characters efficiently
        return '\n'.join([''.join(row) for row in ascii_chars])

    def resize_to_terminal(self, image):
        """
        Downscale the image and fit to terminal size while preserving aspect ratio.
        """
        term_width, term_height = self.get_terminal_size()
        term_height = max(1, term_height - 2)  # Buffer space

        w, h = image.size

        # Apply downscale factor
        w = max(1, w // self.downscale_factor)
        h = max(1, h // self.downscale_factor)

        # Adjust for terminal aspect ratio
        aspect_ratio = h / float(w) * 0.55

        new_width = min(term_width, w)
        new_height = int(new_width * aspect_ratio)

        if new_height > term_height:
            new_height = term_height
            new_width = int(new_height / aspect_ratio)

        return image.resize((max(1, new_width), max(1, new_height)))

    def cleanup(self):
        if os.path.exists(self.audio_path):
            self.stop_audio()
            time.sleep(0.2)
            os.remove(self.audio_path)
        if self.cap.isOpened():
            self.cap.release()

    def play(self):
        print("Extracting audio...")
        self.extract_audio()

        # Setup keyboard listener
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()

        print("\nControls:")
        print("  + or = : Volume Up")
        print("  -      : Volume Down")
        print("  q      : Quit")

        time.sleep(1)

        # Start audio in separate thread
        audio_thread = threading.Thread(target=self.play_audio)
        audio_thread.start()

        print("\nStarting playback...\n")

        # Initialize timing variables
        frame_duration = 1.0 / self.fps
        start_time = time.time()
        SKIP_THRESHOLD = frame_duration * 2

        # Create frame buffer
        frame_buffer = []

        # Pre-buffer some frames
        for _ in range(self.buffer_size):
            ret, frame = self.cap.read()
            if ret:
                ascii_frame = self.frame_to_ascii(frame)
                frame_buffer.append(ascii_frame)

        try:
            while not self.should_stop:
                # Get next frame ready
                ret, next_frame = self.cap.read()
                if ret:
                    ascii_frame = self.frame_to_ascii(next_frame)
                    frame_buffer.append(ascii_frame)

                if not frame_buffer:
                    break

                ideal_time = start_time + (self.frame_index * frame_duration)
                now = time.time()

                # Skip frame if we're too far behind
                if now > ideal_time + SKIP_THRESHOLD:
                    self.frame_index += 1
                    continue

                # Display frame if we're on schedule
                if now <= ideal_time + (frame_duration / 2.0):
                    # Efficient terminal output
                    self.output_buffer.seek(0)
                    self.output_buffer.truncate()
                    self.output_buffer.write('\033[H' + frame_buffer.pop(0))
                    sys.stdout.write(self.output_buffer.getvalue())
                    sys.stdout.flush()

                self.frame_index += 1

                # Precise timing
                now = time.time()
                to_sleep = ideal_time + frame_duration - now
                if to_sleep > 0:
                    time.sleep(max(0, to_sleep - 0.001))
                    while time.time() < ideal_time + frame_duration:
                        pass  # Busy wait for remaining small interval

        except KeyboardInterrupt:
            self.should_stop = True
        finally:
            self.cleanup()
            listener.stop()


def find_video():
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv')
    current_dir = os.path.abspath(os.path.dirname(__file__))
    for file in os.listdir(current_dir):
        if file.lower().endswith(video_extensions):
            return os.path.join(current_dir, file)
    raise FileNotFoundError(
        "No video file (.mp4, .avi, .mkv, .mov, .wmv, .flv) found."
    )


def main():
    # Adjustable parameters
    downscale = 2  # Higher values = fewer characters = faster rendering
    buffer_size = 10  # Higher values = more memory use but smoother playback

    try:
        video_path = find_video()
        print(f"Found video: {os.path.basename(video_path)}")
        player = TerminalVideoPlayer(video_path, downscale_factor=downscale, buffer_size=buffer_size)
        player.play()
    except FileNotFoundError as e:
        print(e)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
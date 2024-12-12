import pyaudio
import queue
import logging
import asyncio


class AsyncMicrophone:
    def __init__(self, chunk=1024, format=pyaudio.paInt16, channels=1, rate=24000):
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.rate = rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
            stream_callback=self.callback,
        )
        self.queue = queue.Queue()
        self.is_recording = False
        self.is_receiving = False
        logging.info("AsyncMicrophone initialized")

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording and not self.is_receiving:
            self.queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start_recording(self):
        self.is_recording = True
        logging.info("Started recording")

    def stop_recording(self):
        self.is_recording = False
        logging.info("Stopped recording")

    def start_receiving(self):
        self.is_receiving = True
        self.is_recording = False
        logging.info("Started receiving assistant response")

    def stop_receiving(self):
        self.is_receiving = False
        logging.info("Stopped receiving assistant response")

    def get_audio_data(self):
        data = b""
        while not self.queue.empty():
            data += self.queue.get()
        return data if data else None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logging.info("AsyncMicrophone closed")

    async def play_audio(self, audio_data):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, output=True)
        stream.write(audio_data)

        # Add a small delay of silence at the end to prevent popping, and weird cuts off sounds
        silence_duration = 0.4
        silence_frames = int(self.rate * silence_duration)
        silence = b"\x00" * (
            silence_frames * self.channels * 2
        )  # 2 bytes per sample for 16-bit audio
        stream.write(silence)

        # Add a small pause before closing the stream to make sure the audio is fully played
        await asyncio.sleep(0.5)

        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.debug("Audio playback completed")
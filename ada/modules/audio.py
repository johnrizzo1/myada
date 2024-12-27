import pyaudio
import queue
import logging
import asyncio
import numpy as np
from __future__ import annotations
from typing import TYPE_CHECKING, BinaryIO
if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from numpy.typing import NDArray
    from collections.abc import Iterable
    import faster_whisper.transcribe
    from .models import TranscriptionWord
    from .asr import FasterWhisperASR


logger = logging.getLogger(__name__)
SAMPLES_PER_SECOND=24000

class AsyncAudio:
    def __init__(self, chunk=1024, format=pyaudio.paInt16, channels=1, rate=SAMPLES_PER_SECOND):
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
        logger.info("AsyncMicrophone initialized")

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording and not self.is_receiving:
            self.queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start_recording(self):
        self.is_recording = True
        logger.info("Started recording")

    def stop_recording(self):
        self.is_recording = False
        logger.info("Stopped recording")

    def start_receiving(self):
        self.is_receiving = True
        self.is_recording = False
        logger.info("Started receiving assistant response")

    def stop_receiving(self):
        self.is_receiving = False
        logger.info("Stopped receiving assistant response")

    def get_audio_data(self):
        data = b""
        while not self.queue.empty():
            data += self.queue.get()
        return data if data else None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logger.info("AsyncMicrophone closed")

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
        logger.debug("Audio playback completed")


# def audio_samples_from_file(file: BinaryIO) -> NDArray[np.float32]:
#     audio_and_sample_rate = sf.read(
#         file,
#         format="RAW",
#         channels=1,
#         samplerate=24000,
#         subtype="PCM_16",
#         dtype="float32",
#         endian="LITTLE",
#     )
#     audio = audio_and_sample_rate[0]
#     return audio  # pyright: ignore[reportReturnType]


class Audio:
    def __init__(
        self,
        data: NDArray[np.float32] = np.array([], dtype=np.float32),
        start: float = 0.0,
    ) -> None:
        self.data = data
        self.start = start

    def __repr__(self) -> str:
        return f"Audio(start={self.start:.2f}, end={self.end:.2f})"

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def duration(self) -> float:
        return len(self.data) / SAMPLES_PER_SECOND

    def after(self, ts: float) -> Audio:
        assert ts <= self.duration
        return Audio(self.data[int(ts * SAMPLES_PER_SECOND) :], start=ts)

    def extend(self, data: NDArray[np.float32]) -> None:
        # logger.debug(f"Extending audio by {len(data) / SAMPLES_PER_SECOND:.2f}s")
        self.data = np.append(self.data, data)
        # logger.debug(f"Audio duration: {self.duration:.2f}s")


class AudioStream(Audio):
    def __init__(
        self,
        data: NDArray[np.float32] = np.array([], dtype=np.float32),
        start: float = 0.0,
    ) -> None:
        super().__init__(data, start)
        self.closed = False
        self.modify_event = asyncio.Event()

    def extend(self, data: NDArray[np.float32]) -> None:
        assert not self.closed
        super().extend(data)
        self.modify_event.set()

    def close(self) -> None:
        assert not self.closed
        self.closed = True
        self.modify_event.set()
        logger.info("AudioStream closed")

    async def chunks(self, min_duration: float) -> AsyncGenerator[NDArray[np.float32], None]:
        i = 0.0  # end time of last chunk
        while True:
            await self.modify_event.wait()
            self.modify_event.clear()

            if self.closed:
                if self.duration > i:
                    yield self.after(i).data
                return
            if self.duration - i >= min_duration:
                # If `i` shouldn't be set to `duration` after the yield
                # because by the time assignment would happen more data might have been added
                i_ = i
                i = self.duration
                # NOTE: probably better to just to a slice
                yield self.after(i_).data


class LocalAgreement:
    def __init__(self) -> None:
        self.unconfirmed = Transcription()

    def merge(self, confirmed: Transcription, incoming: Transcription) -> list[TranscriptionWord]:
        # https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py#L264
        incoming = incoming.after(confirmed.end - 0.1)
        prefix = common_prefix(incoming.words, self.unconfirmed.words)
        logger.debug(f"Confirmed: {confirmed.text}")
        logger.debug(f"Unconfirmed: {self.unconfirmed.text}")
        logger.debug(f"Incoming: {incoming.text}")

        if len(incoming.words) > len(prefix):
            self.unconfirmed = Transcription(incoming.words[len(prefix) :])
        else:
            self.unconfirmed = Transcription()

        return prefix


def needs_audio_after(confirmed: Transcription) -> float:
    full_sentences = to_full_sentences(confirmed.words)
    return full_sentences[-1][-1].end if len(full_sentences) > 0 else 0.0


def prompt(confirmed: Transcription) -> str | None:
    sentences = to_full_sentences(confirmed.words)
    return word_to_text(sentences[-1]) if len(sentences) > 0 else None


async def audio_transcriber(
    asr: FasterWhisperASR,
    audio_stream: AudioStream,
    min_duration: float,
) -> AsyncGenerator[Transcription, None]:
    local_agreement = LocalAgreement()
    full_audio = Audio()
    confirmed = Transcription()
    async for chunk in audio_stream.chunks(min_duration):
        full_audio.extend(chunk)
        audio = full_audio.after(needs_audio_after(confirmed))
        transcription, _ = await asr.transcribe(audio, prompt(confirmed))
        new_words = local_agreement.merge(confirmed, transcription)
        if len(new_words) > 0:
            confirmed.extend(new_words)
            yield confirmed
    logger.debug("Flushing...")
    confirmed.extend(local_agreement.unconfirmed.words)
    yield confirmed
    logger.info("Audio transcriber finished")
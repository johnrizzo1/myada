import whisper
import numpy as np


class SpeachToTextService:
    """
    Service to convert audio to text
    """
    def __init__(self, llm):
        whisper.load_model("turbo")
        self.audio_data = b""
        self.audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self.llm = llm


    def transcribe(self, audio_np: np.ndarray) -> str:
        """
        Transcribes the given audio data using the Whisper speech recognition model.
        Args:
            audio_np (numpy.ndarray): The audio data to be transcribed.
        Returns:
            str: The transcribed text.
        """
        result = whisper.transcribe(audio_np)  # , fp16=True)  # Set fp16=True if using a GPU
        text = result["text"].strip()
        return text


    def record_audio(self):
        data_queue = Queue()  # type: ignore[var-annotated]
        stop_event = threading.Event()
        recording_thread = threading.Thread(
            target=record_audio,
            args=(stop_event, data_queue),
        )
        recording_thread.start()

        input()
        stop_event.set()
        recording_thread.join()

        self.audio_data = b"".join(list(data_queue.queue))
        self.audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )


    def transcribe(self) -> str:
        if audio_np.size > 0:
            return whisper.transcribe(self.audio_data)

        else:
            return ""
        
    def speak(self, text=""):
        sample_rate, audio_array = tts.long_form_synthesize(text, voice_preset="v2/en_speaker_1")
        # console.print(f"[cyan]Assistant: {response}")
        play_audio(sample_rate, audio_array)

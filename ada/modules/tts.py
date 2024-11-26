"""
tts TextToSpeechService 
"""
import warnings
import nltk
import torch
import numpy as np
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    """
    TextToSpeechService
    """
    def __init__(self):
        """
        Initializes the TextToSpeechService class.
        """

        # Downloading the nltk files here so it happens once
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("punkt_tab")

        if torch.cuda.is_available(): self.device = "cuda" 
        elif torch.backends.mps.is_available(): self.device = "mps" 
        else: self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
                v2/en_speaker_6: English, American, male, neutral
                v2/fr_speaker_1: French, female, cheerful
                v2/ja_speaker_3: Japanese, male, calm
                v2/zh_speaker_4: Chinese, female, friendly

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
            # audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []

        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)


from ada.modules.logging import logger, log_tool_call, log_ws_event, log_runtime
from ada.modules.audio import AsyncAudio
from ada.modules.tools import tool_map, tools

import os
import sys
import asyncio
import base64
import torch
from transformers import AutoProcessor, BarkModel
from faster_whisper import WhisperModel


class AdaOpenAIMultiAgent:
    def __init__(self, 
                 device = None,
                 ai_assistant_name="Ada", 
                 human_name="John"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Please set the OPENAI_API_KEY in your .env file.")
            sys.exit(1)
        
        if device is None:
            if torch.cuda.is_available(): self.device = "cuda" 
            elif sys.platform=='darwin' and torch.backends.mps.is_available(): self.device = "mps" 
            else: self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained('suno/bark-small')
        self.ttsmodel = BarkModel.from_pretrained('suno/bark-small')
        self.ttsmodel.to(self.device)
        
        self.mic = AsyncAudio()
        self.exit_event = asyncio.Event()
        self.sttmodel = WhisperModel("turbo")

    async def run(self):
        while True:
            try:
                logger.info(
                    "Conversation started. Speak freely, and the assistant will respond."
                )

                self.mic.start_recording()
                logger.info("Recording started. Listening for speech...")

                await self.send_audio_loop()

                # If execution reaches here without exceptions, exit the loop
                break
            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")
                break  # Exit the loop on unexpected exceptions
            finally:
                self.mic.stop_recording()
                self.mic.close()
        
    async def send_audio_loop(self):
        try:
            while not self.exit_event.is_set():
                await asyncio.sleep(0.1)  # Small delay to accumulate audio data
                if not self.mic.is_receiving:
                    audio_data = self.mic.get_audio_data()
                    if audio_data and len(audio_data) > 0:
                        
                        # base64_audio = base64.b64encode(audio_data).decode('utf-8')
                        # if base64_audio:
                        #     audio_event = {
                        #         "type": "input_audio_buffer.append",
                        #         "audio": base64_audio,
                        #     }
                        #     log_ws_event("Outgoing", audio_event)
                        #     # await websocket.send(json.dumps(audio_event))
                        #     await asyncio.sleep(0.1)
                        # else:
                        #     logger.debug("No audio data to send")
                else:
                    await asyncio.sleep(0.1)  # Wait while receiving assistant response
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Closing the connection.")
        finally:
            self.exit_event.set()
            self.mic.stop_recording()
            self.mic.close()
            # await websocket.close()
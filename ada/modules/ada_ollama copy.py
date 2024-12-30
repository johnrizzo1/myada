import logging
import sys
import threading
import time
from queue import Queue
from typing import Sequence, TypedDict

import bs4
import nltk
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
# from rich.console import Console
from transformers import AutoProcessor, BarkModel
from typing_extensions import Annotated  #, TypedDict
# from ada.modules.logging import logger

class State(TypedDict):
    """Statefully manage chat history"""
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


class AdaOllama:
    """
    Service to convert audio to text
    """
    def __init__(self, device=None):
        self.sttmodel = WhisperModel("turbo") #, device="auto", compute_type="float16")
        self.batched_model = BatchedInferencePipeline(model=self.sttmodel)

        self.audio_data = b""
        self.audio_np = (
            np.frombuffer(self.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        # self.console = Console()

        self.logger = logging.getLogger("ada")

        # Downloading the nltk files here so it happens once
        # nltk.download("punkt")
        # nltk.download("wordnet")
        # nltk.download("omw-1.4")
        # nltk.download("punkt_tab")

        if device is None:
            if torch.cuda.is_available(): self.device = "cuda" 
            elif sys.platform=='darwin' and torch.backends.mps.is_available(): self.device = "mps" 
            else: self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained('suno/bark-small')
        self.ttsmodel = BarkModel.from_pretrained('suno/bark-small')
        self.ttsmodel.to(self.device)

        llm = OllamaLLM(model="llama3.2")
        # llm = OllamaLLM(model="firefunction-v2")
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        ### Construct retriever ###
        loader = WebBaseLoader(
            web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = InMemoryVectorStore.from_documents(
            # documents=splits, embedding=OpenAIEmbeddings()
            documents=splits, embedding=OllamaEmbeddings(model="llama3.2")
        )
        retriever = vectorstore.as_retriever()

        ### Contextualize question ###
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages( [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        ### Answer question ###
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Your name is Ada. [speed: 0.5]"
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self.call_model)

        memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "thread-1"}}
        self.app = workflow.compile(checkpointer=memory)

    def record_audio(self):
        data_queue = Queue()  # type: ignore[var-annotated]
        stop_event = threading.Event()
        recording_thread = threading.Thread(
            target=self.record_audio,
            args=(stop_event, data_queue),
        )
        recording_thread.start()

        input()
        stop_event.set()
        recording_thread.join()

        self.audio_data = b"".join(list(data_queue.queue))
        self.audio_np = (
            np.frombuffer(self.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

    def transcribe(self) -> str:
        if self.audio_np.size > 0:
            return self.transcribe(self.audio_data)
        else:
            return ""

    def call_model(self, state: State):
        """Utility method to get the response
        and return it correctly formatted

        Args:
            state (State): _description_

        Returns:
            _type_: _description_
        """
        response = self.rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }

    def record_audio(self, stop_event, data_queue):
        """
        Captures audio data from the user's microphone and adds it to a queue for further processing.
        Args:
            stop_event (threading.Event): An event that, when set, signals the function to stop recording.
            data_queue (queue.Queue): A queue to which the recorded audio data will be added.
        Returns:
            None
        """
        def callback(indata, frames, time, status):
            if status:
                # self.console.print(status)
                self.logger.log(status)
            data_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback ):
                while not stop_event.is_set():
                    time.sleep(0.1)

    def transcribe(self, audio_np: np.ndarray) -> str:
        """
        Transcribes the given audio data using the Whisper speech recognition model.
        Args:
            audio_np (numpy.ndarray): The audio data to be transcribed.
        Returns:
            str: The transcribed text.
        """
        # result = self.sttmodel.transcribe(audio_np) #, fp16=True)  # Set fp16=True if using a GPU
        # print(result)
        # text = result["text"].strip()

        text = ""
        segments, info = self.sttmodel.transcribe(audio_np, 
                                                  beam_size=5,
                                                  language="en",
                                                  condition_on_previous_text=False)
        for segment in segments:
            # self.logger.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            text += segment.text

        return text

    def get_llm_response(self, text: str) -> str:
        """
        Generates a response to the given text using the Llama-2 language model.
        Args:
            text (str): The input text to be processed.
        Returns:
            str: The generated response.
        """
        response = self.app.invoke({"input": text}, config=self.config, )["answer"]
        return response

    def play_audio(self, sample_rate, audio_array):
        """
        Plays the given audio data using the sounddevice library.
        Args:
            sample_rate (int): The sample rate of the audio data.
            audio_array (numpy.ndarray): The audio data to be played.
        Returns:
            None
        """
        sd.play(audio_array, sample_rate)
        sd.wait()
    
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
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            audio_array = self.ttsmodel.generate(**inputs,
                                                #  attention_mask=attention_mask,
                                                 pad_token_id=self.processor.tokenizer.pad_token_id)
            # audio_array = self.ttsmodel.generate(**inputs, pad_token_id=10000)

        # audio_array = audio_array.cpu().numpy().squeeze()
        if self.device=='mps':
            audio_array = audio_array.mps().numpy().squeeze()
        elif self.device=='cuda':
            audio_array = audio_array.cuda().numpy().squeeze()
        else:
            self.logger.warning("Not using any hardware accellerator")
            audio_array = audio_array.cpu().numpy().squeeze()

        sample_rate = self.ttsmodel.generation_config.sample_rate
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
        silence = np.zeros(int(0.25 * self.ttsmodel.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.ttsmodel.generation_config.sample_rate, np.concatenate(pieces)
    
    def run(self):
        ### Interaction Loop ###
        # self.console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
        self.logger.info("[cyan]Assistant started! Press Ctrl+C to exit.")

        try:
            while True:
                input("Press Enter to start recording, then press Enter again to stop." )

                data_queue = Queue()  # type: ignore[var-annotated]
                stop_event = threading.Event()
                recording_thread = threading.Thread(
                    target=self.record_audio,
                    args=(stop_event, data_queue),
                )
                recording_thread.start()

                input()
                stop_event.set()
                recording_thread.join()

                audio_data = b"".join(list(data_queue.queue))
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                )

                if audio_np.size > 0:
                    self.logger.info("Transcribing...")
                    text = self.transcribe(audio_np)

                    self.logger.info(f"[yellow]You: {text}")

                    self.logger.info("Generating response...")
                    response = self.get_llm_response(text)
                    sample_rate, audio_array = self.long_form_synthesize(response, voice_preset = "v2/en_speaker_1")

                    self.logger.info(f"[cyan]Assistant: {response}")
                    self.play_audio(sample_rate, audio_array)
                else:
                    self.logger.info("[red]No audio recorded. Please ensure your microphone is working.")

        except KeyboardInterrupt:
            self.logger.info("\n[red]Exiting...")

        self.logger.info("[blue]Session ended.")
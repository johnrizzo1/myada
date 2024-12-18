import whisper
import numpy as np
from rich.console import Console
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
import threading
from queue import Queue
import time
import sounddevice as sd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
import bs4

class State(TypedDict):
    """Statefully manage chat history

    Args:
        TypedDict (_type_): _description_
    """
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


class SpeachToTextService:
    """
    Service to convert audio to text
    """
    def __init__(self, llm):
        self.model = whisper.load_model("turbo")
        audio_data = b""
        self.audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self.llm = llm
        self.console = Console()
        # stt = whisper.load_model("base.en")
        # tts = TextToSpeechService() 

        # llm = OllamaLLM(model="llama3.2")
        # llm = OllamaLLM(model="llama3")
        # llm = OllamaLLM(model="firefunction-v2")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        ### Construct retriever ###
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
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
            documents=splits, embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        ### Contextualize question ###
        CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        ### Answer question ###
        SYSTEM_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Your name is Ada. [speed: 0.5]"
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        QA_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self.call_model)

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

    def transcribe(self, audio_np: np.ndarray) -> str:
        """
        Transcribes the given audio data using the Whisper speech recognition model.
        Args:
            audio_np (numpy.ndarray): The audio data to be transcribed.
        Returns:
            str: The transcribed text.
        """
        result = self.model.transcribe(audio_np)  # , fp16=True)  # Set fp16=True if using a GPU
        text = result["text"].strip()
        return text

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
            return self.model.transcribe(self.audio_data)
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
                self.console.print(status)
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
        result = self.model.transcribe(audio_np) #, fp16=True)  # Set fp16=True if using a GPU
        text = result["text"].strip()
        return text

    def get_llm_response(self, text: str) -> str:
        """
        Generates a response to the given text using the Llama-2 language model.
        Args:
            text (str): The input text to be processed.
        Returns:
            str: The generated response.
        """
        response = self.app.invoke(
            {"input": text},
            config=self.config,
        )["answer"]
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
    
    def run(self):
        ### Interaction Loop ###
        self.console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

        try:
            while True:
                self.console.input(
                    "Press Enter to start recording, then press Enter again to stop."
                )

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
                    with self.console.status("Transcribing...", spinner="earth"):
                        text = self.transcribe(audio_np)

                    self.console.print(f"[yellow]You: {text}")

                    with self.console.status("Generating response...", 
                                             spinner="earth"):
                        response = self.get_llm_response(text)
                        sample_rate, audio_array = self.tts.long_form_synthesize(response, voice_preset = "v2/en_speaker_1")

                    self.console.print(f"[cyan]Assistant: {response}")
                    self.play_audio(sample_rate, audio_array)
                else:
                    self.console.print(
                        "[red]No audio recorded. Please ensure your microphone is working."
                    )

        except KeyboardInterrupt:
            self.console.print("\n[red]Exiting...")

        self.console.print("[blue]Session ended.")
import asyncio
import os
import sys

from dotenv import load_dotenv

from ada.modules.ada_openai import AdaOpenAI
from ada.modules.ada_ollama import AdaOllama
from ada.modules.logging import logger

# from ada.modules.audio import AsyncAudio


def main():
    print("Starting ADA, Another Digital Assistant...")
    logger.info("Starting ADA, Another Digital Assistant...")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Please set the OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    # url = os.getenv("OPENAI_WSS_URL")
    # if not url:
    #     logger.error("Please set the OPENAI_WSS_URL in your .env file.")

    model = os.getenv("MODEL")
    if not model:
        model="gpt-4o-realtime-preview-2024-10-01"
        # model="gpt-4o-realtime"
    
    # ada = AdaOpenAI(api_key=api_key, url=url, model=model)
    # ada = AdaOpenAI()
    ada = AdaOllama()
    try:
        asyncio.run(ada.run())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__=='__main__':
    print("Press Ctrl+C to exit the program.")
    main()
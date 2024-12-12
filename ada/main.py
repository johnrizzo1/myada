import asyncio
from dotenv import load_dotenv

from ada.modules.logging import logger
from ada.modules.ada import ADA

load_dotenv()

def main():
    print(f"Starting ADA, Another Digital Assistant...")
    logger.info(f"Starting ADA, Another Digital Assistant...")
    ada = ADA()
    try:
        asyncio.run(ada.run())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__=='__main__':
    print("Press Ctrl+C to exit the program.")
    main()
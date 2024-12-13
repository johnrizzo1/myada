from datetime import datetime
import subprocess
from ada.modules.logging import log_info, logger
import sys



def platform_execute(program_name):
    if sys.platform=='win32':
        open_command = 'start'
    elif sys.platform=='darwin':
        open_command = 'open'
    else:
        open_command = 'xdg-open'
    
    try:
        if sys.platform=='win32':
            subprocess.Popen([open_command, program_name], shell=True)
            # os.startfile(d)
        if sys.platform=='darwin':
            subprocess.Popen([open_command, '-a', program_name])
        else:
            subprocess.Popen([open_command, program_name])
    except Exception as e:
        logger.error(f"Failed to start : {str(e)}")
        return {"status": "Error", "message": f"Failed to start : {str(e)}"}


async def get_current_time():
    return {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


async def start_program(program_name):
    """
    Start a Python subprocess to execute the specified command.
    Args:
        prompt (str): The user's input to determine which subprocess to start.
    """
    log_info(f"📖 start_program()", style="bold magenta")
    logger.info(f"📖 start_program() Opening {str}")
    platform_execute(program_name)
    return {"status": "Program started"}


async def open_browser():
    """
    Open a browser tab with the best-fitting URL based on the user's prompt.

    Args:
        prompt (str): The user's prompt to determine which URL to open.
    """
    log_info(f"📖 open_browser()", style="bold magenta")

    # Open the URL if it's not empty
    logger.info(f"📖 open_browser() Opening URL: {str}")
    platform_execute("http://")
    return {"status": "Browser opened"}


# Map function names to their corresponding functions
tool_map = {
    "get_current_time": get_current_time,
    "open_browser": open_browser,
    "start_program": start_program,
    # "create_note": create_note,
}

# Tools array for session initialization
tools = [
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Returns the current time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "open_browser",
        "description": "Opens a web browser",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "start_program",
        "description": "Starts a program",
        "parameters": {
            "type": "object",
            "properties": {
                "program_name": {
                    "type": "string",
                    "description": "Name of the program to start"
                }
            },
            "required": ["program_name"],
        },
    }
]
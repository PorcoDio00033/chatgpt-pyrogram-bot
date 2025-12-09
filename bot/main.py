import os
import sys
import asyncio

from config import functions_available, model, openai_config, plugin_config, telegram_config
from openai_helper import OpenAIHelper
from plugin_manager import PluginManager
from telegram_bot import ChatGPTTelegramBot
from chill_logging import configure_third_party_loggers, Logger, INFO


def main() -> None:
    logger = Logger(name='', in_subfolder=False, level=INFO, propagate=True, log_file_name="chatgpt_pyrogram_bot").logger

    if sys.platform.startswith("win"):
        try:
            import winloop
            winloop.install()
            logger.info("Using winloop event loop")
        except ImportError:
            logger.info("winloop not installed. Using default asyncio event loop.")
    else:
        try:
            import uvloop
            uvloop.install()
            logger.info("Using uvloop event loop")
        except ImportError:
            logger.info("uvloop not installed. Using default asyncio event loop.")

    # required by pyrogram else RuntimeError
    # https://stackoverflow.com/questions/79795917
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # for testing images in workflow
    # kinda meh, but i'm too lazy to add all the envs as repo secrets
    if os.environ.get("WORKFLOW", "false").lower() == "true":
        logger.info("Workflow mode detected. Exiting gracefully.")
        exit(0)

    # Check if the required environment variables are set
    required_values = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'OPENAI_API_KEY']
    missing_values = [value for value in required_values if not os.environ.get(value)]
    if len(missing_values) > 0:
        logger.error(f'The following environment values are missing in your .env: {", ".join(missing_values)}')
        exit(1)

    if openai_config['enable_functions'] and not functions_available:
        logger.error(
            f'ENABLE_FUNCTIONS is set to true, but the model {model} does not support it. '
            'Please set ENABLE_FUNCTIONS to false or use a model that supports it.'
        )
        exit(1)

    configure_third_party_loggers()

    # Setup and run ChatGPT and Telegram bot
    plugin_manager = PluginManager(config=plugin_config, strict_tools=openai_config['strict_tools'])
    openai_helper = OpenAIHelper(config=openai_config, plugin_manager=plugin_manager)
    telegram_bot = ChatGPTTelegramBot(config=telegram_config, openai=openai_helper)
    telegram_bot.run()


if __name__ == '__main__':
    main()

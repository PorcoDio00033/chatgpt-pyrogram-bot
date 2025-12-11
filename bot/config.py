import datetime
import os
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv

from chill_logging import get_logger_instance

load_dotenv()

# Setup logging
logger = get_logger_instance("config").logger


# Models can be found here: https://platform.openai.com/docs/models/overview
GPT_4_MODELS = ('gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-turbo-preview')
GPT_4_32K_MODELS = ('gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613')
GPT_4_VISION_MODELS = (
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
)
GPT_4_128K_MODELS = (
    'gpt-4-1106-preview',
    'gpt-4-0125-preview',
    'gpt-4-turbo-preview',
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
)
GPT_4O_MODELS = (
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4o-2024-08-06',
    'gpt-4o-2024-05-13',
    'gpt-4o-mini-2024-07-18',
    'gpt-4o-search-preview',
    'gpt-4o-mini-search-preview',
)
GPT_41_MODELS = (
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4.1-2025-04-14',
    'gpt-4.1-mini-2025-04-14',
    'gpt-4.1-nano-2025-04-14',
)
GPT_5_MODELS = (
    'gpt-5.1',
    'gpt-5.1-2025-11-13',
    'gpt-5.1-codex',
    'gpt-5.1-codex-mini',
    'gpt-5.1-chat-latest',
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-nano',
    'gpt-5-2025-08-07',
    'gpt-5-mini-2025-08-07',
    'gpt-5-nano-2025-08-07',
    'gpt-5-chat-latest',
)
GPT_SEARCH_MODELS = (
    'gpt-4o-search-preview',
    'gpt-4o-mini-search-preview',
)
GPT_ALL_MODELS = (
    GPT_4_MODELS
    + GPT_4_32K_MODELS
    + GPT_4_VISION_MODELS
    + GPT_4_128K_MODELS
    + GPT_4O_MODELS
    + GPT_41_MODELS
    + GPT_5_MODELS
    + GPT_SEARCH_MODELS
)


def parse_proxy_url(url: str | None) -> dict | None:
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
        return {
            "scheme": parsed.scheme,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "username": parsed.username,
            "password": parsed.password
        }
    except Exception as e:
        logger.warning(f"Failed to parse proxy URL: {e}")
        return None


def default_max_output_tokens(model: str) -> int:
    """
    Gets the default number of max OUTPUT tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1024
    if model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_128K_MODELS:
        return base * 8
    elif model in GPT_4O_MODELS:
        return base * 16
    elif model in GPT_41_MODELS:
        return base * 32
    elif model in GPT_5_MODELS:
        return base * 32

    return base


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    # Stable models will be updated to support functions on June 27, 2023
    if model in (
        'gpt-4',
        'gpt-4-32k',
        'gpt-4-1106-preview',
        'gpt-4-0125-preview',
        'gpt-4-turbo-preview',
    ):
        return datetime.date.today() > datetime.date(2023, 6, 27)
    if model in GPT_SEARCH_MODELS:
        return False
    return True


def read_prompt_from_file(file_path_env_var, default_path, fallback_env_var, default_value) -> str:
    """
    Read prompt from environment variable, with fallback to file.
    Args:
        file_path_env_var: The environment variable containing the path to the file
        default_path: Default path to use if file_path_env_var is not set
        fallback_env_var: The environment variable to use as primary source
        default_value: Default value to use if neither file nor environment variable exists
    Returns:
        The prompt text
    """
    # Priority 1: Environment variable
    # if not set it will read from file
    # this prevents unecessary volume mounts if using docker/docker compose
    if os.environ.get(fallback_env_var):
        return os.environ[fallback_env_var]

    # Priority 2: File
    file_path_str = os.environ.get(file_path_env_var, default_path)
    if file_path_str:
        file_path = Path(file_path_str)
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding='utf-8').strip()
                if content:
                    logger.info(f'Read prompt from file: {file_path}')
                    return content
            except Exception as e:
                logger.warning(f'Failed to read prompt from file {file_path}: {e}')
    
    # Priority 3: Default value
    return default_value


# Setup configurations
model = os.environ.get('OPENAI_MODEL', 'gpt-4.1-mini')
functions_available = are_functions_available(model=model)
max_output_tokens_default = default_max_output_tokens(model=model)

# Read prompts from files or environment variables
assistant_prompt = read_prompt_from_file('ASSISTANT_PROMPT_FILE', 'prompts/assistant_prompt.txt', 'ASSISTANT_PROMPT', 'You are a helpful assistant.')
whisper_prompt = read_prompt_from_file('WHISPER_PROMPT_FILE', 'prompts/whisper_prompt.txt', 'WHISPER_PROMPT', 'Transcribe this audio.')
reaction_prompt = read_prompt_from_file('REACTION_PROMPT_FILE', 'prompts/reaction_prompt.txt', 'REACTION_PROMPT', '{{{{ User }}}} reacted to your message with {reaction}')
stt_user_prompt = read_prompt_from_file('STT_USER_PROMPT_FILE', 'prompts/stt_user_prompt.txt', 'STT_USER_PROMPT', 'AUDIO TRANSCRIPTION:\n{transcript}\n\n{{{{ User }}}} PROMPT:\n{transcribe_user_prompt}')
pdf_prompt = read_prompt_from_file('ATTACHED_PDF_PROMPT_FILE', 'prompts/attached_pdf_prompt.txt', 'ATTACHED_PDF_PROMPT', 'Summarize this document.')

openai_config = {
    'api_key': os.environ.get('OPENAI_API_KEY'),
    'show_usage': os.environ.get('SHOW_USAGE', 'true').lower() == 'true',
    'stream': os.environ.get('STREAM', 'true').lower() == 'true',
    'proxy': os.environ.get('PROXY', None) or os.environ.get('OPENAI_PROXY', None),
    'openai_base_url': os.environ.get('OPENAI_BASE_URL', None),
    'max_openai_api_retries': int(os.environ.get('MAX_OPENAI_API_RETRIES', 3)),
    'max_history_size': int(os.environ.get('MAX_HISTORY_SIZE', 500)),
    'max_conversation_age_minutes': int(os.environ.get('MAX_CONVERSATION_AGE_MINUTES', 10080)),
    'assistant_prompt': assistant_prompt,
    'max_output_tokens': int(os.environ.get('MAX_OUTPUT_TOKENS', max_output_tokens_default)),
    'max_model_tokens': int(os.environ.get('MAX_MODEL_TOKENS')) if os.environ.get('MAX_MODEL_TOKENS') else None,
    'n_choices': int(os.environ.get('N_CHOICES', 1)),
    'temperature': float(os.environ.get('TEMPERATURE', 1.0)),
    'image_model': os.environ.get('IMAGE_MODEL', 'dall-e-3'),
    'image_quality': os.environ.get('IMAGE_QUALITY', 'standard'),
    'image_style': os.environ.get('IMAGE_STYLE', 'vivid'),
    'image_size': os.environ.get('IMAGE_SIZE', '1024x1024'),
    'model': model,
    'supported_input': [s.strip() for s in os.environ.get('OPENAI_MODEL_SUPPORTED_INPUT', 'text').split(',') if s.strip()],
    'enable_functions': os.environ.get('ENABLE_FUNCTIONS', str(functions_available)).lower() == 'true',
    'functions_max_consecutive_calls': int(os.environ.get('FUNCTIONS_MAX_CONSECUTIVE_CALLS', 25)),
    'strict_tools': os.environ.get('OPENAI_STRICT_TOOLS', 'true').lower() == 'true',
    'presence_penalty': float(os.environ.get('PRESENCE_PENALTY', 0.0)),
    'frequency_penalty': float(os.environ.get('FREQUENCY_PENALTY', 0.0)),
    'bot_language': os.environ.get('BOT_LANGUAGE', 'en'),
    'show_plugins_used': os.environ.get('SHOW_PLUGINS_USED', 'false').lower() == 'true',
    'whisper_prompt': whisper_prompt,
    'whisper_model': os.environ.get('WHISPER_MODEL', 'whisper-1'),
    'vision_model': os.environ.get('VISION_MODEL', 'gpt-4.1-mini'),
    'vision_detail': os.environ.get('VISION_DETAIL', 'auto'),
    'tts_model': os.environ.get('TTS_MODEL', 'tts-1'),
    'tts_voice': os.environ.get('TTS_VOICE', 'nova'),
    'allowed_chat_ids_to_track': set(os.environ.get('ALLOWED_CHAT_IDS_TO_TRACK', '').split(',')),
    'web_search_context_size': os.environ.get('WEB_SEARCH_CONTEXT_SIZE', 'medium'),
    'web_search_support_annotations': os.environ.get('WEB_SEARCH_SUPPORT_ANNOTATIONS', 'true').lower() == 'true',
    'reasoning_effort': os.environ.get('REASONING_EFFORT', 'none').lower(),
    'verbosity': os.environ.get('VERBOSITY', 'low').lower(),
    'reaction_prompt': reaction_prompt,
    'stt_user_prompt': stt_user_prompt,
    'pdf_prompt': pdf_prompt
}

telegram_config = {
    'token': os.environ.get('TELEGRAM_BOT_TOKEN'),
    'telegram_api_id': int(os.environ.get("TELEGRAM_API_ID")),
    'telegram_api_hash': os.environ.get("TELEGRAM_API_HASH"),
    'admin_user_ids': os.environ.get('ADMIN_USER_IDS', '-'),
    'img_gen_access_user_ids': os.environ.get('IMG_GEN_ACCESS_USER_IDS', '-'),
    'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),
    'enable_quoting': os.environ.get('ENABLE_QUOTING', 'true').lower() == 'true',
    'enable_image_generation': os.environ.get('ENABLE_IMAGE_GENERATION', 'true').lower() == 'true',
    'enable_transcription': os.environ.get('ENABLE_TRANSCRIPTION', 'true').lower() == 'true',
    'enable_vision': os.environ.get('ENABLE_VISION', 'true').lower() == 'true',
    'enable_tts_generation': os.environ.get('ENABLE_TTS_GENERATION', 'true').lower() == 'true',
    'budget_period': os.environ.get('BUDGET_PERIOD', 'monthly').lower(),
    'user_budgets': os.environ.get('USER_BUDGETS', os.environ.get('MONTHLY_USER_BUDGETS', '*')),
    'guest_budget': float(os.environ.get('GUEST_BUDGET', os.environ.get('MONTHLY_GUEST_BUDGET', '100.0'))),
    'stream': os.environ.get('STREAM', 'true').lower() == 'true',
    'proxy': parse_proxy_url(os.environ.get('PROXY', None) or os.environ.get('TELEGRAM_PROXY', None)),
    'voice_reply_transcript': os.environ.get('VOICE_REPLY_WITH_TRANSCRIPT_ONLY', 'false').lower() == 'true',
    'voice_reply_prompts': os.environ.get('VOICE_REPLY_PROMPTS', '').split(';'),
    'ignore_group_transcriptions': os.environ.get('IGNORE_GROUP_TRANSCRIPTIONS', 'true').lower() == 'true',
    'ignore_group_vision': os.environ.get('IGNORE_GROUP_VISION', 'true').lower() == 'true',
    'group_trigger_keyword': os.environ.get('GROUP_TRIGGER_KEYWORD', ''),
    'token_price': float(os.environ.get('TOKEN_PRICE', 0.002)),
    'image_prices': [float(i) for i in os.environ.get('IMAGE_PRICES', '0.016,0.018,0.02').split(',')],
    'vision_token_price': float(os.environ.get('VISION_TOKEN_PRICE', '0.01')),
    'image_receive_mode': os.environ.get('IMAGE_FORMAT', 'photo'),
    'tts_model': os.environ.get('TTS_MODEL', 'tts-1-hd'),
    'tts_prices': [float(i) for i in os.environ.get('TTS_PRICES', '0.015,0.030').split(',')],
    'transcription_price': float(os.environ.get('TRANSCRIPTION_PRICE', 0.006)),
    'bot_language': os.environ.get('BOT_LANGUAGE', 'en'),
    'database_url': os.environ.get('DATABASE_URL_TO_DROP_ALL_TABLES'),
    'enable_rate_limit': os.environ.get('ENABLE_RATE_LIMIT', 'true').lower() == 'true',
    'group_rate_limit': int(os.environ.get('GROUP_RATE_LIMIT', '20')),
    'private_rate_limit': float(os.environ.get('PRIVATE_RATE_LIMIT', '1.0')),
    'max_update_frequency': float(os.environ.get('MAX_UPDATE_FREQUENCY', '0.5')),
    'enable_raw_reaction': os.environ.get('ENABLE_RAW_REACTION', 'false').lower() == 'true'
}

if model in GPT_SEARCH_MODELS and openai_config['web_search_support_annotations']:
    # annotations are not supported in streaming mode
    openai_config['stream'] = False
    telegram_config['stream'] = False

plugin_config = {'plugins': os.environ.get('PLUGINS', '').split(',')}
from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4
import asyncpg
from PIL import Image
from pydub import AudioSegment
from pypdf import PdfReader
# used for tgs to mp4 converter
from lottie.parsers.tgs import parse_tgs
from lottie.exporters.video import export_video

from pyrogram import Client, filters, enums, types
from pyrogram.errors import BadRequest, MessageNotModified, FloodWait
from pyrogram.handlers import (
    MessageHandler,
    CallbackQueryHandler,
    InlineQueryHandler,
    ChosenInlineResultHandler,
    MessageReactionUpdatedHandler
)
from pyrogram.types import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    InputMediaPhoto,
    InputMediaDocument,
    Message,
    CallbackQuery,
    InlineQuery,
    ChosenInlineResult
)

from chill_logging import get_logger_instance
from decorators import with_conversation_lock
from openai_helper import OpenAIHelper, localized_text
from usage_tracker import UsageTracker
from utils import (
    add_chat_request_to_usage_tracker,
    edit_message_with_retry,
    encode_image,
    error_handler,
    get_forum_thread_id,
    get_remaining_budget,
    is_quoting_enabled,
    get_stream_cutoff_values,
    handle_direct_result,
    has_image_gen_permission,
    is_allowed,
    is_direct_result,
    is_group_chat,
    is_private_chat,
    is_within_budget,
    message_text,
    split_into_chunks,
    wrap_with_indicator,
    extract_username
)


class RateLimiter:
    """
    Class to handle rate limiting for Telegram API
    """

    def __init__(self, config):
        # Store the configuration
        self.config = config
        self.enabled = config.get('enable_rate_limit', True)
        self.group_limit = config.get('group_rate_limit', 20)  # Messages per minute for groups
        self.private_limit = config.get('private_rate_limit', 1.0)  # Seconds between messages
        self.max_update_frequency = config.get('max_update_frequency', 0.5)  # Maximum updates per second

        # Track last message time per chat
        self.last_update_time: Dict[str, float] = {}
        # Track message count per minute for group chats
        self.group_message_count: Dict[str, int] = {}
        self.group_minute_start: Dict[str, float] = {}
        # Track last update time for streaming
        self.last_stream_update: Dict[str, float] = {}

    async def check_and_wait(self, chat_id: str, is_group: bool = False) -> bool:
        """
        Check if we can send a message and wait if necessary
        Returns True if message can be sent, False if we hit a hard limit
        """
        # If rate limiting is disabled, always allow
        if not self.enabled:
            return True

        current_time = time.time()

        # Initialize tracking for this chat if it doesn't exist
        if chat_id not in self.last_update_time:
            self.last_update_time[chat_id] = 0

        # For group chats, handle the group rate limit (default 20 messages per minute)
        if is_group:
            if chat_id not in self.group_message_count:
                self.group_message_count[chat_id] = 0
                self.group_minute_start[chat_id] = current_time

            # Reset counter if a minute has passed
            if current_time - self.group_minute_start[chat_id] > 60:
                self.group_message_count[chat_id] = 0
                self.group_minute_start[chat_id] = current_time

            # Check if we've hit the group message limit
            if self.group_message_count[chat_id] >= self.group_limit:
                # We've hit the hard limit for this minute
                return False

            # Increment the group message counter
            self.group_message_count[chat_id] += 1

        # Calculate time to wait to meet rate limit
        time_since_last_message = current_time - self.last_update_time[chat_id]
        if time_since_last_message < self.private_limit:
            await asyncio.sleep(self.private_limit - time_since_last_message)

        # Update the last message time
        self.last_update_time[chat_id] = time.time()
        return True

    def should_update(self, chat_id: str, is_group: bool, current_length: int, prev_length: int, cutoff: int) -> bool:
        """
        Decide if we should update the message based on rate limiting and content change size
        This is used for streaming to avoid unnecessary updates
        """
        # If rate limiting is disabled, always update
        if not self.enabled:
            return True

        current_time = time.time()

        # Initialize tracking
        if chat_id not in self.last_update_time:
            self.last_update_time[chat_id] = 0
            return True

        if chat_id not in self.last_stream_update:
            self.last_stream_update[chat_id] = 0

        # Check max update frequency for streaming
        time_since_last_stream_update = current_time - self.last_stream_update[chat_id]
        if time_since_last_stream_update < self.max_update_frequency:
            # If we've updated very recently, only update if significant changes (2x cutoff)
            significant_change = abs(current_length - prev_length) > (cutoff * 2)
            if not significant_change:
                return False

        # For group chats, be more conservative with updates
        if is_group:
            # Check if we're approaching the group message limit
            if chat_id in self.group_message_count:
                # If we're at 80% of the limit, be more selective
                if self.group_message_count[chat_id] >= 0.8 * self.group_limit:
                    # Only update if significant changes (2x cutoff)
                    return abs(current_length - prev_length) > (cutoff * 2)

            # Check time since last update
            time_since_last_message = current_time - self.last_update_time[chat_id]

            # For groups, prefer fewer updates
            if time_since_last_message < self.private_limit * 2:
                # If it's been less than 2x the rate limit,
                # only update if significant changes
                return abs(current_length - prev_length) > (cutoff * 1.5)

        # For private chats, be more frequent but still respect limits
        time_since_last_message = current_time - self.last_update_time[chat_id]
        if time_since_last_message < self.private_limit:
            # If it's been less than the rate limit, only update if significant changes
            return abs(current_length - prev_length) > cutoff

        # If we decide to update, update the last stream update time
        self.last_stream_update[chat_id] = current_time

        # Otherwise update is fine
        return True


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        self.logger = get_logger_instance("telegram_bot").logger
        self.rate_limiter = RateLimiter(config)
        bot_language = self.config['bot_language']
        self.commands = [
            # BotCommand(
            #     command='help',
            #     description=localized_text('help_description', bot_language),
            # ),
            BotCommand(
                command='reset',
                description=localized_text('reset_description', bot_language),
            ),
            # BotCommand(
            #     command='stats',
            #     description=localized_text('stats_description', bot_language),
            # ),
            # BotCommand(
            #     command='resend',
            #     description=localized_text('resend_description', bot_language),
            # ),
        ]
        # If imaging is enabled, add the "image" command to the list
        if self.config.get('enable_image_generation', False):
            self.commands.append(
                BotCommand(command='image', description='Generate image from prompt (e.g. /image cat)')
            )

        if self.config.get('enable_tts_generation', False):
            self.commands.append(
                BotCommand(
                    command='tts',
                    description=localized_text('tts_description', bot_language),
                )
            )

        self.group_commands = (
            [
                # BotCommand(
                #     command='chat',
                #     description=localized_text('chat_description', bot_language),
                # )
            ]
            + self.commands
        )
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}
        self.image_prompts_cache = {}  # Cache for storing image prompts
        self.image_quality_cache = {}
        self.image_to_edit_cache = {}  # Cache for storing image to edit data
        self.replies_tracker = {}
        self.bot_message_ids = set()
        self.pending_quality_confirmations = {}  # Store pending confirmations

        # Initialize Pyrogram Client
        self.client = Client(
            "chatgpt_pyrogram_bot",
            api_id=self.config['telegram_api_id'],
            api_hash=self.config['telegram_api_hash'],
            bot_token=self.config['token'],
            proxy=self.config['proxy'],
            workdir="data"
        )

    def get_thread_id(self, message: Message) -> str:
        c = message.chat.id
        m = message
        if not m:
            raise ValueError('No message found in update')

        if is_private_chat(message):
            return f'{c}'

        if not m.reply_to_message:
            return f'{c}_{m.id}'

        self.replies_tracker[m.id] = (
            self.replies_tracker[m.reply_to_message.id]
            if m.reply_to_message.id in self.replies_tracker
            else m.reply_to_message.id
        )

        thread_id = self.replies_tracker[m.id]
        return f'{c}_{thread_id}'

    def get_real_thread_id(self, message: Message) -> Optional[int]:
        m = message
        if not m:
            raise ValueError('No message found in update')

        if not m.reply_to_message:
            return m.id

        self.replies_tracker[m.id] = (
            self.replies_tracker[m.reply_to_message.id]
            if m.reply_to_message.id in self.replies_tracker
            else m.reply_to_message.id
        )

        return self.replies_tracker[m.id]

    def save_reply(self, msg: Message, original_message: Message):
        if not msg:
            return
        self.bot_message_ids.add((msg.chat.id, msg.id))

        if is_private_chat(original_message):
            return

        self.replies_tracker[msg.id] = self.get_real_thread_id(original_message)

    async def help(self, client: Client, message: Message) -> None:
        """
        Shows the help menu.
        """
        commands = self.group_commands if is_group_chat(message) else self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        bot_language = self.config['bot_language']
        help_text = (
            localized_text('help_text', bot_language)[0]
            + '\n\n'
            + '\n'.join(commands_description)
            + '\n\n'
            + localized_text('help_text', bot_language)[1]
            + '\n\n'
            + localized_text('help_text', bot_language)[2]
        )
        await message.reply_text(help_text, link_preview_options=types.LinkPreviewOptions(is_disabled=True))

    async def stats(self, client: Client, message: Message):
        """
        Returns token usage statistics for current day and month.
        """
        if not await is_allowed(self.config, client, message):
            self.logger.warning(
                f'User {extract_username(message.from_user)} (id: {message.from_user.id}) '
                'is not allowed to request their usage statistics'
            )
            await self.send_disallowed_message(client, message)
            return

        self.logger.info(
            f'User {extract_username(message.from_user)} (id: {message.from_user.id}) requested their usage statistics'
        )

        user_id = message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, extract_username(message.from_user))

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        (
            transcribe_minutes_today,
            transcribe_seconds_today,
            transcribe_minutes_month,
            transcribe_seconds_month,
        ) = self.usage[user_id].get_current_transcription_duration()
        vision_today, vision_month = self.usage[user_id].get_current_vision_tokens()
        characters_today, characters_month = self.usage[user_id].get_current_tts_usage()
        current_cost = self.usage[user_id].get_current_cost()

        chat_id = message.chat.id
        chat_messages, chat_token_length = await self.openai.get_conversation_stats(chat_id)
        remaining_budget = get_remaining_budget(self.config, self.usage, message)
        bot_language = self.config['bot_language']

        text_current_conversation = (
            f'*{localized_text("stats_conversation", bot_language)[0]}*:\n'
            f'{chat_messages} {localized_text("stats_conversation", bot_language)[1]}\n'
            f'{chat_token_length} {localized_text("stats_conversation", bot_language)[2]}\n'
            '----------------------------\n'
        )

        # Check if image generation is enabled and, if so, generate the image statistics for today
        text_today_images = ''
        if self.config.get('enable_image_generation', False):
            text_today_images = f'{images_today} {localized_text("stats_images", bot_language)}\n'

        text_today_vision = ''
        if self.config.get('enable_vision', False):
            text_today_vision = f'{vision_today} {localized_text("stats_vision", bot_language)}\n'

        text_today_tts = ''
        if self.config.get('enable_tts_generation', False):
            text_today_tts = f'{characters_today} {localized_text("stats_tts", bot_language)}\n'

        text_today = (
            f'*{localized_text("usage_today", bot_language)}:*\n'
            f'{tokens_today} {localized_text("stats_tokens", bot_language)}\n'
            f'{text_today_images}'  # Include the image statistics for today if applicable
            f'{text_today_vision}'
            f'{text_today_tts}'
            f'{transcribe_minutes_today} {localized_text("stats_transcribe", bot_language)[0]} '
            f'{transcribe_seconds_today} {localized_text("stats_transcribe", bot_language)[1]}\n'
            f'{localized_text("stats_total", bot_language)}{current_cost["cost_today"]:.2f}\n'
            '----------------------------\n'
        )

        text_month_images = ''
        if self.config.get('enable_image_generation', False):
            text_month_images = f'{images_month} {localized_text("stats_images", bot_language)}\n'

        text_month_vision = ''
        if self.config.get('enable_vision', False):
            text_month_vision = f'{vision_month} {localized_text("stats_vision", bot_language)}\n'

        text_month_tts = ''
        if self.config.get('enable_tts_generation', False):
            text_month_tts = f'{characters_month} {localized_text("stats_tts", bot_language)}\n'

        # Check if image generation is enabled and, if so, generate the image statistics for the month
        text_month = (
            f'*{localized_text("usage_month", bot_language)}:*\n'
            f'{tokens_month} {localized_text("stats_tokens", bot_language)}\n'
            f'{text_month_images}'  # Include the image statistics for the month if applicable
            f'{text_month_vision}'
            f'{text_month_tts}'
            f'{transcribe_minutes_month} {localized_text("stats_transcribe", bot_language)[0]} '
            f'{transcribe_seconds_month} {localized_text("stats_transcribe", bot_language)[1]}\n'
            f'{localized_text("stats_total", bot_language)}{current_cost["cost_month"]:.2f}'
        )

        # text_budget filled with conditional content
        text_budget = '\n\n'
        budget_period = self.config['budget_period']
        if remaining_budget < float('inf'):
            text_budget += (
                f'{localized_text("stats_budget", bot_language)}'
                f'{localized_text(budget_period, bot_language)}: '
                f'${remaining_budget:.2f}.\n'
            )
        # No longer works as of July 21st 2023, as OpenAI has removed the billing API
        # add OpenAI account information for admin request
        # if is_admin(self.config, user_id):
        #     text_budget += (
        #         f"{localized_text('stats_openai', bot_language)}"
        #         f"{self.openai.get_billing_current_month():.2f}"
        #     )

        usage_text = text_current_conversation + text_today + text_month + text_budget
        await message.reply_text(usage_text, parse_mode=enums.ParseMode.HTML)

    async def resend(self, client: Client, message: Message):
        """
        Resend the last request
        """
        if not await is_allowed(self.config, client, message):
            self.logger.warning(
                f'User {extract_username(message.from_user)}  (id: {message.from_user.id})'
                ' is not allowed to resend the message'
            )
            await self.send_disallowed_message(client, message)
            return

        chat_id = message.chat.id
        if chat_id not in self.last_message:
            self.logger.warning(
                f'User {extract_username(message.from_user)} (id: {message.from_user.id})'
                ' does not have anything to resend'
            )
            await message.reply_text(
                text=localized_text('resend_failed', self.config['bot_language']),
                #message_thread_id=get_forum_thread_id(message)
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        self.logger.info(
            f'Resending the last prompt from user: {extract_username(message.from_user)} (id: {message.from_user.id})'
        )
        
        # Pyrogram can't modify the message object in place like PTB's _unfrozen()
        # Pyrogram objects are mutable.
        message.text = self.last_message.pop(chat_id)

        await self.prompt(client, message)

    async def reset(self, client: Client, message: Message):
        """
        Resets the conversation.
        """
        if not await is_allowed(self.config, client, message):
            self.logger.warning(
                f'User {extract_username(message.from_user)} (id: {message.from_user.id}) '
                'is not allowed to reset the conversation'
            )
            await self.send_disallowed_message(client, message)
            return

        ai_context_id = self.get_thread_id(message)
        self.logger.info(f'Resetting the conversation for {ai_context_id}.')

        reset_content = message_text(message)
        await self.openai.reset_chat_history(chat_id=ai_context_id, content=reset_content)
        sent_msg = await message.reply_text(
            text=localized_text('reset_done', self.config['bot_language']),
            #message_thread_id=get_forum_thread_id(message)
        )
        self.save_reply(sent_msg, message)

    def _get_quality_reply_markup(self, prompt_id):
        if prompt_id not in self.image_quality_cache or 'highest' not in self.image_quality_cache[prompt_id]:
            return None
        highest = self.image_quality_cache[prompt_id]['highest']
        keyboard = []
        # Always show LOW
        row = [InlineKeyboardButton('LOW', callback_data=f'show_quality:{prompt_id}:low')]
        if highest in ('medium', 'high'):
            row.append(InlineKeyboardButton('MEDIUM', callback_data=f'show_quality:{prompt_id}:medium'))
        if highest == 'high':
            row.append(InlineKeyboardButton('HIGH', callback_data=f'show_quality:{prompt_id}:high'))
        keyboard.append(row)
        # Add improve button if not at highest
        if highest == 'low':
            keyboard.append(
                [
                    InlineKeyboardButton(
                        '🖼️ Improve to Medium Quality ($0.1)', callback_data=f'improve_quality:{prompt_id}:medium'
                    )
                ]
            )
        elif highest == 'medium':
            keyboard.append(
                [
                    InlineKeyboardButton(
                        '❗ High Quality Upgrade ($0.2)', callback_data=f'confirm_quality:{prompt_id}:high'
                    )
                ]
            )
        return InlineKeyboardMarkup(keyboard)

    def _get_confirmation_markup(self, prompt_id):
        keyboard = [
            [
                InlineKeyboardButton('❌ Cancel', callback_data=f'cancel_quality:{prompt_id}'),
                InlineKeyboardButton('✅ Confirm ($0.2)', callback_data=f'improve_quality:{prompt_id}:high'),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    async def handle_show_quality(self, client: Client, query: CallbackQuery):
        await query.answer()

        if not has_image_gen_permission(self.config, query.from_user.id):
            return

        parts = query.data.split(':')
        prompt_id = parts[1]
        target_quality = parts[2]

        if prompt_id not in self.image_quality_cache or target_quality not in self.image_quality_cache[prompt_id]:
            await query.answer('Sorry, this quality version is no longer available.')
            return

        file_id = self.image_quality_cache[prompt_id][target_quality]['file_id']
        caption = self.image_quality_cache[prompt_id][target_quality]['caption']
        reply_markup = self._get_quality_reply_markup(prompt_id)

        if self.config['image_receive_mode'] == 'photo':
            await client.edit_message_media(
                chat_id=query.message.chat.id,
                message_id=query.message.id,
                media=InputMediaPhoto(media=file_id, caption=caption),
                reply_markup=reply_markup,
            )
        elif self.config['image_receive_mode'] == 'document':
            await client.edit_message_media(
                chat_id=query.message.chat.id,
                message_id=query.message.id,
                media=InputMediaDocument(media=file_id, caption=caption),
                reply_markup=reply_markup,
            )

    async def handle_quality_confirmation(self, client: Client, query: CallbackQuery):
        await query.answer()

        if not has_image_gen_permission(self.config, query.from_user.id):
            return

        parts = query.data.split(':')
        prompt_id = parts[1]
        target_quality = parts[2]

        confirmation_text = (
            '⚠️ Premium Feature: High Quality Generation\n'
            'Cost: $0.2 (= 4 medium quality images)\n\n'
            'High quality images provide:\n'
            '• 1024x1024 resolution\n'
            '• Enhanced details and clarity\n'
            '• Better handling of complex scenes\n\n'
            'Are you sure you want to proceed?'
        )

        # Store the confirmation state
        self.pending_quality_confirmations[prompt_id] = {
            'user_id': query.from_user.id,
            'timestamp': datetime.now(),
            'target_quality': target_quality,
        }

        # Update the message with confirmation dialog
        await client.edit_message_caption(
            chat_id=query.message.chat.id,
            message_id=query.message.id,
            caption=confirmation_text,
            reply_markup=self._get_confirmation_markup(prompt_id),
        )

    async def handle_quality_cancel(self, client: Client, query: CallbackQuery):
        await query.answer()

        parts = query.data.split(':')
        prompt_id = parts[1]

        # Remove from pending confirmations
        if prompt_id in self.pending_quality_confirmations:
            del self.pending_quality_confirmations[prompt_id]

        # Restore original markup
        original_caption = self.image_quality_cache[prompt_id]['medium']['caption']
        await client.edit_message_caption(
            chat_id=query.message.chat.id,
            message_id=query.message.id,
            caption=original_caption,
            reply_markup=self._get_quality_reply_markup(prompt_id),
        )

    async def image(self, client: Client, message: Message):
        """
        Generates an image for the given prompt using DALL·E or GPT Image APIs
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(
            client, message
        ):
            return

        bot_language = self.config['bot_language']

        if not has_image_gen_permission(self.config, message.from_user.id):
            self.logger.warning(
                f'User {extract_username(message.from_user)} (id: {message.from_user.id}) '
                'is not allowed to generate images'
            )
            return

        image_query = message_text(message)

        if not image_query:
            await message.reply_text(
                text=localized_text('image_no_prompt', self.config['bot_language']),
                #message_thread_id=get_forum_thread_id(message)
            )
            return

        reply = message.reply_to_message
        image_to_edit_attachment = None
        image_to_edit = None
        
        if reply:
            if reply.photo:
                image_to_edit_attachment = reply.photo
            elif reply.document and reply.document.mime_type.startswith('image/'):
                image_to_edit_attachment = reply.document

        try:
            if image_to_edit_attachment:
                image_to_edit = await client.download_media(image_to_edit_attachment, in_memory=True)
                # image_to_edit is BytesIO object
                image_to_edit.seek(0)
        except Exception as e:
            self.logger.exception(e)
            await message.reply_text(
                text=(
                    f'{localized_text("media_download_fail", bot_language)[0]}: '
                    f'{str(e)}. {localized_text("media_download_fail", bot_language)[1]}'
                ),
                parse_mode=enums.ParseMode.HTML,
                reply_parameters=is_quoting_enabled(self.config, message),
                #message_thread_id=get_forum_thread_id(message)
            )
            return

        user_id = message.from_user.id

        action_msg = 'EDITING' if image_to_edit else 'GENERATING'
        self.logger.info(
            f'New image {action_msg} request received from user {extract_username(message.from_user)} (id: {user_id})'
        )

        async def _generate():
            nonlocal user_id
            try:
                image_bytes, image_size, price = await self.openai.generate_image(
                    prompt=image_query, image_to_edit=image_to_edit, user_id=str(user_id)
                )

                prompt_id = str(uuid4())
                self.image_prompts_cache[prompt_id] = image_query
                self.image_quality_cache[prompt_id] = {'highest': 'low'}

                # Store image_to_edit in cache if it exists
                if image_to_edit:
                    # Create a copy of the image data for later use
                    image_to_edit.seek(0)
                    image_copy = io.BytesIO(image_to_edit.read())
                    self.image_to_edit_cache[prompt_id] = image_copy

                # Add username to price caption
                price_with_user = f'{price}\n\nby {extract_username(message.from_user)}'

                reply_markup = self._get_quality_reply_markup(prompt_id)
                if self.config['image_receive_mode'] == 'photo':
                    sent_msg = await message.reply_photo(
                        photo=image_bytes,
                        caption=price_with_user,
                        reply_markup=reply_markup,
                        reply_parameters=is_quoting_enabled(self.config, message),
                        #message_thread_id=get_forum_thread_id(message)
                    )
                    file_id = sent_msg.photo.file_id
                elif self.config['image_receive_mode'] == 'document':
                    sent_msg = await message.reply_document(
                        document=image_bytes,
                        caption=price_with_user,
                        reply_markup=reply_markup,
                        reply_parameters=is_quoting_enabled(self.config, message),
                        #message_thread_id=get_forum_thread_id(message)
                    )
                    file_id = sent_msg.document.file_id
                else:
                    raise Exception(
                        f'env variable IMAGE_FORMAT has invalid value {self.config["image_receive_mode"]}'
                    )

                self.image_quality_cache[prompt_id]['low'] = {'file_id': file_id, 'caption': price_with_user}

                user_id = message.from_user.id
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, extract_username(message.from_user))

                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage['guests'].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                self.logger.exception(e)
                await message.reply_text(
                    text=f'{localized_text("image_fail", self.config["bot_language"])}: {str(e)}',
                    parse_mode=enums.ParseMode.HTML,
                    reply_parameters=is_quoting_enabled(self.config, message),
                    #message_thread_id=get_forum_thread_id(message)
                )

        await wrap_with_indicator(client, message, _generate, enums.ChatAction.UPLOAD_PHOTO)

    async def handle_improve_quality(self, client: Client, query: CallbackQuery):
        await query.answer()

        if not has_image_gen_permission(self.config, query.from_user.id):
            return

        parts = query.data.split(':')
        prompt_id = parts[1]
        target_quality = parts[2]

        # Check if this is a confirmed action for high quality
        if target_quality == 'high' and prompt_id not in self.pending_quality_confirmations:
            # If not confirmed, show confirmation dialog
            await self.handle_quality_confirmation(client, query)
            return

        # Clean up confirmation state if it exists
        if prompt_id in self.pending_quality_confirmations:
            del self.pending_quality_confirmations[prompt_id]

        if prompt_id not in self.image_prompts_cache:
            await query.answer('Sorry, the prompt is no longer available.')
            return

        prompt = self.image_prompts_cache[prompt_id]

        # Get image_to_edit from cache if it exists
        image_to_edit = None
        if prompt_id in self.image_to_edit_cache:
            image_to_edit = self.image_to_edit_cache[prompt_id]
            image_to_edit.seek(0)

        loading_keyboard = [[InlineKeyboardButton('⏳ Generating...', callback_data='loading')]]
        loading_markup = InlineKeyboardMarkup(loading_keyboard)
        await client.edit_message_reply_markup(
            chat_id=query.message.chat.id, message_id=query.message.id, reply_markup=loading_markup
        )

        async def _generate():
            try:
                user_id = query.from_user.id

                quality_param = 'high' if target_quality == 'high' else 'medium'
                image_bytes, image_size, price = await self.openai.generate_image(
                    prompt=prompt, quality=quality_param, image_to_edit=image_to_edit, user_id=str(user_id)
                )

                # Add username to price caption
                username = query.from_user.username or query.from_user.first_name
                price_with_user = f'{price}\n\nby @{username}'

                self.image_quality_cache[prompt_id]['highest'] = quality_param

                reply_markup = self._get_quality_reply_markup(prompt_id)
                if self.config['image_receive_mode'] == 'photo':
                    sent_msg = await client.edit_message_media(
                        chat_id=query.message.chat.id,
                        message_id=query.message.id,
                        media=InputMediaPhoto(image_bytes, caption=price_with_user),
                        reply_markup=reply_markup,
                    )
                    file_id = sent_msg.photo.file_id
                else:
                    sent_msg = await client.edit_message_media(
                        chat_id=query.message.chat.id,
                        message_id=query.message.id,
                        media=InputMediaDocument(image_bytes, caption=price_with_user),
                        reply_markup=reply_markup,
                    )
                    file_id = sent_msg.document.file_id

                self.image_quality_cache[prompt_id][quality_param] = {'file_id': file_id, 'caption': price_with_user}

                user_id = query.from_user.id
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, extract_username(query.from_user))

                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage['guests'].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                self.logger.exception(e)
                await client.send_message(
                    chat_id=query.message.chat.id,
                    text=f'Failed to improve image quality: {str(e)}',
                    reply_parameters=types.ReplyParameters(query.message.id),
                )

        await wrap_with_indicator(client, query.message, _generate, enums.ChatAction.UPLOAD_PHOTO)

    async def tts(self, client: Client, message: Message):
        """
        Generates an speech for the given input using TTS APIs
        """
        if not self.config['enable_tts_generation'] or not await self.check_allowed_and_within_budget(client, message):
            return

        tts_query = message_text(message)
        if message.reply_to_message and message.reply_to_message.text:
            reply_text = message_text(message.reply_to_message)
            tts_query = f'{reply_text} {tts_query}'.strip()

        if not tts_query:
            await message.reply_text(
                text=localized_text('tts_no_prompt', self.config['bot_language']),
                #message_thread_id=get_forum_thread_id(message)
            )
            return

        self.logger.info(
            f'New speech generation request received from user {extract_username(message.from_user)} '
            f'(id: {message.from_user.id})'
        )

        async def _generate():
            try:
                speech_file, text_length = await self.openai.generate_speech(text=tts_query)

                sent_msg = await message.reply_voice(
                    voice=speech_file,
                    reply_parameters=is_quoting_enabled(self.config, message),
                    #message_thread_id=get_forum_thread_id(message)
                )
                self.save_reply(sent_msg, message)
                speech_file.close()
                # add image request to users usage tracker
                user_id = message.from_user.id
                self.usage[user_id].add_tts_request(text_length, self.config['tts_model'], self.config['tts_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage['guests'].add_tts_request(
                        text_length, self.config['tts_model'], self.config['tts_prices']
                    )

            except Exception as e:
                self.logger.exception(e)
                await message.reply_text(
                    text=f'{localized_text("tts_fail", self.config["bot_language"])}: {str(e)}',
                    parse_mode=enums.ParseMode.HTML,
                    reply_parameters=is_quoting_enabled(self.config, message),
                    #message_thread_id=get_forum_thread_id(message)
                )

        await wrap_with_indicator(client, message, _generate, enums.ChatAction.UPLOAD_AUDIO)

    async def transcribe(self, client: Client, message: Message):
        """
        Transcribe audio messages.
        """
        if not self.config['enable_transcription'] or not await self.check_allowed_and_within_budget(client, message):
            return

        if is_group_chat(message) and self.config['ignore_group_transcriptions']:
            self.logger.info('Transcription coming from group chat, ignoring...')
            return
        
        if message.command:
            transcribe_user_prompt = ' '.join(message.command[1:]) # extract text after /stt
        else:
            transcribe_user_prompt = message.text
        
        if message.reply_to_message:
            target_message = message.reply_to_message
        else:
            target_message = message

        ai_context_id = self.get_thread_id(message)
        
        # Pyrogram doesn't have file_unique_id directly on message, it's on the media object
        media = target_message.voice or target_message.audio or target_message.video or target_message.video_note or target_message.document
        if not media:
            return
            
        filename = media.file_unique_id

        # TODO: add env "ALWAYS_TRANSCODE_TO_MP3" to be able to upload original files as provided by users instead of always transcoding them
        async def _execute():
            bot_language = self.config['bot_language']
            
            try:
                # Download the media file
                media_file = await client.download_media(target_message, in_memory=True)
                
                # Convert to MP3
                mp3_file, duration = self._convert_media_to_mp3(media_file)

                if not mp3_file:
                    await message.reply_text(
                        text=f'{localized_text("transcribe_fail", bot_language)}: Could not convert media to MP3.',
                        parse_mode=enums.ParseMode.HTML,
                        reply_parameters=is_quoting_enabled(self.config, message),
                    )
                    return

                self.logger.info(
                    f'New transcribe request received from user {extract_username(message.from_user)} '
                    f'(id: {message.from_user.id})'
                )

                user_id = message.from_user.id
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, extract_username(message.from_user))

                transcript = await self.openai.transcribe(mp3_file, transcribe_user_prompt)

                transcription_price = self.config['transcription_price']
                self.usage[user_id].add_transcription_seconds(duration, transcription_price)

                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage['guests'].add_transcription_seconds(duration, transcription_price)

                    # check if transcript starts with any of the prefixes
                    response_to_transcription = any(
                        transcript.lower().startswith(prefix.lower()) if prefix else False
                        for prefix in self.config['voice_reply_prompts']
                    )

                    if self.config['voice_reply_transcript'] and not response_to_transcription:
                        # Split into chunks of 4096 characters (Telegram's message limit)
                        transcript_output = f'<i>{localized_text("transcript", bot_language)}:</i>\n"{transcript}"'
                        chunks = split_into_chunks(transcript_output)

                        for index, transcript_chunk in enumerate(chunks):
                            sent_msg = await message.reply_text(
                                text=transcript_chunk,
                                parse_mode=enums.ParseMode.HTML,
                                reply_parameters=is_quoting_enabled(self.config, message) if index == 0 else None,
                                #message_thread_id=get_forum_thread_id(message)
                            )
                            self.save_reply(sent_msg, message)
                    else:
                        # when user input text after /stt command, for example if they want a more detailed transcriptions with timestamps or similar
                        if transcribe_user_prompt:
                            full_query = str(self.openai.config['stt_user_prompt']).format(transcript=transcript, transcribe_user_prompt=transcribe_user_prompt)
                        else:
                            full_query = transcript

                        # Get the response of the transcript
                        response, total_tokens = await self.openai.get_chat_response(
                            chat_id=ai_context_id, query=full_query, user_id=str(user_id)
                        )

                        self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                        if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                            self.usage['guests'].add_chat_tokens(total_tokens, self.config['token_price'])

                        # Split into chunks of 4096 characters (Telegram's message limit)
                        transcript_output = (
                            f'<i>{localized_text("transcript", bot_language)}:</i>\n"{transcript}"\n\n'
                            f'<i>{localized_text("answer", bot_language)}:</i>\n{response}'
                        )
                        chunks = split_into_chunks(transcript_output)

                        for index, transcript_chunk in enumerate(chunks):
                            sent_msg = await message.reply_text(
                                text=transcript_chunk,
                                parse_mode=enums.ParseMode.HTML,
                                link_preview_options=types.LinkPreviewOptions(is_disabled=True),
                                reply_parameters=is_quoting_enabled(self.config, message) if index == 0 else None,
                                #message_thread_id=get_forum_thread_id(message)
                            )
                            self.save_reply(sent_msg, message)

            except Exception as e:
                self.logger.exception(e)
                await message.reply_text(
                    text=f'{localized_text("transcribe_fail", bot_language)}: {str(e)}',
                    parse_mode=enums.ParseMode.HTML,
                    reply_parameters=is_quoting_enabled(self.config, message),
                    #message_thread_id=get_forum_thread_id(message)
                )

        await wrap_with_indicator(client, message, _execute, enums.ChatAction.TYPING)

    @with_conversation_lock
    async def vision(self, client: Client, message: Message, reply: Message = None):
        await self._vision_no_lock(client, message, reply)

    async def _vision_no_lock(self, client: Client, message: Message, reply: Message = None):
        """
        Interpret image using vision model.
        """
        if not self.config['enable_vision'] or not await self.check_allowed_and_within_budget(client, message):
            return

        ai_context_id = self.get_thread_id(message)
        chat_id = message.chat.id

        if reply is None:
            prompt = message.caption
        else:
            prompt = message_text(message)

        if reply is None and is_group_chat(message):
            if self.config['ignore_group_vision']:
                self.logger.info('Vision coming from group chat, ignoring...')
                return
            else:
                no_reply = (
                    message.reply_to_message is None
                    or message.reply_to_message.from_user.id != client.me.id
                )

                trigger_keyword = self.config['group_trigger_keyword']
                no_keyword = (prompt is None and trigger_keyword != '') or (
                    prompt is not None and not prompt.lower().startswith(trigger_keyword.lower())
                )

                if no_reply and no_keyword:
                    self.logger.info('Vision coming from group chat with wrong keyword, ignoring...')
                    return
        elif reply and is_group_chat(message):
            trigger_keyword = self.config['group_trigger_keyword']
            no_keyword = (prompt is None and trigger_keyword != '') or (
                prompt is not None and not prompt.lower().startswith(trigger_keyword.lower())
            )

            if no_keyword:
                self.logger.info('Vision coming from group chat with wrong keyword, ignoring...')
                return

        target_msg = reply if reply else message
        image = target_msg.photo or target_msg.document
        
        if not image:
            return

        async def _execute():
            bot_language = self.config['bot_language']
            total_tokens = 0

            try:
                # Pyrogram download_media
                temp_file = await client.download_media(image, in_memory=True)
                # temp_file is BytesIO
            except Exception as e:
                self.logger.exception(e)
                await message.reply_text(
                    text=(
                        f'{localized_text("media_download_fail", bot_language)[0]}: '
                        f'{str(e)}. {localized_text("media_download_fail", bot_language)[1]}'
                    ),
                    parse_mode=enums.ParseMode.HTML,
                    reply_parameters=is_quoting_enabled(self.config, message),
                    #message_thread_id=get_forum_thread_id(message)
                )
                return

            # convert jpg from telegram to png as understood by openai

            temp_file_png = io.BytesIO()

            try:
                original_image = Image.open(temp_file)

                original_image.save(temp_file_png, format='PNG')
                self.logger.info(
                    f'New vision request received from user {extract_username(message.from_user)} '
                    f'(id: {message.from_user.id})'
                )

            except Exception as e:
                self.logger.exception(e)

            user_id = message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, extract_username(message.from_user))

            if self.config['stream']:
                stream_response = self.openai.get_chat_response_stream(
                    chat_id=ai_context_id, image=encode_image(temp_file_png), query=prompt, user_id=str(user_id)
                )
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                processed_chunks = []  # Track which chunks have been processed
                is_group = is_group_chat(message)
                str_chat_id = str(chat_id)

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, message, content, self.save_reply)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        # Keep track of the last chunk as current content
                        content = stream_chunks[-1]

                        # Process any new complete chunks
                        for chunk_idx in range(len(processed_chunks), len(stream_chunks) - 1):
                            try:
                                # If we have a message already, edit it with the current complete chunk
                                if sent_message is not None:
                                    # Check rate limits before sending
                                    can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                    if not can_send:
                                        self.logger.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                        continue

                                    await edit_message_with_retry(
                                        client,
                                        chat_id,
                                        sent_message.id,
                                        stream_chunks[chunk_idx],
                                    )

                                # Create a new message for the next chunk (current content)
                                # Check rate limits before sending
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                if not can_send:
                                    self.logger.warning(f'Rate limit reached for chat {chat_id}, skipping new message')
                                    # Mark this chunk as processed anyway to avoid creating multiple messages later
                                    processed_chunks.append(chunk_idx)
                                    continue

                                sent_message = await message.reply_text(
                                    text=content if len(content) > 0 else '...',
                                    #message_thread_id=get_forum_thread_id(message)
                                )
                                self.save_reply(sent_message, message)
                                processed_chunks.append(chunk_idx)
                            except Exception as e:
                                self.logger.error(f'Error handling chunk: {e}')
                                pass

                        # If we've processed all complete chunks, continue streaming with the last chunk
                        if len(processed_chunks) == len(stream_chunks) - 1:
                            # We've handled all complete chunks, continue with normal streaming for the last chunk
                            pass
                        else:
                            # We still have unprocessed complete chunks, skip this iteration
                            continue

                    cutoff = get_stream_cutoff_values(message, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            # Check rate limits before sending first message
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                self.logger.warning(f'Rate limit reached for chat {chat_id}, waiting for next update')
                                continue

                            if sent_message is not None:
                                await client.delete_messages(
                                    chat_id=sent_message.chat.id,
                                    message_ids=sent_message.id,
                                )
                            sent_message = await message.reply_text(
                                text=content,
                                reply_parameters=is_quoting_enabled(self.config, message),
                                #message_thread_id=get_forum_thread_id(message)
                            )
                            self.save_reply(sent_message, message)
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            # Instead of waiting, check if we should update the message
                            should_update = tokens != 'not_finished' or self.rate_limiter.should_update(
                                str_chat_id, is_group, len(content), len(prev), cutoff
                            )

                            # If we shouldn't update, skip this iteration
                            if not should_update:
                                continue

                            # Otherwise, check rate limits and update if possible
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                self.logger.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                continue

                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(
                                client,
                                chat_id,
                                sent_message.id,
                                text=content,
                                markdown=use_markdown,
                            )

                        except FloodWait as e:
                            backoff += 5
                            await asyncio.sleep(e.value)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        # Add a small delay between updates
                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                try:
                    interpretation, total_tokens = await self.openai.get_chat_response(
                        ai_context_id, prompt, image=encode_image(temp_file_png), user_id=str(user_id)
                    )

                    try:
                        # Check rate limits before sending
                        is_group = is_group_chat(message)
                        str_chat_id = str(chat_id)
                        can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                        if can_send:
                            sent_msg = await message.reply_text(
                                text=interpretation,
                                parse_mode=enums.ParseMode.HTML,
                                reply_parameters=is_quoting_enabled(self.config, message),
                                #message_thread_id=get_forum_thread_id(message)
                            )
                            self.save_reply(sent_msg, message)
                        else:
                            # If rate limit reached, try without markdown
                            self.logger.warning(f'Rate limit reached for chat {chat_id}, trying again in 1 second')
                            await asyncio.sleep(1)
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                            if can_send:
                                sent_msg = await message.reply_text(
                                    text=interpretation,
                                    reply_parameters=is_quoting_enabled(self.config, message),
                                    #message_thread_id=get_forum_thread_id(message)
                                )
                                self.save_reply(sent_msg, message)
                            else:
                                self.logger.error('Failed to send vision response due to rate limits')
                    except BadRequest:
                        try:
                            # Check rate limits before retrying
                            is_group = is_group_chat(message)
                            str_chat_id = str(chat_id)
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                            if can_send:
                                sent_msg = await message.reply_text(
                                    text=interpretation,
                                    reply_parameters=is_quoting_enabled(self.config, message),
                                    #message_thread_id=get_forum_thread_id(message)
                                )
                                self.save_reply(sent_msg, message)
                            else:
                                self.logger.error('Failed to send vision response due to rate limits')
                        except Exception as e:
                            self.logger.exception(e)
                            await message.reply_text(
                                text=f'{localized_text("vision_fail", bot_language)}: {str(e)}',
                                parse_mode=enums.ParseMode.HTML,
                                reply_parameters=is_quoting_enabled(self.config, message),
                                #message_thread_id=get_forum_thread_id(message)
                            )
                except Exception as e:
                    self.logger.exception(e)
                    await message.reply_text(
                        text=f'{localized_text("vision_fail", bot_language)}: {str(e)}',
                        parse_mode=enums.ParseMode.HTML,
                        reply_parameters=is_quoting_enabled(self.config, message),
                        #message_thread_id=get_forum_thread_id(message)
                    )
            vision_token_price = self.config['vision_token_price']
            self.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)

            allowed_user_ids = self.config['allowed_user_ids'].split(',')
            if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                self.usage['guests'].add_vision_tokens(total_tokens, vision_token_price)

        await wrap_with_indicator(client, message, _execute, enums.ChatAction.TYPING)

    async def reaction(self, client: Client, update: types.MessageReactionUpdated):
        """
        React to incoming reactions and respond accordingly.
        """
        # Pyrogram MessageReactionUpdated
        reaction_msg_key = (update.chat.id, update.message_id)
        if reaction_msg_key not in self.bot_message_ids:
            # prevent action on non-bot messages
            # prevent action on old messages which are not in the memory anymore
            return

        if not update.new_reaction:
            return

        emoji_to_message = {
            '👍': 'Yes.',
            '👎': 'No.',
            '❤️': 'I really like this.',
            '🔥': 'This is awesome!',
            '🥰': 'So sweet, thanks!',
            '👏': 'Great idea!',
            '😁': 'Glad to hear that!',
            '🤔': 'Let me think...',
            '🤯': 'Wow, that blew my mind!',
            '😱': "That's shocking!",
            '🤬': 'This is outrageous!',
            '😢': "I'm sorry to hear that.",
            '🎉': 'Congratulations!',
            '🤩': 'Wow, impressive!',
            '🤮': "That's disgusting.",
            '💩': "That's really bad.",
            '🙏': 'Please, go on.',
            '👌': 'I agree, perfect.',
            '🕊': 'Peace and calm.',
            '🤡': 'Is this a joke?',
            '🥱': "I'm bored...",
            '🥴': "I don't quite understand.",
            '😍': "I'm thrilled!",
            '🐳': 'Interesting, tell me more.',
            '❤️‍🔥': 'True passion!',
            '🌚': 'Hmm, mysterious.',
            '🌭': 'Odd choice, but okay.',
            '💯': 'Totally support that.',
            '🤣': "Haha, that's funny!",
            '⚡': "That's very energetic!",
            '🍌': 'Unexpected!',
            '🏆': 'Great achievement!',
            '💔': "That's sad.",
            '🤨': 'That seems doubtful to me.',
            '😐': 'Neutral stance.',
            '🍓': 'I love it.',
            '🍾': 'Time to celebrate!',
            '💋': 'Sending love!',
            '🖕': "That's rude!",
            '😈': "Alright, let's play naughty.",
            '😴': 'I need to rest.',
            '😭': 'Very touching.',
            '🤓': 'Interesting fact, thanks!',
            '👻': 'Was there a ghost here?',
            '👨‍💻': "Let's code!",
            '👀': "I'm watching closely.",
            '🎃': 'Happy Halloween!',
            '🙈': "I didn't see that.",
            '😇': 'Good idea!',
            '😨': "That's scary.",
            '🤝': 'Agreed.',
            '✍': 'Noting it down.',
            '🤗': 'Hugs!',
            '🫡': 'Order received!',
            '🎅': 'Merry Christmas!',
            '🎄': 'Festive mood!',
            '☃': 'Winter wonderland.',
            '💅': 'Stylish indeed.',
            '🤪': 'A bit crazy?',
            '🗿': 'No emotions...',
            '🆒': 'Very cool!',
            '💘': 'In love!',
            '🙉': "I don't want to hear that.",
            '🦄': 'Something magical!',
            '😘': 'Kisses!',
            '💊': 'Need some help?',
            '🙊': "Won't say a thing.",
            '😎': 'Cool and confident.',
            '👾': 'Exciting!',
            '🤷‍♂️': "Don't know what to say.",
            '🤷': 'No opinion yet.',
            '🤷‍♀️': "I'm not sure.",
            '😡': 'This annoys me.',
        }

        new_reactions = {r.emoji for r in update.new_reaction if isinstance(r, types.ReactionTypeEmoji)}

        if self.config.get('enable_raw_reaction'):
            reaction_prompt = self.openai.config.get('reaction_prompt')
            text_parts = []
            for emoji in new_reactions:
                try:
                    formatted = reaction_prompt.format(reaction=emoji)
                except Exception:
                    formatted = reaction_prompt.replace('{reaction}', emoji)
                
                text_parts.append(formatted)
            text = '\n'.join(text_parts)
        else:
            text = ''.join(emoji_to_message.get(emoji, '') for emoji in new_reactions)

        if not text.strip():
            return

        self.logger.info(f'New reaction received from user {extract_username(update.user)} (TEXT: {text})')

        fake_message = Message(
            id=update.message_id,
            date=update.date,
            chat=update.chat,
            from_user=update.user,
            text=text,
            reply_to_message=Message(
                id=update.message_id,
                date=update.date,
                chat=update.chat,
                from_user=client.me, # fake bot's message
            ),
            client=client
        )

        # now call with fake compatible update
        await self.prompt(client, fake_message)

    @with_conversation_lock
    async def prompt(self, client: Client, message: Message):
        await self._prompt_no_lock(client, message)

    async def _prompt_no_lock(self, client: Client, message: Message):
        """
        React to incoming messages and respond accordingly.
        """
        if message.edit_date or not message.text or message.via_bot:
            return

        if not await self.check_allowed_and_within_budget(client, message):
            return

        ai_context_id = self.get_thread_id(message)
        self.logger.info(f'New message received from user {extract_username(message.from_user)} (CTX: {ai_context_id})')
        chat_id = message.chat.id
        user_id = message.from_user.id
        prompt = message_text(message)
        self.last_message[chat_id] = prompt

        if message.reply_to_message and (message.reply_to_message.document or message.reply_to_message.photo):
            attachment = message.reply_to_message.document or message.reply_to_message.photo
            if isinstance(attachment, types.Document) and attachment.mime_type == 'application/pdf':
                message.reply_to_message.caption = prompt
                return await self.handle_pdf(client, message.reply_to_message)

            if message.reply_to_message.photo or message.reply_to_message.document:
                 return await self._vision_no_lock(client, message, message.reply_to_message)

        if is_group_chat(message):
            trigger_keyword = self.config['group_trigger_keyword']

            if prompt.lower().startswith(trigger_keyword.lower()) or message.text.lower().startswith('/chat'):
                if prompt.lower().startswith(trigger_keyword.lower()):
                    prompt = prompt[len(trigger_keyword) :].strip()

                if (
                    message.reply_to_message
                    and message.reply_to_message.text
                    and message.reply_to_message.from_user.id != client.me.id
                ):
                    reply_text = message_text(message.reply_to_message)
                    prompt = f'"{reply_text}"\n---\n{prompt}'
            else:
                if message.reply_to_message and message.reply_to_message.from_user.id == client.me.id:
                    self.logger.info('Message is a reply to the bot, allowing...')
                else:
                    self.logger.warning('Message does not start with trigger keyword, ignoring...')
                    return

        try:
            total_tokens = 0

            if self.config['stream']:
                await client.send_chat_action(
                    chat_id=message.chat.id,
                    action=enums.ChatAction.TYPING,
                    message_thread_id=get_forum_thread_id(message)
                )

                stream_response = self.openai.get_chat_response_stream(
                    chat_id=ai_context_id, query=prompt, user_id=str(user_id)
                )
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                processed_chunks = []  # Track which chunks have been processed
                is_group = is_group_chat(message)
                str_chat_id = str(chat_id)

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, message, content, self.save_reply)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        # Keep track of the last chunk as current content
                        content = stream_chunks[-1]

                        # Process any new complete chunks
                        for chunk_idx in range(len(processed_chunks), len(stream_chunks) - 1):
                            try:
                                # If we have a message already, edit it with the current complete chunk
                                if sent_message is not None:
                                    # Check rate limits before sending
                                    can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                    if not can_send:
                                        self.logger.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                        continue
    
                                    await edit_message_with_retry(
                                        client,
                                        chat_id,
                                        sent_message.id,
                                        stream_chunks[chunk_idx],
                                    )

                                # Create a new message for the next chunk (current content)
                                # Check rate limits before sending
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                if not can_send:
                                    self.logger.warning(f'Rate limit reached for chat {chat_id}, skipping new message')
                                    # Mark this chunk as processed anyway to avoid creating multiple messages later
                                    processed_chunks.append(chunk_idx)
                                    continue
    
                                sent_message = await message.reply_text(
                                    text=content if len(content) > 0 else '...',
                                    #message_thread_id=get_forum_thread_id(message)
                                )
                                self.save_reply(sent_message, message)
                                processed_chunks.append(chunk_idx)
                            except Exception as e:
                                self.logger.error(f'Error handling chunk: {e}')
                                pass
    
                        # If we've processed all complete chunks, continue streaming with the last chunk
                        if len(processed_chunks) == len(stream_chunks) - 1:
                            # We've handled all complete chunks, continue with normal streaming for the last chunk
                            pass
                        else:
                            # We still have unprocessed complete chunks, skip this iteration
                            continue

                    cutoff = get_stream_cutoff_values(message, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            # Check rate limits before sending first message
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                self.logger.warning(f'Rate limit reached for chat {chat_id}, waiting for next update')
                                continue

                            if sent_message is not None:
                                await client.delete_messages(
                                    chat_id=sent_message.chat.id,
                                    message_ids=sent_message.id,
                                )
                            sent_message = await message.reply_text(
                                text=content,
                                reply_parameters=is_quoting_enabled(self.config, message),
                                #message_thread_id=get_forum_thread_id(message)
                            )
                            self.save_reply(sent_message, message)
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            # Instead of waiting, check if we should update the message
                            should_update = tokens != 'not_finished' or self.rate_limiter.should_update(
                                str_chat_id, is_group, len(content), len(prev), cutoff
                            )

                            # If we shouldn't update, skip this iteration
                            if not should_update:
                                continue

                            # Otherwise, check rate limits and update if possible
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                self.logger.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                continue
    
                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(
                                client,
                                chat_id,
                                sent_message.id,
                                text=content,
                                markdown=use_markdown,
                            )

                        except FloodWait as e:
                            backoff += 5
                            await asyncio.sleep(e.value)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        # Add a small delay between updates
                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:

                async def _reply():
                    nonlocal total_tokens
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=ai_context_id, query=prompt, user_id=str(user_id)
                    )

                    if is_direct_result(response):
                        return await handle_direct_result(self.config, message, response, self.save_reply)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = split_into_chunks(response)

                    # Check if we're in a group
                    is_group = is_group_chat(message)
                    str_chat_id = str(chat_id)

                    for index, chunk in enumerate(chunks):
                        try:
                            # Check rate limits before sending
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                # If rate limit reached, add a delay and notify
                                self.logger.warning(f'Rate limit reached for chat {chat_id}, waiting...')
                                if index > 0:
                                    # Only add this notification for subsequent chunks
                                    await message.reply_text(
                                        text='⚠️ Rate limit reached. Remaining response will be sent shortly.',
                                        #message_thread_id=get_forum_thread_id(message)
                                    )
                                await asyncio.sleep(60)  # Wait for a minute
                                # Try again after waiting
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                            if can_send:
                                sent_msg = await message.reply_text(
                                    text=chunk,
                                    parse_mode=enums.ParseMode.HTML,
                                    link_preview_options=types.LinkPreviewOptions(is_disabled=True),
                                    reply_parameters=is_quoting_enabled(self.config, message) if index == 0 else None,
                                    #message_thread_id=get_forum_thread_id(message)
                                )
                                self.save_reply(sent_msg, message)
                            else:
                                self.logger.error('Failed to send chunk due to rate limits even after waiting')
                        except Exception:
                            try:
                                # Check rate limits before retrying
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                if not can_send:
                                    self.logger.warning(f'Rate limit reached for chat {chat_id}, skipping chunk')
                                    continue

                                sent_msg = await message.reply_text(
                                    text=chunk,
                                    link_preview_options=types.LinkPreviewOptions(is_disabled=True),
                                    reply_parameters=is_quoting_enabled(self.config, message) if index == 0 else None,
                                    #message_thread_id=get_forum_thread_id(message)
                                )
                                self.save_reply(sent_msg, message)
                            except Exception as exception:
                                raise exception

                await wrap_with_indicator(client, message, _reply, enums.ChatAction.TYPING)

            add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            self.logger.exception(e)
            await message.reply_text(
                text=f'{localized_text("chat_fail", self.config["bot_language"])} {str(e)}',
                parse_mode=enums.ParseMode.HTML,
                reply_parameters=is_quoting_enabled(self.config, message),
                #message_thread_id=get_forum_thread_id(message)
            )

    async def inline_query(self, client: Client, inline_query: InlineQuery) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        query = inline_query.query
        user_id = inline_query.from_user.id
        name = extract_username(inline_query.from_user)

        if len(query) < 3:
            return

        if not await self.check_allowed_and_within_budget(client, inline_query, is_inline=True):
            self.logger.warning(f'User {name} (id: {user_id}) not allowed or over budget')
            return

        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        await self.send_inline_query_result(client, inline_query, result_id, message_content=query)

    async def send_inline_query_result(self, client: Client, inline_query: InlineQuery, result_id, message_content, callback_data=''):
        """
        Send inline query result with a placeholder message that will be updated with the actual response
        """
        try:
            bot_language = self.config['bot_language']
            loading_tr = localized_text('loading', bot_language)
            answer_tr = localized_text('answer', bot_language)

            placeholder_text = f'{message_content}\n\n<i>{answer_tr}:</i>\n{loading_tr}'

            # Add a placeholder button
            reply_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton('⏳ Generating...', callback_data='generating')]]
            )

            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text('ask_chatgpt', bot_language),
                input_message_content=InputTextMessageContent(placeholder_text, parse_mode=enums.ParseMode.HTML),
                description=message_content,
                thumbnail_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea-b02a7a32149a.png',
                reply_markup=reply_markup,
            )

            await inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            self.logger.error(f'Failed to send inline result for result_id {result_id}: {str(e)}')
            self.logger.exception(e)

    async def handle_chosen_inline_result(self, client: Client, chosen_inline_result: ChosenInlineResult) -> None:
        """
        Handle the chosen inline result and generate the response
        """
        if not chosen_inline_result:
            self.logger.warning('Received empty chosen_inline_result')
            return

        result_id = chosen_inline_result.result_id
        inline_message_id = chosen_inline_result.inline_message_id
        user_id = chosen_inline_result.from_user.id
        name = extract_username(chosen_inline_result.from_user)

        # Retrieve the query from cache
        query = self.inline_queries_cache.get(result_id)
        if not query:
            self.logger.error(f'Query not found in cache for result_id: {result_id}')
            error_message = f'{localized_text("error", self.config["bot_language"])}. {localized_text("try_again", self.config["bot_language"])}'
            await edit_message_with_retry(
                client, chat_id=None, message_id=inline_message_id, text=error_message, is_inline=True
            )
            return

        self.logger.info(f'User {name} (id: {user_id}) selected result_id: {result_id} ({query})')
        self.inline_queries_cache.pop(result_id)

        bot_language = self.config['bot_language']
        answer_tr = localized_text('answer', bot_language)
        loading_tr = localized_text('loading', bot_language)
        total_tokens = 0
        str_user_id = str(user_id)  # Use user_id as chat_id for inline messages

        try:
            if self.config['stream']:
                stream_response = self.openai.get_chat_response_stream(
                    chat_id=str(user_id), query=query, user_id=str(user_id)
                )
                i = 0
                prev = ''
                backoff = 0
                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        self.logger.info('Received direct result, not supported in inline mode')
                        unavailable_message = localized_text('function_unavailable_in_inline_mode', bot_language)
                        await edit_message_with_retry(
                            client,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=f'{query}\n\n<i>{answer_tr}:</i>\n{unavailable_message}',
                            is_inline=True,
                        )
                        return

                    if len(content.strip()) == 0:
                        continue

                    cutoff = get_stream_cutoff_values(chosen_inline_result, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            # Check rate limits before sending first update
                            can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                            if not can_send:
                                self.logger.warning(f'Rate limit reached for user {user_id}, waiting for next update')
                                continue

                            await edit_message_with_retry(
                                client,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=f'{query}\n\n{answer_tr}:\n{content}',
                                is_inline=True,
                            )
                        except Exception:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content
                        try:
                            # Instead of waiting, check if we should update the message
                            should_update = tokens != 'not_finished' or self.rate_limiter.should_update(
                                str_user_id, False, len(content), len(prev), cutoff
                            )

                            # If we shouldn't update, skip this iteration
                            if not should_update:
                                continue

                            # Check rate limits before updating message
                            can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                            if not can_send:
                                self.logger.warning(f'Rate limit reached for user {user_id}, skipping update')
                                continue

                            use_markdown = tokens != 'not_finished'
                            text = f'{query}\n\n<i>{answer_tr}:</i>\n{content}'

                            # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                            text = text[:4096]

                            await edit_message_with_retry(
                                client,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=text,
                                markdown=use_markdown,
                                is_inline=True,
                            )

                        except FloodWait as e:
                            backoff += 5
                            await asyncio.sleep(e.value)
                            continue
                        except Exception:
                            backoff += 5
                            continue

                        # Add delay between updates to respect rate limits
                        await asyncio.sleep(0.1)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                # Show loading message
                # Check rate limits before sending
                can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                if can_send:
                    await client.edit_message_text(
                        inline_message_id=inline_message_id,
                        text=f'{query}\n\n<i>{answer_tr}:</i>\n{loading_tr}',
                        parse_mode=enums.ParseMode.HTML,
                    )

                response, total_tokens = await self.openai.get_chat_response(
                    chat_id=str(user_id), query=query, user_id=str(user_id)
                )

                if is_direct_result(response):
                    self.logger.info('Received direct result, not supported in inline mode')
                    unavailable_message = localized_text('function_unavailable_in_inline_mode', bot_language)

                    # Check rate limits before sending final message
                    can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                    if can_send:
                        await edit_message_with_retry(
                            client,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=f'{query}\n\n<i>{answer_tr}:</i>\n{unavailable_message}',
                            is_inline=True,
                        )
                    return

                text_content = f'{query}\n\n<i>{answer_tr}:</i>\n{response}'
                # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                text_content = text_content[:4096]

                # Check rate limits before sending final message
                can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                if can_send:
                    await edit_message_with_retry(
                        client,
                        chat_id=None,
                        message_id=inline_message_id,
                        text=text_content,
                        is_inline=True,
                    )
                else:
                    self.logger.warning(f'Rate limit reached for user {user_id}, waiting to send final response')
                    await asyncio.sleep(1)  # Wait a bit
                    # Try one more time
                    can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                    if can_send:
                        await edit_message_with_retry(
                            client,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=text_content,
                            is_inline=True,
                        )

            add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            self.logger.error(f'Failed to respond to an inline query: {str(e)}')
            self.logger.exception(e)
            localized_answer = localized_text('chat_fail', self.config['bot_language'])
            await edit_message_with_retry(
                client,
                chat_id=None,
                message_id=inline_message_id,
                text=f'{query}\n\n<i>{answer_tr}:</i>\n{localized_answer} {str(e)}',
                is_inline=True,
            )

    async def check_allowed_and_within_budget(
        self, client: Client, update: [Message, InlineQuery], is_inline=False
    ) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :param is_inline: Boolean flag for inline queries
        :return: Boolean indicating if the user is allowed to use the bot
        """
        name = extract_username(update.from_user)
        user_id = update.from_user.id

        if not await is_allowed(self.config, client, update, is_inline=is_inline):
            self.logger.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            return False
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            self.logger.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(client, update, is_inline)
            return False

        return True

    async def send_disallowed_message(self, client: Client, update: [Message, InlineQuery], is_inline=False):
        """
        Sends the disallowed message to the user.
        """
        if not is_inline:
            await update.reply_text(
                text=self.disallowed_message,
                link_preview_options=types.LinkPreviewOptions(is_disabled=True),
                #message_thread_id=get_forum_thread_id(update)
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(client, update, result_id, message_content=self.disallowed_message)

    async def send_budget_reached_message(self, client: Client, update: [Message, InlineQuery], is_inline=False):
        """
        Sends the budget reached message to the user.
        """
        if not is_inline:
            await update.reply_text(
                text=self.budget_limit_message,
                #message_thread_id=get_forum_thread_id(update)
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(client, update, result_id, message_content=self.budget_limit_message)

    async def post_init(self, client: Client) -> None:
        """
        Post initialization hook for the bot.
        """
        await client.set_bot_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await client.set_bot_commands(self.commands)

        if self.config['database_url']:
            self.openai.db_pool = await asyncpg.create_pool(dsn=self.config['database_url'])
            async with self.openai.db_pool.acquire() as connection:
                await connection.execute('drop schema public cascade')
                await connection.execute('create schema public')

    async def post_shutdown(self, client: Client) -> None:
        if self.openai.db_pool:
            await self.openai.db_pool.close()

    def _convert_media_to_mp3(self, media_file: io.BytesIO) -> tuple[Optional[io.BytesIO], float]:
        """
        Converts a media file to MP3 format.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_input:
                temp_input.write(media_file.read())
                temp_input_path = temp_input.name

            temp_output_path = temp_input_path + ".mp3"

            try:
                audio = AudioSegment.from_file(temp_input_path)
                duration = audio.duration_seconds
                audio.export(temp_output_path, format="mp3")

                if os.path.exists(temp_output_path):
                    with open(temp_output_path, 'rb') as f:
                        mp3_data = f.read()
                    
                    mp3_file = io.BytesIO(mp3_data)
                    mp3_file.name = "audio.mp3"
                    return mp3_file, duration
                return None, 0.0
            finally:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        except Exception as e:
            self.logger.error(f"Error converting media to MP3: {str(e)}")
            return None, 0.0

    def _convert_tgs_to_webm(self, tgs_data: bytes) -> Optional[bytes]:
        """
        Converts TGS (Lottie JSON) data to WEBM using lottie[video].
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.tgs', delete=False) as temp_input:
                temp_input.write(tgs_data)
                temp_input_path = temp_input.name

            temp_output_path = temp_input_path + ".webm"

            try:
                # Parse TGS
                with open(temp_input_path, 'rb') as f:
                    anim = parse_tgs(f)
                
                # Export to Video
                export_video(anim, temp_output_path, format="webm")

                if os.path.exists(temp_output_path):
                    with open(temp_output_path, 'rb') as f:
                        mp4_data = f.read()
                    return mp4_data
                return None
            finally:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        except Exception as e:
            self.logger.error(f"Error converting TGS to MP4: {str(e)}")
            return None

    async def _handle_multimodal_input(self, client: Client, message: Message, media_type: str = None) -> bool:
        """
        Unified handler for multimodal input (audio, video, pdf, stickers, animations).
        Returns True if handled (or rejected due to budget), False if not supported.
        """
        media = (
            message.audio
            or message.voice
            or message.video
            or message.video_note
            or message.document
            or message.sticker
            or message.animation
        )
        if not media:
            return False

        # Determine media type if not provided
        if not media_type:
            if message.audio or message.voice:
                media_type = 'audio'
            elif message.video or message.video_note or message.animation:
                media_type = 'video'
            elif message.sticker:
                if message.sticker.is_animated or message.sticker.is_video:
                    media_type = 'video'
                else:
                    media_type = 'image'
            elif message.document:
                mime = (media.mime_type or '').lower()
                if mime == 'application/pdf':
                    media_type = 'pdf'
                elif mime.startswith('audio/'):
                    media_type = 'audio'
                elif mime.startswith('video/'):
                    media_type = 'video'
                elif mime.startswith('image/'):
                    media_type = 'image'
                else:
                    return False  # Unknown document type

        # Check if media type is supported
        # For images (stickers), we check enable_vision instead of supported_input
        if media_type == 'image':
            if not self.config.get('enable_vision', False):
                return False
        elif media_type not in self.openai.config['supported_input']:
            return False

        if not await self.check_allowed_and_within_budget(client, message):
            return True  # Handled (rejected)

        self.logger.info(
            f'New {media_type} request received from user {extract_username(message.from_user)} (id: {message.from_user.id})'
        )

        async def _execute():
            bot_language = self.config['bot_language']
            try:
                temp_file = await client.download_media(media, in_memory=True)
                temp_file.seek(0)
                media_bytes = temp_file.read()
                # Check if we need to convert TGS to MP4
                if message.sticker and message.sticker.is_animated:
                    converted_bytes = await asyncio.to_thread(self._convert_tgs_to_webm, media_bytes)
                    if converted_bytes:
                        media_bytes = converted_bytes
                    else:
                        self.logger.warning("TGS conversion failed, sending original data.")

                media_base64 = base64.b64encode(media_bytes).decode('utf-8')

                # Use actual mime type from Telegram object
                mime_type = getattr(media, 'mime_type', '')

                # Fallback for voice/video_note/sticker/animation if mime_type is missing
                if not mime_type:
                    if message.voice:
                        mime_type = 'audio/ogg'
                    elif message.video_note:
                        mime_type = 'video/mp4'
                    elif message.sticker:
                        mime_type = 'image/webp'
                    elif message.animation:
                        mime_type = 'video/mp4'

                # Fallback format if still empty
                if not mime_type:
                    if media_type == 'audio':
                        mime_type = 'audio/mp3'
                    elif media_type == 'video':
                        mime_type = 'video/mp4'
                    elif media_type == 'pdf':
                        mime_type = 'application/pdf'
                    elif media_type == 'image':
                        mime_type = 'image/webp'

                kwargs = {
                    'chat_id': self.get_thread_id(message),
                    'query': message.caption or message.text or "",
                    'user_id': str(message.from_user.id),
                }

                if media_type == 'image':
                    kwargs['image'] = f"data:{mime_type};base64,{media_base64}"
                else:
                    kwargs[media_type] = {'data': media_base64, 'format': mime_type}

                response, total_tokens = await self.openai.get_chat_response(**kwargs)

                chunks = split_into_chunks(response)
                for index, chunk in enumerate(chunks):
                    sent_msg = await message.reply_text(
                        text=chunk,
                        parse_mode=enums.ParseMode.HTML,
                        reply_parameters=is_quoting_enabled(self.config, message) if index == 0 else None,
                    )
                    self.save_reply(sent_msg, message)

                user_id = message.from_user.id
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, extract_username(message.from_user))
                self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])

            except Exception as e:
                self.logger.exception(e)
                await message.reply_text(
                    text=f'{localized_text("error", bot_language)}: {str(e)}',
                    parse_mode=enums.ParseMode.HTML,
                    reply_parameters=is_quoting_enabled(self.config, message),
                )

        await wrap_with_indicator(client, message, _execute, enums.ChatAction.TYPING)
        return True

    @with_conversation_lock
    async def handle_media(self, client: Client, message: Message):
        """
        Unified handler for media messages (audio, video, document).
        """
        # Try to handle as multimodal input first
        if await self._handle_multimodal_input(client, message):
            return

        # Fallback logic for PDF: Manual Text Extraction
        if message.document and message.document.mime_type == 'application/pdf':
            await self._handle_pdf_legacy(client, message)
            return

        # Handle image documents via vision
        if message.document and (message.document.mime_type or '').startswith('image/'):
            await self._vision_no_lock(client, message)

    async def _handle_pdf_legacy(self, client: Client, message: Message):
        """
        Extract text from PDF files and process as prompt.
        Legacy method for when PDF is not supported as multimodal input.
        """
        if not await self.check_allowed_and_within_budget(client, message):
            return

        caption = message.caption or ''
        if is_group_chat(message):
            trigger_keyword = self.config['group_trigger_keyword']

            if not caption.lower().startswith(trigger_keyword.lower()):
                # If it's a reply to bot, allow, otherwise ignore
                if message.reply_to_message and message.reply_to_message.from_user.id == client.me.id:
                    self.logger.info('PDF is a reply to the bot, allowing...')
                else:
                    self.logger.warning('PDF caption does not start with trigger keyword, ignoring...')
                    return

        self.logger.info(f'New PDF received from user {extract_username(message.from_user)} (id: {message.from_user.id})')

        async def _process_pdf():
            try:
                # Pyrogram download_media
                temp_path = await client.download_media(message.document)

                extracted_text = ''
                try:
                    reader = PdfReader(temp_path)
                    number_of_pages = len(reader.pages)

                    for i in range(number_of_pages):
                        page = reader.pages[i]
                        page_text = page.extract_text() or ''
                        extracted_text += f'[Page number {i}]' + page_text + '\n\n'

                        # Limit the total text to prevent excessive token usage
                        if len(extracted_text) > 30000:
                            extracted_text = extracted_text[:30000]
                            extracted_text += '\n\n[Text truncated due to length]'
                            break

                except Exception as e:
                    self.logger.exception(e)
                    await message.reply_text(
                        text=f'Error extracting text from PDF: {str(e)}',
                        #message_thread_id=get_forum_thread_id(message)
                    )
                    return
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                if not extracted_text.strip():
                    await message.reply_text(
                        text='No text could be extracted from the PDF. It might be scanned or contain only images.',
                        #message_thread_id=get_forum_thread_id(message)
                    )
                    return

                prompt = (
                    f'{caption}\n---\nPDF Content:\n{extracted_text}' if caption else f'PDF Content:\n{extracted_text}'
                )

                # Modify message text to be the prompt
                message.text = prompt

                await self._prompt_no_lock(client, message)

            except Exception as e:
                self.logger.exception(e)
                await message.reply_text(
                    text=f'Failed to process PDF: {str(e)}',
                    #message_thread_id=get_forum_thread_id(message)
                )

        await wrap_with_indicator(client, message, _process_pdf, enums.ChatAction.TYPING)

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        # Register handlers
        self.client.add_handler(MessageHandler(self.reset, filters.command("reset")))
        self.client.add_handler(MessageHandler(self.image, filters.command("image")))
        self.client.add_handler(MessageHandler(self.tts, filters.command("tts")))
        self.client.add_handler(MessageHandler(self.transcribe, filters.command("stt")))

        self.client.add_handler(MessageHandler(self.vision, filters.photo))
        self.client.add_handler(
            MessageHandler(
                self.handle_media,
                filters.audio
                | filters.voice
                | filters.video
                | filters.video_note
                | filters.document
                | filters.sticker
                | filters.animation,
            )
        )
        
        # should be similar to PTB filters.COMMAND
        async def is_any_command(_, __, message):
            text = message.text or message.caption
            if not isinstance(text, str) or not text.startswith("/"):
                return False
            cmd = text.split()[0][1:]
            if "@" in cmd:
                cmd_name, bot = cmd.split("@", 1)
                return bot.lower() == message._client.name.lower()
            return True
        
        self.client.add_handler(MessageHandler(self.prompt, filters.text & ~filters.create(is_any_command)))
        
        self.client.add_handler(
            MessageReactionUpdatedHandler(
                self.reaction
            )
        )
        
        self.client.add_handler(CallbackQueryHandler(self.handle_improve_quality, filters.regex('^improve_quality:')))
        self.client.add_handler(CallbackQueryHandler(self.handle_show_quality, filters.regex('^show_quality:')))
        self.client.add_handler(CallbackQueryHandler(self.handle_quality_confirmation, filters.regex('^confirm_quality:')))
        self.client.add_handler(CallbackQueryHandler(self.handle_quality_cancel, filters.regex('^cancel_quality:')))
        
        self.client.add_handler(
            InlineQueryHandler(
                self.inline_query
            )
        )
        
        self.client.add_handler(ChosenInlineResultHandler(self.handle_chosen_inline_result))

        # Start the client
        self.logger.info("Starting bot...")
        self.client.run()

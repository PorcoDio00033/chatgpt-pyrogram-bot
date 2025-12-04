from __future__ import annotations

import asyncio
import base64
import itertools
import json
from typing import Callable, Optional, Union

from pyrogram import Client, enums, types
from pyrogram.errors import BadRequest, MessageNotModified
from usage_tracker import UsageTracker

from chill_logging import get_logger_instance

logger = get_logger_instance("utils").logger


def message_text(message: types.Message) -> str:
    """
    Returns the text of a message, excluding any bot commands.
    """
    text = message.text or message.caption or ""
    if not text:
        return ""

    entities = message.entities or message.caption_entities or []
    
    # Sort entities by offset in descending order to remove them without affecting other offsets
    for entity in sorted(entities, key=lambda x: x.offset, reverse=True):
        if entity.type == enums.MessageEntityType.BOT_COMMAND:
            text = text[:entity.offset] + text[entity.offset + entity.length:]

    return text.strip()


async def is_user_in_group(client: Client, chat_id: int, user_id: int) -> bool:
    """
    Checks if user_id is a member of the group
    """
    try:
        chat_member = await client.get_chat_member(chat_id, user_id)
        return chat_member.status in [
            enums.ChatMemberStatus.OWNER,
            enums.ChatMemberStatus.ADMINISTRATOR,
            enums.ChatMemberStatus.MEMBER,
        ]
    except BadRequest as e:
        if 'User not found' in str(e):
            return False
        else:
            raise e
    except Exception as e:
        raise e


def get_forum_thread_id(message: Union[types.Message, types.Update]) -> int | None:
    """
    Gets the message thread id for the update, if any
    """
    if isinstance(message, types.Message):
        return message.message_thread_id

    return None


def get_stream_cutoff_values(update: types.Message, content: str) -> int:
    """
    Gets the stream cutoff values for the message length
    """
    if is_group_chat(update):
        # group chats have stricter flood limits
        return 180 if len(content) > 1000 else 120 if len(content) > 200 else 90 if len(content) > 50 else 50
    return 90 if len(content) > 1000 else 45 if len(content) > 200 else 25 if len(content) > 50 else 15


def is_group_chat(message: types.Message) -> bool:
    """
    Checks if the message was sent from a group chat
    """
    if not hasattr(message, 'chat') or not message.chat:
        return False
    return message.chat.type in [
        enums.ChatType.GROUP,
        enums.ChatType.SUPERGROUP,
    ]


def is_private_chat(message: types.Message) -> bool:
    """
    Checks if the message was sent from a private chat
    """
    if not message.chat:
        return False
    return message.chat.type == enums.ChatType.PRIVATE


def split_into_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """
    Splits a string into chunks of a given size.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


async def wrap_with_indicator(
    client: Client,
    message: types.Message,
    coroutine,
    chat_action: enums.ChatAction = enums.ChatAction.TYPING,
    is_inline=False,
):
    """
    Wraps a coroutine while repeatedly sending a chat action to the user.
    """
    # Create the task for the coroutine
    task = asyncio.create_task(coroutine())
    
    while not task.done():
        if not is_inline:
            try:
                await client.send_chat_action(
                    chat_id=message.chat.id,
                    action=chat_action,
                    message_thread_id=get_forum_thread_id(message)
                )
            except Exception:
                # Ignore errors sending chat action (e.g. if bot was kicked)
                pass
                
        try:
            await asyncio.wait_for(asyncio.shield(task), 4.5)
        except asyncio.TimeoutError:
            pass
    
    return await task


async def edit_message_with_retry(
    client: Client,
    chat_id: int | None,
    message_id: int,
    text: str,
    markdown: bool = True,
    is_inline: bool = False,
):
    """
    Edit a message with retry logic in case of failure (e.g. broken markdown)
    :param context: The context to use
    :param chat_id: The chat id to edit the message in
    :param message_id: The message id to edit
    :param text: The text to edit the message with
    :param markdown: Whether to use markdown parse mode
    :param is_inline: Whether the message to edit is an inline message
    :return: None
    """
    try:
        kwargs = {
            "text": text,
            "parse_mode": enums.ParseMode.DEFAULT, # pyrogram DEFAULT=markdown+html
            #"parse_mode": enums.ParseMode.HTML if markdown else enums.ParseMode.DEFAULT,
            "link_preview_options": types.LinkPreviewOptions(is_disabled=True),
        }
        if is_inline:
            await client.edit_inline_text(inline_message_id=str(message_id), **kwargs)
        else:
            await client.edit_message_text(chat_id=chat_id, message_id=message_id, **kwargs)
    except MessageNotModified:
        return
    except BadRequest as e:
        # Fallback to plain text if markdown fails
        try:
            kwargs["parse_mode"] = None
            if is_inline:
                await client.edit_inline_text(inline_message_id=str(message_id), **kwargs)
            else:
                await client.edit_message_text(chat_id=chat_id, message_id=message_id, **kwargs)
        except Exception as e:
            logger.warning(f'Failed to edit message: {str(e)}')
            raise e
    except Exception as e:
        logger.warning(str(e))
        raise e


async def error_handler(client: Client, update: types.Update, error: Exception) -> None:
    """
    Handles errors.
    """
    logger.error(f'Exception while handling an update: {error}', exc_info=error)


async def is_allowed(config, client: Client, update: Union[types.Message, types.InlineQuery], is_inline=False) -> bool:
    """
    Checks if the user is allowed to use the bot.
    """
    if config['allowed_user_ids'] == '*':
        return True

    user = update.from_user
    if not user:
        return False
        
    user_id = user.id
    if is_admin(config, user_id):
        return True
        
    name = extract_username(user)
    allowed_user_ids = config['allowed_user_ids'].split(',')
    
    # Check if user is allowed
    if str(user_id) in allowed_user_ids:
        return True
        
    # Check if it's a group chat with at least one authorized member
    # Note: update is usually Message here if not inline
    if not is_inline and isinstance(update, types.Message) and is_group_chat(update):
        admin_user_ids = config['admin_user_ids'].split(',')
        for allowed_user in itertools.chain(allowed_user_ids, admin_user_ids):
            if not allowed_user.strip():
                continue
            try:
                # Check if the allowed user is in this group
                if await is_user_in_group(client, update.chat.id, int(allowed_user)):
                    logger.info(f'{allowed_user} is a member. Allowing group chat message...')
                    return True
            except ValueError:
                continue
                
        logger.info(f'Group chat messages from user {name} (id: {user_id}) are not allowed')
    return False


def is_admin(config, user_id: int, log_no_admin=False) -> bool:
    """
    Checks if the user is the admin of the bot.
    The first user in the user list is the admin.
    """
    if config['admin_user_ids'] == '-':
        if log_no_admin:
            logger.info('No admin user defined.')
        return False

    admin_user_ids = config['admin_user_ids'].split(',')

    # Check if user is in the admin user list
    if str(user_id) in admin_user_ids:
        return True

    return False


def has_image_gen_permission(config, user_id: int) -> bool:
    if config['img_gen_access_user_ids'] == '-':
        return False

    img_gen_access_user_ids = config['img_gen_access_user_ids'].split(',')
    if str(user_id) in img_gen_access_user_ids:
        return True

    return False


def get_user_budget(config, user_id) -> float | None:
    """
    Get the user's budget based on their user ID and the bot configuration.
    :param config: The bot configuration object
    :param user_id: User id
    :return: The user's budget as a float, or None if the user is not found in the allowed user list
    """

    # no budget restrictions for admins and '*'-budget lists
    if is_admin(config, user_id) or config['user_budgets'] == '*':
        return float('inf')

    user_budgets = config['user_budgets'].split(',')
    if config['allowed_user_ids'] == '*':
        # same budget for all users, use value in first position of budget list
        if len(user_budgets) > 1:
            logger.warning(
                'multiple values for budgets set with unrestricted user list '
                'only the first value is used as budget for everyone.'
            )
        return float(user_budgets[0])

    allowed_user_ids = config['allowed_user_ids'].split(',')
    if str(user_id) in allowed_user_ids:
        user_index = allowed_user_ids.index(str(user_id))
        if len(user_budgets) <= user_index:
            logger.warning(f'No budget set for user id: {user_id}. Budget list shorter than user list.')
            return 0.0
        return float(user_budgets[user_index])
    return None


def get_remaining_budget(config, usage, update: Union[types.Message, types.InlineQuery], is_inline=False) -> float:
    """
    Calculate the remaining budget for a user based on their current usage.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: The remaining budget for the user as a float
    """
    # Mapping of budget period to cost period
    budget_cost_map = {
        'monthly': 'cost_month',
        'daily': 'cost_today',
        'all-time': 'cost_all_time',
    }

    user = update.from_user
    user_id = user.id
    name = extract_username(user)
    
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)

    # Get budget for users
    user_budget = get_user_budget(config, user_id)
    budget_period = config['budget_period']
    if user_budget is not None:
        cost = usage[user_id].get_current_cost()[budget_cost_map[budget_period]]
        return user_budget - cost

    # Get budget for guests
    if 'guests' not in usage:
        usage['guests'] = UsageTracker('guests', 'all guest users in group chats')
    cost = usage['guests'].get_current_cost()[budget_cost_map[budget_period]]
    return config['guest_budget'] - cost


def is_within_budget(config, usage, update: Union[types.Message, types.InlineQuery], is_inline=False) -> bool:
    """
    Checks if the user reached their usage limit.
    Initializes UsageTracker for user and guest when needed.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: Boolean indicating if the user has a positive budget
    """
    user = update.from_user
    user_id = user.id
    name = extract_username(user)
    
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)
    remaining_budget = get_remaining_budget(config, usage, update, is_inline=is_inline)
    return remaining_budget > 0


def add_chat_request_to_usage_tracker(usage, config, user_id, used_tokens):
    """
    Add chat request to usage tracker
    :param usage: The usage tracker object
    :param config: The bot configuration object
    :param user_id: The user id
    :param used_tokens: The number of tokens used
    """
    try:
        if int(used_tokens) == 0:
            logger.warning('No tokens used. Not adding chat request to usage tracker.')
            return
        # add chat request to users usage tracker
        usage[user_id].add_chat_tokens(used_tokens, config['token_price'])
        # add guest chat request to guest usage tracker
        allowed_user_ids = config['allowed_user_ids'].split(',')
        if str(user_id) not in allowed_user_ids and 'guests' in usage:
            usage['guests'].add_chat_tokens(used_tokens, config['token_price'])
    except Exception as e:
        logger.warning(f'Failed to add tokens to usage_logs: {str(e)}')
        pass


# def get_reply_to_message_id(config, message: types.Message):
#     """
#     Returns the message id of the message to reply to
#     :param config: Bot configuration object
#     :param update: Telegram update object
#     :return: Message id of the message to reply to, or None if quoting is disabled
#     """
#     if config['enable_quoting'] or is_group_chat(message):
#         return message.id
#     return None

# pyrotgfork requires types.ReplyParameters() obj for quoting
def is_quoting_enabled(config, message: types.Message):
    if config['enable_quoting'] or is_group_chat(message):
        return types.ReplyParameters(message_id=message.id)
    return None

# similar to user.name of PTB
def extract_username(user: types.User):
    if isinstance(user, types.User):
        return user.first_name if not user.username else f"@{user.username}"
    return None


def is_direct_result(response: any) -> bool:
    """
    Checks if the dict contains a direct result that can be sent directly to the user
    :param response: The response value
    :return: Boolean indicating if the result is a direct result
    """
    if isinstance(response, list):
        # we do use lists to return multiple direct results from parallel function calls
        return True

    if not isinstance(response, dict):
        try:
            json_response = json.loads(response)
            return json_response.get('direct_result', False)
        except:
            return False
    else:
        return response.get('direct_result', False)


async def handle_direct_result(config, update: types.Message, response: any, save_reply: Optional[Callable] = None):
    if isinstance(response, list):
        for resp in response[:10]:  # limit to first 10 direct results to avoid flooding
            await __handle_direct_result(config, update, resp, save_reply)
    else:
        await __handle_direct_result(config, update, response, save_reply)


async def __handle_direct_result(config, message: types.Message, response: any, save_reply: Optional[Callable] = None):
    """
    Handles a direct result from a plugin
    """
    if not isinstance(response, dict):
        response = json.loads(response)

    result = response['direct_result']
    kind = result['kind']

    common_args = {
        # pyrotgfork reply_* auto handles message_thread_id
        #'message_thread_id': get_forum_thread_id(message),
        'reply_parameters': is_quoting_enabled(config, message),
    }

    sent_msg = None
    if kind == 'photo':
        sent_msg = await message.reply_photo(
            photo=result['photo'], caption=result.get('caption'), **common_args
        )
    elif kind == 'album':
        media = [types.InputMediaPhoto(media=photo) for photo in result['photos']]
        if 'caption' in result:
            media[0] = types.InputMediaPhoto(media=result['photos'][0], caption=result['caption'])

        sent_msgs = await message.reply_media_group(media=media, **common_args)
        sent_msg = sent_msgs[-1] if sent_msgs else None
    elif kind in {'gif', 'file', 'document'}:
        sent_msg = await message.reply_document(
            document=result['document'], caption=result.get('caption'), **common_args
        )
    elif kind == 'reaction':
        await message._client.set_reaction(chat_id=message.chat.id, message_id=message.id, reaction=result['reaction'])
    elif kind == 'voice':
        sent_msg = await message.reply_voice(
            voice=result['voice'], caption=result.get('caption'), **common_args
        )
    elif kind == 'video_note':
        sent_msg = await message.reply_video_note(
            video_note=result['video_note'], **common_args
        )
    elif kind == 'video':
        sent_msg = await message.reply_video(
            video=result['video'], caption=result.get('caption'), **common_args
        )
    elif kind == 'audio':
        sent_msg = await message.reply_audio(
            audio=result['audio'], caption=result.get('caption'), **common_args
        )
    elif kind == 'dice':
        sent_msg = await message._client.send_dice(chat_id=message.chat.id, emoji=result['emoji'], **common_args)
    elif kind == 'poll':
        sent_msg = await message.reply_poll(
            question=result['question'],
            options=result['options'],
            is_anonymous=result.get('is_anonymous', False),
            allows_multiple_answers=result.get('allows_multiple_answers', False),
            **common_args
        )
    elif kind == 'location':
        sent_msg = await message.reply_location(
            latitude=result['latitude'],
            longitude=result['longitude'],
            **common_args
        )
    elif kind == 'venue':
        sent_msg = await message.reply_venue(
            latitude=result['latitude'],
            longitude=result['longitude'],
            title=result['title'],
            address=result['address'],
            **common_args
        )
    elif kind == 'contact':
        sent_msg = await message.reply_contact(
            phone_number=result['phone_number'],
            first_name=result['first_name'],
            last_name=result['last_name'],
            **common_args
        )

    if save_reply and sent_msg:
        save_reply(sent_msg, message)


# Function to encode the image
def encode_image(fileobj):
    image = base64.b64encode(fileobj.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{image}'


def decode_image(imgbase64):
    image = imgbase64[len('data:image/jpeg;base64,') :]
    return base64.b64decode(image)



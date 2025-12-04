import functools
from typing import Callable, TypeVar, Union

from pyrogram import Client, types

T = TypeVar('T')


def with_conversation_lock(method: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that handles conversation locking for bot methods.
    It gets the thread ID and uses it to acquire a conversation lock before executing the method.
    """

    @functools.wraps(method)
    async def wrapper(self, client: Client, update: Union[types.Message, types.CallbackQuery, types.Update], *args, **kwargs):
        ai_context_id = self.get_thread_id(update)
        async with self.openai.get_conversation_lock(ai_context_id):
            return await method(self, client, update, *args, **kwargs)

    return wrapper

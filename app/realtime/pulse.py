# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""In-process publish/subscribe for the live pulse (one queue per SSE client)."""

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


class PulseBroker:
    def __init__(self, queue_size: int = 32) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._queue_size = queue_size

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: str, data: Any) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                # A slow client loses its oldest event rather than blocking the rest.
                queue.get_nowait()
                log.debug("Dropped a pulse event for a slow subscriber")
            queue.put_nowait((event, data))

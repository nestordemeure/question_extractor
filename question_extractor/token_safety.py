import time
import asyncio

class Throttler_token:
    """
    Throttler class for limiting the number of tokens per minute.
    """
    def __init__(self, rate):
        self.rate = rate
        self.start_time = time.time()
        self.tokens_consumed = 0
        self.lock = asyncio.Lock()

    async def acquire(self, tokens):
        """
        Method to acquire tokens and block until the rate limit allows processing.
        """
        async with self.lock:
            elapsed_time = time.time() - self.start_time
            available_tokens = self.rate * elapsed_time / 60 - self.tokens_consumed
            sleep_time = max(0, (tokens - available_tokens) * 60 / self.rate)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self.tokens_consumed += tokens

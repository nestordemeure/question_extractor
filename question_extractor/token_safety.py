import asyncio
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = asyncio.get_event_loop().time()

    async def consume(self, amount=1):
        while self._tokens < amount:
            await self._refill()
        self._tokens -= amount

    async def _refill(self):
        now = asyncio.get_event_loop().time()
        time_delta = now - self._last_refill
        refill_amount = time_delta * self.rate
        self._tokens = min(self.capacity, self._tokens + refill_amount)
        self._last_refill = now
        await asyncio.sleep(1/self.rate)
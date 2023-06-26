import time
import asyncio
import collections

class Throttler_token:
    """
    Throttler class for limiting the number of tokens per minute.
    """
    def __init__(self, rate):
        self.time_slot=5
        self.frac = self.time_slot/60
        self.rate = rate
        self.tokens_consumed = collections.deque()  # store tokens with their timestamps
        self.tokens_consumed.append((time.monotonic(),0))
        self.lock = asyncio.Lock()
        self.output_file = open("output.log", "w")  # output file

    async def acquire(self, tokens):
        """
        Method to acquire tokens and block until the rate limit allows processing.
        """
        async with self.lock:
            current_time = time.monotonic()
            sleep_time = 0
            flag=False
            # Remove tokens consumed more than 60 seconds ago
            while (current_time - self.tokens_consumed[0][0])>self.time_slot and (len(self.tokens_consumed)>1):
                self.tokens_consumed.popleft()
                flag=True
            if flag:
                # Calculate consumed tokens in the last 60 seconds
                tokens_used = sum(token for _,token in self.tokens_consumed)
                used_time = current_time - self.tokens_consumed[0][0]
                real_rate=60*tokens_used/used_time
                self.output_file.write(f"tokens_in_last_minute: {tokens_used}\n")
                self.output_file.write(f"used_time: {used_time}\n")
                self.output_file.write(f"real_rate: {real_rate}\n")
                sleep_time = min(max(0, (60*tokens_used/self.rate-used_time)),self.time_slot)
                self.output_file.write(f"Sleep Time:{sleep_time}\n")
                self.output_file.write('---------------------------------------------------------------------------------------------\n')

                print(f"tokens_in_last_minute: {tokens_used}")
                print(f"used_time: {used_time}")
                print(f"real_rate: {real_rate}")
                print(f"Sleep Time:{sleep_time}")
                print('---------------------------------------------------------------------------------------------')


                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Add new tokens to the consumed queue
            self.tokens_consumed.append((current_time + sleep_time,tokens))  # add sleep_time to reflect the real consumption time
    def __del__(self):
        self.output_file.close()

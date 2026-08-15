"""Queue that serializes access to the GPU: only one XTTS job runs at a time."""
import asyncio
import queue
import threading
from concurrent.futures import Future

import torch

from config import MAX_GPU_QUEUE_SIZE

_STREAM_END = object()


class GpuQueueFull(Exception):
    pass


class GpuOutOfMemory(Exception):
    pass


class GpuWorker:
    """Runs GPU jobs one at a time on a dedicated thread, to avoid saturating the card's memory."""

    def __init__(self):
        self._jobs = queue.Queue(maxsize=MAX_GPU_QUEUE_SIZE)
        threading.Thread(target=self._process_jobs, daemon=True).start()

    async def run(self, fn, *args, **kwargs):
        """Queues a task that returns a single result and awaits it without blocking the event loop."""
        future = Future()
        self._enqueue(fn, args, kwargs, future)
        return await asyncio.wrap_future(future)

    def open_stream(self, generator_fn, *args, **kwargs):
        """Queues a generator task right away; returns an async iterable to read its chunks.

        Queuing synchronously (not inside a coroutine) lets GpuQueueFull propagate before
        the endpoint builds the response, avoiding breaking a stream that's already started.
        """
        output = queue.Queue()

        def produce():
            try:
                for chunk in generator_fn(*args, **kwargs):
                    output.put(chunk)
            except Exception as error:
                output.put(error)
            finally:
                output.put(_STREAM_END)

        self._enqueue(produce, (), {}, Future())
        return self._consume(output)

    async def _consume(self, output: queue.Queue):
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, output.get)
            if item is _STREAM_END:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _enqueue(self, fn, args, kwargs, future: Future):
        try:
            self._jobs.put_nowait((fn, args, kwargs, future))
        except queue.Full:
            raise GpuQueueFull("The GPU queue is full, try again in a few seconds.")

    def _process_jobs(self):
        while True:
            fn, args, kwargs, future = self._jobs.get()
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                future.set_exception(GpuOutOfMemory(f"No GPU memory available: {error}"))
            except Exception as error:
                future.set_exception(error)


gpu_worker = GpuWorker()

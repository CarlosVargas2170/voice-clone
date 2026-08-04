"""Cola que serializa el acceso a la GPU: un solo trabajo de XTTS corre a la vez."""
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
    """Ejecuta trabajos de GPU uno a la vez en un hilo dedicado para no saturar la memoria de la tarjeta."""

    def __init__(self):
        self._jobs = queue.Queue(maxsize=MAX_GPU_QUEUE_SIZE)
        threading.Thread(target=self._process_jobs, daemon=True).start()

    async def run(self, fn, *args, **kwargs):
        """Encola una tarea que devuelve un único resultado y espera sin bloquear el event loop."""
        future = Future()
        self._enqueue(fn, args, kwargs, future)
        return await asyncio.wrap_future(future)

    def open_stream(self, generator_fn, *args, **kwargs):
        """Encola de inmediato una tarea generadora; devuelve un iterable async para leer sus chunks.

        Encolar de forma síncrona (no dentro de una corrutina) permite que GpuQueueFull se propague
        antes de que el endpoint arme la respuesta, evitando romper un stream ya iniciado.
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
            raise GpuQueueFull("La cola de la GPU está llena, intenta de nuevo en unos segundos.")

    def _process_jobs(self):
        while True:
            fn, args, kwargs, future = self._jobs.get()
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                future.set_exception(GpuOutOfMemory(f"Sin memoria de GPU disponible: {error}"))
            except Exception as error:
                future.set_exception(error)


gpu_worker = GpuWorker()

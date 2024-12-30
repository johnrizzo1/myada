#!/usr/bin/env python3
"""
Creating an asyncio generator for blocks of audio data.

This example shows how a generator can be used to analyze audio input blocks.
In addition, it shows how a generator can be created that yields not only input
blocks but also output blocks where audio data can be written to.

You need Python 3.7 or newer to run this.
"""
import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd

from datetime import datetime, timedelta

import logging

logging.basicConfig(level=logging.INFO)


class LocalStreamingAda:
  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self._stream = None
    self.logger = logging.getLogger(__name__)

  async def stream_generator(self, blocksize, *, channels=1, dtype='float32',
                            pre_fill_blocks=10, **kwargs):
    """
    Generator that yields blocks of input/output data as NumPy arrays.

    The output blocks are uninitialized and have to be filled with
    appropriate audio signals.
    """
    assert blocksize != 0
    q_in = asyncio.Queue()
    q_out = queue.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, outdata, frame_count, time_info, status):
      loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
      outdata[:] = q_out.get_nowait()

    # pre-fill output queue
    for _ in range(pre_fill_blocks):
      q_out.put(np.zeros((blocksize, channels), dtype=dtype))

    stream = sd.Stream(blocksize=blocksize, callback=callback, dtype=dtype,
                      channels=channels, **kwargs)
    with stream:
      while True:
        indata, status = await q_in.get()
        outdata = np.empty((blocksize, channels), dtype=dtype)
        yield indata, outdata, status
        q_out.put_nowait(outdata)

  async def wire_coro(self, **kwargs):
    """
    Create a connection between audio inputs and outputs.

    Asynchronously iterates over a stream generator and for each block
    simply copies the input data into the output block.
    """
    conversation_status = 'Not Started'
    last_started_set_time = datetime.now()
    async for indata, outdata, status in self.stream_generator(**kwargs):
      if status:
        self.logger.info(status)
      if conversation_status == 'Not Started' and indata.max() >= 0.1:
        self.logger.info(f"Since Conversation Started: {datetime.now() - last_started_set_time}")
        if (datetime.now() - last_started_set_time) > timedelta(seconds=0.005):
          conversation_status = 'Started'
        last_started_set_time = datetime.now()
      elif conversation_status == 'Started' and indata.max() < 0.1:
        if (datetime.now() - last_started_set_time) > timedelta(seconds=3):
          conversation_status = 'Not Started'
      self.logger.info(f"min:{indata.min()}\tmax:{indata.max()}\tavg:{indata.mean()}\tstatus:{conversation_status}")
      outdata[:] = indata

  async def main(self, **kwargs):
    self.logger.info('Enough of that, activating wire ...')
    audio_task = asyncio.create_task(self.wire_coro(**kwargs))
    await asyncio.sleep(10)
    audio_task.cancel()
    try:
      await audio_task
    except asyncio.CancelledError:
      self.logger.info('wire was cancelled')


if __name__ == "__main__":
  try:
    ada = LocalStreamingAda()
    asyncio.run(ada.main(blocksize=1024))
  except KeyboardInterrupt:
    sys.exit('\nInterrupted by user')
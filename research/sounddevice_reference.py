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


async def stream_generator(blocksize, *, channels=1, dtype='float32',
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


async def wire_coro(**kwargs):
  """
  Create a connection between audio inputs and outputs.

  Asynchronously iterates over a stream generator and for each block
  simply copies the input data into the output block.

  """
  conversation_status = 'Not Started'
  last_started_set_time = datetime.now()
  async for indata, outdata, status in stream_generator(**kwargs):
    if status:
      print(status)
    if conversation_status == 'Not Started' and indata.max() >= 0.1:
      print(f"Since Conversation Started: {datetime.now() - last_started_set_time}")
      if (datetime.now() - last_started_set_time) > timedelta(seconds=0.005):
        conversation_status = 'Started'
      last_started_set_time = datetime.now()
    elif conversation_status == 'Started' and indata.max() < 0.1:
      if (datetime.now() - last_started_set_time) > timedelta(seconds=0.05):
        conversation_status = 'Not Started'
    print('min:', indata.min(), '\t', 'max:', indata.max(), '\t', 'avg:', indata.mean(), '\t', 'status:', conversation_status)

    outdata[:] = indata


async def main(**kwargs):
  print('\nEnough of that, activating wire ...\n')
  audio_task = asyncio.create_task(wire_coro(**kwargs))
  await asyncio.sleep(10)
  audio_task.cancel()
  try:
    await audio_task
  except asyncio.CancelledError:
    print('\nwire was cancelled')


if __name__ == "__main__":
  try:
    asyncio.run(main(blocksize=1024))
  except KeyboardInterrupt:
    sys.exit('\nInterrupted by user')
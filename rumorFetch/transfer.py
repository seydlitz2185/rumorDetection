#!/usr/bin/env python3

import asyncio
import websockets
import sys
import wave
import redis



async def run_test(uri,file):
    async with websockets.connect(uri) as websocket:

        wf = wave.open(file, "rb")
        await websocket.send('{ "config" : { "sample_rate" : %d } }' % (wf.getframerate()))
        buffer_size = int(wf.getframerate() * 0.2) # 0.2 seconds of audio
        ret =''
        while True:
            data = wf.readframes(buffer_size)

            if len(data) == 0:
                break

            await websocket.send(data)
            message = eval(await websocket.recv())
            if "text" in message:
                ret+= message["text"]+'\n'

        await websocket.send('{"eof" : 1}')
        ret+=eval(await websocket.recv())["text"]
        #print(ret)

        return ret

#asyncio.run(run_test('ws://localhost:2700'))

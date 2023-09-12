import asyncio
import logging
import os

import aiohttp
import azure.functions as func


async def fetch(session):
    async with session.get(
        url=f"{os.environ['URL']}/api/mlcenitex",
        json={"text": [""], "mode": ["General"]},
    ) as resp:
        return await resp.read()


async def main(mytimer: func.TimerRequest) -> None:
    logging.info("*******Starting ping function*******")
    try:
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *(fetch(session) for _ in range(int(os.environ["PING_NUM"])))
            )

    except ConnectionError as e:
        logging.error(f"Connection Error: {e}")

from typing import Any
import json

import aioredis

from custom_interface.core.settinngs import settings
from custom_interface.core.schemas import BarrierCameraStatus
from custom_interface.core.enums import BarrierCurrentStatuses


class Redis:
    def __init__(self):
        self._redis = aioredis.from_url(settings.REDIS_URL)

    async def _get(self, key: Any):
        res = await self._redis.get(key)
        if res:
            return json.loads(res)

    async def get_barrier(self, key: Any) -> BarrierCameraStatus:
        res = await self._get(key)
        if res:
            return BarrierCameraStatus(**res)

    async def check_ready_barrier(self, barrier_pk: str) -> bool:
        barrier = await self.get_barrier(barrier_pk)
        if not barrier:
            return True
        elif barrier:
            if barrier.status_code == BarrierCurrentStatuses.available.value:
                return True
        return False

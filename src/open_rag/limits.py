"""Per-tenant rate limiting and daily LLM token budgets.

With Redis the counters are shared by every replica (the limit is cluster-wide, not
"limit x replicas"). Without Redis an in-process fallback is used (dev / single replica).
Redis errors fail *open* and are logged: a limiter outage must not become an API outage.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class Limits:
    def __init__(self, rate_per_min: int, daily_tokens: int, redis_client=None) -> None:
        self.rate, self.daily_tokens, self.r = rate_per_min, daily_tokens, redis_client
        self._mem: dict[str, tuple[float, int]] = {}  # key -> (expires_at, value)
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings, redis_client=None) -> Limits:
        if redis_client is None and settings.redis_url:
            import redis

            redis_client = redis.Redis.from_url(settings.redis_url, socket_timeout=0.5)
        return cls(settings.rate_limit_per_min, settings.tenant_daily_tokens, redis_client)

    def _incr(self, key: str, n: int, ttl_s: int) -> int:
        if self.r is not None:
            try:
                pipe = self.r.pipeline()
                pipe.incrby(key, n)
                pipe.expire(key, ttl_s)
                return int(pipe.execute()[0])
            except Exception:
                log.warning("limiter backend unavailable, failing open", exc_info=True)
                return 0
        now = time.time()
        with self._lock:
            if len(self._mem) > 10_000:  # bound memory: drop expired counters
                self._mem = {k: v for k, v in self._mem.items() if v[0] > now}
            _, value = self._mem.get(key, (0.0, 0))
            self._mem[key] = (now + ttl_s, value + n)
            return value + n

    def _get(self, key: str) -> int:
        if self.r is not None:
            try:
                return int(self.r.get(key) or 0)
            except Exception:
                log.warning("limiter backend unavailable, failing open", exc_info=True)
                return 0
        with self._lock:
            expires, value = self._mem.get(key, (0.0, 0))
            return value if expires > time.time() else 0

    def allow_request(self, tenant: str) -> bool:
        """Fixed 60 s window. Returns False when the tenant exceeded its rate."""
        if self.rate <= 0:
            return True
        window = int(time.time() // 60)
        return self._incr(f"rag:rl:{tenant}:{window:012d}", 1, 120) <= self.rate

    def _day_key(self, tenant: str) -> str:
        return f"rag:tok:{tenant}:{time.strftime('%Y%m%d', time.gmtime())}"

    def has_token_budget(self, tenant: str) -> bool:
        return self.daily_tokens <= 0 or self._get(self._day_key(tenant)) < self.daily_tokens

    def spend_tokens(self, tenant: str, n: int) -> None:
        if n > 0:
            self._incr(self._day_key(tenant), n, 2 * 86400)

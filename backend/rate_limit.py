"""
rate_limit.py — Redis-backed Distributed Rate Limiting
=========================================================
Redis-based rate limiter for multi-worker deployments.
"""

import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import redis.asyncio as redis

from .config import get_settings

log = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None


class RedisRateLimiter:
    """Redis-backed sliding window rate limiter."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_limit: int = 100,
        default_window: int = 60,
    ):
        self.redis_url = redis_url
        self.default_limit = default_limit
        self.default_window = default_window
        self._client: Optional[redis.Redis] = None
    
    async def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client
    
    def _make_key(self, identifier: str, endpoint: str) -> str:
        """Create Redis key for rate limit."""
        return f"ratelimit:{endpoint}:{identifier}"
    
    async def check_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
    ) -> RateLimitResult:
        """
        Check if request is within rate limit.
        
        Uses sliding window algorithm with Redis sorted sets.
        """
        client = await self._get_client()
        key = self._make_key(identifier, endpoint)
        limit = limit or self.default_limit
        window = window or self.default_window
        now = time.time()
        window_start = now - window
        
        # Lua script for atomic sliding window check
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window = tonumber(ARGV[4])
        
        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
        
        -- Count current requests
        local current = redis.call('ZCARD', key)
        
        if current >= limit then
            -- Rate limited
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local reset_time = 0
            if #oldest > 0 then
                reset_time = tonumber(oldest[2]) + window
            end
            return {0, current, reset_time}
        else
            -- Add current request
            redis.call('ZADD', key, now, now .. '-' .. math.random())
            redis.call('EXPIRE', key, window + 1)
            return {1, current + 1, now + window}
        end
        """
        
        try:
            result = await client.eval(
                lua_script,
                1,
                key,
                now,
                window_start,
                limit,
                window,
            )
            
            allowed = bool(result[0])
            remaining = max(0, limit - int(result[1]))
            reset_time = float(result[2])
            
            retry_after = None
            if not allowed:
                retry_after = max(0, reset_time - now)
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
            )
        except Exception as e:
            log.warning(f"Rate limit check failed, allowing request: {e}")
            # Fail open - allow request if Redis is unavailable
            return RateLimitResult(
                allowed=True,
                remaining=limit,
                reset_time=now + window,
            )
    
    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


def get_rate_limiter() -> RedisRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379")
        _rate_limiter = RedisRateLimiter(
            redis_url=redis_url,
            default_limit=settings.RATE_LIMIT_DEFAULT_LIMIT if hasattr(settings, 'RATE_LIMIT_DEFAULT_LIMIT') else 100,
            default_window=settings.RATE_LIMIT_DEFAULT_WINDOW if hasattr(settings, 'RATE_LIMIT_DEFAULT_WINDOW') else 60,
        )
    return _rate_limiter


async def rate_limit_dependency(
    identifier: str,
    endpoint: str,
    limit: Optional[int] = None,
    window: Optional[int] = None,
) -> RateLimitResult:
    """FastAPI dependency for rate limiting."""
    limiter = get_rate_limiter()
    return await limiter.check_limit(identifier, endpoint, limit, window)
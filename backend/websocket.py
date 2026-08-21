"""
websocket.py — WebSocket Manager for Real-time Updates
========================================================
Manages WebSocket connections for real-time job progress updates.
"""

import json
import logging
from dataclasses import dataclass, field

from fastapi import WebSocket

log = logging.getLogger(__name__)


@dataclass
class ConnectionManager:
    """Manages active WebSocket connections."""

    # job_id -> set of websockets
    _job_connections: dict[str, set[WebSocket]] = field(default_factory=dict)
    # websocket -> job_id (for cleanup)
    _ws_to_job: dict[WebSocket, str] = field(default_factory=dict)

    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept a new WebSocket connection for a job."""
        await websocket.accept()

        if job_id not in self._job_connections:
            self._job_connections[job_id] = set()

        self._job_connections[job_id].add(websocket)
        self._ws_to_job[websocket] = job_id

        log.info(f"WebSocket connected for job {job_id}. "
                 f"Active connections: {len(self._job_connections[job_id])}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        job_id = self._ws_to_job.pop(websocket, None)
        if job_id and job_id in self._job_connections:
            self._job_connections[job_id].discard(websocket)
            if not self._job_connections[job_id]:
                del self._job_connections[job_id]

        log.info(f"WebSocket disconnected for job {job_id}. "
                 f"Remaining: {len(self._job_connections.get(job_id, set()))}")

    async def send_progress(self, job_id: str, progress: int, status: str,
                            result: dict | None = None, error: str | None = None):
        """Send progress update to all connections for a job."""
        if job_id not in self._job_connections:
            return

        message = {
            "job_id": job_id,
            "progress": progress,
            "status": status,
            "result": result,
            "error": error,
        }

        dead_connections = []
        for ws in self._job_connections[job_id]:
            try:
                await ws.send_text(json.dumps(message))
            except Exception as e:
                log.warning(f"Failed to send WebSocket message: {e}")
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(ws)

    async def send_to_job(self, job_id: str, message: dict):
        """Send a raw message to all connections for a job."""
        if job_id not in self._job_connections:
            return

        dead_connections = []
        for ws in self._job_connections[job_id]:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead_connections.append(ws)

        for ws in dead_connections:
            self.disconnect(ws)

    def get_connection_count(self, job_id: str) -> int:
        """Get number of active connections for a job."""
        return len(self._job_connections.get(job_id, set()))

    def get_all_job_ids(self) -> list[str]:
        """Get all job IDs with active connections."""
        return list(self._job_connections.keys())


# Global connection manager
_manager: ConnectionManager | None = None


def get_ws_manager() -> ConnectionManager:
    """Get the global WebSocket manager instance."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager

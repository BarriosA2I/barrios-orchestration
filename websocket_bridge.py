#!/usr/bin/env python3
"""
WebSocket Bridge - Real-time event streaming to frontend
Bridges Event Bus to WebSocket connections

Author: Barrios A2I Architecture Team
"""

import asyncio
import json
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import structlog
from fastapi import WebSocket, WebSocketDisconnect

from event_bus import TypedEventBus, BaseEvent, EventType

logger = structlog.get_logger("websocket_bridge")


@dataclass
class ConnectedClient:
    """Represents a connected WebSocket client"""
    client_id: str
    websocket: WebSocket
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.now)
    subscribed_events: Set[str] = field(default_factory=set)


class WebSocketBridge:
    """
    Bridges Event Bus to WebSocket connections.
    Handles client management, filtering, and delivery.
    """

    def __init__(self, event_bus: TypedEventBus):
        self.event_bus = event_bus
        self._clients: Dict[str, ConnectedClient] = {}
        self._session_clients: Dict[str, Set[str]] = {}  # session_id -> client_ids
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._unsubscribers = []

    async def start(self):
        """Start the WebSocket bridge"""
        self._running = True

        # Subscribe to events we want to forward
        events_to_forward = [
            EventType.STREAM_TOKEN,
            EventType.STREAM_CHUNK,
            EventType.STREAM_COMPLETE,
            EventType.REQUEST_COMPLETED,
            EventType.REQUEST_FAILED,
            EventType.VIDEO_PROGRESS,
            EventType.ALERT,
        ]

        for event_type in events_to_forward:
            unsub = self.event_bus.subscribe(event_type, self._forward_event)
            self._unsubscribers.append(unsub)

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("WebSocket bridge started")

    async def stop(self):
        """Stop the WebSocket bridge"""
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from events
        for unsub in self._unsubscribers:
            unsub()

        # Close all connections
        for client in list(self._clients.values()):
            try:
                await client.websocket.close()
            except Exception:
                pass

        self._clients.clear()
        self._session_clients.clear()

        logger.info("WebSocket bridge stopped")

    async def connect(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Connect a new WebSocket client"""
        await websocket.accept()

        client_id = str(uuid.uuid4())
        client = ConnectedClient(
            client_id=client_id,
            websocket=websocket,
            session_id=session_id,
            user_id=user_id,
        )

        self._clients[client_id] = client

        # Track by session
        if session_id:
            if session_id not in self._session_clients:
                self._session_clients[session_id] = set()
            self._session_clients[session_id].add(client_id)

        logger.info(
            "Client connected",
            client_id=client_id,
            session_id=session_id,
            total_clients=len(self._clients),
        )

        # Send connection confirmation
        await self._send_to_client(client_id, {
            "event_type": "connected",
            "client_id": client_id,
            "session_id": session_id,
        })

        return client_id

    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        client = self._clients.pop(client_id, None)
        if client:
            # Remove from session tracking
            if client.session_id and client.session_id in self._session_clients:
                self._session_clients[client.session_id].discard(client_id)
                if not self._session_clients[client.session_id]:
                    del self._session_clients[client.session_id]

            logger.info(
                "Client disconnected",
                client_id=client_id,
                session_id=client.session_id,
                total_clients=len(self._clients),
            )

    async def _forward_event(self, event: BaseEvent):
        """Forward an event to relevant WebSocket clients"""
        # Determine which clients should receive this event
        target_clients = set()

        if event.session_id:
            # Send to all clients in this session
            session_clients = self._session_clients.get(event.session_id, set())
            target_clients.update(session_clients)

        if not target_clients and not event.session_id:
            # Broadcast to all if no session specified
            target_clients = set(self._clients.keys())

        # Send to each target client
        for client_id in target_clients:
            await self._send_to_client(client_id, event.to_dict())

    async def _send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to a specific client"""
        client = self._clients.get(client_id)
        if not client:
            return

        try:
            await client.websocket.send_json(data)
        except Exception as e:
            logger.warning(
                "Failed to send to client",
                client_id=client_id,
                error=str(e),
            )
            # Remove disconnected client
            await self.disconnect(client_id)

    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        for client_id in list(self._clients.keys()):
            await self._send_to_client(client_id, data)

    async def send_to_session(self, session_id: str, data: Dict[str, Any]):
        """Send data to all clients in a session"""
        client_ids = self._session_clients.get(session_id, set())
        for client_id in client_ids:
            await self._send_to_client(client_id, data)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to all clients"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds

                heartbeat = {
                    "event_type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                }

                await self.broadcast(heartbeat)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            "total_clients": len(self._clients),
            "total_sessions": len(self._session_clients),
            "clients_by_session": {
                sid: len(cids) for sid, cids in self._session_clients.items()
            },
        }


# Entry point for standalone testing
if __name__ == "__main__":
    async def main():
        # Create event bus
        event_bus = TypedEventBus()
        await event_bus.start()

        # Create bridge
        bridge = WebSocketBridge(event_bus)
        await bridge.start()

        print("WebSocket Bridge ready")
        print(f"Stats: {bridge.get_stats()}")

        # Simulate some events
        print("\n=== Simulating events ===")
        await event_bus.publish_stream_token(
            request_id="test-123",
            content="Hello ",
            index=0,
            session_id="session-abc",
        )

        await event_bus.publish_request_completed(
            request_id="test-123",
            answer="Hello World!",
            confidence=0.95,
            model_used="claude-3-haiku",
            latency_ms=150.0,
            session_id="session-abc",
        )

        print(f"Stats after events: {bridge.get_stats()}")

        await bridge.stop()
        await event_bus.stop()

    asyncio.run(main())

#!/usr/bin/env python3
"""
Event Bus - Typed Event System with Redis Pub/Sub
Real-time streaming for the Nervous System

Author: Barrios A2I Architecture Team
"""

import asyncio
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable, TypeVar, Generic
from datetime import datetime
import uuid
import structlog

logger = structlog.get_logger("event_bus")


class EventType(Enum):
    """All event types in the system"""
    # Request lifecycle
    REQUEST_RECEIVED = "request.received"
    REQUEST_ROUTING = "request.routing"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"

    # Streaming
    STREAM_START = "stream.start"
    STREAM_TOKEN = "stream.token"
    STREAM_CHUNK = "stream.chunk"
    STREAM_COMPLETE = "stream.complete"
    STREAM_ERROR = "stream.error"

    # Video generation
    VIDEO_QUEUED = "video.queued"
    VIDEO_PROGRESS = "video.progress"
    VIDEO_COMPLETE = "video.complete"
    VIDEO_FAILED = "video.failed"

    # Intelligence
    INTEL_START = "intel.start"
    INTEL_PROGRESS = "intel.progress"
    INTEL_COMPLETE = "intel.complete"

    # System
    HEALTH_CHECK = "system.health"
    ALERT = "system.alert"
    HEARTBEAT = "heartbeat"


@dataclass
class BaseEvent:
    """Base event structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseEvent":
        return cls.from_dict(json.loads(json_str))


@dataclass
class StreamChunkEvent(BaseEvent):
    """Event for streaming tokens"""
    event_type: str = EventType.STREAM_CHUNK.value
    content: str = ""
    index: int = 0
    is_final: bool = False


@dataclass
class RequestCompletedEvent(BaseEvent):
    """Event when request completes"""
    event_type: str = EventType.REQUEST_COMPLETED.value
    answer: str = ""
    confidence: float = 0.0
    model_used: str = ""
    latency_ms: float = 0.0


@dataclass
class AlertEvent(BaseEvent):
    """System alert event"""
    event_type: str = EventType.ALERT.value
    severity: str = "info"  # info, warning, error, critical
    message: str = ""
    source: str = ""


EventHandler = Callable[[BaseEvent], None]


class TypedEventBus:
    """
    Redis-backed event bus with typed events.
    Supports pub/sub, persistence, and replay.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        enable_persistence: bool = True,
        max_history: int = 1000,
    ):
        self.redis_url = redis_url
        self.enable_persistence = enable_persistence
        self.max_history = max_history

        self._running = False
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._event_history: List[BaseEvent] = []
        self._redis = None
        self._pubsub = None
        self._listener_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the event bus"""
        self._running = True
        logger.info("Event bus started", persistence=self.enable_persistence)

        # In production, connect to Redis here
        # For demo, use in-memory operation

    async def stop(self):
        """Stop the event bus"""
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")

    def subscribe(self, event_type: EventType, handler: EventHandler) -> Callable:
        """
        Subscribe to an event type.
        Returns unsubscribe function.
        """
        type_str = event_type.value
        if type_str not in self._handlers:
            self._handlers[type_str] = []

        self._handlers[type_str].append(handler)
        logger.debug("Subscribed to event", event_type=type_str)

        def unsubscribe():
            if type_str in self._handlers:
                self._handlers[type_str].remove(handler)

        return unsubscribe

    async def publish(self, event: BaseEvent):
        """Publish an event to all subscribers"""
        event_type = event.event_type
        logger.debug(
            "Publishing event",
            event_type=event_type,
            event_id=event.event_id,
        )

        # Store in history if persistence enabled
        if self.enable_persistence:
            self._event_history.append(event)
            # Trim to max_history
            if len(self._event_history) > self.max_history:
                self._event_history = self._event_history[-self.max_history:]

        # Dispatch to local handlers
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(
                    "Handler error",
                    event_type=event_type,
                    error=str(e),
                )

    async def publish_stream_token(
        self,
        request_id: str,
        content: str,
        index: int,
        session_id: Optional[str] = None,
    ):
        """Convenience method for publishing stream tokens"""
        event = StreamChunkEvent(
            request_id=request_id,
            session_id=session_id,
            content=content,
            index=index,
        )
        await self.publish(event)

    async def publish_request_completed(
        self,
        request_id: str,
        answer: str,
        confidence: float,
        model_used: str,
        latency_ms: float,
        session_id: Optional[str] = None,
    ):
        """Convenience method for publishing request completion"""
        event = RequestCompletedEvent(
            request_id=request_id,
            session_id=session_id,
            answer=answer,
            confidence=confidence,
            model_used=model_used,
            latency_ms=latency_ms,
        )
        await self.publish(event)

    async def publish_alert(
        self,
        message: str,
        severity: str = "info",
        source: str = "system",
    ):
        """Convenience method for publishing alerts"""
        event = AlertEvent(
            message=message,
            severity=severity,
            source=source,
        )
        await self.publish(event)

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[BaseEvent]:
        """Get event history with optional filters"""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type.value]

        if session_id:
            events = [e for e in events if e.session_id == session_id]

        return events[-limit:]

    async def replay(
        self,
        session_id: str,
        from_event_id: Optional[str] = None,
    ) -> List[BaseEvent]:
        """Replay events for a session"""
        events = self.get_history(session_id=session_id)

        if from_event_id:
            # Find the starting point
            start_idx = None
            for i, e in enumerate(events):
                if e.event_id == from_event_id:
                    start_idx = i + 1
                    break
            if start_idx:
                events = events[start_idx:]

        return events


class EventHandlers:
    """Built-in event handlers for common operations"""

    @staticmethod
    def log_handler(event: BaseEvent):
        """Log all events"""
        logger.info(
            "Event received",
            event_type=event.event_type,
            event_id=event.event_id,
            request_id=event.request_id,
        )

    @staticmethod
    async def metrics_handler(event: BaseEvent):
        """Update metrics based on events"""
        # In production, update Prometheus metrics here
        pass

    @staticmethod
    async def persistence_handler(event: BaseEvent):
        """Persist events to database"""
        # In production, write to PostgreSQL here
        pass


# Entry point for standalone testing
if __name__ == "__main__":
    async def main():
        bus = TypedEventBus()
        await bus.start()

        # Test subscription
        received_events = []

        def test_handler(event: BaseEvent):
            received_events.append(event)
            print(f"Received: {event.event_type} - {event.payload}")

        bus.subscribe(EventType.STREAM_CHUNK, test_handler)
        bus.subscribe(EventType.REQUEST_COMPLETED, test_handler)

        # Test publishing
        print("\n=== Publishing stream tokens ===")
        for i in range(5):
            await bus.publish_stream_token(
                request_id="test-123",
                content=f"token_{i} ",
                index=i,
            )
            await asyncio.sleep(0.1)

        # Test completion
        print("\n=== Publishing completion ===")
        await bus.publish_request_completed(
            request_id="test-123",
            answer="This is the complete answer",
            confidence=0.92,
            model_used="claude-3-sonnet-20240229",
            latency_ms=1234.5,
        )

        # Test alert
        print("\n=== Publishing alert ===")
        await bus.publish_alert(
            message="System test completed",
            severity="info",
            source="test",
        )

        print(f"\n=== Total events received: {len(received_events)} ===")
        print(f"History size: {len(bus.get_history())}")

        await bus.stop()

    asyncio.run(main())

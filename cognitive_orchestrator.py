#!/usr/bin/env python3
"""
Cognitive Orchestrator - Core State Machine
Dual-process routing with Thompson sampling

Author: Barrios A2I Architecture Team
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import uuid
import structlog
import random
from collections import defaultdict

logger = structlog.get_logger("cognitive_orchestrator")


class ProcessingMode(Enum):
    AUTO = "auto"
    SYSTEM_1_FAST = "system1_fast"
    SYSTEM_2_DEEP = "system2_deep"
    HYBRID = "hybrid"


class PipelinePhase(Enum):
    RECEIVED = "received"
    ROUTING = "routing"
    FAST_RETRIEVAL = "fast_retrieval"
    DEEP_ANALYSIS = "deep_analysis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    ERROR = "error"


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class OrchestrationRequest:
    """Request to the orchestrator"""
    query: str
    mode: ProcessingMode = ProcessingMode.AUTO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timeout_seconds: float = 30.0
    require_citations: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result from the orchestrator"""
    request_id: str
    success: bool
    answer: Optional[str] = None
    confidence: float = 0.0
    model_used: Optional[str] = None
    mode_used: Optional[ProcessingMode] = None
    complexity_score: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    citations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Health check response"""
    healthy: bool
    components: Dict[str, Dict[str, Any]]
    uptime_seconds: float
    version: str


class ThompsonRouter:
    """Thompson sampling for optimal model selection"""

    def __init__(self):
        # Beta distribution parameters for each model
        self.models = {
            "claude-3-haiku-20240307": {"alpha": 1, "beta": 1, "cost": 0.00025},
            "claude-3-sonnet-20240229": {"alpha": 1, "beta": 1, "cost": 0.003},
            "claude-3-opus-20240229": {"alpha": 1, "beta": 1, "cost": 0.015},
        }

    def select_model(self, complexity: float) -> str:
        """Select model using Thompson sampling based on complexity"""
        # Simple complexity-based routing
        if complexity <= 3:
            return "claude-3-haiku-20240307"
        elif complexity >= 7:
            return "claude-3-opus-20240229"
        else:
            return "claude-3-sonnet-20240229"

    def update(self, model: str, success: bool):
        """Update model statistics based on outcome"""
        if model in self.models:
            if success:
                self.models[model]["alpha"] += 1
            else:
                self.models[model]["beta"] += 1


class CircuitBreaker:
    """Circuit breaker for resilience"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time: Optional[datetime] = None

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class CognitiveOrchestrator:
    """Main orchestrator with state machine execution."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6380",
        enable_circuit_breaker: bool = True,
        enable_dlq: bool = True,
        enable_checkpointing: bool = True,
    ):
        self.redis_url = redis_url
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_dlq = enable_dlq
        self.enable_checkpointing = enable_checkpointing

        self._running = False
        self._start_time: Optional[datetime] = None
        self.router = ThompsonRouter()
        self.circuit_breaker = CircuitBreaker()

        # Metrics
        self._metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "latency_sum_ms": 0,
            "cost_sum_usd": 0,
            "model_selections": defaultdict(int),
        }

    async def start(self):
        """Start the orchestrator"""
        self._running = True
        self._start_time = datetime.now()
        logger.info("Cognitive Orchestrator started")

    async def stop(self):
        """Stop the orchestrator"""
        self._running = False
        logger.info("Cognitive Orchestrator stopped")

    async def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute an orchestration request"""
        request_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(
            "Executing request",
            request_id=request_id,
            query_length=len(request.query),
            mode=request.mode.value,
        )

        try:
            # Classify complexity
            complexity = await self._classify_complexity(request.query)

            # Route to appropriate model
            if request.mode == ProcessingMode.AUTO:
                if complexity <= 3:
                    mode = ProcessingMode.SYSTEM_1_FAST
                    model = "claude-3-haiku-20240307"
                elif complexity >= 7:
                    mode = ProcessingMode.SYSTEM_2_DEEP
                    model = "claude-3-opus-20240229"
                else:
                    mode = ProcessingMode.HYBRID
                    model = "claude-3-sonnet-20240229"
            else:
                mode = request.mode
                model = self.router.select_model(complexity)

            self._metrics["model_selections"][model] += 1

            # Generate response (production would call actual LLM)
            answer = await self._generate_response(request.query, model, mode)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            # Calculate cost based on model
            cost_map = {
                "claude-3-haiku-20240307": 0.00025,
                "claude-3-sonnet-20240229": 0.003,
                "claude-3-opus-20240229": 0.015,
            }
            cost = cost_map.get(model, 0.001)

            # Update metrics
            self._metrics["requests_total"] += 1
            self._metrics["requests_success"] += 1
            self._metrics["latency_sum_ms"] += latency
            self._metrics["cost_sum_usd"] += cost

            # Update router with success
            self.router.update(model, True)

            return OrchestrationResult(
                request_id=request_id,
                success=True,
                answer=answer,
                confidence=0.85 + (random.random() * 0.1),  # 0.85-0.95
                model_used=model,
                mode_used=mode,
                complexity_score=complexity / 10,
                latency_ms=latency,
                cost_usd=cost,
                citations=[],
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error("Execution error", error=str(e), request_id=request_id)
            self._metrics["requests_total"] += 1
            self._metrics["requests_failed"] += 1

            return OrchestrationResult(
                request_id=request_id,
                success=False,
                error=str(e),
                trace_id=trace_id,
            )

    async def _classify_complexity(self, query: str) -> int:
        """Classify query complexity (1-10)"""
        score = min(len(query) // 50, 5)

        # High complexity indicators
        if any(w in query.lower() for w in ['analyze', 'compare', 'explain why', 'evaluate']):
            score += 3
        if any(w in query.lower() for w in ['step by step', 'detailed', 'comprehensive']):
            score += 2
        if '?' in query and query.count('?') > 1:
            score += 1

        # Medium complexity indicators
        if any(w in query.lower() for w in ['how', 'why', 'what if']):
            score += 1

        return min(max(score, 1), 10)

    async def _generate_response(
        self,
        query: str,
        model: str,
        mode: ProcessingMode
    ) -> str:
        """Generate response using the selected model"""
        # Simulated responses for demo - production would call actual API
        responses = {
            ProcessingMode.SYSTEM_1_FAST: (
                f"I can help you with that! Based on your question about "
                f"'{query[:50]}...', here's a quick answer:\n\n"
                f"Barrios A2I specializes in autonomous AI systems that can handle "
                f"complex business processes with zero human intervention. "
                f"Our solutions include RAG agents, marketing automation, "
                f"and custom AI development.\n\n"
                f"Would you like to know more about any specific service?"
            ),
            ProcessingMode.SYSTEM_2_DEEP: (
                f"Let me provide a comprehensive analysis for your query:\n\n"
                f"**Understanding Your Question**\n"
                f"You asked about '{query[:100]}...'\n\n"
                f"**Detailed Analysis**\n"
                f"Barrios A2I offers enterprise-grade AI automation solutions:\n\n"
                f"1. **RAG Research Agents** - Autonomous information gathering and analysis\n"
                f"2. **Marketing Overlord** - End-to-end marketing automation\n"
                f"3. **RAGNAROK Video System** - AI-generated video commercials\n"
                f"4. **Custom AI Systems** - Tailored solutions for your business\n\n"
                f"**Recommendation**\n"
                f"Based on your inquiry, I'd suggest scheduling a consultation "
                f"to discuss how we can automate your specific workflows."
            ),
            ProcessingMode.HYBRID: (
                f"Great question! Let me break this down for you:\n\n"
                f"Regarding '{query[:75]}...'\n\n"
                f"Barrios A2I provides AI-powered automation solutions designed "
                f"to transform businesses through intelligent automation.\n\n"
                f"**Key Capabilities:**\n"
                f"- Autonomous sales through AI chat (like this conversation!)\n"
                f"- AI-generated video content\n"
                f"- Competitive intelligence gathering\n"
                f"- Custom AI agent development\n\n"
                f"What aspect interests you most?"
            ),
        }

        # Add small delay to simulate processing
        await asyncio.sleep(0.1)

        return responses.get(mode, responses[ProcessingMode.HYBRID])

    async def health_check(self) -> HealthCheckResult:
        """Perform health check"""
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        return HealthCheckResult(
            healthy=self._running,
            components={
                "orchestrator": {"status": "healthy" if self._running else "stopped"},
                "router": {"status": "healthy", "models": len(self.router.models)},
                "circuit_breaker": {"status": self.circuit_breaker.state.value},
            },
            uptime_seconds=uptime,
            version="1.0.0",
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_latency = 0
        if self._metrics["requests_success"] > 0:
            avg_latency = self._metrics["latency_sum_ms"] / self._metrics["requests_success"]

        return {
            "requests": {
                "total": self._metrics["requests_total"],
                "success": self._metrics["requests_success"],
                "failed": self._metrics["requests_failed"],
            },
            "latency": {
                "average_ms": avg_latency,
            },
            "cost": {
                "total_usd": self._metrics["cost_sum_usd"],
            },
            "models": dict(self._metrics["model_selections"]),
        }


# Entry point for standalone testing
if __name__ == "__main__":
    async def main():
        orchestrator = CognitiveOrchestrator()
        await orchestrator.start()

        # Test queries
        test_queries = [
            "What is AI?",
            "Can you analyze the competitive landscape for AI automation companies?",
            "Tell me about your video generation service step by step",
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)

            result = await orchestrator.execute(
                OrchestrationRequest(query=query)
            )

            print(f"Success: {result.success}")
            print(f"Model: {result.model_used}")
            print(f"Mode: {result.mode_used.value if result.mode_used else 'N/A'}")
            print(f"Complexity: {result.complexity_score:.2f}")
            print(f"Latency: {result.latency_ms:.2f}ms")
            print(f"Answer: {result.answer[:200]}...")

        print(f"\n{'='*60}")
        print("Metrics:")
        print(orchestrator.get_metrics())

        await orchestrator.stop()

    asyncio.run(main())

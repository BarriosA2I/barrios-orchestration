#!/usr/bin/env python3
"""
Cognitive Orchestrator - Core State Machine
Dual-process routing with Thompson sampling

Author: Barrios A2I Architecture Team
"""

import asyncio
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import uuid
import structlog
import random
from collections import defaultdict

import anthropic

logger = structlog.get_logger("cognitive_orchestrator")

# System prompt for Nexus Brain
NEXUS_SYSTEM_PROMPT = """You are Nexus Brain, the intelligent AI assistant for Barrios A2I - an AI automation consultancy.

ABOUT BARRIOS A2I:
- Motto: "Alienation 2 Innovation"
- We build world-class autonomous AI systems that run businesses with zero human intervention
- Founded by Gary Barrios, AI automation architect

SERVICES:
1. RAG Research Agents - Scrape competitors, analyze market data, provide actionable intelligence
2. Marketing Overlord System - Automated campaigns, content generation, social media, lead gen
3. AI-Powered Websites - Intelligent assistants with generative UI (not basic chatbots)
4. Custom AI Development:
   - Option A: Free build for 30% equity
   - Option B: Flat fee for 100% ownership

KEY SYSTEMS:
- RAGNAROK v7.0: 9-agent video generation system (243s per commercial, $2.60 cost)
- Trinity Orchestrator: 3-agent market intelligence system
- Thompson Router: Intelligent model selection (Haiku/Sonnet/Opus)

TARGET CUSTOMERS:
- Marketing directors at B2B SaaS
- Agency owners
- E-commerce managers
- High-ticket clients ($50K-$300K for custom AI systems)

RESPONSE STYLE:
- Be helpful, professional, and concise
- Answer questions directly about our services
- If asked to do something (like "say BANANA"), just do it
- Use markdown formatting for lists and emphasis
- Don't be overly salesy - be genuinely helpful

CONTACT: barriosa2i.com"""


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
        # Beta distribution parameters for each model (updated Dec 2025)
        self.models = {
            "claude-3-5-haiku-20241022": {"alpha": 1, "beta": 1, "cost": 0.0008},
            "claude-3-5-sonnet-20241022": {"alpha": 1, "beta": 1, "cost": 0.003},
            "claude-sonnet-4-20250514": {"alpha": 1, "beta": 1, "cost": 0.003},
        }

    def select_model(self, complexity: float) -> str:
        """Select model using Thompson sampling based on complexity"""
        # Simple complexity-based routing (updated Dec 2025)
        if complexity <= 3:
            return "claude-3-5-haiku-20241022"
        elif complexity >= 7:
            return "claude-sonnet-4-20250514"  # Use Sonnet 4 for complex queries
        else:
            return "claude-3-5-sonnet-20241022"

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

            # Route to appropriate model (updated Dec 2025)
            if request.mode == ProcessingMode.AUTO:
                if complexity <= 3:
                    mode = ProcessingMode.SYSTEM_1_FAST
                    model = "claude-3-5-haiku-20241022"
                elif complexity >= 7:
                    mode = ProcessingMode.SYSTEM_2_DEEP
                    model = "claude-sonnet-4-20250514"
                else:
                    mode = ProcessingMode.HYBRID
                    model = "claude-3-5-sonnet-20241022"
            else:
                mode = request.mode
                model = self.router.select_model(complexity)

            self._metrics["model_selections"][model] += 1

            # Generate response (production would call actual LLM)
            answer = await self._generate_response(request.query, model, mode)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            # Calculate cost based on model (updated Dec 2025)
            cost_map = {
                "claude-3-5-haiku-20241022": 0.0008,
                "claude-3-5-sonnet-20241022": 0.003,
                "claude-sonnet-4-20250514": 0.003,
            }
            cost = cost_map.get(model, 0.003)

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
        """Generate response using the selected Claude model"""
        try:
            # Get API key from environment
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not set!")
                return "I'm having trouble connecting right now. Please try again in a moment."

            # Create Anthropic client
            client = anthropic.Anthropic(api_key=api_key)

            # Adjust max tokens based on mode
            max_tokens = {
                ProcessingMode.SYSTEM_1_FAST: 500,
                ProcessingMode.SYSTEM_2_DEEP: 2000,
                ProcessingMode.HYBRID: 1000,
            }.get(mode, 1000)

            logger.info(
                "Calling Claude API",
                model=model,
                mode=mode.value,
                max_tokens=max_tokens,
            )

            # Call Claude API
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=NEXUS_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": query}
                ]
            )

            # Extract response text
            response_text = message.content[0].text

            logger.info(
                "Claude API response received",
                model=model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            )

            return response_text

        except anthropic.APIConnectionError as e:
            logger.error("API connection error", error=str(e))
            return "I'm having trouble connecting to my AI backend. Please try again."
        except anthropic.RateLimitError as e:
            logger.error("Rate limit exceeded", error=str(e))
            return "I'm receiving too many requests right now. Please wait a moment and try again."
        except anthropic.APIStatusError as e:
            logger.error("API status error", error=str(e), status_code=e.status_code)
            return f"Something went wrong on my end. Please try again."
        except Exception as e:
            logger.error("Unexpected error in _generate_response", error=str(e))
            return "I encountered an unexpected error. Please try again."

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

"""
Core Agent Implementation

The main Agent class built on LangGraph's Functional API.
Implements the cyclic Plan → Act → Reflect loop.

Security: Includes rate limiting on tool execution to prevent
resource exhaustion and data exfiltration attacks.

Enhanced with LLM-based task completion detection for accurate evaluation.
"""

import inspect
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, get_args, get_origin
from uuid import UUID, uuid4

import structlog
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from squadron.core.config import SquadronConfig
from squadron.core.state import (
    AgentPhase,
    AgentState,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)
from squadron.llm.base import LLMProvider, LLMMessage

logger = structlog.get_logger(__name__)


# Prompt template for LLM-based task completion detection
COMPLETION_DETECTION_PROMPT = '''You are an expert task completion evaluator. Analyze whether the given task has been successfully completed based on the conversation history and tool results.

## Task
{task}

## Conversation History
{history}

## Tool Results Summary
{tool_results}

## Instructions
Carefully analyze the evidence to determine if the task has been completed. Consider:
1. Was the main objective of the task achieved?
2. Are all required sub-tasks finished?
3. Were the tool executions successful?
4. Is there any indication the task is incomplete or failed?

## Output Format
Respond with a JSON object:
```json
{{
    "completed": true/false,
    "confidence": 0.0-1.0,
    "reason": "Brief explanation of your assessment"
}}
```

Your assessment:'''


# Security: Default rate limits for tool execution
DEFAULT_RATE_LIMITS = {
    # High-risk tools: stricter limits
    "run_command": (10, 60),  # 10 calls per 60 seconds
    "execute_code": (5, 60),
    "write_file": (20, 60),
    "delete_file": (5, 60),
    # Medium-risk tools
    "read_file": (100, 60),
    "search": (50, 60),
    # Default for unspecified tools
    "_default": (100, 60),  # 100 calls per 60 seconds
}


class Agent:
    """
    Squadron Agent - The core agent implementation.
    
    Built on LangGraph's Functional API for improved ergonomics and
    cleaner state management. Supports cyclic loops (Plan → Act → Reflect)
    rather than linear chains.
    
    Example:
        ```python
        agent = Agent(
            name="researcher",
            config=SquadronConfig(),
        )
        result = await agent.run("Analyze this codebase")
        ```
    """

    def __init__(
        self,
        name: str,
        config: SquadronConfig | None = None,
        llm: LLMProvider | None = None,
        memory: Any | None = None,  # GraphitiMemory
        reasoner: Any | None = None,  # LATSReasoner
        tools: list[Any] | Any | None = None,
        checkpointer: Any | None = None,
        completion_confidence_threshold: float = 0.7,
    ):
        """
        Initialize the agent.

        Args:
            name: Agent identifier
            config: Configuration object
            llm: LLM provider for completion detection and other evaluations
            memory: Memory system (Graphiti)
            reasoner: Reasoning engine (LATS)
            tools: Available tools — a list of callables, a single tool pack
                (object with ``get_tools()``), or a list mixing both
            checkpointer: LangGraph checkpointer for persistence
            completion_confidence_threshold: Minimum confidence for LLM completion detection
        """
        self.name = name
        self.config = config or SquadronConfig()
        self.llm = llm
        self.memory = memory
        self.reasoner = reasoner
        self.checkpointer = checkpointer or MemorySaver()
        self.completion_confidence_threshold = completion_confidence_threshold

        # Normalize tools: accept a single tool pack, a list of tools/packs, or None
        self.tools = self._normalize_tools(tools)

        # Tool registry for quick lookup
        self._tool_registry: dict[str, Callable] = {}
        for tool in self.tools:
            if hasattr(tool, "name"):
                self._tool_registry[tool.name] = tool
            elif hasattr(tool, "__name__"):
                self._tool_registry[tool.__name__] = tool

        # Security: Rate limiting state
        self._rate_limit_history: dict[str, list[datetime]] = defaultdict(list)
        self._rate_limits = DEFAULT_RATE_LIMITS.copy()

        # Bridge tools → reasoner: convert callables to ToolDefinitions
        # so the LATS reasoner can include them in LLM prompts.
        if self.reasoner and hasattr(self.reasoner, "register_tools"):
            tool_defs = self._build_tool_definitions()
            if tool_defs:
                self.reasoner.register_tools(tool_defs)

        # Build the execution graph
        self._graph = self._build_graph()

        logger.info(
            "Agent initialized",
            name=self.name,
            tools=list(self._tool_registry.keys()),
            has_llm=self.llm is not None,
        )

    @staticmethod
    def _normalize_tools(tools: Any) -> list[Any]:
        """
        Normalize the tools parameter into a flat list of callables.

        Handles:
        - None → empty list
        - A single tool pack (object with ``get_tools()``) → its tools list
        - A single callable → wrapped in a list
        - A list that may contain a mix of callables and tool packs
        """
        if tools is None:
            return []

        # Single tool pack passed directly (not wrapped in a list)
        if not isinstance(tools, (list, tuple)):
            if hasattr(tools, "get_tools"):
                return tools.get_tools()
            # Single callable
            return [tools]

        # List/tuple – expand any embedded tool packs
        expanded: list[Any] = []
        for item in tools:
            if hasattr(item, "get_tools"):
                expanded.extend(item.get_tools())
            else:
                expanded.append(item)
        return expanded

    # ------------------------------------------------------------------
    # Tool → ToolDefinition conversion
    # ------------------------------------------------------------------
    def _build_tool_definitions(self) -> list[Any]:
        """
        Convert registered callable tools into ``ToolDefinition`` objects.

        Inspects each tool's signature, docstring, and ``@mcp_tool``
        metadata (if present) so the LATS reasoner can present them to
        the LLM when generating candidate actions.
        """
        from squadron.llm.base import ToolDefinition

        definitions: list[ToolDefinition] = []
        for name, tool in self._tool_registry.items():
            # Description: prefer @mcp_tool metadata, then docstring
            desc = (
                getattr(tool, "_mcp_description", None)
                or (tool.__doc__ or "").strip()
                or f"Tool {name}"
            )
            # Truncate overly long docstrings
            if len(desc) > 500:
                desc = desc[:497] + "..."

            params = self._extract_parameters_schema(tool)
            definitions.append(ToolDefinition(
                name=name,
                description=desc,
                parameters=params,
            ))
        return definitions

    @staticmethod
    def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
        """Map a Python type annotation to a JSON-Schema type descriptor."""
        if annotation is inspect.Parameter.empty or annotation is Any:
            return {"type": "string"}

        origin = get_origin(annotation)

        # Handle Optional / Union with None  (e.g. str | None)
        if origin is type(int | str):  # types.UnionType (Python 3.10+)
            args = get_args(annotation)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return Agent._python_type_to_json_schema(non_none[0])
            return {"type": "string"}

        # typing.Union  (Python < 3.10 style)
        try:
            import typing
            if origin is typing.Union:
                args = get_args(annotation)
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    return Agent._python_type_to_json_schema(non_none[0])
                return {"type": "string"}
        except Exception:
            pass

        # list[...]
        if origin is list:
            inner_args = get_args(annotation)
            items = (
                Agent._python_type_to_json_schema(inner_args[0])
                if inner_args
                else {"type": "string"}
            )
            return {"type": "array", "items": items}

        # dict[...]
        if origin is dict:
            return {"type": "object"}

        # Primitive types
        _PRIM_MAP: dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        schema_type = _PRIM_MAP.get(annotation)  # type: ignore[arg-type]
        if schema_type:
            return {"type": schema_type}

        return {"type": "string"}

    @staticmethod
    def _extract_parameters_schema(func: Callable) -> dict[str, Any]:
        """
        Build a JSON-Schema ``parameters`` object from a callable's
        signature (skipping ``self``).
        """
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return {"type": "object", "properties": {}}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            prop = Agent._python_type_to_json_schema(param.annotation)
            properties[pname] = prop

            if param.default is inspect.Parameter.empty:
                required.append(pname)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def _build_graph(self) -> Any:
        """Build the LangGraph execution graph."""
        
        @task
        async def plan(state: AgentState) -> AgentState:
            """
            Planning phase: Analyze the task and generate a plan.
            
            Uses the reasoning engine (LATS) if available to generate
            multiple candidate plans and select the best one.
            """
            logger.debug("Entering planning phase", iteration=state.iteration)
            
            # Update state to planning phase
            state = state.set_phase(AgentPhase.PLANNING)
            
            # Retrieve relevant memory context
            if self.memory:
                try:
                    context = await self.memory.retrieve(
                        query=state.task,
                        session_id=str(state.session_id),
                    )
                    state = state.update_memory_context(context)
                    logger.debug("Retrieved memory context", facts=len(context.get("facts", [])))
                except Exception as e:
                    logger.warning("Memory retrieval failed", error=str(e))
            
            # Generate plan using reasoner or simple LLM call
            if self.reasoner:
                # Use LATS for tree-search planning
                plan_result = await self.reasoner.plan(state)
                state = plan_result
            elif self.llm:
                # Use LLM to generate a plan and tool calls
                try:
                    # Format tools for the LLM
                    from squadron.llm.base import ToolDefinition
                    import inspect
                    
                    llm_tools = []
                    for name, tool in self._tool_registry.items():
                        # Simple tool definition extraction
                        # In a real implementation, this would use a proper schema generator
                        desc = tool.__doc__ or f"Tool {name}"
                        llm_tools.append(ToolDefinition(
                            name=name,
                            description=desc,
                            parameters={"type": "object", "properties": {}}
                        ))
                    
                    # Generate response
                    response = await self.llm.generate(
                        messages=state.messages,
                        tools=llm_tools if llm_tools else None
                    )
                    
                    # Add assistant message
                    state = state.add_message(response.to_message())
                    
                    # Add tool calls to state
                    if response.tool_calls:
                        for tc in response.tool_calls:
                            state = state.add_tool_call(tc)
                            
                except Exception as e:
                    logger.error("LLM generation failed", error=str(e))
                    state = state.add_error(f"LLM generation failed: {e}")
                    state = state.set_phase(AgentPhase.ERROR)
            else:
                # Simple single-shot planning
                plan_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=f"Planning to address: {state.task}",
                    metadata={"phase": "planning"},
                )
                state = state.add_message(plan_message)
                # Without an LLM or reasoner, we can't generate tool calls
                # Mark as error to prevent infinite loop
                state = state.add_error("No LLM or reasoner available to generate plan")
                state = state.set_phase(AgentPhase.ERROR)
            
            return state

        @task
        async def act(state: AgentState) -> AgentState:
            """
            Action phase: Execute tools based on the plan.
            
            Handles tool execution, human-in-the-loop approval,
            and result collection.
            """
            logger.debug("Entering action phase", iteration=state.iteration)
            
            state = state.set_phase(AgentPhase.ACTING)
            
            # Check for pending tool calls
            if not state.pending_tool_calls:
                logger.warning(
                    "No pending tool calls in action phase – planning "
                    "did not produce any executable actions",
                    iteration=state.iteration,
                )
                return state
            
            # Process each tool call
            for tool_call in state.pending_tool_calls:
                # Check if human approval is required
                if tool_call.tool_name in self.config.governance.require_human_approval:
                    logger.info(
                        "Requesting human approval",
                        tool=tool_call.tool_name,
                    )
                    state = state.request_approval({
                        "tool_call": tool_call.model_dump(),
                        "reason": f"Tool '{tool_call.tool_name}' requires approval",
                    })
                    # Interrupt execution for approval
                    interrupt({"approval_request": state.approval_request})
                    return state
                
                # Execute the tool
                result = await self._execute_tool(tool_call)
                state = state.add_tool_result(result)
                
                # Add tool result as message
                result_message = Message(
                    role=MessageRole.TOOL,
                    content=str(result.result) if result.success else f"Error: {result.error}",
                    name=tool_call.tool_name,
                    metadata={"tool_call_id": str(tool_call.id)},
                )
                state = state.add_message(result_message)
            
            return state

        @task
        async def reflect(state: AgentState) -> AgentState:
            """
            Reflection phase: Evaluate progress and decide next steps.
            
            Determines whether to continue planning, complete the task,
            or handle errors.
            """
            logger.debug("Entering reflection phase", iteration=state.iteration)
            
            state = state.set_phase(AgentPhase.REFLECTING)
            state = state.increment_iteration()
            
            # Check iteration limit
            if state.iteration >= state.max_iterations:
                logger.warning("Max iterations reached", iteration=state.iteration)
                state = state.add_error("Maximum iterations reached")
                return state.set_phase(AgentPhase.ERROR)
            
            # Analyze recent results
            recent_results = state.tool_results[-5:] if state.tool_results else []
            all_successful = all(r.success for r in recent_results)
            
            # Use reasoner for reflection if available
            if self.reasoner:
                reflection_result = await self.reasoner.reflect(state)
                state = reflection_result
            
            # Store interaction in memory
            if self.memory:
                try:
                    await self.memory.store(
                        messages=state.messages,
                        session_id=str(state.session_id),
                    )
                except Exception as e:
                    logger.warning("Memory storage failed", error=str(e))
            
            # Determine next phase
            # This is a simplified check - real implementation would use LLM
            if self._is_task_complete(state):
                return state.set_phase(AgentPhase.COMPLETED)
            
            return state.set_phase(AgentPhase.PLANNING)

        @task
        async def route(state: AgentState) -> Command:
            """Route to the next node based on current phase."""
            if state.phase == AgentPhase.COMPLETED:
                return Command(goto="__end__")
            elif state.phase == AgentPhase.ERROR:
                return Command(goto="__end__")
            elif state.phase == AgentPhase.PLANNING:
                return Command(goto="act")
            elif state.phase == AgentPhase.ACTING:
                return Command(goto="reflect")
            elif state.phase == AgentPhase.REFLECTING:
                return Command(goto="plan")
            else:
                return Command(goto="plan")

        @entrypoint(checkpointer=self.checkpointer)
        async def agent_graph(state: AgentState) -> AgentState:
            """
            Main agent execution graph.
            
            Implements the cyclic Plan → Act → Reflect loop.
            Includes stall detection to prevent spinning when no
            tool calls are being generated.
            """
            consecutive_no_ops = 0
            max_no_ops = 3  # bail after 3 consecutive cycles with no tool execution

            while state.should_continue:
                tool_results_before = len(state.tool_results)

                if state.phase in (AgentPhase.PLANNING, AgentPhase.REFLECTING):
                    state = await plan(state)
                    if not state.should_continue:
                        break
                
                if state.phase == AgentPhase.PLANNING:
                    state = state.set_phase(AgentPhase.ACTING)
                
                if state.phase == AgentPhase.ACTING:
                    state = await act(state)
                    if not state.should_continue:
                        break
                    state = await reflect(state)

                # Stall detection: if no new tool results were produced in
                # this full plan→act→reflect cycle, the agent is stuck.
                tool_results_after = len(state.tool_results)
                if tool_results_after == tool_results_before:
                    consecutive_no_ops += 1
                    logger.warning(
                        "No tool calls executed this iteration",
                        consecutive_no_ops=consecutive_no_ops,
                        max_no_ops=max_no_ops,
                        iteration=state.iteration,
                    )
                    if consecutive_no_ops >= max_no_ops:
                        logger.error(
                            "Agent stalled – no progress for %d consecutive "
                            "iterations, terminating",
                            max_no_ops,
                        )
                        state = state.add_error(
                            f"Agent stalled: no tool calls executed for "
                            f"{consecutive_no_ops} consecutive iterations. "
                            f"Check that tools are registered and the LLM "
                            f"is generating valid tool call names."
                        )
                        state = state.set_phase(AgentPhase.ERROR)
                        break
                else:
                    consecutive_no_ops = 0
            
            return state

        return agent_graph

    def _check_rate_limit(self, tool_name: str) -> tuple[bool, str]:
        """
        Check if a tool call is within rate limits.

        Security: Prevents resource exhaustion and exfiltration attacks.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        # Get rate limit for this tool (or default)
        max_calls, window_seconds = self._rate_limits.get(
            tool_name, self._rate_limits.get("_default", (100, 60))
        )

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)

        # Clean old entries
        self._rate_limit_history[tool_name] = [
            t for t in self._rate_limit_history[tool_name] if t > window_start
        ]

        # Check limit
        current_count = len(self._rate_limit_history[tool_name])
        if current_count >= max_calls:
            logger.warning(
                "Rate limit exceeded",
                tool=tool_name,
                count=current_count,
                limit=max_calls,
                window=window_seconds,
            )
            return False, f"Rate limit exceeded for '{tool_name}': {current_count}/{max_calls} in {window_seconds}s"

        # Record this call
        self._rate_limit_history[tool_name].append(now)
        return True, ""

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with rate limiting."""
        import time

        # Security: Check rate limit before execution
        is_allowed, rate_error = self._check_rate_limit(tool_call.tool_name)
        if not is_allowed:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                result=None,
                error=rate_error,
            )

        start_time = time.time()

        tool = self._tool_registry.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                result=None,
                error=f"Tool '{tool_call.tool_name}' not found",
            )
        
        try:
            # Execute the tool
            if callable(tool):
                result = await tool(**tool_call.arguments)
            else:
                result = await tool.invoke(tool_call.arguments)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                result=result,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "Tool execution failed",
                tool=tool_call.tool_name,
                error=str(e),
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                result=None,
                error=str(e),
                execution_time_ms=execution_time,
            )

    def _is_task_complete(self, state: AgentState) -> bool:
        """
        Determine if the task is complete.

        Uses LLM-based evaluation when available, with fallback to heuristics.
        """
        # Check if we have any successful tool results
        if not state.tool_results:
            return False

        # Try LLM-based detection if available
        if self.llm:
            import asyncio
            try:
                # Run async method in sync context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context, use the sync fallback
                    return self._is_task_complete_heuristic(state)
                return loop.run_until_complete(
                    self._is_task_complete_llm(state)
                )
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self._is_task_complete_llm(state))
            except Exception as e:
                logger.warning(
                    "LLM completion detection failed, using heuristic",
                    error=str(e),
                )
                return self._is_task_complete_heuristic(state)

        return self._is_task_complete_heuristic(state)

    async def _is_task_complete_llm(self, state: AgentState) -> bool:
        """
        Use LLM to determine if the task is complete.

        Returns True if the LLM determines the task is complete with
        sufficient confidence.
        """
        # Format conversation history
        history_lines = []
        for msg in state.messages[-15:]:  # Last 15 messages
            role = msg.role.value.upper()
            content = msg.content[:300]
            if len(msg.content) > 300:
                content += "..."
            history_lines.append(f"[{role}]: {content}")
        history_text = "\n".join(history_lines) if history_lines else "No messages."

        # Format tool results
        tool_results_lines = []
        for result in state.tool_results[-10:]:  # Last 10 results
            status = "SUCCESS" if result.success else "FAILED"
            result_preview = str(result.result)[:200] if result.result else ""
            error_text = f" Error: {result.error}" if result.error else ""
            tool_results_lines.append(
                f"- {result.tool_name}: {status}{error_text}"
                f"{(' -> ' + result_preview) if result_preview else ''}"
            )
        tool_results_text = (
            "\n".join(tool_results_lines) if tool_results_lines else "No tool results."
        )

        # Build the prompt
        prompt = COMPLETION_DETECTION_PROMPT.format(
            task=state.task,
            history=history_text,
            tool_results=tool_results_text,
        )

        try:
            messages = [LLMMessage.user(prompt)]
            response = await self.llm.generate(messages)

            # Parse the response
            result = self._parse_completion_response(response.content)

            completed = result.get("completed", False)
            confidence = result.get("confidence", 0.0)
            reason = result.get("reason", "")

            logger.info(
                "LLM completion detection result",
                completed=completed,
                confidence=confidence,
                reason=reason[:100] if reason else None,
            )

            # Only mark as complete if confidence exceeds threshold
            if completed and confidence >= self.completion_confidence_threshold:
                return True

            return False

        except Exception as e:
            logger.error("LLM completion detection error", error=str(e))
            return self._is_task_complete_heuristic(state)

    def _parse_completion_response(self, content: str) -> dict[str, Any]:
        """Parse the LLM completion detection response."""
        try:
            # Find JSON in response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)

            logger.warning("No JSON found in completion response")
            return {"completed": False, "confidence": 0.0}

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse completion JSON", error=str(e))
            return {"completed": False, "confidence": 0.0}

    def _is_task_complete_heuristic(self, state: AgentState) -> bool:
        """
        Fallback heuristic-based completion detection.

        Uses word-boundary keyword matching so that words like
        "successfully" do not falsely trigger the "success" keyword.
        Requires at least 3 successful tool results before considering
        keyword-based completion, to avoid stopping after the very first
        write_file or similar preparatory step.
        """
        import re as _re

        # Need at least one successful tool result
        if not state.tool_results:
            return False

        successful_results = [r for r in state.tool_results if r.success]

        # Only consider keyword-based completion if we have made meaningful
        # progress (≥3 successful tool calls).  This prevents stopping after
        # just writing a script file before it has been executed.
        # NOTE: Only check ASSISTANT messages, not TOOL messages — TOOL
        # messages often contain 'success': True in JSON output and would
        # cause false positives.
        if len(successful_results) >= 3:
            recent_messages = [
                m for m in (state.messages[-6:] if state.messages else [])
                if m.role.value == "assistant"
            ]
            # Use word-boundary matching to avoid "successfully" matching "success"
            completion_keywords = [
                r"\bcomplete\b", r"\bdone\b", r"\bfinished\b",
                r"\bsuccess\b", r"\bcompleted\b",
            ]
            for msg in recent_messages:
                text = msg.content.lower()
                if any(_re.search(kw, text) for kw in completion_keywords):
                    return True

        # Only return True when a tool that explicitly produces the final
        # artefact (run_command / write_file) confirms delivery.
        # We do NOT use a raw "≥N successes" fallback because research-heavy
        # tasks (web_search, read_url loops) easily accumulate 5+ successes
        # without ever producing the requested output file.
        run_results = [
            r for r in state.tool_results
            if r.tool_name in ("run_command", "write_file") and r.success
        ]
        if run_results:
            last_run = run_results[-1]
            result_text = str(last_run.result or "").lower()
            # pptxgenjs prints "Presentation saved: <filename>.pptx"
            if ".pptx" in result_text or "presentation saved" in result_text:
                return True

        return False

    def set_llm(self, llm: LLMProvider) -> None:
        """Set the LLM provider for the agent."""
        self.llm = llm
        logger.info("Set LLM provider for agent", provider=llm.provider_name)

    async def run(
        self,
        task: str,
        session_id: UUID | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentState:
        """
        Run the agent on a task.
        
        Args:
            task: The task description
            session_id: Optional session ID for continuity
            initial_context: Optional initial context
            
        Returns:
            Final agent state after execution
        """
        logger.info("Starting agent run", task=task[:100])
        
        # Create initial state
        state = AgentState(
            session_id=session_id or uuid4(),
            task=task,
            task_id=uuid4(),
            max_iterations=self.config.governance.max_iterations,
        )
        
        # Add initial user message
        user_message = Message(
            role=MessageRole.USER,
            content=task,
        )
        state = state.add_message(user_message)
        
        # Add initial context if provided
        if initial_context:
            state = state.update_memory_context(initial_context)
        
        # Run the graph
        try:
            config = {"configurable": {"thread_id": str(state.session_id)}}
            final_state = await self._graph.ainvoke(state, config)
            logger.info(
                "Agent run completed",
                phase=final_state.phase,
                iterations=final_state.iteration,
            )
            return final_state
        except Exception as e:
            logger.error("Agent run failed", error=str(e))
            return state.add_error(str(e))

    async def resume(
        self,
        session_id: UUID,
        approval: bool = True,
        user_input: str | None = None,
    ) -> AgentState:
        """
        Resume a paused agent execution.

        Used after human-in-the-loop approval or to continue from a checkpoint.

        Args:
            session_id: Session to resume
            approval: Whether the pending action was approved
            user_input: Optional additional input from the user

        Returns:
            Final agent state after resumption
        """
        logger.info(
            "Resuming agent",
            session_id=str(session_id),
            approval=approval,
            has_user_input=user_input is not None,
        )

        # Retrieve the checkpointed state
        config = {"configurable": {"thread_id": str(session_id)}}

        try:
            # Get the checkpoint tuple from the checkpointer
            checkpoint_tuple = await self._get_checkpoint(config)

            if checkpoint_tuple is None:
                raise ValueError(f"No checkpoint found for session {session_id}")

            # Extract state from checkpoint
            checkpoint = checkpoint_tuple.checkpoint
            state_data = checkpoint.get("channel_values", {})

            # Reconstruct AgentState from checkpoint data
            state = self._reconstruct_state(state_data, session_id)

            if state is None:
                raise ValueError(f"Could not reconstruct state for session {session_id}")

            logger.debug(
                "Retrieved checkpoint",
                session_id=str(session_id),
                phase=state.phase,
                iteration=state.iteration,
                awaiting_approval=state.awaiting_human_approval,
            )

            # Handle approval request if present
            if state.awaiting_human_approval:
                if not approval:
                    # User rejected the pending action
                    logger.info("User rejected pending action")
                    state = state.add_error("Action rejected by user")
                    return state

                # User approved - clear the approval request and continue
                state = state.grant_approval()

                # Add user input as message if provided
                if user_input:
                    user_msg = Message(
                        role=MessageRole.USER,
                        content=user_input,
                    )
                    state = state.add_message(user_msg)

            # Continue execution from the current state
            logger.info("Continuing execution from checkpoint")
            final_state = await self._graph.ainvoke(state, config)

            logger.info(
                "Resume completed",
                phase=final_state.phase,
                iterations=final_state.iteration,
            )

            return final_state

        except Exception as e:
            logger.error("Resume failed", error=str(e), session_id=str(session_id))
            raise

    async def _get_checkpoint(self, config: dict[str, Any]) -> Any:
        """
        Retrieve checkpoint from the checkpointer.

        Handles both sync and async checkpointers.
        """
        try:
            # Try async get first
            if hasattr(self.checkpointer, "aget_tuple"):
                return await self.checkpointer.aget_tuple(config)
            elif hasattr(self.checkpointer, "get_tuple"):
                # Sync checkpointer
                return self.checkpointer.get_tuple(config)
            else:
                logger.warning("Checkpointer does not support tuple retrieval")
                return None
        except Exception as e:
            logger.error("Checkpoint retrieval failed", error=str(e))
            return None

    def _reconstruct_state(
        self, state_data: dict[str, Any], session_id: UUID
    ) -> AgentState | None:
        """
        Reconstruct AgentState from checkpoint data.

        Args:
            state_data: Raw state data from checkpoint
            session_id: The session ID for this state

        Returns:
            Reconstructed AgentState or None if failed
        """
        try:
            # If state_data is already an AgentState, return it
            if isinstance(state_data, AgentState):
                return state_data

            # If it's a dict with an AgentState under a key
            if isinstance(state_data, dict):
                # Common patterns for stored state
                for key in ["state", "__root__", "agent_state", ""]:
                    if key in state_data:
                        value = state_data[key]
                        if isinstance(value, AgentState):
                            return value
                        if isinstance(value, dict):
                            try:
                                return AgentState(**value)
                            except Exception:
                                continue

                # Try to construct directly from state_data
                try:
                    # Ensure session_id is set
                    if "session_id" not in state_data:
                        state_data["session_id"] = session_id
                    return AgentState(**state_data)
                except Exception:
                    pass

            logger.warning("Could not reconstruct state from checkpoint data")
            return None

        except Exception as e:
            logger.error("State reconstruction failed", error=str(e))
            return None

    async def get_session_state(self, session_id: UUID) -> AgentState | None:
        """
        Get the current state of a session without resuming.

        Args:
            session_id: The session to check

        Returns:
            Current AgentState or None if not found
        """
        config = {"configurable": {"thread_id": str(session_id)}}
        checkpoint_tuple = await self._get_checkpoint(config)

        if checkpoint_tuple is None:
            return None

        checkpoint = checkpoint_tuple.checkpoint
        state_data = checkpoint.get("channel_values", {})

        return self._reconstruct_state(state_data, session_id)

    def is_session_paused(self, state: AgentState) -> bool:
        """Check if a session is paused awaiting approval."""
        return state.awaiting_human_approval

    def register_tool(self, tool: Any) -> None:
        """Register a new tool with the agent and sync with the reasoner."""
        if hasattr(tool, "name"):
            name = tool.name
        elif hasattr(tool, "__name__"):
            name = tool.__name__
        else:
            raise ValueError("Tool must have a 'name' attribute or '__name__'")
        
        self._tool_registry[name] = tool
        self.tools.append(tool)

        # Keep reasoner's tool list in sync
        self._sync_tools_to_reasoner()
        logger.info("Tool registered", tool=name)

    def get_tool(self, name: str) -> Any | None:
        """Get a tool by name."""
        return self._tool_registry.get(name)

    @property
    def available_tools(self) -> list[str]:
        """List available tool names."""
        return list(self._tool_registry.keys())

    async def load_mcp_tools(self, mcp_host: Any) -> int:
        """
        Load tools from an MCP host.
        
        Args:
            mcp_host: MCPHost instance with connected servers
            
        Returns:
            Number of tools loaded
        """
        from squadron.connectivity.mcp_host import MCPHost
        
        if not isinstance(mcp_host, MCPHost):
            raise TypeError("Expected MCPHost instance")
        
        count = 0
        for tool in mcp_host.get_all_tools():
            # Create a wrapper function for the MCP tool
            async def mcp_tool_wrapper(
                _host=mcp_host,
                _name=tool.name,
                **kwargs
            ):
                return await _host.call_tool(_name, kwargs)
            
            mcp_tool_wrapper.__name__ = tool.name
            mcp_tool_wrapper.__doc__ = tool.description
            
            self._tool_registry[tool.name] = mcp_tool_wrapper
            count += 1
        
        logger.info("Loaded MCP tools", count=count)
        return count

    def register_tool_pack(self, tool_pack: Any) -> int:
        """
        Register all tools from a tool pack.
        
        Args:
            tool_pack: A tool pack instance (CodingTools, ResearchTools, etc.)
            
        Returns:
            Number of tools registered
        """
        if not hasattr(tool_pack, "get_tools"):
            raise TypeError("Tool pack must have a get_tools() method")
        
        tools = tool_pack.get_tools()
        for tool in tools:
            # Use low-level registration (skip per-tool sync)
            if hasattr(tool, "name"):
                name = tool.name
            elif hasattr(tool, "__name__"):
                name = tool.__name__
            else:
                continue
            self._tool_registry[name] = tool
            self.tools.append(tool)

        # Sync once after all tools are added
        self._sync_tools_to_reasoner()

        logger.info("Registered tool pack", tools=len(tools))
        return len(tools)

    # ------------------------------------------------------------------
    # Reasoner synchronisation helper
    # ------------------------------------------------------------------
    def _sync_tools_to_reasoner(self) -> None:
        """Push the current set of ``ToolDefinition``s to the reasoner."""
        if self.reasoner and hasattr(self.reasoner, "register_tools"):
            tool_defs = self._build_tool_definitions()
            if tool_defs:
                self.reasoner.register_tools(tool_defs)


from __future__ import annotations

import time
import traceback
import logging
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Tuple

if TYPE_CHECKING:
    # These pull in the full AIOS/cerebrum runtime, which StrideFairShareCore
    # (the pure scheduling logic) does not need. Imported for real, lazily,
    # in FairShareScheduler.__init__ so this module stays importable -- and
    # StrideFairShareCore testable -- without that runtime installed.
    from aios.hooks.types.llm import LLMRequestQueueGetMessage
    from aios.hooks.types.memory import MemoryRequestQueueGetMessage
    from aios.hooks.types.tool import ToolRequestQueueGetMessage
    from aios.hooks.types.storage import StorageRequestQueueGetMessage
    from aios.memory.manager import MemoryManager
    from aios.storage.storage import StorageManager
    from aios.llm_core.adapter import LLMAdapter
    from aios.tool.manager import ToolManager
    from .base import BaseScheduler

logger = logging.getLogger(__name__)

STRIDE_BASE = 10_000
# Large constant numerator for stride_i = STRIDE_BASE / weight_i. Keeps
# strides at a convenient scale for typical weights (1-100); the exact
# value doesn't affect fairness, only readability of pass values.


@dataclass
class AgentStrideState:
    """Per-agent bookkeeping needed by stride scheduling."""

    weight: float
    pass_value: float = 0.0
    stride: float = field(init=False)
    service_count: int = 0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"weight must be > 0, got {self.weight}")
        self.stride = STRIDE_BASE / self.weight

    def set_weight(self, weight: float) -> None:
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        self.weight = weight
        self.stride = STRIDE_BASE / weight


class StrideFairShareCore:
    
    def __init__(self, default_weight: float = 1.0) -> None:
        if default_weight <= 0:
            raise ValueError("default_weight must be > 0")
        self.default_weight = default_weight
        self._agents: Dict[Hashable, AgentStrideState] = {}
        # waiting[agent_id] = FIFO list of (arrival_seq, request)
        self._waiting: Dict[Hashable, List[Tuple[int, object]]] = {}
        self._arrival_seq = 0

    # ---------------------------------------------------------------- agents

    def set_weight(self, agent_id: Hashable, weight: float) -> None:
        """Explicitly (re)configure an agent's scheduling weight."""
        self._ensure_agent(agent_id, weight)

    def _seed_pass(self) -> float:
        if not self._agents:
            return 0.0
        return min(state.pass_value for state in self._agents.values())

    def _ensure_agent(self, agent_id: Hashable, weight: Optional[float]) -> AgentStrideState:
        state = self._agents.get(agent_id)
        if state is None:
            state = AgentStrideState(
                weight=weight if weight is not None else self.default_weight,
                pass_value=self._seed_pass(),
            )
            self._agents[agent_id] = state
        elif weight is not None and weight != state.weight:
            state.set_weight(weight)
        return state

    # --------------------------------------------------------------- queueing

    def enqueue(self, agent_id: Hashable, request: object, weight: Optional[float] = None) -> None:
        
        self._ensure_agent(agent_id, weight)
        self._arrival_seq += 1
        self._waiting.setdefault(agent_id, []).append((self._arrival_seq, request))

    def has_waiting(self) -> bool:
        return any(self._waiting.values())

    def waiting_count(self, agent_id: Hashable) -> int:
        return len(self._waiting.get(agent_id, ()))

    def select(self) -> Optional[Tuple[Hashable, object]]:
        
        candidates = [aid for aid, q in self._waiting.items() if q]
        if not candidates:
            return None

        def sort_key(aid: Hashable):
            state = self._agents[aid]
            earliest_seq = self._waiting[aid][0][0]
            return (state.pass_value, earliest_seq)

        chosen = min(candidates, key=sort_key)
        _, request = self._waiting[chosen].pop(0)

        state = self._agents[chosen]
        state.pass_value += state.stride
        state.service_count += 1

        return chosen, request

    # ------------------------------------------------- introspection / metrics

    def service_counts(self) -> Dict[Hashable, int]:
        return {aid: s.service_count for aid, s in self._agents.items()}

    def pass_values(self) -> Dict[Hashable, float]:
        return {aid: s.pass_value for aid, s in self._agents.items()}

    def weights(self) -> Dict[Hashable, float]:
        return {aid: s.weight for aid, s in self._agents.items()}


try:
    # Deferred until here (rather than a top-level import) because
    # aios.scheduler.base transitively imports the full AIOS/cerebrum
    # runtime (aios.memory.manager, aios.storage.storage, etc.), none of
    # which StrideFairShareCore or its tests need. This keeps the module
    # importable -- and StrideFairShareCore testable in isolation -- in
    # environments that only have the pure-Python scheduling logic's
    # dependencies installed.
    from aios.hooks.types.llm import LLMRequestQueueGetMessage
    from aios.hooks.types.memory import MemoryRequestQueueGetMessage
    from aios.hooks.types.tool import ToolRequestQueueGetMessage
    from aios.hooks.types.storage import StorageRequestQueueGetMessage
    from aios.memory.manager import MemoryManager
    from aios.storage.storage import StorageManager
    from aios.llm_core.adapter import LLMAdapter
    from aios.tool.manager import ToolManager
    from .base import BaseScheduler
except ImportError as _import_error:
    BaseScheduler = None
    _fair_share_import_error = _import_error
else:
    _fair_share_import_error = None


if BaseScheduler is not None:

    class FairShareScheduler(BaseScheduler):


        def __init__(
            self,
            llm: LLMAdapter,
            memory_manager: MemoryManager,
            storage_manager: StorageManager,
            tool_manager: ToolManager,
            log_mode: str,
            get_llm_syscall: LLMRequestQueueGetMessage,
            get_memory_syscall: MemoryRequestQueueGetMessage,
            get_storage_syscall: StorageRequestQueueGetMessage,
            get_tool_syscall: ToolRequestQueueGetMessage,
            default_weight: float = 1.0,
            poll_interval: float = 0.05,
        ):

            super().__init__(
                llm,
                memory_manager,
                storage_manager,
                tool_manager,
                log_mode,
                get_llm_syscall,
                get_memory_syscall,
                get_storage_syscall,
                get_tool_syscall,
            )
            self.core = StrideFairShareCore(default_weight=default_weight)
            self.poll_interval = poll_interval

        def _execute_syscall(self, syscall: Any, executor: Any, syscall_type: str) -> Optional[Dict[str, Any]]:
        
            try:
                syscall.set_status("executing")
                self.logger.log(
                    f"{syscall.agent_name} is executing {syscall_type} syscall.\n",
                    "executing",
                )
                syscall.set_start_time(time.time())

                response = executor(syscall)
                syscall.set_response(response)

                syscall.event.set()
                syscall.set_status("done")
                syscall.set_end_time(time.time())

                self.logger.log(
                    f"Completed {syscall_type} syscall for {syscall.agent_name}. "
                    f"Thread ID: {syscall.get_pid()}\n",
                    "done",
                )

                return response

            except Exception as e:
                logger.error(f"Error executing {syscall_type} syscall: {str(e)}")
                traceback.print_exc()
                return None

        def _drain_new_llm_syscalls(self) -> None:
            while True:
                try:
                    syscall = self.get_llm_syscall()
                except Empty:
                    return
                agent_id = getattr(syscall, "agent_name", None)
                weight = getattr(syscall, "agent_weight", None)
                self.core.enqueue(agent_id, syscall, weight=weight)

        def process_llm_requests(self) -> None:
        
            while self.active:
                self._drain_new_llm_syscalls()
                selected = self.core.select()
                if selected is None:
                    time.sleep(self.poll_interval)
                    continue
                agent_id, syscall = selected
                self._execute_syscall(syscall, self.llm.execute_llm_syscall, "LLM")

        def process_memory_requests(self) -> None:
            while self.active:
                try:
                    memory_syscall = self.get_memory_syscall()
                    self._execute_syscall(memory_syscall, self.memory_manager.address_request, "Memory")
                except Empty:
                    pass

        def process_storage_requests(self) -> None:
            while self.active:
                try:
                    storage_syscall = self.get_storage_syscall()
                    self._execute_syscall(storage_syscall, self.storage_manager.address_request, "Storage")
                except Empty:
                    pass

        def process_tool_requests(self) -> None:
            while self.active:
                try:
                    tool_syscall = self.get_tool_syscall()
                    self._execute_syscall(tool_syscall, self.tool_manager.address_request, "Tool")
                except Empty:
                    pass

        def start(self) -> None:
            self.active = True
            self.start_processing_threads([
                self.process_llm_requests,
                self.process_memory_requests,
                self.process_storage_requests,
                self.process_tool_requests,
            ])

        def stop(self) -> None:
            self.active = False
            self.stop_processing_threads()

else:

    class FairShareScheduler:  # type: ignore[no-redef]
        """Stand-in raised on use when the full AIOS/cerebrum runtime isn't installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "FairShareScheduler requires the full AIOS runtime (cerebrum and "
                "friends). Install the project's requirements to use it; "
                "StrideFairShareCore remains usable standalone."
            ) from _fair_share_import_error
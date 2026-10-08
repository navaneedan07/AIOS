"""Integration tests for ``FairShareScheduler`` against the real AIOS interfaces.

The weighted-stride *decisions* are covered without any AIOS dependency in
``tests/test_fair_share_scheduler.py`` (``StrideFairShareCore``). These tests
cover the adapter half: that the kernel-facing scheduler dispatches LLM syscalls
through the adapter, completes their lifecycle, and hands pending requests back
when the policy is swapped at runtime.

``fair_share_scheduler`` subclasses ``BaseScheduler``, which pulls in the whole
AIOS component stack, so importing ``cerebrum`` is a prerequisite and these
tests are skipped when it is missing.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from queue import Queue
from typing import Any, List, Optional

import pytest

pytest.importorskip("cerebrum")

from aios.scheduler.fair_share_scheduler import (  # noqa: E402
    FairShareScheduler,
    StrideFairShareCore,
)
from aios.scheduler.registry import (  # noqa: E402
    POLICY_FAIR_SHARE,
    create_scheduler,
    policy_options,
)


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------
class FakeSyscall:
    """Minimal stand-in for ``aios.syscall.Syscall``."""

    def __init__(self, agent_name: str, priority: Any = None):
        self.agent_name = agent_name
        self.priority = priority
        self.agent_weight: Optional[float] = None
        self.status: Optional[str] = None
        self.response: Any = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.event = threading.Event()

    def get_priority(self):
        return self.priority

    def set_status(self, value):
        self.status = value

    def set_response(self, value):
        self.response = value

    def set_start_time(self, value):
        self.start_time = value

    def set_end_time(self, value):
        self.end_time = value

    def get_pid(self):
        return id(self)


class RecordingLLM:
    """Stand-in for ``LLMAdapter``: the adapter's batch LLM entry point."""

    def __init__(self):
        self.dispatched: List[str] = []

    def execute_llm_syscalls(self, llm_syscalls):
        for syscall in llm_syscalls:
            self.dispatched.append(syscall.agent_name)
            syscall.set_status("done")
            syscall.set_response("ok")
            syscall.set_end_time(time.time())
            syscall.event.set()


class FailingLLM:
    """Adapter whose backend is unavailable."""

    def execute_llm_syscalls(self, llm_syscalls):
        raise RuntimeError("backend unavailable")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_queue_getter(queue: Queue, timeout: float = 0.05):
    def getter():
        return queue.get(block=True, timeout=timeout)

    return getter


def make_scheduler(llm: Any, queue: Queue, **kwargs) -> FairShareScheduler:
    return FairShareScheduler(
        llm=llm,
        memory_manager=None,
        storage_manager=None,
        tool_manager=None,
        log_mode="console",
        get_llm_syscall=make_queue_getter(queue),
        get_memory_syscall=None,
        get_storage_syscall=None,
        get_tool_syscall=None,
        **kwargs,
    )


@contextmanager
def llm_dispatch_thread(scheduler: FairShareScheduler):
    """Run only the LLM processor in a thread, as ``start`` would."""
    scheduler.active = True
    thread = threading.Thread(target=scheduler.process_llm_requests, daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        scheduler.active = False
        thread.join(timeout=5)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------
def test_a_syscall_is_dispatched_and_its_agent_unblocked():
    """The regression guard: the scheduler must call the adapter's batch method.

    An earlier version called ``execute_llm_syscall`` (singular, and not an
    adapter method), which raised inside the scheduler thread and left the
    calling agent blocked on its event forever.
    """
    llm = RecordingLLM()
    queue = Queue()
    scheduler = make_scheduler(llm, queue)

    syscall = FakeSyscall("agentA")
    with llm_dispatch_thread(scheduler):
        queue.put(syscall)
        assert syscall.event.wait(timeout=5), "the agent was left blocked"

    assert llm.dispatched == ["agentA"]
    assert syscall.status == "done"
    assert syscall.response == "ok"
    assert scheduler.dispatched_count == 1


def test_equal_weight_agents_take_turns():
    llm = RecordingLLM()
    queue = Queue()
    scheduler = make_scheduler(llm, queue)

    with llm_dispatch_thread(scheduler):
        for _ in range(2):
            queue.put(FakeSyscall("agentA"))
            queue.put(FakeSyscall("agentB"))
        assert wait_until(lambda: len(llm.dispatched) >= 4)

    assert llm.dispatched == ["agentA", "agentB", "agentA", "agentB"]


def test_a_failed_dispatch_does_not_leave_the_agent_blocked():
    queue = Queue()
    scheduler = make_scheduler(FailingLLM(), queue)

    syscall = FakeSyscall("agentA")
    with llm_dispatch_thread(scheduler):
        queue.put(syscall)
        assert syscall.event.wait(timeout=5), "the agent was left blocked"

    assert syscall.status == "error"


# ----------------------------------------------------------------------
# Runtime hand-over
# ----------------------------------------------------------------------
def test_pending_requests_are_handed_back_for_a_policy_swap():
    """Accepted-but-undispatched requests must survive being replaced."""
    scheduler = make_scheduler(RecordingLLM(), Queue())
    core = scheduler.core
    for agent in ("agentA", "agentB", "agentA"):
        core.enqueue(agent, FakeSyscall(agent))

    pending = scheduler.drain_pending_syscalls()

    assert sorted(syscall.agent_name for syscall in pending) == [
        "agentA",
        "agentA",
        "agentB",
    ]
    assert scheduler.drain_pending_syscalls() == []


# ----------------------------------------------------------------------
# Registry integration
# ----------------------------------------------------------------------
def test_policy_name_is_the_registry_key():
    assert POLICY_FAIR_SHARE == "fair_share"
    assert policy_options({"fair_share": {"default_weight": 3.0}}, POLICY_FAIR_SHARE) == {
        "default_weight": 3.0
    }


def test_create_scheduler_builds_the_configured_fair_share_policy():
    scheduler = create_scheduler(
        "fair_share",
        llm=RecordingLLM(),
        memory_manager=None,
        storage_manager=None,
        tool_manager=None,
        log_mode="console",
        get_llm_syscall=None,
        get_memory_syscall=None,
        get_storage_syscall=None,
        get_tool_syscall=None,
        default_weight=4.0,
    )

    assert isinstance(scheduler, FairShareScheduler)
    assert isinstance(scheduler.core, StrideFairShareCore)
    assert scheduler.core.default_weight == 4.0
    assert scheduler.active is False

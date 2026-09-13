# Design: Priority-based Agent-level Scheduling for AIOS

| | |
|---|---|
| **Status** | Draft — design review welcome |
| **Related issue** | #127 (*AIOS Semantic Scheduling* → *Priority-based Agent-level Scheduling*) |
| **Tracking issue** | `<FILL IN> ` |
| **Authors** | Group 5 — university operating systems course project |
| **Scope** | `aios/scheduler/`, `aios/hooks/types/`, `runtime/launch.py`, `aios/config/`, `tests/` |

---

## 1. Summary

AIOS is an agent operating system whose central responsibility is deciding which agent gets
the LLM next. Today that decision is made by only two policies — first-come-first-served
(`FIFOScheduler`) and round-robin (`RRScheduler`) — and it cannot express importance or
fairness.

This document proposes an **opt-in, non-breaking** scheduling feature that adds:

1. **Agent priority levels** that influence dispatch order.
2. **Weighted fair-share** so no agent can monopolize LLM capacity.
3. **Aging**, which guarantees a low-priority agent still runs within a bounded wait.
4. **A policy registry**, so the active policy is chosen by name from configuration.
5. **Per-agent accounting and observability**, without which none of the above can be
   evaluated.

Existing behaviour is unchanged: `FIFOScheduler` and `RRScheduler` are left untouched, the
new policies are **off by default**, and there are **no new third-party dependencies**.

---

## 2. Motivation

The scheduler is the reason an "agent operating system" exists, but the current
implementation cannot handle three situations that arise in normal use:

- **Head-of-line blocking.** One long-running agent blocks a short interactive one
  completely. A chat request waits behind a multi-minute report generation.
- **Noisy neighbours.** When several agents or users share one AIOS deployment, nothing
  bounds a heavy agent's share of LLM capacity. There are no weights, shares, or usage
  counters anywhere in the codebase.
- **No protection against starvation.** With no notion of priority, there is nothing that
  could starve — but there is also nothing that lets an urgent agent be served first. Adding
  priority *without* aging would introduce starvation, which is why aging is part of this
  design rather than a follow-up.

The gap is also publicly acknowledged: *Priority-based Agent-level Scheduling for AIOS* has
been listed under *AIOS Semantic Scheduling* in roadmap issue #127 since May 2024,
unassigned and unimplemented.

---

## 3. Current state in AIOS (verified)

| Item | Current state |
|---|---|
| `aios/scheduler/` | `__init__.py`, `base.py`, `fifo_scheduler.py`, `rr_scheduler.py` — two policies, no priority, no fairness |
| Policy selection | Determined internally by the unrelated `use_context_manager` flag in `runtime/launch.py` |
| `priority` field | `aios/syscall/__init__.py` defines `self.priority`, `set_priority()`, `get_priority()` — **no call sites anywhere in the repository** |
| Scheduler settings | `SchedulerParams` (`aios/hooks/types/scheduler.py`) accepts only the four managers, `log_mode` and four queue-reader callbacks |
| `scheduler:` config | `aios/config/config.yaml.example` contains only `log_mode` |
| Accounting | None — no per-agent usage tracking |
| Scheduler metrics | `SyscallExecutor._execute_syscall` computes waiting/turnaround times, but returns on the success path before recording them, so the recorded lists are always empty |
| Scheduler tests | None — `tests/` contains no scheduler tests |

The `priority` field and the config surface were evidently anticipated by the existing
design; the policy itself was never implemented.

---

## 4. Goals and non-goals

### Goals

- An agent can be labelled `high` / `normal` / `low`, and this affects dispatch order.
- Each agent can be given a weight guaranteeing a proportional share of dispatched work.
- A waiting request is guaranteed to be dispatched within a bounded number of scheduling
  ticks, regardless of how many high-priority requests arrive.
- The active policy is selectable by name in `config.yaml`.
- Per-agent usage, waiting time, turnaround time, and a fairness score are observable.
- The policy logic is unit-testable in isolation, without a live LLM backend.

### Non-goals

- Replacing or modifying `FIFOScheduler` / `RRScheduler`, or changing default behaviour.
- Preemptive interruption of an in-flight LLM call. This design schedules **which request
  is dispatched next**; it does not interrupt running generations. (On LLM backends, an
  interrupted generation is not free to resume, so preemption is deliberately a separate
  design problem.)
- Admission control / bounded queues. Complementary work, out of scope here.
- Changes to the agent SDK. The `priority` field already exists on every request; this
  design intends to *consume* it, not extend the API.

---

## 5. Design

### 5.1 Architecture

```
                       config.yaml  (scheduler.policy = "priority")
                            │
                            ▼
                   SchedulerRegistry ──── name → class
                            │
                            ▼
  Agent ──► SyscallExecutor ──► request queues ──► SelectedPolicy  (BaseScheduler subclass)
                                                       │
                                          ┌────────────┴────────────┐
                                          ▼                         ▼
                                  AgentLedger               PolicyDecision
                              (per-agent usage,           (which syscall
                               priority, weight)           is dispatched next)
                                          │                         │
                                          └────────────┬────────────┘
                                                       ▼
                                            LLMAdapter.execute_llm_syscalls(batch)
                                                       │
                                                       ▼
                                      MetricsCollector ──► per-agent stats + fairness score
```

The three existing schedulers keep their shape. A new policy is a `BaseScheduler` subclass
that implements the four `process_*_requests` methods plus `start` / `stop`, and — instead
of draining the queue in arrival order — builds a batch using its policy decision function.

### 5.2 Components

**`SchedulerRegistry`** (`aios/scheduler/registry.py`, new)
Maps a policy name to its class: `fcfs`, `round_robin`, `priority`, `fair_share`. Adding a
future policy becomes a one-line registration. This is the policy/mechanism split: the
dispatch machinery stays fixed while the rule becomes pluggable.

**`AgentLedger`** (`aios/scheduler/accounting.py`, new)
The per-agent record that makes fairness possible. Agents are identified by
`syscall.agent_name`, which AIOS already uses as the agent identity throughout the syscall
path.

```python
@dataclass
class AgentRecord:
    name: str
    priority: int          # 0 = high, 1 = normal, 2 = low
    weight: int            # fair-share weight, default 1
    dispatched: int = 0    # LLM dispatches charged to this agent
    wait_ticks: int = 0    # ticks spent waiting since last dispatch
    pass_value: float = 0.0  # virtual time, used by fair-share
    waiting_since: float | None = None
    total_wait: float = 0.0
    total_turnaround: float = 0.0
```

The ledger is guarded by a single lock; it is small, and fairness is impossible without a
consistent view of it.

**`MetricsCollector`** (`aios/utils/metrics.py`, new)
Aggregates per-agent waiting/turnaround and computes a **Jain fairness index** over
normalized dispatches:

```
J = (Σ xᵢ)² / (n · Σ xᵢ²)        xᵢ = dispatched_i / weight_i
```

`J = 1` means perfectly proportional; lower means a heavier share went to some agents.

### 5.3 Policies

#### `priority` — priority with aging

Each queued request carries a base level from `priority` (`high=0, normal=1, low=2`).

```
effective_level(r) = max(0, r.base_level - r.wait_ticks // aging_interval)
select = the queued request minimising (effective_level, arrival_time)
```

**Starvation bound.** A request whose base level is `L` escalates one level every
`aging_interval` ticks, so it reaches the top level within

```
(L - 0) × aging_interval  ticks        # 2 × aging_interval for a low-priority request
```

Ticks are scheduling rounds, not wall-clock seconds, which keeps the bound independent of
how slow an individual LLM call happens to be. This bound is a testable property, and the
test suite asserts it.

#### `fair_share` — weighted fair queueing with virtual time (stride scheduling)

Standard stride scheduling: agent `i` has weight `wᵢ`; each dispatch advances its virtual
time by the inverse of its weight, and the agent with the smallest virtual time runs next.

```
stride_i  = STRIDE_BASE / w_i
pass_i   += stride_i                 # on dispatch
select    = the queued request whose agent has the smallest pass_i
tie-break = priority level, then arrival_time
```

This guarantees proportional dispatch counts over time. A long-running heavy agent cannot
push a small-weight agent out: the small agent's `pass` stays low and it is chosen quickly.

**What is charged per dispatch.** v1 charges one unit per LLM syscall dispatched. This gives
accurate *call-count* fairness, which is the right model while different backends report
inconsistent token usage. Charging by tokens (via the response usage fields, where
available) is a natural v2 extension and is noted as such rather than half-implemented.

#### `fcfs` / `round_robin`

Retained exactly as they are today, re-exposed through the registry so that
`scheduler.policy` can select them by name. Their behaviour must not change — a regression
test asserts this.

### 5.4 Configuration

```yaml
scheduler:
  log_mode: "console"
  policy: "fcfs"            # fcfs | round_robin | priority | fair_share  (default: fcfs)
  batch_interval: 1.0       # seconds between scheduling rounds
  aging_interval: 5         # ticks before a waiting request escalates one level
  default_priority: "normal" # high | normal | low
  fair_share:
    default_weight: 1
    agents:                 # optional per-agent weights
      research_agent: 3
      chat_agent: 1
```

Defaults reproduce today's behaviour exactly, so an existing `config.yaml` is unaffected.

### 5.5 Compatibility and rollout

- New policies ship **opt-in**; `policy` defaults to `fcfs`.
- `FIFOScheduler` and `RRScheduler` are not modified.
- No new dependencies.
- Agent priority is read from the existing `Syscall.priority`; agents that never set it get
  `default_priority`.
- The new policies pass a **correctly-formed list** of syscalls to
  `LLMAdapter.execute_llm_syscalls(batch)`, which is the documented contract of that method.

### 5.6 Files

| File | Change |
|---|---|
| `aios/scheduler/registry.py` | **New** — policy name → class registry |
| `aios/scheduler/accounting.py` | **New** — `AgentLedger`, `AgentRecord` |
| `aios/scheduler/priority_scheduler.py` | **New** — priority policy with aging |
| `aios/scheduler/fair_share_scheduler.py` | **New** — stride-based fair-share policy |
| `aios/utils/metrics.py` | **New** — fairness index and per-agent reporting |
| `aios/hooks/types/scheduler.py` | Extend `SchedulerParams` with the new fields |
| `runtime/launch.py` | Select the policy from config via the registry |
| `aios/config/config.yaml.example` | Document the new keys |
| `tests/modules/scheduler/` | **New** — unit, property and regression tests |
| `docs/scheduling/priority_scheduling.md` | **This document** |

---

## 6. Testing

| Level | What it covers |
|---|---|
| Unit | Priority ordering; tie-breaks; ledger accounting; registry lookup and unknown-policy handling |
| Property (`hypothesis`) | **Starvation bound**: for arbitrary arrival sequences, a low-priority request is dispatched within `2 × aging_interval` ticks. Also: share adherence within tolerance for arbitrary weight vectors |
| Regression | `fcfs` behaves identically to `FIFOScheduler` on the same input trace |
| Metrics | Waiting/turnaround recorded for every completed syscall (today they are never recorded) |

The policy classes are written so they can be driven with synthetic request objects, so the
entire test suite runs without an LLM backend. This is deliberate: it keeps CI fast and makes
the guarantees checkable in isolation.

---

## 7. Evaluation plan

**Workloads.** Scripted agents with mixed task lengths (short interactive, medium, long
batch) and a Poisson arrival process; plus an overload regime where longer agents are
continuously present.

**Metrics.** Per-agent waiting and turnaround time; worst-case wait of the lowest-priority
agent (the starvation bound in practice); P50/P95/P99 turnaround of short agents; fairness
score; throughput (completed LLM syscalls per minute); scheduler decision overhead in
microseconds.

**Baselines.** `fcfs` as shipped (the default users actually run); `round_robin`; `priority`
**without** aging, to isolate what aging contributes; and an oracle lower bound.

**Expected result.** `priority` sharply reduces short-agent response time; `fair_share`
raises the fairness score while keeping short-agent latency low; the starvation bound holds
under overload; scheduler overhead is negligible relative to LLM call latency.

---

## 8. Open questions for the maintainers

1. Should each new policy be a separate `BaseScheduler` subclass (our assumption), or should
   the priority logic extend `FIFOScheduler`?
2. Preferred naming for the priority levels and config keys?
3. Any constraints from the `aios-rs` rewrite we should respect so the interface stays
   portable?
4. Is `Syscall.priority` intended to be set by the agent SDK, or derived by the kernel from
   agent configuration? We would like to use it as intended.

---

## 9. Prior work

Priority and multi-level scheduling for agent systems has been explored before — notably
*AgentRM: An OS-Inspired Resource Manager for LLM Agent Systems* (arXiv:2603.13110, 2026),
which uses a Multi-Level Feedback Queue. This design does **not** claim priority scheduling
as a novel concept. The contribution is a working, configurable implementation inside the
AIOS kernel consistent with roadmap #127, with an aging-based starvation guarantee and a
measured evaluation. MLFQ is a natural additional policy behind the same registry.

---

## 10. Task checklist

```
[x] Design document
[ ] SchedulerRegistry + config plumbing
[ ] AgentLedger
[ ] Priority policy with aging
[ ] Property test: starvation bound
[ ] Fair-share policy (stride)
[ ] Property test: share adherence
[ ] Metrics + fairness score reporting
[ ] Regression test: fcfs unchanged
[ ] Benchmark harness (scripted workloads)
[ ] Evaluation write-up
```

# agentops-sdk compatibility

This page covers which AOS Control servers each `agentops-sdk` release works
with, how long the server keeps accepting the SDK's current wire format, and
the plan for an OpenTelemetry-based `agentops-sdk` 1.0.

## What 0.x sends

`agentops-sdk` 0.x wraps the Langfuse **v2** Python client (`langfuse>=2,<3`).
Version 2.60.10 is the final v2 release, so v2 is frozen. The SDK talks to
AOS Control over these routes:

| Traffic                                       | Route                                                               | Auth                                                                                     |
| --------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Traces, observations, scores (tracing)        | `POST /api/public/ingestion` (batched JSON)                         | HTTP Basic `pk:sk`                                                                       |
| Prompts, content (`cms/`), tables (`tables/`) | `GET /api/public/v2/prompts/{name}`                                 | HTTP Basic `pk:sk`                                                                       |
| `check_guardrails`, `get_file`                | `POST /api/public/guardrails`, `GET /api/public/files`              | HTTP Basic `pk:sk` from 0.4.1 (0.4.0 sent Bearer, see "Fixed in 0.4.1")                  |
| `check_tool_call`, `check_tool_result`        | `POST /api/public/guardrails` with `stage` `pre_tool` / `post_tool` | Needs a server with the tool stages (#867); an older server answers 400 for these stages |
| `Guardrails.from_agent`                       | `GET /api/public/agents?name=`, `GET /api/public/agents/{id}`       | HTTP Basic `pk:sk`                                                                       |

The ingestion batch holds these event types, and the server must keep
accepting all of them:

| Event type                               | Emitted by                                                                             | Server behaviour                                                                    |
| ---------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `trace-create`                           | `Langfuse().trace()`, `trace.update()`, `@observe` start/end, `update_current_trace`   | Upserts the trace. The v2 SDK uses `trace-create` for updates too.                  |
| `span-create`, `span-update`             | `.span()`, `.update()`, `.end()`                                                       | Merged into one `SPAN` observation                                                  |
| `generation-create`, `generation-update` | `.generation()`, `@observe(as_type="generation")`, `update_current_observation`        | Merged into one `GENERATION`: `model`, `usageDetails`/legacy `usage`, `costDetails` |
| `event-create`                           | `.event()`                                                                             | `EVENT` observation                                                                 |
| `score-create`                           | `.score()`, `Langfuse().score()`                                                       | Numeric, boolean and categorical scores                                             |
| `sdk-log`                                | Memory debug reports (`PYTHONTRACEMALLOC=1` + `LANGFUSE_DEBUG_MEMORY_REPORT_INTERVAL`) | Accepted, logged and dropped. It is in neither `successes` nor `errors`.            |

## Compatibility promise

1. **The server keeps accepting v2 ingestion.** `POST /api/public/ingestion`
   keeps accepting and materialising every event type above, in the shape the
   final v2 client sends. That holds until at least **12 months after
   `agentops-sdk` 1.0 is generally available**, and never before
   **2027-12-31**. Removal gets a deprecation notice in two consecutive server
   minor releases first. _(The dates are a proposal. Alex decides; see "Open
   decisions".)_
2. **0.x only gets additive changes.** New helpers can be added. Existing
   functions keep their signatures, and a server feature that is missing
   degrades with a warning, never silently (see `Guardrails.from_agent`).
3. **Tests enforce it.** Changing either side is a breaking change until the
   deprecation window has run. These tests fail when a change would break
   the promise:
   - `sdk/python/tests/fixtures/langfuse_v2_ingestion/*.json` are batches the
     real `langfuse==2.60.10` client sent to a capture server. They're
     recorded by `sdk/python/scripts/record_langfuse_v2_fixtures.py` in three
     scenarios:
     - `lowlevel`: Base-Agent's handle-based client
     - `decorator`: Securitas's `@observe` + `update_current_*`
     - `sdk_log`
   - `web/src/__tests__/langfuse-v2-sdk-ingestion-contract.servertest.ts`
     replays them against the real ingestion handler. It checks for a 207
     with every event a 201 success, then checks that each entity's events
     are in blob storage and an ingestion job is queued.
   - `worker/src/services/IngestionService/tests/langfuseV2SdkContract.integration.test.ts`
     runs them through `processEventBatch` and the real ingestion queue
     processor. It asserts the ClickHouse results: trace name, session, user,
     tags, metadata and IO; span nesting; generation model, usage, cost and
     `completionStartTime`; event level; and numeric and categorical scores.
   - `sdk/python/tests/test_langfuse_v2_fixtures.py` re-records the scenarios
     with the installed client. It fails if the wire shape drifted from the
     committed fixtures.

### Version matrix

| `agentops-sdk` | Tracing (ingestion)                                                                                                         | Other features                                                                                                                                                                                                                                            |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.4.0          | Any AOS Control server                                                                                                      | `client.update_current_observation` is missing from the published wheel. It falls through to the Langfuse client and raises `AttributeError`, which Securitas works around with `_agentops_patch.py`. `check_guardrails` and `get_file` always get a 401. |
| 0.4.1          | Any AOS Control server. Servers with this change merge same-millisecond `trace-create` events in arrival order (see below). | Ships `update_current_observation` on `AgentOps`, `AsyncAgentOps` and the `agentops` module. With #581: `check_guardrails`/`get_file` use Basic auth, and `Guardrails.from_agent` uses the Agent Registry API and warns instead of silently no-oping.     |

**Same-millisecond merges.** The v2 `@observe` decorator sends up to three
`trace-create` events for one trace: start, `update_current_trace` and end.
They often fall within the same millisecond. Before this change the worker's
event sort reversed same-millisecond creates, so the **first** event won the
merge. `update_current_trace(name=...)` then lost to the function name. The
worker now keeps arrival order on ties.

### Fixed in 0.4.1

- `check_guardrails` and `get_file` (sync and async) sent
  `Authorization: Bearer <secret key>` in 0.4.0. The public API only
  resolves public keys from Bearer headers and needs a project-scoped Basic
  key for those routes, so both calls always got a 401. Fixed in 0.4.1 by
  #581: every public REST call in the SDK uses Basic `pk:sk`. Upgrade from
  0.4.0 to get working server-side guardrail checks and file downloads.

  0.4.1 is released once #581 and this change have both merged (#581 first).

## ADR: `agentops-sdk` 1.0 on OpenTelemetry

**Status:** proposed. Not implemented. **Date:** 2026-09-23.

### Context

- Langfuse v2 is frozen at 2.60.10. Langfuse v3 (mid-2025) and v4 rebuilt the
  Python SDK on OpenTelemetry: an OTel `TracerProvider` with an OTLP/HTTP
  span exporter, `@observe`, `start_as_current_span`/`start_as_current_generation`,
  and `update_current_span`/`update_current_generation`/`update_current_trace`.
  Staying on v2 means no upstream fixes, a growing Python-version risk and an
  SDK shaped unlike the rest of the ecosystem.
- AOS Control already ingests OTLP at `POST /api/public/otel/v1/traces`.
  `OtelIngestionProcessor` understands `langfuse.observation.*` and
  `langfuse.trace.*` attributes (`packages/shared/src/server/otel/attributes.ts`),
  and mappers plug into `ObservationTypeMapperRegistry`.
- AOS-specific helpers live in `agentops` today: local guardrails, tables and
  CMS, Talk correlation, Cloud Run auth, and (with #457)
  `agentops.governance`, which already exports AGT decision spans over
  OTLP to the same endpoint. So 1.0 would have one OTel pipeline for
  tracing and governance, not a v2 batch client next to an OTel exporter.
- The extension-framework design (#571) plans `agentops.extensions` in this
  package for phase 2. An OTel base makes AOS semantic conventions an
  extension point (`otelMappers`) instead of ingestion-schema work.

### Decision

1. `agentops-sdk` 1.0 depends on the **Langfuse v3/v4 Python SDK**, pinned
   to a major version. It configures that SDK's exporter for AOS: base URL
   `AGENTOPS_HOST`, Basic `pk:sk`, and the Cloud Run OIDC header when enabled.
   We keep using Langfuse's decorator and context propagation rather than
   rebuild them.
2. AOS additions stay thin modules on top, each optional:
   - `agentops.guardrails`: unchanged local checks, plus `check_guardrails`
     on Basic auth
   - `agentops.tables` and `get_content`: the prompts REST API, unchanged
   - `agentops.talk`: the turn id as a trace attribute
   - `agentops.governance`: #457, sharing the SDK's `TracerProvider` rather
     than attaching a second exporter
   - `agentops.extensions`: #571, phase 2
3. The top-level surface stays source-compatible where the semantics allow:
   - `agentops.init()`/`AgentOps(...)`
   - `observe`
   - `update_current_trace(...)`
   - `update_current_observation(...)`, which routes to
     `update_current_generation` or `update_current_span` depending on the
     active observation type
   - `flush()`
   - `get_prompt`/`get_content`/`get_table*`

   Langfuse-v2-only APIs (`Langfuse().trace()` handles, `langfuse_context`)
   are not re-exported.

4. The server keeps both transports. v2 batch ingestion stays under the
   promise above. 1.0 gets its own recorded-fixture contract test: OTLP
   protobuf and JSON exports from the pinned v3/v4 client, replayed against
   `/api/public/otel/v1/traces`.
5. Releases: `0.4.x` gets bug fixes only. `1.0.0b*` ships alongside it from
   the same `sdk/python` tree, as a separate build target or branch cut at
   release time. 1.0 goes GA once Base-Agent and Securitas have run on it in
   staging. The v2 deprecation clock starts at 1.0 GA.

### Alternatives considered

- **Stay on v2 indefinitely.** No migration work, but the dependency is
  frozen and diverges further from OTel-based tools, and governance (#457)
  already brings OTel in anyway. Rejected as an end state. It's the plan
  until 1.0 is GA.
- **Plain OpenTelemetry SDK with AOS semantic conventions, no Langfuse
  dependency.** Leanest dependency tree. But we'd rebuild `@observe`,
  generation typing, prompt linking and context helpers, and lose drop-in
  compatibility with Langfuse docs and integrations. Worth revisiting if
  Langfuse's OTel SDK becomes a liability.
- **Fork Langfuse v2.** It keeps working, but we'd own an HTTP batching
  client with no upstream. Rejected.

### Consequences

- The OTel path becomes the primary path for agent telemetry. Gaps G1 to G3
  (Claude Code spans, OTLP logs, metrics) in the dogfooding plan (#568) get
  more important.
- Contract tests double: v2 batches and OTLP exports. Both run against the
  same materialisation assertions.
- Cost and usage come from `langfuse.observation.usage_details` and
  `cost_details` span attributes, so model matching must keep working on the
  OTel path. It's covered by the existing processor.

### Migration notes

**Base-Agent** (`packages/agents-observability-agentops`) uses the v2
handle API directly (see its ADR-0011):

| Base-Agent today (v2)                                    | With 1.0 (v3/v4 base)                                                                             |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `Langfuse().trace(name, session_id, metadata)`           | `client.start_span(name=...)` as the root, then `span.update_trace(session_id=..., metadata=...)` |
| `trace.generation(model, input, metadata)`               | `parent.start_generation(name=..., model=..., input=..., metadata=...)`                           |
| `gen.end(output, usage_details=...)`                     | `gen.update(output=..., usage_details=...)`, then `gen.end()`                                     |
| `trace.update(**fields)`                                 | `root.update_trace(**fields)`                                                                     |
| `Langfuse().flush()`, `atexit` flush                     | `client.flush()` (also `shutdown()` on exit)                                                      |
| `Langfuse().get_prompt(name, label)` in `AgentOpsPrompt` | Same call on the v3/v4 client, or `agentops` `get_prompt`. The prompt cache semantics carry over. |
| `_env_bridge.py` (`AGENTOPS_*` to `LANGFUSE_*`)          | Still needed. v3/v4 also reads `LANGFUSE_BASE_URL`, and `agentops` 1.0 sets it.                   |

The handle-based design (no contextvars) maps one to one because v3/v4
spans are explicit objects, so the observer keeps its "never crash the
agent loop" wrapper. The main change is where trace-level fields go: they
move from `trace.update` to `span.update_trace`. The `agentops_sdk` wheel
URL pin moves to the 1.0 release.

**Securitas** (`core/app/utils/observability/observe.py`) uses
`from agentops import AgentOps, observe`, `client.update_current_trace(...)`,
`client.update_current_observation(...)`, `get_prompt` and
`check_guardrails`:

1. **Now, on 0.4.1:** bump the wheel pin to `v0.4.1` and delete
   `_agentops_patch.py` and its import. The native method makes the shim a
   no-op (its idempotency check returns early), so deleting it is safe
   before or after the bump.
2. **On 1.0:** bump the pin. `@observe(as_type="generation",
capture_input=False, capture_output=False)`, `update_current_trace` and
   `update_current_observation` keep working through `agentops` 1.0's
   routing. Direct `langfuse.decorators` imports, if any are added in the
   meantime, must move to `agentops` or `langfuse` v3/v4 equivalents. The
   `AGENTOPS_*` to `LANGFUSE_*` env copy in `observe.py` stays valid.

### Open decisions

- The v2 ingestion sunset window (proposed: 12 months after 1.0 GA, not
  before 2027-12-31).
- Langfuse v3 or v4 as the 1.0 base (v4 is current, but check the Python
  and OTel versions that Base-Agent's `requires-python==3.12.10` and
  Securitas can take).
- Whether 1.0 lives in `sdk/python` with a separate build target or on a
  release branch.

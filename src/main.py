"""
Oryonix ResearchBot2 — Main Service
====================================
Service:     main
Application: researchbot2
Repository:  researchbot2-main

Pipeline stages (serial, Temporal-durable):
  1. structural_decomposition
  2. evidence_extraction
  3. risk_analysis
  4. comparative_context_generation
  5. executive_summary_synthesis

Human approval gate with unlimited iterative refinement loop.
Streaming via generator-yield; STREAM_COMPLETED appended by SDK automatically.
All @onix.compute and @onix.flow calls use keyword arguments exclusively.
"""

import json
import urllib.error
import urllib.request

import onix


# ---------------------------------------------------------------------------
# Global approval registry
# Keyed by flow_id (equals raw request_id — the Temporal workflow ID).
# Written exclusively by the @onix.signal submit_approval.
# Read exclusively by onix.wait_for_condition inside run_research_pipeline.
# Replayed deterministically by Temporal's event-sourced execution model.
# ---------------------------------------------------------------------------
_approval_registry: dict = {}


# ---------------------------------------------------------------------------
# Gemini API constants
# ---------------------------------------------------------------------------
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    # "gemini-3-flash-preview:generateContent"
    "gemini-2.5-flash-lite:generateContent"
)
_GEMINI_MODEL_LABEL = "gemini"

# ---------------------------------------------------------------------------
# Compute error contract
#
# Context: the Oryonix compute worker has a known bug (unfixed in master) where
# any exception that escapes a @onix.compute WASM boundary is incorrectly reported
# to Temporal as ActivityExecutionResult(Cancellation(...)) rather than
# ActivityExecutionResult(Failure(...)).  Temporal then rejects the completion
# with "unable to mark activity as canceled without activity being request
# canceled first" because no cancellation was requested — it was a plain
# failure.  This kills the workflow entirely.
#
# Workaround: exceptions must never escape @onix.compute functions.  Instead,
# each compute catches Exception internally, encodes the error as a prefixed
# return string, and returns normally.  The flow inspects every result with
# _is_compute_error() and, on a positive match, yields a pipeline_error chunk
# and returns — cleanly terminating the generator from within the flow worker,
# which handles failure correctly.
#
# We catch Exception (not BaseException) so that platform-level signals
# (SystemExit, KeyboardInterrupt) — which the WASM runtime may use to
# forcefully terminate a compute task — are never intercepted.  All real
# application errors (HTTP, JSON, value, connection) are Exception subclasses.
# ---------------------------------------------------------------------------
_COMPUTE_ERROR_PREFIX = "__COMPUTE_ERROR__:"


def _is_compute_error(result: str) -> bool:
    """Return True if the compute function encoded a failure into its return value."""
    return isinstance(result, str) and result.startswith(_COMPUTE_ERROR_PREFIX)


def _extract_compute_error(result: str) -> str:
    """Strip the prefix and return the raw error repr string."""
    return result[len(_COMPUTE_ERROR_PREFIX):]


# ---------------------------------------------------------------------------
# Internal helpers (not onix primitives — pure deterministic utilities)
# ---------------------------------------------------------------------------


def _build_chunk(
    flow_id: str,
    sequence: int,
    stage: str,
    status: str,
    attempt: int,
    artifact: dict,
) -> dict:
    """
    Construct a streaming chunk conforming to the mandatory schema.
    'sequence' is the authoritative ordering field — supplied by the caller
    from a locally-maintained counter.
    'timestamp' is set to an empty string. Generating a live timestamp inside
    a @onix.flow is non-deterministic (violates durable execution replay
    guarantees). The field is informational only and never used for ordering
    or routing; the platform streaming DB holds authoritative server-side times.
    """
    return {
        "flow_id": flow_id,
        "sequence": sequence,
        "stage": stage,
        "status": status,
        "attempt": attempt,
        "artifact": artifact,
        "model": _GEMINI_MODEL_LABEL,
        "timestamp": "",
    }


def _gemini_generate(gemini_api_key: str, prompt: str) -> str:
    """
    gemini-2.5-flash-lite synchronous request.

    Design preserved:
    - stdlib urllib only (WASM safe)
    - fully buffered
    - raises on HTTP or schema error
    """

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
            "topP": 0.9,
        },
    }

    body_bytes = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=_GEMINI_ENDPOINT,
        data=body_bytes,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": gemini_api_key,
        },
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        raw = response.read().decode("utf-8")

    parsed = json.loads(raw)

    try:
        text: str = parsed["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected Gemini response structure: {exc!r}. Raw: {raw[:500]}"
        ) from exc

    # Gemini often wraps JSON responses in markdown code fences
    # (e.g. ```json\n{...}\n```).  Strip them so downstream code and the UI
    # always receive clean text.  We handle both ```json and plain ``` fences.
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove the opening fence line (```json, ```JSON, or just ```)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        # Remove the closing fence if present
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
        text = stripped

    return text


# ---------------------------------------------------------------------------
# Pipeline stage compute functions
# Each is a @onix.compute (Temporal activity):
#   - Accepts gemini_api_key; never hardcodes credentials.
#   - Calls Gemini via _gemini_generate with a stage-specific prompt.
#   - Fully buffers and returns the response text.
#   - Raises on failure; platform handles retry.
#   - Must NOT call other compute functions or flows.
# ---------------------------------------------------------------------------

@onix.compute
def structural_decomposition(abstract: str, gemini_api_key: str) -> str:
    """
    Stage 1 — Structural Decomposition.
    Identifies: research question / hypothesis, methodology, key variables,
    scope and limitations.
    """
    prompt = (
        "You are a rigorous research analyst performing structural decomposition.\n\n"
        "Given the following abstract, identify and extract:\n"
        "1. The central research question or hypothesis.\n"
        "2. The methodology employed (experimental, observational, meta-analytic, etc.).\n"
        "3. Key variables: independent, dependent, and control.\n"
        "4. Scope boundaries and explicitly stated limitations.\n\n"
        "Return your analysis as a valid JSON object with keys:\n"
        "  research_question, hypothesis, methodology, key_variables, "
        "scope, limitations.\n\n"
        f"Abstract:\n{abstract}"
    )
    try:
        return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)
    except Exception as exc:
        return f"{_COMPUTE_ERROR_PREFIX}{exc!r}"


@onix.compute
def evidence_extraction(
    abstract: str,
    decomposition: str,
    gemini_api_key: str,
) -> str:
    """
    Stage 2 — Evidence Extraction.
    Extracts all evidence claims with strength, source type, and confidence.
    Accepts structural decomposition from stage 1.
    """
    prompt = (
        "You are a rigorous research analyst performing evidence extraction.\n\n"
        "Using the structural decomposition as context, examine the abstract and "
        "extract every evidence claim present. For each claim provide:\n"
        "  - claim: verbatim or paraphrased statement.\n"
        "  - evidence_strength: (strong / moderate / weak / anecdotal).\n"
        "  - source_type: (empirical / theoretical / comparative / inferential).\n"
        "  - confidence_level: percentage (0–100).\n"
        "  - notes: any caveats.\n\n"
        "Return a valid JSON object with a top-level key 'evidence_claims' "
        "containing an array of claim objects.\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Structural Decomposition:\n{decomposition}"
    )
    try:
        return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)
    except Exception as exc:
        return f"{_COMPUTE_ERROR_PREFIX}{exc!r}"


@onix.compute
def risk_analysis(
    abstract: str,
    evidence: str,
    gemini_api_key: str,
) -> str:
    """
    Stage 3 — Risk Analysis.
    Identifies methodological, bias, generalizability, and replication risks.
    Accepts evidence extraction output from stage 2.
    """
    prompt = (
        "You are a critical research evaluator performing risk analysis.\n\n"
        "Evaluate the following abstract and its extracted evidence for research risks. "
        "Identify and score each risk category on a scale of 1 (minimal) to 10 (critical):\n"
        "  - methodological_risk: flaws in research design or execution.\n"
        "  - bias_risk: selection, confirmation, publication, or measurement bias.\n"
        "  - generalizability_risk: limitations of applying findings broadly.\n"
        "  - replication_risk: likelihood findings would not replicate.\n"
        "  - confounding_risk: uncontrolled variables that could explain results.\n\n"
        "For each risk provide: score (1–10), rationale, and mitigation_suggestions.\n\n"
        "Return a valid JSON object with a top-level key 'risk_profile' "
        "containing an object for each risk category.\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Extracted Evidence:\n{evidence}"
    )
    try:
        return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)
    except Exception as exc:
        return f"{_COMPUTE_ERROR_PREFIX}{exc!r}"


@onix.compute
def comparative_context_generation(
    abstract: str,
    risk: str,
    gemini_api_key: str,
) -> str:
    """
    Stage 4 — Comparative Context Generation.
    Positions the research relative to established findings and academic landscape.
    Accepts risk analysis output from stage 3.
    """
    prompt = (
        "You are a research contextualization expert generating comparative context.\n\n"
        "For the abstract and risk profile below, produce:\n"
        "  - field_positioning: Where this research sits in the current academic landscape.\n"
        "  - alignment_with_consensus: Areas where findings agree with established literature.\n"
        "  - divergence_from_consensus: Areas where findings conflict or extend existing knowledge.\n"
        "  - novelty_assessment: What is genuinely new, if anything.\n"
        "  - comparable_studies: Description of analogous published research (do not fabricate citations).\n"
        "  - impact_potential: Short-term and long-term impact forecast.\n\n"
        "Return a valid JSON object with the above keys.\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Risk Profile:\n{risk}"
    )
    try:
        return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)
    except Exception as exc:
        return f"{_COMPUTE_ERROR_PREFIX}{exc!r}"


@onix.compute
def executive_summary_synthesis(
    abstract: str,
    decomposition: str,
    evidence: str,
    risk: str,
    context: str,
    gemini_api_key: str,
) -> str:
    """
    Stage 5 — Executive Summary Synthesis (initial attempt).
    Synthesises all prior stage outputs into a decision-ready executive summary.
    All five prior outputs are passed in.
    """
    prompt = (
        "You are a senior research director producing a definitive executive summary.\n\n"
        "Synthesise the following research analysis stages into a concise, "
        "decision-ready executive summary. Include:\n"
        "  - executive_summary: A clear 200–300 word narrative suitable for senior stakeholders.\n"
        "  - key_findings: Bullet array of the 3–5 most important findings.\n"
        "  - overall_confidence: Overall confidence score (0–100) with rationale.\n"
        "  - aggregate_risk_rating: (low / medium / high / critical) with justification.\n"
        "  - recommendation: One of (accept / accept_with_conditions / reject) with explanation.\n"
        "  - conditions: If accept_with_conditions, list required conditions.\n\n"
        "Return a valid JSON object with the above keys.\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Structural Decomposition:\n{decomposition}\n\n"
        f"Evidence Extraction:\n{evidence}\n\n"
        f"Risk Analysis:\n{risk}\n\n"
        f"Comparative Context:\n{context}"
    )
    try:
        return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)
    except Exception as exc:
        return f"{_COMPUTE_ERROR_PREFIX}{exc!r}"


@onix.compute
def refined_executive_summary_synthesis(
    abstract: str,
    last_summary: str,
    critique: str,
    gemini_api_key: str,
) -> str:
    """
    Refinement compute — called on each rejection iteration.

    Per spec §12: feeds ONLY the original abstract, last executive summary,
    and human critique. Does NOT re-feed full prior pipeline output.
    This keeps the refinement prompt focused and avoids compounding prior
    analysis noise across iterations.
    """
    prompt = (
        "You are a senior research director revising an executive summary based on "
        "a human reviewer's critique.\n\n"
        "A prior version of your executive summary has been rejected. "
        "Carefully read the critique and produce a revised executive summary that "
        "directly addresses every point of feedback.\n\n"
        "Return a valid JSON object with exactly these keys:\n"
        "  - executive_summary: Revised narrative (200–300 words).\n"
        "  - key_findings: Updated bullet array of 3–5 findings.\n"
        "  - overall_confidence: Updated confidence score (0–100) with rationale.\n"
        "  - aggregate_risk_rating: Updated (low / medium / high / critical) with justification.\n"
        "  - recommendation: Updated (accept / accept_with_conditions / reject) with explanation.\n"
        "  - conditions: If accept_with_conditions, updated conditions list.\n"
        "  - revision_notes: Explicit description of what was changed and why.\n\n"
        f"Original Abstract:\n{abstract}\n\n"
        f"Previous Executive Summary:\n{last_summary}\n\n"
        f"Human Critique:\n{critique}"
    )
    try:
        return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)
    except Exception as exc:
        return f"{_COMPUTE_ERROR_PREFIX}{exc!r}"


# ---------------------------------------------------------------------------
# Signal — Human approval gate
#
# Decorated with @onix.signal, making it a direct API entrypoint.
# operationId in server.yaml MUST match function name exactly: submit_approval
#
# The endpoint is PATCH /api/approval.
# flow_id is supplied by the caller in the X-Flow-ID request header — NOT in
# the request body. The platform extracts the header value and passes it to
# this function as the flow_id keyword argument, then delivers it as a signal
# to the workflow identified by that flow_id.
#
# Mutates _approval_registry global in-memory state.
# The parent flow polls this state via onix.wait_for_condition.
#
# MUST NOT be wrapped in a flow.
# MUST NOT spawn new flows.
# MUST NOT modify the flow_id.
# ---------------------------------------------------------------------------

@onix.signal
def submit_approval(flow_id: str, approved: bool, critique: str) -> None:
    """
    Receive human approval decision for the given flow.
    Writes approval state into _approval_registry keyed by flow_id.
    The orchestrating flow unblocks when it detects its flow_id present.

    Signal payload:
      flow_id  — sourced from X-Flow-ID request header; format "research:{request_id}"
      approved — true: pipeline accepted; false: trigger refinement iteration
      critique — human feedback text; used in refinement prompt on rejection
    """
    _approval_registry[flow_id] = {
        "approved": approved,
        "critique": critique if critique is not None else "",
    }


# ---------------------------------------------------------------------------
# Flow — Primary pipeline orchestrator
#
# operationId in server.yaml MUST match function name: run_research_pipeline
#
# Generator flow: yields streaming chunks that the platform persists to its
# streaming DB. Chunks are served to the client via GET /api/stream.
# SDK appends STREAM_COMPLETED sentinel automatically.
# Developer MUST NOT append STREAM_COMPLETED.
#
# flow_id is constructed as "research:{request_id}" — a stable, unique,
# human-readable identifier embedded in every chunk. The client reads it
# from the first chunk and uses it verbatim in PATCH /api/approval X-Flow-ID.
# The signal endpoint routes by this value via the Oryonix signal registry.
#
# Sequence counter is local to this invocation; does not rely on DB chunk_id.
# All compute and flow calls use keyword arguments exclusively.
# No parallelisation — stages execute strictly serially.
# ---------------------------------------------------------------------------

@onix.flow
def run_research_pipeline(request_id: str, abstract: str, gemini_api_key: str):
    """
    Orchestrate the full research evaluation pipeline.

    Execution model:
      1. Run five serial pipeline stage compute functions.
      2. Yield streaming chunks after each stage (and sub-steps within stages).
      3. Block at the human approval gate via onix.wait_for_condition.
      4. On approval -> emit final chunk and return.
      5. On rejection -> increment attempt, call refinement compute, loop.
      6. Loop is unlimited; no attempt cap.
    """

    # flow_id is a stable, unique identifier for this pipeline run.
    # Constructed from request_id so it is deterministic and human-readable.
    # Embedded in every chunk; the client reads it from the first chunk and
    # sends it back verbatim in the PATCH /api/approval X-Flow-ID header.
    flow_id: str = f"research:{request_id}"

    # Local sequence counter — authoritative ordering for this stream.
    # Never derived from DB chunk_id; never assumed contiguous externally.
    _sequence: int = 0

    # Attempt counter — starts at 1 for the initial pipeline execution.
    attempt: int = 1

    # ------------------------------------------------------------------
    # Local helpers bound to this flow invocation's mutable state.
    # ------------------------------------------------------------------

    def _next_seq() -> int:
        nonlocal _sequence
        _sequence += 1
        return _sequence

    def _chunk(stage: str, status: str, artifact: dict) -> dict:
        return _build_chunk(
            flow_id=flow_id,
            sequence=_next_seq(),
            stage=stage,
            status=status,
            attempt=attempt,
            artifact=artifact,
        )

    def _error_chunk(failed_stage: str, error_msg: str) -> dict:
        return _build_chunk(
            flow_id=flow_id,
            sequence=_next_seq(),
            stage="pipeline_error",
            status="error",
            attempt=attempt,
            artifact={
                "message": (
                    f"Pipeline failed at stage '{failed_stage}'. "
                    f"Error: {error_msg}"
                ),
                "failed_stage": failed_stage,
            },
        )

    # ==================================================================
    # PIPELINE START
    # ==================================================================

    yield _chunk(
        stage="pipeline_started",
        status="running",
        artifact={
            "message": "Research evaluation pipeline initiated.",
            "request_id": request_id,
            "flow_id": flow_id,
        },
    )

    # ==================================================================
    # STAGE 1 — Structural Decomposition
    # ==================================================================

    yield _chunk(
        stage="structural_decomposition",
        status="started",
        artifact={"message": "Stage 1 of 5: Structural decomposition starting."},
    )
    yield _chunk(
        stage="structural_decomposition",
        status="processing",
        artifact={"message": "Dispatching structural decomposition to Gemini."},
    )

    decomposition: str = structural_decomposition(
        abstract=abstract,
        gemini_api_key=gemini_api_key,
    )
    if _is_compute_error(decomposition):
        yield _error_chunk("structural_decomposition", _extract_compute_error(decomposition))
        return

    yield _chunk(
        stage="structural_decomposition",
        status="processing",
        artifact={"message": "Structural decomposition response received; validating."},
    )
    yield _chunk(
        stage="structural_decomposition",
        status="complete",
        artifact={
            "message": "Structural decomposition complete.",
            "result": decomposition,
        },
    )

    # ==================================================================
    # STAGE 2 — Evidence Extraction
    # ==================================================================

    yield _chunk(
        stage="evidence_extraction",
        status="started",
        artifact={"message": "Stage 2 of 5: Evidence extraction starting."},
    )
    yield _chunk(
        stage="evidence_extraction",
        status="processing",
        artifact={"message": "Dispatching evidence extraction to Gemini."},
    )

    evidence: str = evidence_extraction(
        abstract=abstract,
        decomposition=decomposition,
        gemini_api_key=gemini_api_key,
    )
    if _is_compute_error(evidence):
        yield _error_chunk("evidence_extraction", _extract_compute_error(evidence))
        return

    yield _chunk(
        stage="evidence_extraction",
        status="processing",
        artifact={"message": "Evidence extraction response received; validating."},
    )
    yield _chunk(
        stage="evidence_extraction",
        status="complete",
        artifact={
            "message": "Evidence extraction complete.",
            "result": evidence,
        },
    )

    # ==================================================================
    # STAGE 3 — Risk Analysis
    # ==================================================================

    yield _chunk(
        stage="risk_analysis",
        status="started",
        artifact={"message": "Stage 3 of 5: Risk analysis starting."},
    )
    yield _chunk(
        stage="risk_analysis",
        status="processing",
        artifact={"message": "Dispatching risk analysis to Gemini."},
    )

    risk: str = risk_analysis(
        abstract=abstract,
        evidence=evidence,
        gemini_api_key=gemini_api_key,
    )
    if _is_compute_error(risk):
        yield _error_chunk("risk_analysis", _extract_compute_error(risk))
        return

    yield _chunk(
        stage="risk_analysis",
        status="processing",
        artifact={"message": "Risk analysis response received; validating."},
    )
    yield _chunk(
        stage="risk_analysis",
        status="complete",
        artifact={
            "message": "Risk analysis complete.",
            "result": risk,
        },
    )

    # ==================================================================
    # STAGE 4 — Comparative Context Generation
    # ==================================================================

    yield _chunk(
        stage="comparative_context_generation",
        status="started",
        artifact={"message": "Stage 4 of 5: Comparative context generation starting."},
    )
    yield _chunk(
        stage="comparative_context_generation",
        status="processing",
        artifact={"message": "Dispatching comparative context generation to Gemini."},
    )

    context: str = comparative_context_generation(
        abstract=abstract,
        risk=risk,
        gemini_api_key=gemini_api_key,
    )
    if _is_compute_error(context):
        yield _error_chunk("comparative_context_generation", _extract_compute_error(context))
        return

    yield _chunk(
        stage="comparative_context_generation",
        status="processing",
        artifact={"message": "Comparative context response received; validating."},
    )
    yield _chunk(
        stage="comparative_context_generation",
        status="complete",
        artifact={
            "message": "Comparative context generation complete.",
            "result": context,
        },
    )

    # ==================================================================
    # STAGE 5 — Executive Summary Synthesis (initial)
    # ==================================================================

    yield _chunk(
        stage="executive_summary_synthesis",
        status="started",
        artifact={"message": "Stage 5 of 5: Executive summary synthesis starting."},
    )
    yield _chunk(
        stage="executive_summary_synthesis",
        status="processing",
        artifact={"message": "Dispatching executive summary synthesis to Gemini."},
    )

    summary: str = executive_summary_synthesis(
        abstract=abstract,
        decomposition=decomposition,
        evidence=evidence,
        risk=risk,
        context=context,
        gemini_api_key=gemini_api_key,
    )
    if _is_compute_error(summary):
        yield _error_chunk("executive_summary_synthesis", _extract_compute_error(summary))
        return

    yield _chunk(
        stage="executive_summary_synthesis",
        status="processing",
        artifact={"message": "Executive summary response received; validating."},
    )
    yield _chunk(
        stage="executive_summary_synthesis",
        status="complete",
        artifact={
            "message": "Executive summary synthesis complete.",
            "result": summary,
        },
    )

    # ==================================================================
    # HUMAN APPROVAL GATE — Iterative refinement loop
    # ==================================================================

    while True:

        yield _chunk(
            stage="awaiting_approval",
            status="pending",
            artifact={
                "message": (
                    f"Pipeline paused. Awaiting human approval for attempt {attempt}. "
                    f"Submit signal to flow_id '{flow_id}'."
                ),
                "current_summary": summary,
                "attempt": attempt,
            },
        )

        onix.wait_for_condition(lambda: flow_id in _approval_registry)

        approval_record: dict = _approval_registry.pop(flow_id)
        approved: bool = approval_record["approved"]
        critique: str = approval_record["critique"]

        if approved:
            yield _chunk(
                stage="approval_received",
                status="approved",
                artifact={
                    "message": f"Approval received for attempt {attempt}: accepted.",
                },
            )
            yield _chunk(
                stage="pipeline_complete",
                status="approved",
                artifact={
                    "message": (
                        f"Pipeline approved by human reviewer on attempt {attempt}. "
                        "Research evaluation finalised."
                    ),
                    "final_summary": summary,
                    "total_attempts": attempt,
                },
            )
            return

        yield _chunk(
            stage="approval_received",
            status="rejected",
            artifact={
                "message": (
                    f"Approval received for attempt {attempt}: rejected. "
                    "Beginning refinement."
                ),
                "critique": critique,
            },
        )

        attempt += 1

        yield _chunk(
            stage="refinement_started",
            status="running",
            artifact={
                "message": (
                    f"Attempt {attempt - 1} rejected. "
                    f"Beginning refinement iteration (attempt {attempt})."
                ),
                "critique": critique,
            },
        )
        yield _chunk(
            stage="refined_executive_summary_synthesis",
            status="started",
            artifact={
                "message": (
                    f"Stage 5 (refined): Refined executive summary synthesis "
                    f"starting (attempt {attempt})."
                ),
            },
        )
        yield _chunk(
            stage="refined_executive_summary_synthesis",
            status="processing",
            artifact={
                "message": (
                    f"Dispatching refined executive summary synthesis to Gemini "
                    f"(attempt {attempt})."
                ),
            },
        )

        summary = refined_executive_summary_synthesis(
            abstract=abstract,
            last_summary=summary,
            critique=critique,
            gemini_api_key=gemini_api_key,
        )
        if _is_compute_error(summary):
            yield _error_chunk("refined_executive_summary_synthesis", _extract_compute_error(summary))
            return

        yield _chunk(
            stage="refined_executive_summary_synthesis",
            status="processing",
            artifact={
                "message": (
                    f"Refined executive summary response received; validating "
                    f"(attempt {attempt})."
                ),
            },
        )
        yield _chunk(
            stage="refined_executive_summary_synthesis",
            status="complete",
            artifact={
                "message": f"Refined executive summary complete (attempt {attempt}).",
                "result": summary,
            },
        )

        # Loop back to emit awaiting_approval and block again.

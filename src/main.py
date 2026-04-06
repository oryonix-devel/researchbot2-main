"""
Oryonix ResearchBot2 — Main Service
====================================
Service:     main
Application: researchbot2
Repository:  researchbot2-main

Architecture: stage-per-child-flow
  Each of the five pipeline stages runs as an independent @onix.flow (Temporal
  child workflow), started asynchronously by the main orchestrator. Stages 2–5
  consume their predecessor's platform stream via onix.Stream to extract the
  prior result, then call their own @onix.compute.

  Child flow ID scheme (all deterministic from request_id):
    research:{request_id}:s1  — structural_decomposition_stage
    research:{request_id}:s2  — evidence_extraction_stage
    research:{request_id}:s3  — risk_analysis_stage
    research:{request_id}:s4  — comparative_context_generation_stage
    research:{request_id}:s5  — executive_summary_synthesis_stage

  The main orchestrator (run_research_pipeline) launches all five child flows
  then consumes their streams in order, re-yielding every chunk to the client
  with a globally re-sequenced counter.

  Execution order is serially enforced through stream blocking:
    s2 blocks on s1 stream → s3 blocks on s2 stream → etc.
  Even though all five are launched async, they execute in a strict chain.

  Parent-close policy (Temporal default: terminate): if the main flow returns
  early (pipeline_error), all child workflows are automatically terminated by
  the platform. No manual cleanup needed.

Pipeline stages (serial chain via stream blocking):
  1. structural_decomposition
  2. evidence_extraction
  3. risk_analysis
  4. comparative_context_generation
  5. executive_summary_synthesis

Human approval gate with unlimited iterative refinement loop.
Refinement runs directly in the main flow (no child flow needed).
Streaming via generator-yield; STREAM_COMPLETED appended by SDK automatically.
All @onix.compute and @onix.flow calls use keyword arguments exclusively.
"""

import json
import urllib.error
import urllib.request

import onix


# ---------------------------------------------------------------------------
# Global approval registry
# Keyed by flow_id — format "research:{request_id}" (the Temporal workflow ID).
# Written exclusively by the @onix.signal submit_approval.
# Read exclusively by onix.wait_for_condition inside run_research_pipeline.
# Signals share execution context with their parent flow, so both the signal
# and the flow operate on the same in-process global state.
# Replayed deterministically by Temporal's event-sourced execution model.
# ---------------------------------------------------------------------------
_approval_registry: dict = {}


# ---------------------------------------------------------------------------
# Gemini API constants
# ---------------------------------------------------------------------------
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
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

    Design:
    - stdlib urllib only (WASM safe)
    - fully buffered response
    - raises RuntimeError on HTTP error (with Gemini's response body included)
    - raises on schema error
    """

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 4096,
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

    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Capture the Gemini error response body for actionable error messages.
        # Common bodies: {"error":{"code":400,"message":"API key not valid..."}}
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"Gemini API HTTP {exc.code} {exc.reason}. Body: {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Gemini API connection error: {exc.reason}"
        ) from exc

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
#   - Never lets exceptions escape (compute error contract above).
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
    Accepts structural decomposition from stage 1 as context.
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
    Accepts evidence extraction output from stage 2 as context.
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
    Accepts risk analysis output from stage 3 as context.
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
    Refinement compute — called on each rejection iteration directly from the
    main orchestrator flow (not as a child flow).

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
# The platform routes the signal to the workflow identified by the X-Flow-ID
# request header. The signal function's parameters are populated from the
# request body: flow_id, approved, and critique map directly to the three
# keyword arguments below.
#
# Mutates _approval_registry global in-memory state.
# Signals share execution context with their parent flow, so the flow's
# onix.wait_for_condition lambda can read this same global immediately.
#
# MUST NOT be wrapped in a flow.
# MUST NOT spawn new flows.
# MUST NOT modify the flow_id.
# ---------------------------------------------------------------------------

@onix.signal
def submit_approval(flow_id: str, approved: bool, critique: str) -> None:
    """
    Receive human approval decision for the given workflow.
    Writes approval state into _approval_registry keyed by flow_id.
    The orchestrating flow unblocks when it detects its flow_id present.

    Parameters (all sourced from the PATCH /api/approval request body):
      flow_id  — format "research:{request_id}"; must match the flow_id emitted
                 in every chunk and supplied in the X-Flow-ID header
      approved — true: pipeline accepted; false: trigger refinement iteration
      critique — human feedback text; used in refinement prompt on rejection
    """
    _approval_registry[flow_id] = {
        "approved": approved,
        "critique": critique if critique is not None else "",
    }


# ---------------------------------------------------------------------------
# Child stage flows
#
# Each stage is an independent @onix.flow (Temporal child workflow).
# Started asynchronously by run_research_pipeline.
# Stages 2–5 read their predecessor's stream via onix.Stream to extract
# the prior result, then call their own @onix.compute.
#
# Design rules for every child stage flow:
#   - flow_id in all emitted chunks is "research:{request_id}" — the MAIN
#     workflow's ID. This ensures the client sees a single consistent flow_id
#     across all streamed chunks regardless of which child emitted them.
#   - attempt is always 1 (child flows run once; refinement runs in main flow).
#   - sequence counter is local; the main flow overwrites it when re-yielding.
#   - Never append STREAM_COMPLETED — SDK does this automatically.
#   - The flow_id kwarg passed at call-site is the child's own Temporal
#     workflow ID (e.g. "research:{request_id}:s1"). This is a platform SDK
#     convention (temporary API — will change in a future SDK release) and is
#     distinct from the flow_id embedded in the chunk payload.
#   - All calls to computes and flows use keyword arguments exclusively.
# ---------------------------------------------------------------------------


@onix.flow
def structural_decomposition_stage(request_id: str, abstract: str, gemini_api_key: str):
    """
    Stage 1 child flow — Structural Decomposition.

    No predecessor stream. Calls the structural_decomposition compute directly.
    Yields 4 chunks: started → processing → processing(received) → complete.
    On compute error yields pipeline_error and returns.
    """
    # flow_id in chunks is always the MAIN flow's ID — unified client identity.
    flow_id: str = f"research:{request_id}"
    _sequence: int = 0
    attempt: int = 1

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


@onix.flow
def evidence_extraction_stage(request_id: str, abstract: str, gemini_api_key: str):
    """
    Stage 2 child flow — Evidence Extraction.

    Reads the stage 1 stream (research:{request_id}:s1) via onix.Stream,
    blocking in real-time until structural_decomposition_stage emits its
    complete chunk. Extracts the decomposition result, then calls the
    evidence_extraction compute.

    If stage 1's stream ends without a complete chunk (stage 1 failed),
    emits pipeline_error and returns. The main flow will have already
    detected stage 1's failure and returned; this child is then terminated
    by the platform's parent-close policy.

    Yields 4 chunks: started → processing → processing(received) → complete.
    """
    flow_id: str = f"research:{request_id}"
    s1_id: str = f"research:{request_id}:s1"
    _sequence: int = 0
    attempt: int = 1

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

    # Block on stage 1's stream. Break as soon as the complete chunk arrives.
    # If stage 1 emitted pipeline_error and terminated without a complete chunk,
    # the for loop ends naturally and decomposition remains None.
    decomposition: str = None
    for chunk in onix.StreamConsumer(stream_type=onix.StreamType.Workflow, args=s1_id):
        if (
            chunk.get("stage") == "structural_decomposition"
            and chunk.get("status") == "complete"
        ):
            decomposition = chunk.get("artifact", {}).get("result")
            break

    if decomposition is None:
        yield _error_chunk(
            "evidence_extraction",
            "Predecessor stage 1 (structural_decomposition) did not produce a result. "
            "Stage 1 may have failed.",
        )
        return

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


@onix.flow
def risk_analysis_stage(request_id: str, abstract: str, gemini_api_key: str):
    """
    Stage 3 child flow — Risk Analysis.

    Reads the stage 2 stream (research:{request_id}:s2) via onix.Stream,
    blocking until evidence_extraction_stage emits its complete chunk.
    Stage 2 itself blocked on stage 1 — so by the time stage 3 unblocks,
    both stages 1 and 2 have completed. Calls the risk_analysis compute.

    Yields 4 chunks: started → processing → processing(received) → complete.
    """
    flow_id: str = f"research:{request_id}"
    s2_id: str = f"research:{request_id}:s2"
    _sequence: int = 0
    attempt: int = 1

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

    evidence: str = None
    for chunk in onix.StreamConsumer(stream_type=onix.StreamType.Workflow, args=s2_id):
        if (
            chunk.get("stage") == "evidence_extraction"
            and chunk.get("status") == "complete"
        ):
            evidence = chunk.get("artifact", {}).get("result")
            break

    if evidence is None:
        yield _error_chunk(
            "risk_analysis",
            "Predecessor stage 2 (evidence_extraction) did not produce a result. "
            "Stage 2 may have failed.",
        )
        return

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


@onix.flow
def comparative_context_generation_stage(request_id: str, abstract: str, gemini_api_key: str):
    """
    Stage 4 child flow — Comparative Context Generation.

    Reads the stage 3 stream (research:{request_id}:s3) via onix.Stream,
    blocking until risk_analysis_stage emits its complete chunk.
    Calls the comparative_context_generation compute.

    Yields 4 chunks: started → processing → processing(received) → complete.
    """
    flow_id: str = f"research:{request_id}"
    s3_id: str = f"research:{request_id}:s3"
    _sequence: int = 0
    attempt: int = 1

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

    risk: str = None
    for chunk in onix.StreamConsumer(stream_type=onix.StreamType.Workflow, args=s3_id):
        if (
            chunk.get("stage") == "risk_analysis"
            and chunk.get("status") == "complete"
        ):
            risk = chunk.get("artifact", {}).get("result")
            break

    if risk is None:
        yield _error_chunk(
            "comparative_context_generation",
            "Predecessor stage 3 (risk_analysis) did not produce a result. "
            "Stage 3 may have failed.",
        )
        return

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


@onix.flow
def executive_summary_synthesis_stage(request_id: str, abstract: str, gemini_api_key: str):
    """
    Stage 5 child flow — Executive Summary Synthesis.

    Reads predecessor streams for all four prior stages to collect results.
    Reading order: s1, s2, s3, s4.

    Streams s1, s2, s3 are already-completed durable streams by the time this
    flow reads them (because s4 cannot complete until s3 completes, s3 until s2,
    s2 until s1). Reading s4 last is therefore the only real blocking operation.

    Then calls executive_summary_synthesis with all four prior results.

    Yields 4 chunks: started → processing → processing(received) → complete.
    On any missing predecessor result or compute error yields pipeline_error.
    """
    flow_id: str = f"research:{request_id}"
    s1_id: str = f"research:{request_id}:s1"
    s2_id: str = f"research:{request_id}:s2"
    s3_id: str = f"research:{request_id}:s3"
    s4_id: str = f"research:{request_id}:s4"
    _sequence: int = 0
    attempt: int = 1

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

    # s1: instant replay (always done before s4).
    decomposition: str = None
    for chunk in onix.StreamConsumer(stream_type=onix.StreamType.Workflow, args=s1_id):
        if (
            chunk.get("stage") == "structural_decomposition"
            and chunk.get("status") == "complete"
        ):
            decomposition = chunk.get("artifact", {}).get("result")
            break

    # s2: instant replay.
    evidence: str = None
    for chunk in onix.StreamConsumer(stream_type=onix.StreamType.Workflow, args=s2_id):
        if (
            chunk.get("stage") == "evidence_extraction"
            and chunk.get("status") == "complete"
        ):
            evidence = chunk.get("artifact", {}).get("result")
            break

    # s3: instant replay.
    risk: str = None
    for chunk in onix.StreamConsumer(stream_type=onix.StreamType.Workflow, args=s3_id):
        if (
            chunk.get("stage") == "risk_analysis"
            and chunk.get("status") == "complete"
        ):
            risk = chunk.get("artifact", {}).get("result")
            break

    # s4: the real blocking read — waits for comparative_context_generation_stage.
    context: str = None
    for chunk in onix.StreamConsumer(stream_type=onix.StreamType.Workflow, args=s4_id):
        if (
            chunk.get("stage") == "comparative_context_generation"
            and chunk.get("status") == "complete"
        ):
            context = chunk.get("artifact", {}).get("result")
            break

    # All four must be present. If any predecessor failed its stream ends
    # without a complete chunk and its result is None.
    missing = [
        name
        for name, val in [
            ("decomposition", decomposition),
            ("evidence", evidence),
            ("risk", risk),
            ("context", context),
        ]
        if val is None
    ]
    if missing:
        yield _error_chunk(
            "executive_summary_synthesis",
            f"Missing predecessor results: {', '.join(missing)}. "
            "One or more upstream stages may have failed.",
        )
        return

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


# ---------------------------------------------------------------------------
# Flow — Primary pipeline orchestrator
#
# operationId in server.yaml MUST match function name: run_research_pipeline
#
# Launches all five stage child flows asynchronously, then consumes their
# streams in order, re-yielding every chunk to the client with a globally
# re-sequenced counter. The HTTP connection (POST /api/run response body)
# stays open across onix.wait_for_condition suspension and resumes delivering
# chunks when submit_approval unblocks the flow.
# SDK appends STREAM_COMPLETED sentinel automatically after the generator
# returns. Developer MUST NOT append STREAM_COMPLETED.
#
# Child flow ID scheme (derived deterministically from request_id):
#   s1_id = research:{request_id}:s1  — structural_decomposition_stage
#   s2_id = research:{request_id}:s2  — evidence_extraction_stage
#   s3_id = research:{request_id}:s3  — risk_analysis_stage
#   s4_id = research:{request_id}:s4  — comparative_context_generation_stage
#   s5_id = research:{request_id}:s5  — executive_summary_synthesis_stage
#
# flow_id (the main workflow ID) = "research:{request_id}".
# This is what the client receives in every chunk and sends back verbatim
# in the PATCH /api/approval X-Flow-ID header.
#
# The flow_id kwarg on each child flow call is a temporary SDK API for
# specifying the child's Temporal workflow ID. This will change in a future
# SDK release.
#
# Platform parent-close policy (terminate): if this flow returns early on
# pipeline_error, all child workflows are automatically terminated. No manual
# cancellation logic needed.
#
# All compute and flow calls use keyword arguments exclusively.
# ---------------------------------------------------------------------------

@onix.flow
def run_research_pipeline(request_id: str, abstract: str, gemini_api_key: str):
    """
    Orchestrate the full research evaluation pipeline.

    Execution model:
      1. Emit pipeline_started chunk.
      2. Launch all five stage child flows asynchronously via keyword flow_id.
         Stages 2–5 immediately block on their predecessor's stream; effective
         execution is serial despite async launch.
      3. Consume each child's stream in pipeline order, re-yielding chunks to
         the client with a globally re-sequenced counter. Abort on pipeline_error.
      4. Confirm all child workflows have completed via onix.wait_all.
      5. Block at the human approval gate via onix.wait_for_condition.
      6. On approval -> emit final chunks and return.
      7. On rejection -> increment attempt, call refinement compute directly,
         loop back to approval gate. Loop is unlimited; no attempt cap.
    """

    # Main workflow's stable identity — embedded in every chunk.
    flow_id: str = f"research:{request_id}"

    # Deterministic child workflow IDs.
    s1_id: str = f"research:{request_id}:s1"
    s2_id: str = f"research:{request_id}:s2"
    s3_id: str = f"research:{request_id}:s3"
    s4_id: str = f"research:{request_id}:s4"
    s5_id: str = f"research:{request_id}:s5"

    # Global sequence counter — authoritative ordering for the client stream.
    _sequence: int = 0

    # Attempt counter — starts at 1 for the initial pipeline run.
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
    # Launch all five child stage flows asynchronously.
    #
    # The flow_id kwarg sets each child's Temporal workflow ID. This is a
    # temporary SDK API — it will change in a future SDK release.
    #
    # Stages 2–5 will immediately block on their predecessor's platform
    # stream. Effective execution order is serial (s1 → s2 → s3 → s4 → s5)
    # enforced through stream blocking, not through sequential launching.
    #
    # Handles are stored for onix.wait_all after stream consumption.
    # ==================================================================

    h1 = structural_decomposition_stage(
        request_id=request_id,
        abstract=abstract,
        gemini_api_key=gemini_api_key,
        flow_id=s1_id,
    )
    h2 = evidence_extraction_stage(
        request_id=request_id,
        abstract=abstract,
        gemini_api_key=gemini_api_key,
        flow_id=s2_id,
    )
    h3 = risk_analysis_stage(
        request_id=request_id,
        abstract=abstract,
        gemini_api_key=gemini_api_key,
        flow_id=s3_id,
    )
    h4 = comparative_context_generation_stage(
        request_id=request_id,
        abstract=abstract,
        gemini_api_key=gemini_api_key,
        flow_id=s4_id,
    )
    h5 = executive_summary_synthesis_stage(
        request_id=request_id,
        abstract=abstract,
        gemini_api_key=gemini_api_key,
        flow_id=s5_id,
    )

    # ==================================================================
    # Consume and re-yield chunks from each child flow in pipeline order.
    #
    # Every chunk is shallow-copied and its sequence field is overwritten
    # with the main flow's global counter. All other fields (flow_id, stage,
    # status, attempt, artifact, model, timestamp) are preserved as emitted
    # by the child.
    #
    # On pipeline_error: return immediately. The platform terminates all
    # remaining child workflows via parent-close policy (terminate).
    #
    # summary is captured inline when stage 5's complete chunk passes through.
    # ==================================================================

    summary: str = None
    pipeline_failed: bool = False

    for child_stream_id in [s1_id, s2_id, s3_id, s4_id, s5_id]:
        consumer = onix.StreamConsumer(
            stream_type=onix.StreamType.Workflow,
            args=child_stream_id,
        )
        for chunk in consumer:
            chunk = dict(chunk)
            chunk["sequence"] = _next_seq()
            yield chunk

            if chunk.get("stage") == "pipeline_error":
                pipeline_failed = True
                break

            if (
                chunk.get("stage") == "executive_summary_synthesis"
                and chunk.get("status") == "complete"
            ):
                summary = chunk.get("artifact", {}).get("result", "")

        if pipeline_failed:
            return

    # ==================================================================
    # All five stage streams consumed. Confirm child workflows are done.
    # At this point all streams have been read to completion so wait_all
    # returns immediately — it is a correctness guarantee, not a wait.
    # ==================================================================

    onix.wait_all(handles=[h1, h2, h3, h4, h5])

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

        onix.wait_for_condition(lambda fid=flow_id: fid in _approval_registry)

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
            yield _error_chunk(
                "refined_executive_summary_synthesis",
                _extract_compute_error(summary),
            )
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
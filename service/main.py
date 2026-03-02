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
All @fc.compute and @fc.flow calls use keyword arguments exclusively.
"""

import datetime
import json
import urllib.error
import urllib.request

import fc


# ---------------------------------------------------------------------------
# Global approval registry
# Keyed by flow_id (equals raw request_id — the Temporal workflow ID).
# Written exclusively by the @fc.signal submit_approval.
# Read exclusively by fc.wait_for_condition inside run_research_pipeline.
# Replayed deterministically by Temporal's event-sourced execution model.
# ---------------------------------------------------------------------------
_approval_registry: dict = {}


# ---------------------------------------------------------------------------
# Gemini API constants
# ---------------------------------------------------------------------------
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-pro:generateContent"
)
_GEMINI_MODEL_LABEL = "gemini"


# ---------------------------------------------------------------------------
# Internal helpers (not fc primitives — pure deterministic utilities)
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time in ISO-8601 format with Z suffix."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _build_chunk(
    flow_id: str,
    sequence: int,
    stage: str,
    status: str,
    attempt: int,
    artifact: dict,
) -> dict:
    """
    Construct a streaming chunk conforming to the mandatory schema (§13).
    'sequence' must be supplied by the caller from a locally-maintained counter.
    'timestamp' is set at emission time; it is informational only and not used
    for routing or ordering — sequence is the authoritative ordering field.
    """
    return {
        "flow_id": flow_id,
        "sequence": sequence,
        "stage": stage,
        "status": status,
        "attempt": attempt,
        "artifact": artifact,
        "model": _GEMINI_MODEL_LABEL,
        "timestamp": _utc_now(),
    }


def _gemini_generate(gemini_api_key: str, prompt: str) -> str:
    """
    Perform a synchronous, fully-buffered HTTP POST to the Gemini
    generateContent endpoint.

    Design choices:
    - Uses stdlib urllib only (no external packages; WASM-safe).
    - Raises on any HTTP error or unexpected response shape.
      Exceptions propagate to the Temporal activity runtime, which retries
      according to platform-configured retry policy.
    - Fully buffers the response body before returning.
    - Does not stream tokens.
    """
    url = f"{_GEMINI_ENDPOINT}?key={gemini_api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
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
        url=url,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # urllib.request.urlopen raises urllib.error.HTTPError on 4xx/5xx,
    # propagating naturally to the Temporal activity for retry.
    with urllib.request.urlopen(request) as response:
        raw = response.read().decode("utf-8")

    parsed = json.loads(raw)

    # Validate expected response shape; raise on unexpected structure so
    # Temporal can retry rather than returning corrupted data silently.
    try:
        text: str = parsed["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected Gemini response structure: {exc!r}. Raw: {raw[:500]}"
        ) from exc

    return text


# ---------------------------------------------------------------------------
# Pipeline stage compute functions
# Each is a @fc.compute (Temporal activity):
#   - Accepts gemini_api_key; never hardcodes credentials.
#   - Calls Gemini via _gemini_generate with a stage-specific prompt.
#   - Fully buffers and returns the response text.
#   - Raises on failure; platform handles retry.
#   - Must NOT call other compute functions or flows.
# ---------------------------------------------------------------------------

@fc.compute
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
    return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)


@fc.compute
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
    return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)


@fc.compute
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
    return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)


@fc.compute
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
    return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)


@fc.compute
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
    return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)


@fc.compute
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
    return _gemini_generate(gemini_api_key=gemini_api_key, prompt=prompt)


# ---------------------------------------------------------------------------
# Signal — Human approval gate
#
# Decorated with @fc.signal, making it a direct API entrypoint.
# operationId in server.yaml MUST match function name exactly: submit_approval
#
# Mutates _approval_registry global in-memory state.
# The parent flow polls this state via fc.wait_for_condition.
#
# MUST NOT be wrapped in a flow.
# MUST NOT spawn new flows.
# MUST NOT modify the flow_id.
# ---------------------------------------------------------------------------

@fc.signal
def submit_approval(flow_id: str, approved: bool, critique: str) -> None:
    """
    Receive human approval decision for the given flow.
    Writes approval state into _approval_registry keyed by flow_id.
    The orchestrating flow unblocks when it detects its flow_id present.

    Signal payload (§11):
      flow_id  — equals request_id supplied to POST /api/run
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
# Returns a generator → SDK auto-detects and streams each yielded chunk.
# SDK appends STREAM_COMPLETED sentinel automatically.
# Developer MUST NOT append STREAM_COMPLETED.
#
# Sequence counter is local to this invocation; does not rely on DB chunk_id.
# All compute and flow calls use keyword arguments exclusively.
# No UUID generation inside the flow.
# No parallelisation — stages execute strictly serially.
# ---------------------------------------------------------------------------

@fc.flow
def run_research_pipeline(request_id: str, abstract: str, gemini_api_key: str):
    """
    Orchestrate the full research evaluation pipeline.

    Flow ID: research:{request_id}  (set externally at flow start time)
    The flow_id IS the request_id — the platform uses it as the Temporal workflow ID.

    Execution model:
      1. Run five serial pipeline stage compute functions.
      2. Yield streaming chunks after each stage (and sub-steps within stages).
      3. Block at the human approval gate via fc.wait_for_condition.
      4. On approval → emit final chunk and return.
      5. On rejection → increment attempt, call refinement compute, loop.
      6. Loop is unlimited; no attempt cap.
    """

    # flow_id is embedded in every yielded chunk. The UI extracts it from the
    # first chunk of the POST /api/run SSE stream rather than constructing it
    # client-side, so any value is fine as long as it is stable and unique.
    # We use "research:{request_id}" as a human-readable namespaced identifier.
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
        """
        Emit a chunk with the current flow_id, auto-incremented sequence,
        and current attempt. Reads 'attempt' from enclosing scope at call time,
        so refinement iterations correctly reflect the incremented value.
        """
        return _build_chunk(
            flow_id=flow_id,
            sequence=_next_seq(),
            stage=stage,
            status=status,
            attempt=attempt,
            artifact=artifact,
        )

    def _error_chunk(failed_stage: str, exc: Exception) -> dict:
        """
        Emit a pipeline_error chunk when a @fc.compute activity raises after
        exhausting Temporal retries. Surfaces the raw exception string in the
        SSE stream so the client can display it instead of a silent stream close.
        """
        return _build_chunk(
            flow_id=flow_id,
            sequence=_next_seq(),
            stage="pipeline_error",
            status="error",
            attempt=attempt,
            artifact={
                "message": (
                    f"Pipeline failed at stage '{failed_stage}' after retries. "
                    f"Error: {exc!r}"
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

    try:
        decomposition: str = structural_decomposition(
            abstract=abstract,
            gemini_api_key=gemini_api_key,
        )
    except BaseException as exc:
        yield _error_chunk("structural_decomposition", exc)
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

    try:
        evidence: str = evidence_extraction(
            abstract=abstract,
            decomposition=decomposition,
            gemini_api_key=gemini_api_key,
        )
    except BaseException as exc:
        yield _error_chunk("evidence_extraction", exc)
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

    try:
        risk: str = risk_analysis(
            abstract=abstract,
            evidence=evidence,
            gemini_api_key=gemini_api_key,
        )
    except BaseException as exc:
        yield _error_chunk("risk_analysis", exc)
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

    try:
        context: str = comparative_context_generation(
            abstract=abstract,
            risk=risk,
            gemini_api_key=gemini_api_key,
        )
    except BaseException as exc:
        yield _error_chunk("comparative_context_generation", exc)
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

    try:
        summary: str = executive_summary_synthesis(
            abstract=abstract,
            decomposition=decomposition,
            evidence=evidence,
            risk=risk,
            context=context,
            gemini_api_key=gemini_api_key,
        )
    except BaseException as exc:
        yield _error_chunk("executive_summary_synthesis", exc)
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
    #
    # Loop is semantically:
    #   1. Emit awaiting_approval chunk.
    #   2. Block via fc.wait_for_condition until submit_approval fires.
    #   3. Pop and inspect the approval record.
    #   4. If approved → emit final chunk and return (ends generator).
    #   5. If rejected → increment attempt, run refined_executive_summary_synthesis,
    #      update 'summary', continue loop.
    #
    # The flow_id is stable across all iterations.
    # No new flow is spawned on rejection.
    # Only abstract + last_summary + critique are fed to refinement (§12).
    # The loop is unlimited; no attempt cap (§12).
    # ==================================================================

    while True:

        # Emit the awaiting_approval chunk before blocking.
        # This ensures the client receives the current summary and can present
        # it to the human reviewer.
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

        # Block deterministically until submit_approval writes to _approval_registry.
        # Temporal replays this condition check against replayed signal events.
        # The lambda captures flow_id (immutable string) — safe for replay.
        fc.wait_for_condition(lambda: flow_id in _approval_registry)

        # Consume the approval record atomically.
        # Using dict.pop ensures the entry is removed exactly once.
        # If the flow replays and the signal has not re-fired yet,
        # wait_for_condition will correctly block until it does.
        approval_record: dict = _approval_registry.pop(flow_id)
        approved: bool = approval_record["approved"]
        critique: str = approval_record["critique"]

        if approved:
            # Emit approval_received first so the UI can close the approval modal.
            yield _chunk(
                stage="approval_received",
                status="approved",
                artifact={
                    "message": (
                        f"Approval received for attempt {attempt}: accepted."
                    ),
                },
            )
            # Human accepted — pipeline is complete.
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
            # Returning from the generator signals normal completion.
            # The SDK appends STREAM_COMPLETED automatically.
            return

        # ------------------------------------------------------------------
        # Rejection path — begin a new refinement iteration.
        # ------------------------------------------------------------------

        # Emit approval_received/rejected so UI closes the approval modal
        # and switches back to the pipeline view before refinement begins.
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

        attempt += 1  # Increment before emitting so chunks reflect new attempt.

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

        # Refinement compute: abstract + last summary + critique ONLY (§12).
        try:
            summary = refined_executive_summary_synthesis(
                abstract=abstract,
                last_summary=summary,
                critique=critique,
                gemini_api_key=gemini_api_key,
            )
        except BaseException as exc:
            yield _error_chunk("refined_executive_summary_synthesis", exc)
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

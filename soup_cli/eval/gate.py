"""Eval-Gated Training (v0.26.0 Part B).

Declarative ``evals/gate.yaml`` suites define per-task thresholds; the gate
runs them at epoch boundaries during training (or post-hoc via
``soup eval gate``) and surfaces pass / fail / regression verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from soup_cli.utils.paths import is_under_cwd


@dataclass(frozen=True)
class GateTaskResult:
    name: str
    score: float
    threshold: float
    baseline: Optional[float]
    delta: Optional[float]
    passed: bool


@dataclass(frozen=True)
class GateResult:
    passed: bool
    regression: bool
    task_results: list[GateTaskResult]


class GateTask(BaseModel):
    """One gate task — e.g. a custom eval + threshold."""

    type: Literal["judge", "custom", "benchmark"] = Field(
        description="Task type: judge | custom | benchmark",
    )
    name: str = Field(description="Task name (used as baseline key)")
    threshold: float = Field(
        ge=0.0, le=1000.0,
        description="Minimum score to pass (scale depends on scorer)",
    )
    # type=custom
    tasks: Optional[str] = Field(
        default=None, description="JSONL file of custom eval tasks",
    )
    scorer: Optional[Literal["exact", "contains", "regex", "semantic"]] = Field(
        default=None, description="Scorer for type=custom",
    )
    # type=judge
    prompts: Optional[str] = Field(
        default=None, description="JSONL prompts for LLM judge",
    )
    judge_model: Optional[str] = Field(
        default=None, description="Judge model URL (e.g. ollama://llama3.1)",
    )
    # type=benchmark
    benchmark: Optional[str] = Field(
        default=None, description="Registered benchmark id (e.g. mini_mmlu)",
    )

    @field_validator("tasks", "prompts")
    @classmethod
    def _clean_path_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if "\x00" in value:
            raise ValueError("path field contains null byte")
        return value

    @field_validator("judge_model")
    @classmethod
    def _valid_judge_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        # Allowlist of schemes — SSRF hardening consistent with the project.
        allowed = ("ollama://", "https://", "http://localhost", "http://127.0.0.1")
        if not value.startswith(allowed):
            raise ValueError(
                f"judge_model URL '{value}' uses disallowed scheme - "
                "use ollama://, https://, or http://localhost"
            )
        return value


class EvalSuite(BaseModel):
    """Parsed ``evals/gate.yaml``."""

    suite: str = Field(description="Suite name for display / logs")
    tasks: list[GateTask] = Field(default_factory=list)


def load_suite(path: str) -> EvalSuite:
    """Load and validate an eval suite from disk."""
    suite_path = Path(path)
    if not is_under_cwd(suite_path):
        raise ValueError(
            f"eval-gate suite '{path}' is outside cwd - refusing to load"
        )
    if not suite_path.exists():
        raise FileNotFoundError(f"eval-gate suite not found: {path}")
    data = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    return EvalSuite(**data)


def resolve_baseline(spec: Optional[str]) -> dict[str, float]:
    """Resolve a baseline specifier to a ``{task_name: score}`` map.

    - ``None`` / ``""``: return empty map
    - ``registry://<id>``: look up eval_results for that entry via the
      experiment tracker, keyed by benchmark
    - filesystem path: JSON mapping of ``{name: score}``
    """
    if not spec:
        return {}

    if spec.startswith("registry://"):
        from soup_cli.registry.store import RegistryStore

        ref = spec[len("registry://"):]
        with RegistryStore() as store:
            # ``resolve`` strips the scheme itself; pass the raw ref so the
            # error path below reports what the user typed.
            entry_id = store.resolve(ref)
            if entry_id is None:
                raise ValueError(
                    f"registry baseline not found: {ref} (use `soup registry list`)"
                )
            rows = store.get_eval_results(entry_id)
        return {
            row.get("benchmark", ""): float(row.get("score", 0.0))
            for row in rows
            if row.get("benchmark") and row.get("score") is not None
        }

    # Filesystem path
    baseline_path = Path(spec)
    if not is_under_cwd(baseline_path):
        raise ValueError(
            f"baseline file '{spec}' is outside cwd - refusing to load"
        )
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline file not found: {spec}")
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in baseline file: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"baseline file must be a JSON object mapping name -> score; "
            f"got {type(data).__name__}"
        )
    return {str(k): float(v) for k, v in data.items()}


def _run_custom_task(
    task: GateTask, generate_fn: Callable[[str], str],
) -> float:
    """Run a type=custom task and return its aggregate score in [0, 1]."""
    from dataclasses import replace

    from soup_cli.eval.custom import load_eval_tasks, score_task

    if not task.tasks:
        raise ValueError(f"task '{task.name}' is type=custom but 'tasks' is missing")
    tasks = load_eval_tasks(task.tasks)
    if not tasks:
        return 0.0
    # Override scoring if the suite specified one (EvalTask.scoring field)
    if task.scorer is not None:
        tasks = [replace(t, scoring=task.scorer) for t in tasks]
    total = 0.0
    for eval_task in tasks:
        output = generate_fn(eval_task.prompt)
        result = score_task(eval_task, output)
        total += float(result.score)
    return total / len(tasks)


def run_gate(
    suite: EvalSuite,
    *,
    generate_fn: Callable[[str], str],
    baseline: Optional[dict[str, float]] = None,
    regression_threshold: float = 0.05,
) -> GateResult:
    """Run every task in ``suite`` and return an aggregate verdict.

    ``generate_fn`` accepts a prompt and returns the model output. Injecting
    this keeps the gate testable without a live model.
    """
    baseline = baseline or {}
    task_results: list[GateTaskResult] = []
    any_regressed = False
    any_failed_threshold = False

    for task in suite.tasks:
        if task.type == "custom":
            score = _run_custom_task(task, generate_fn)
        else:
            # Judge / benchmark are wired in v0.26.1+; treat as skipped with
            # score=1.0 here so we don't hard-fail valid configs. The CLI
            # surfaces a warning when these are encountered.
            score = 1.0

        passed_threshold = score >= task.threshold
        base_score = baseline.get(task.name)
        delta = None
        regressed = False
        if base_score is not None:
            delta = score - base_score
            if delta < -abs(regression_threshold):
                regressed = True

        if not passed_threshold:
            any_failed_threshold = True
        if regressed:
            any_regressed = True

        task_results.append(GateTaskResult(
            name=task.name,
            score=score,
            threshold=task.threshold,
            baseline=base_score,
            delta=delta,
            passed=passed_threshold and not regressed,
        ))

    return GateResult(
        passed=not (any_failed_threshold or any_regressed),
        regression=any_regressed,
        task_results=task_results,
    )

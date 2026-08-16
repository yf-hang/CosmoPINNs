from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import ollama
import optuna
from optuna.importance import get_param_importances
from optuna.trial import FrozenTrial, TrialState


# ============================================================
# 1. Configuration
# ============================================================

# Default local Ollama model. Override it at startup with --model or with the
# COSMO_AGENT_MODEL environment variable, then switch interactively if needed.
DEFAULT_MODEL_NAME = (
    os.environ.get("COSMO_AGENT_MODEL", "phi4-mini:latest").strip()
    or "phi4-mini:latest"
)
ACTIVE_MODEL_NAME = DEFAULT_MODEL_NAME
MODEL_CAPABILITIES_CACHE: dict[str, set[str]] = {}

# The real Phase 0 Optuna database included with this project.
# Resolve it relative to this file so the agent works from any
# current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASE_CONFIG_PATH = REPO_ROOT / "config.json"
TRAINING_RUNS_DIR = SCRIPT_DIR / "training_runs"
TRAINING_RUNNER_PATH = SCRIPT_DIR / "run_training_job.py"
DATABASE_PATH = (
    SCRIPT_DIR
    / "phase0_optuna_results"
    / "eps_0"
    / "study.sqlite3"
)
STORAGE = f"sqlite:///{DATABASE_PATH.as_posix()}"

# The real study name stored in the database above.
STUDY_NAME = "phase0_tree_level_eps_0_lr_cde_bc_ratio"

# Example relative SQLite URI:
#
# STORAGE = "sqlite:///cosmopinns_optuna.db"
#
# Example absolute SQLite URI:
# STORAGE = "sqlite:////Users/your_name/project/cosmopinns_optuna.db"

# Maximum number of model/tool turns allowed per answer:
# Ollama -> tool -> Ollama -> tool -> ...
MAX_AGENT_TURNS = 8

# Maximum number of trials returned by get_top_trials.
MAX_TOP_TRIALS = 10

# Context window used for the persistent conversation. The model
# supports a larger window, but 16K keeps local memory use practical.
NUM_CONTEXT_TOKENS = 16384

# Print tool calls and abbreviated tool results in the terminal.
TRACE_TOOL_CALLS = True

# Only one Agent-launched training job is allowed to run at a time. This
# prevents two conversational requests from silently competing for a GPU.
ACTIVE_TRAINING_STATES = {"queued", "running"}


# ============================================================
# 2. JSON utilities
# ============================================================

def to_json(data: Any) -> str:
    """
    Convert Python data to readable JSON.

    default=str prevents uncommon Optuna attribute types from
    causing JSON serialization failures.
    """
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


# ============================================================
# 3. Ollama model selection
# ============================================================

def _installed_ollama_model_names() -> list[str]:
    response = ollama.list()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    names: list[str] = []
    for model in models or []:
        name = getattr(model, "model", None)
        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if name:
            names.append(str(name))
    return sorted(set(names), key=str.lower)


def _resolve_installed_model_name(requested: str) -> str:
    requested = str(requested).strip()
    if not requested:
        raise ValueError("model_name cannot be empty.")

    installed = _installed_ollama_model_names()
    exact = {name.lower(): name for name in installed}
    if requested.lower() in exact:
        return exact[requested.lower()]

    latest = f"{requested}:latest".lower()
    if latest in exact:
        return exact[latest]

    prefix_matches = [
        name
        for name in installed
        if name.split(":", 1)[0].lower() == requested.lower()
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    available = ", ".join(installed) if installed else "(none)"
    raise ValueError(
        f"Ollama model {requested!r} is not installed or is ambiguous. "
        f"Installed models: {available}"
    )


def _get_model_capabilities(model_name: str, *, refresh: bool = False) -> set[str]:
    if not refresh and model_name in MODEL_CAPABILITIES_CACHE:
        return set(MODEL_CAPABILITIES_CACHE[model_name])

    model_info = ollama.show(model_name)
    capabilities = set(getattr(model_info, "capabilities", None) or [])
    if not capabilities and isinstance(model_info, dict):
        capabilities = set(model_info.get("capabilities", []) or [])
    MODEL_CAPABILITIES_CACHE[model_name] = set(capabilities)
    return capabilities


def get_active_model_name() -> str:
    return ACTIVE_MODEL_NAME


def _model_think_setting(model_name: str, capabilities: set[str]) -> str | bool:
    # Ollama's named effort levels are model-specific. gpt-oss supports "low";
    # use disabled thinking for other tool-capable models for broad compatibility.
    if "thinking" in capabilities and model_name.lower().startswith("gpt-oss:"):
        return "low"
    return False


def list_ollama_models() -> str:
    """
    List locally installed Ollama models and their CosmoAgent compatibility.

    Returns:
        JSON containing installed names, capabilities, active state, and
        whether each model supports the tool calling required by CosmoAgent.
    """
    installed = _installed_ollama_model_names()
    entries: list[dict[str, Any]] = []
    for name in installed:
        try:
            capabilities = _get_model_capabilities(name)
            error = None
        except Exception as exc:
            capabilities = set()
            error = f"{type(exc).__name__}: {exc}"
        entries.append({
            "name": name,
            "active": name == ACTIVE_MODEL_NAME,
            "capabilities": sorted(capabilities),
            "cosmoagent_compatible": "tools" in capabilities,
            "thinking_mode": _model_think_setting(name, capabilities),
            "inspection_error": error,
        })
    return to_json({
        "active_model": ACTIVE_MODEL_NAME,
        "default_model": DEFAULT_MODEL_NAME,
        "models": entries,
        "selection": (
            "Use switch_ollama_model, the /model command, --model at startup, "
            "or COSMO_AGENT_MODEL."
        ),
    })


def _activate_compatible_model(model_name: str) -> tuple[str, set[str], str]:
    global ACTIVE_MODEL_NAME

    resolved = _resolve_installed_model_name(model_name)
    capabilities = _get_model_capabilities(resolved)
    if "tools" not in capabilities:
        raise ValueError(
            f"Model {resolved!r} cannot be used by CosmoAgent because it "
            f"does not advertise Ollama tool support. Capabilities: "
            f"{sorted(capabilities)}"
        )
    previous = ACTIVE_MODEL_NAME
    ACTIVE_MODEL_NAME = resolved
    return resolved, capabilities, previous


def switch_ollama_model(model_name: str) -> str:
    """
    Switch CosmoAgent to another locally installed tool-capable Ollama model.

    Call this tool only when the user explicitly asks to change models.

    Args:
        model_name: Installed Ollama model name, with or without :latest.

    Returns:
        JSON confirming the active model or explaining incompatibility.
    """
    try:
        resolved, capabilities, previous = _activate_compatible_model(model_name)
    except ValueError as exc:
        return to_json({
            "status": "not_switched",
            "requested_model": model_name,
            "reason": str(exc),
        })

    return to_json({
        "status": "switched",
        "previous_model": previous,
        "active_model": resolved,
        "capabilities": sorted(capabilities),
        "thinking_mode": _model_think_setting(resolved, capabilities),
        "conversation_history_preserved": True,
    })


# ============================================================
# 4. Optuna study utilities
# ============================================================

def load_current_study() -> optuna.Study:
    """
    Load the configured Optuna study.

    Raises:
        RuntimeError: If the study cannot be loaded or is
        a multi-objective study.
    """
    try:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=STORAGE,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the Optuna study.\n"
            f"STORAGE = {STORAGE}\n"
            f"STUDY_NAME = {STUDY_NAME}\n"
            f"Original error: {exc}"
        ) from exc

    # This implementation currently supports single-objective studies.
    if len(study.directions) != 1:
        raise RuntimeError(
            "This version of CosmoAgent supports only "
            "single-objective Optuna studies."
        )

    return study


def get_direction_name(study: optuna.Study) -> str:
    """Return MINIMIZE or MAXIMIZE."""
    return study.directions[0].name


def get_completed_trials(
    study: optuna.Study,
) -> list[FrozenTrial]:
    """Return completed trials with finite objective values."""
    return [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]


def find_trial(
    study: optuna.Study,
    trial_number: int,
) -> FrozenTrial:
    """
    Find one trial by its Optuna trial number.

    Raises:
        ValueError: If no matching trial exists.
    """
    for trial in study.trials:
        if trial.number == trial_number:
            return trial

    raise ValueError(
        f"Trial number {trial_number} does not exist."
    )


def duration_seconds(
    duration: timedelta | None,
) -> float | None:
    """Convert a timedelta to seconds."""
    if duration is None:
        return None

    return duration.total_seconds()


def trial_to_dict(
    trial: FrozenTrial,
    include_intermediate_values: bool = False,
) -> dict[str, Any]:
    """
    Convert an Optuna FrozenTrial to a JSON-friendly dictionary.
    """
    data: dict[str, Any] = {
        "trial_number": trial.number,
        "state": trial.state.name,
        "objective_value": (
            float(trial.value)
            if trial.value is not None
            else None
        ),
        "parameters": trial.params,
        "user_attributes": trial.user_attrs,
        "system_attributes": trial.system_attrs,
        "datetime_start": (
            trial.datetime_start.isoformat()
            if trial.datetime_start is not None
            else None
        ),
        "datetime_complete": (
            trial.datetime_complete.isoformat()
            if trial.datetime_complete is not None
            else None
        ),
        "duration_seconds": duration_seconds(trial.duration),
    }

    if include_intermediate_values:
        # Keep only the latest 50 values to limit context growth.
        items = sorted(trial.intermediate_values.items())
        data["intermediate_values"] = {
            str(step): float(value)
            for step, value in items[-50:]
        }

    return data


# ============================================================
# 4. Read-only tools exposed to Ollama
# ============================================================

def get_study_summary() -> str:
    """
    Return an overall summary of the current Optuna study.

    The result includes the optimization direction, trial counts,
    current best trial, objective value, and available user-attribute
    metric names.

    Returns:
        A JSON string containing the study summary.
    """
    study = load_current_study()
    completed = get_completed_trials(study)

    state_counts = {
        state.name: sum(
            trial.state == state
            for trial in study.trials
        )
        for state in TrialState
    }

    metric_counts: dict[str, int] = {}

    for trial in completed:
        for key in trial.user_attrs:
            metric_counts[key] = metric_counts.get(key, 0) + 1

    if completed:
        best_trial = study.best_trial

        best_result: dict[str, Any] | None = {
            "trial_number": best_trial.number,
            "objective_value": float(best_trial.value),
            "parameters": best_trial.params,
            "user_attributes": best_trial.user_attrs,
        }
    else:
        best_result = None

    result = {
        "study_name": study.study_name,
        "storage": STORAGE,
        "optimization_direction": get_direction_name(study),
        "total_trials": len(study.trials),
        "trial_state_counts": state_counts,
        "completed_trials_with_objective": len(completed),
        "best_trial": best_result,
        "recorded_user_attribute_keys": metric_counts,
        "warnings": [
            (
                "The best single trial is not necessarily a robust "
                "configuration unless repeated-seed evidence exists."
            ),
            (
                "Optuna parameter importance describes association "
                "within this study and does not establish causality."
            ),
        ],
    }

    return to_json(result)


def _current_study_grounding_snapshot() -> str:
    """Return compact authoritative study evidence for every Agent turn."""
    summary = json.loads(get_study_summary())
    compact = {
        "data_source": str(DATABASE_PATH),
        "study_name": summary["study_name"],
        "optimization_direction": summary["optimization_direction"],
        "total_trials": summary["total_trials"],
        "trial_state_counts": summary["trial_state_counts"],
        "completed_trials_with_objective": summary[
            "completed_trials_with_objective"
        ],
        "best_trial": summary["best_trial"],
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _direct_grounded_study_answer(
    user_message: str,
    grounding_snapshot: str,
) -> str | None:
    """Answer simple best-trial lookups deterministically from SQLite data."""
    text = str(user_message).strip()
    lowered = text.lower()
    robustness_terms = (
        "robust",
        "reliable",
        "repeat",
        "稳健",
        "可靠",
        "重复",
    )
    if any(term in lowered for term in robustness_terms):
        return None

    asks_best_trial = bool(
        re.search(r"\bbest\s+(completed\s+)?trial\b", lowered)
        or re.search(r"\bwhich\b.*\bbest\b.*\btrial\b", lowered)
        or re.search(r"\bwhich\s+one\b.*\bbest\b", lowered)
        or re.search(r"(最佳|最优|最好).*(trial|试验)", lowered)
        or re.search(r"(哪个|哪一个).*(trial|试验).*(最佳|最优|最好)", lowered)
    )
    if not asks_best_trial:
        return None

    snapshot = json.loads(grounding_snapshot)
    best = snapshot.get("best_trial")
    if not isinstance(best, dict):
        return "The configured Optuna study has no completed finite trial."

    number = int(best["trial_number"])
    objective = float(best["objective_value"])
    parameters = best.get("parameters", {})
    direction = str(snapshot["optimization_direction"]).upper()
    criterion = "lowest" if direction == "MINIMIZE" else "highest"
    parameter_text = ", ".join(
        f"{name}={value}"
        for name, value in parameters.items()
    )
    is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)
    if is_chinese:
        criterion_cn = "最低" if direction == "MINIMIZE" else "最高"
        return (
            f"当前最佳的已完成 trial 是 Trial {number}。"
            f"该 study 的方向是 {direction}，它的{criterion_cn}目标值为 "
            f"{objective}；参数为：{parameter_text}。"
        )
    return (
        f"The best completed trial is Trial {number}. "
        f"The study direction is {direction}, and it has the {criterion} "
        f"objective value, {objective}. Parameters: {parameter_text}."
    )


def get_top_trials(n: int = 5) -> str:
    """
    Return the top completed trials ranked by objective value.

    Args:
        n: Number of top trials to return. Must be between 1 and 10.

    Returns:
        A JSON string containing ranked trials, objective differences,
        parameters, and recorded user attributes.
    """
    study = load_current_study()
    completed = get_completed_trials(study)

    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 5

    n = max(1, min(n, MAX_TOP_TRIALS))

    if not completed:
        return to_json({
            "status": "no_completed_trials",
        })

    direction = get_direction_name(study)
    reverse = direction == "MAXIMIZE"

    ranked = sorted(
        completed,
        key=lambda trial: float(trial.value),
        reverse=reverse,
    )[:n]

    best_value = float(ranked[0].value)
    output: list[dict[str, Any]] = []

    for rank, trial in enumerate(ranked, start=1):
        value = float(trial.value)

        if direction == "MINIMIZE":
            signed_difference = value - best_value
        else:
            signed_difference = best_value - value

        if best_value != 0:
            percentage_difference = (
                100.0
                * signed_difference
                / abs(best_value)
            )
        else:
            percentage_difference = None

        output.append({
            "rank": rank,
            "trial_number": trial.number,
            "objective_value": value,
            "absolute_difference_from_best": signed_difference,
            "percentage_difference_from_best": (
                percentage_difference
            ),
            "parameters": trial.params,
            "user_attributes": trial.user_attrs,
            "duration_seconds": duration_seconds(
                trial.duration
            ),
        })

    return to_json({
        "optimization_direction": direction,
        "number_requested": n,
        "trials": output,
    })


def get_trial_details(
    trial_number: int,
) -> str:
    """
    Return detailed information for one Optuna trial.

    Args:
        trial_number: The integer trial number shown by Optuna.

    Returns:
        A JSON string containing trial parameters, objective,
        attributes, timing, state, and recent intermediate values.
    """
    study = load_current_study()

    try:
        trial_number = int(trial_number)
    except (TypeError, ValueError) as exc:
        return to_json({
            "status": "error",
            "reason": (
                "trial_number must be an integer."
            ),
            "details": str(exc),
        })

    try:
        trial = find_trial(study, trial_number)
    except ValueError as exc:
        return to_json({
            "status": "not_found",
            "reason": str(exc),
        })

    return to_json(
        trial_to_dict(
            trial,
            include_intermediate_values=True,
        )
    )


def compare_trials(
    trial_numbers_csv: str,
) -> str:
    """
    Compare selected Optuna trials.

    Args:
        trial_numbers_csv: Comma-separated trial numbers,
            for example "13,7,18".

    Returns:
        A JSON string containing the selected trials ranked
        according to the study optimization direction.
    """
    study = load_current_study()

    try:
        numbers = [
            int(item.strip())
            for item in trial_numbers_csv.split(",")
            if item.strip()
        ]
    except (AttributeError, TypeError, ValueError) as exc:
        return to_json({
            "status": "error",
            "reason": (
                "Provide comma-separated integer trial numbers."
            ),
            "details": str(exc),
        })

    # Remove duplicate trial numbers while preserving their order.
    numbers = list(dict.fromkeys(numbers))

    if not numbers:
        return to_json({
            "status": "error",
            "reason": "No trial numbers were provided.",
        })

    if len(numbers) > 10:
        return to_json({
            "status": "error",
            "reason": (
                "At most 10 trials can be compared at once."
            ),
        })

    selected: list[FrozenTrial] = []
    missing: list[int] = []

    for number in numbers:
        try:
            selected.append(find_trial(study, number))
        except ValueError:
            missing.append(number)

    complete_selected = [
        trial
        for trial in selected
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
    ]

    direction = get_direction_name(study)
    reverse = direction == "MAXIMIZE"

    ranked = sorted(
        complete_selected,
        key=lambda trial: float(trial.value),
        reverse=reverse,
    )

    result = {
        "optimization_direction": direction,
        "requested_trial_numbers": numbers,
        "missing_trial_numbers": missing,
        "ranked_completed_trials": [
            trial_to_dict(
                trial,
                include_intermediate_values=False,
            )
            for trial in ranked
        ],
        "noncompleted_trials": [
            trial_to_dict(
                trial,
                include_intermediate_values=False,
            )
            for trial in selected
            if trial.state != TrialState.COMPLETE
            or trial.value is None
        ],
    }

    return to_json(result)


def get_parameter_importance() -> str:
    """
    Estimate parameter importance for the current Optuna study.

    Returns:
        A JSON string containing normalized importance scores.

    Notes:
        Importance scores are study-dependent associations.
        They must not be described as causal physical effects.
    """
    study = load_current_study()
    completed = get_completed_trials(study)

    if len(completed) < 2:
        return to_json({
            "status": "unavailable",
            "reason": (
                "At least two completed trials are required."
            ),
        })

    try:
        importance = get_param_importances(study)
    except Exception as exc:
        return to_json({
            "status": "unavailable",
            "reason": str(exc),
            "possible_causes": [
                "Too few completed trials.",
                "Conditional parameters are not shared by enough trials.",
                "The objective contains invalid or nonfinite values.",
            ],
        })

    return to_json({
        "status": "available",
        "importance_scores": importance,
        "interpretation": (
            "Higher scores indicate a stronger association with "
            "objective variation inside this Optuna study."
        ),
        "warning": (
            "These values do not prove that a parameter physically "
            "or causally produced an improvement."
        ),
    })


# ============================================================
# 5. CosmoPINNs training tools
# ============================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _normalise_training_phase(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "p0": "phase0",
        "phase0": "phase0",
        "phase_0": "phase0",
        "p0_p1": "phase0_phase1",
        "phase0_phase1": "phase0_phase1",
        "phase_0_phase_1": "phase0_phase1",
        "p0_p2": "phase0_phase2",
        "phase0_phase2": "phase0_phase2",
        "phase_0_phase_2": "phase0_phase2",
    }
    if text not in aliases:
        raise ValueError(
            "phase must be one of: phase0, phase0_phase1, phase0_phase2"
        )
    return aliases[text]


def _normalise_training_device(value: str) -> str:
    device = str(value).strip().lower()
    if device not in {"config", "auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: config, auto, cpu, cuda, mps")
    return device


def _best_training_trial() -> tuple[optuna.Study, FrozenTrial]:
    study = load_current_study()
    completed = get_completed_trials(study)
    if not completed:
        raise RuntimeError("The configured Optuna study has no completed finite trial.")
    trial = study.best_trial
    required = {"learning_rate", "cde_bc_ratio"}
    missing = sorted(required - set(trial.params))
    if missing:
        raise RuntimeError(
            "The best trial is missing required Phase 0 parameters: "
            + ", ".join(missing)
        )
    return study, trial


def _validate_study_compatibility(
    study: optuna.Study,
    config: dict[str, Any],
) -> None:
    signature = study.user_attrs.get("phase0_signature")
    if not isinstance(signature, dict):
        raise RuntimeError(
            "The Optuna study has no phase0_signature, so its physical/model "
            "setup cannot be verified against config.json."
        )

    expected = {
        "eps_global": float(config["eps_global"]),
        "output_part": str(config.get("phase0_output_part", "re")).strip().lower(),
        "domain": [
            float(config["x1_min"]),
            float(config["x1_max"]),
            float(config["x2_min"]),
            float(config["x2_max"]),
        ],
        "hidden_size": int(config["hidden_size"]),
        "n_hidden_layers": int(config["n_hidden_layers"]),
    }
    observed = {
        "eps_global": float(signature["eps_global"]),
        "output_part": str(signature["output_part"]).strip().lower(),
        "domain": [float(value) for value in signature["domain"]],
        "hidden_size": int(signature["hidden_size"]),
        "n_hidden_layers": int(signature["n_hidden_layers"]),
    }
    if observed != expected:
        raise RuntimeError(
            "The Optuna study is incompatible with the current Phase 0 "
            "physical/model configuration. Run a matching study before "
            f"training. Study signature={observed}; config signature={expected}"
        )


def _build_training_config(
    *,
    phase: str,
    phase0_epochs: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = _normalise_training_phase(phase)
    device = _normalise_training_device(device)
    try:
        phase0_epochs = int(phase0_epochs)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase0_epochs must be an integer.") from exc
    if phase0_epochs < 0:
        raise ValueError("phase0_epochs must be zero or positive.")

    config = _read_json_object(BASE_CONFIG_PATH)
    study, trial = _best_training_trial()
    _validate_study_compatibility(study, config)
    learning_rate = float(trial.params["learning_rate"])
    cde_bc_ratio = float(trial.params["cde_bc_ratio"])

    # Train from scratch with production data/epoch settings from config.json,
    # replacing only the Phase 0 hyperparameters selected by Optuna.
    overrides: dict[str, Any] = {
        "learning_rate_p0": learning_rate,
        "lambda1": cde_bc_ratio,
        "lambda2": 1.0,
        "use_local_config": False,
        "reuse_saved_models": False,
        "reuse_eval_bundle": False,
        "phase0_model_load_path": "",
        "phase0_history_load_path": "",
        "phase0_eval_bundle_load_path": "",
        "phase1_model_load_path": "",
        "phase1_history_load_path": "",
        "phase1_eval_bundle_load_path": "",
        "phase2_model_load_path": "",
        "phase2_history_load_path": "",
        "phase2_eval_bundle_load_path": "",
        "run_phase1_only": False,
        "run_phase2_only": False,
        "enable_phase1": phase == "phase0_phase1",
        "enable_phase2": phase == "phase0_phase2",
        "save_phase_artifacts": True,
    }
    if phase0_epochs > 0:
        overrides["phase0_epochs"] = phase0_epochs
    if device != "config":
        overrides["device"] = device
    config.update(overrides)

    selection = {
        "study_name": study.study_name,
        "trial_number": int(trial.number),
        "objective_value": float(trial.value),
        "optuna_parameters": {
            "learning_rate": learning_rate,
            "cde_bc_ratio": cde_bc_ratio,
        },
        "config_mapping": {
            "learning_rate_p0": learning_rate,
            "lambda1": cde_bc_ratio,
            "lambda2": 1.0,
        },
        "phase": phase,
        "phase0_epochs": int(config["phase0_epochs"]),
        "device": str(config["device"]),
    }
    return config, selection


def preview_cosmopinns_training(
    phase: str = "phase0",
    phase0_epochs: int = 0,
    device: str = "config",
) -> str:
    """
    Preview a CosmoPINNs training configuration without starting training.

    Args:
        phase: Training pipeline: phase0, phase0_phase1, or phase0_phase2.
        phase0_epochs: Optional Phase 0 epoch override. Use 0 to keep config.json.
        device: config, auto, cpu, cuda, or mps.

    Returns:
        JSON describing the selected Optuna trial and effective training setup.
    """
    config, selection = _build_training_config(
        phase=phase,
        phase0_epochs=phase0_epochs,
        device=device,
    )
    return to_json({
        "status": "preview",
        "training_started": False,
        "selection": selection,
        "production_settings": {
            "n_coll": config.get("n_coll"),
            "phase0_epochs": config.get("phase0_epochs"),
            "warmup_epochs_p0": config.get("warmup_epochs_p0"),
            "phase0_output_part": config.get("phase0_output_part"),
            "eps_global": config.get("eps_global"),
            "enable_phase1": config.get("enable_phase1"),
            "enable_phase2": config.get("enable_phase2"),
        },
        "note": (
            "The Optuna checkpoint is not resumed. Final training starts from "
            "a new model using the selected hyperparameters."
        ),
    })


def _is_pid_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
        if pid_int <= 0:
            return False
        os.kill(pid_int, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _job_metadata_paths() -> list[Path]:
    if not TRAINING_RUNS_DIR.is_dir():
        return []
    return sorted(
        TRAINING_RUNS_DIR.glob("*/job.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )


def _refresh_job(payload: dict[str, Any], metadata_path: Path) -> dict[str, Any]:
    if payload.get("status") == "running" and not _is_pid_running(payload.get("pid")):
        payload["status"] = "terminated_unknown"
        payload["finished_at"] = payload.get("finished_at") or _utc_now()
        payload["status_note"] = (
            "The recorded runner process is no longer active and did not "
            "write a final exit status."
        )
        _write_json_atomic(metadata_path, payload)
    return payload


def _active_training_jobs() -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for path in _job_metadata_paths():
        payload = _refresh_job(_read_json_object(path), path)
        status = payload.get("status")
        if status == "queued":
            try:
                created = datetime.fromisoformat(str(payload["created_at"]))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if (now - created).total_seconds() > 300:
                    payload["status"] = "launch_failed"
                    payload["finished_at"] = _utc_now()
                    payload["status_note"] = "Runner did not start within five minutes."
                    _write_json_atomic(path, payload)
                    continue
            except (KeyError, TypeError, ValueError):
                pass
        if payload.get("status") in ACTIVE_TRAINING_STATES:
            active.append(payload)
    return active


def start_cosmopinns_training(
    phase: str = "phase0",
    phase0_epochs: int = 0,
    device: str = "config",
) -> str:
    """
    Start a background CosmoPINNs run using the best Optuna Phase 0 trial.

    Call this state-changing tool only when the user explicitly asks to start
    or run training. It creates an isolated config, log, and results directory.

    Args:
        phase: Training pipeline: phase0, phase0_phase1, or phase0_phase2.
        phase0_epochs: Optional Phase 0 epoch override. Use 0 to keep config.json.
        device: config, auto, cpu, cuda, or mps.

    Returns:
        JSON with the launched job ID and paths used to monitor the run.
    """
    active = _active_training_jobs()
    if active:
        return to_json({
            "status": "not_started",
            "reason": "Another CosmoAgent training job is already active.",
            "active_jobs": [
                {
                    "job_id": item.get("job_id"),
                    "status": item.get("status"),
                    "pid": item.get("pid"),
                }
                for item in active
            ],
        })

    config, selection = _build_training_config(
        phase=phase,
        phase0_epochs=phase0_epochs,
        device=device,
    )
    phase_name = selection["phase"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{phase_name}-{timestamp}-{uuid.uuid4().hex[:8]}"
    job_dir = TRAINING_RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    config_path = job_dir / "config.json"
    metadata_path = job_dir / "job.json"
    log_path = job_dir / "training.log"
    results_path = job_dir / "results"
    results_root = results_path.relative_to(REPO_ROOT).as_posix()

    _write_json_atomic(config_path, config)
    metadata: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _utc_now(),
        "pid": None,
        "exit_code": None,
        "selection": selection,
        "paths": {
            "job_dir": str(job_dir),
            "config": str(config_path),
            "log": str(log_path),
            "results": str(results_path),
        },
    }
    _write_json_atomic(metadata_path, metadata)

    command = [
        sys.executable,
        str(TRAINING_RUNNER_PATH),
        "--job-dir",
        str(job_dir),
        "--results-root",
        results_root,
    ]
    try:
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception as exc:
        metadata.update({
            "status": "launch_failed",
            "finished_at": _utc_now(),
            "launch_error": f"{type(exc).__name__}: {exc}",
        })
        _write_json_atomic(metadata_path, metadata)
        return to_json(metadata)

    return to_json({
        "status": "started",
        "job_id": job_id,
        "runner_pid": process.pid,
        "selection": selection,
        "paths": metadata["paths"],
        "next_action": (
            "Use get_training_status with this job_id to read status and log output."
        ),
    })


def _resolve_job_metadata(job_id: str) -> Path:
    paths = _job_metadata_paths()
    if str(job_id).strip().lower() == "latest":
        if not paths:
            raise FileNotFoundError("No CosmoAgent training jobs exist.")
        return paths[0]

    job_id = str(job_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid job_id.")
    path = TRAINING_RUNS_DIR / job_id / "job.json"
    if not path.is_file():
        raise FileNotFoundError(f"Training job does not exist: {job_id}")
    return path


def get_training_status(
    job_id: str = "latest",
    tail_lines: int = 30,
) -> str:
    """
    Return status and recent log lines for a CosmoAgent training job.

    Args:
        job_id: Job ID returned by start_cosmopinns_training, or latest.
        tail_lines: Number of recent log lines to return, from 0 through 100.

    Returns:
        JSON containing persisted process state, paths, and the log tail.
    """
    try:
        tail_lines = max(0, min(int(tail_lines), 100))
    except (TypeError, ValueError):
        tail_lines = 30
    metadata_path = _resolve_job_metadata(job_id)
    payload = _refresh_job(_read_json_object(metadata_path), metadata_path)
    log_path = Path(payload.get("paths", {}).get("log", ""))
    log_tail: list[str] = []
    if tail_lines and log_path.is_file():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            log_tail = handle.readlines()[-tail_lines:]
        log_tail = [line.rstrip("\n") for line in log_tail]
    payload["log_tail"] = log_tail
    return to_json(payload)


def list_training_jobs(limit: int = 10) -> str:
    """
    List recent CosmoAgent-launched CosmoPINNs training jobs.

    Args:
        limit: Maximum number of recent jobs to return, from 1 through 20.

    Returns:
        JSON containing compact job status records.
    """
    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = 10
    jobs: list[dict[str, Any]] = []
    for path in _job_metadata_paths()[:limit]:
        payload = _refresh_job(_read_json_object(path), path)
        jobs.append({
            "job_id": payload.get("job_id"),
            "status": payload.get("status"),
            "created_at": payload.get("created_at"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "exit_code": payload.get("exit_code"),
            "selection": payload.get("selection"),
            "paths": payload.get("paths"),
        })
    return to_json({"jobs": jobs, "count": len(jobs)})


# ============================================================
# 6. Tool registry
# ============================================================

AVAILABLE_FUNCTIONS: dict[
    str,
    Callable[..., str],
] = {
    "list_ollama_models": list_ollama_models,
    "switch_ollama_model": switch_ollama_model,
    "get_study_summary": get_study_summary,
    "get_top_trials": get_top_trials,
    "get_trial_details": get_trial_details,
    "compare_trials": compare_trials,
    "get_parameter_importance": get_parameter_importance,
    "preview_cosmopinns_training": preview_cosmopinns_training,
    "start_cosmopinns_training": start_cosmopinns_training,
    "get_training_status": get_training_status,
    "list_training_jobs": list_training_jobs,
}

# Ollama Python SDK can derive tool schemas from signatures/docstrings
TOOLS = list(AVAILABLE_FUNCTIONS.values())


PSEUDO_TOOL_PATTERN = re.compile(
    r"\[DATA\]\s*(.*?)\s*\[/DATA\]",
    flags=re.IGNORECASE | re.DOTALL,
)


def _invoke_registered_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    arguments = dict(arguments or {})
    function = AVAILABLE_FUNCTIONS.get(tool_name)
    if function is None:
        return to_json({
            "status": "error",
            "reason": f"Unknown tool: {tool_name}",
        })
    try:
        return function(**arguments)
    except TypeError as exc:
        return to_json({
            "status": "error",
            "tool": tool_name,
            "reason": "Invalid tool arguments.",
            "details": str(exc),
        })
    except Exception as exc:
        return to_json({
            "status": "error",
            "tool": tool_name,
            "reason": str(exc),
        })


def _extract_pseudo_tool_call(
    content: str,
) -> tuple[str, dict[str, Any]] | None:
    """
    Parse text-style tool requests emitted by some smaller Ollama models.

    Example accepted form: [DATA]GET_STUDY_SUMMARY[/DATA]
    """
    match = PSEUDO_TOOL_PATTERN.search(content or "")
    if match is None:
        return None

    body = match.group(1).strip()
    arguments: dict[str, Any] = {}
    if body.startswith("{"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return ("__invalid_pseudo_tool__", {})
        name = str(payload.get("name", "")).strip()
        raw_arguments = payload.get("arguments", {})
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
    else:
        name_match = re.fullmatch(
            r"([A-Za-z0-9_]+)(?:\s*:\s*(\{.*\}))?",
            body,
            flags=re.DOTALL,
        )
        if name_match is None:
            return ("__invalid_pseudo_tool__", {})
        name = name_match.group(1)
        if name_match.group(2):
            try:
                parsed_arguments = json.loads(name_match.group(2))
            except json.JSONDecodeError:
                return ("__invalid_pseudo_tool__", {})
            if isinstance(parsed_arguments, dict):
                arguments = parsed_arguments

    normalized = name.strip().lower()
    return normalized, arguments


# ============================================================
# 7. Agent instructions
# ============================================================

SYSTEM_PROMPT = """
You are CosmoAgent, a scientific machine-learning experiment-analysis
and training-operations agent for CosmoPINNs.

You have tools that query an existing Optuna study and tools that can
launch and monitor isolated CosmoPINNs training jobs.

Core behavior:
1. Understand the user's question.
2. Decide which tools are needed.
3. Call only the necessary tools.
4. Inspect the returned study data.
5. If more evidence is required, call another appropriate tool.
6. Stop when enough evidence has been collected.
7. Answer in the same language as the user.

Scientific rules:
- Obtain every numerical statement about the Optuna study from tools.
- Never invent trials, objective values, parameters, metrics, or
  physical explanations.
- Respect the study's MINIMIZE or MAXIMIZE direction.
- Say "lowest objective value" for a MINIMIZE study.
- A best trial is not automatically a robust configuration.
- Do not claim robustness without repeated-seed evidence.
- Parameter importance indicates association, not causality.
- Clearly distinguish observations from hypotheses.
- If the study lacks the data needed to answer, say so explicitly.
- Do not recommend a larger dataset unless dataset information was
  actually provided.
- Never modify Optuna trials or the Optuna database.
- Start training only when the user explicitly asks to start, run, or
  launch training. Questions about whether training is possible, requests
  for explanations, and hypothetical discussions do not authorize a run.
- When a user explicitly asks to train with the best Optuna result, call
  start_cosmopinns_training. Do not merely describe shell commands.
- The default training pipeline is phase0. Use phase0_phase1 or
  phase0_phase2 only when the user explicitly requests that transfer stage.
- phase0_epochs=0 means keep the production value from config.json.
- Use preview_cosmopinns_training when the user asks to inspect the
  effective configuration without launching work.
- Training runs in the background. After launching, report the job ID and
  tell the user that get_training_status can monitor it.
- Use get_training_status for progress or completion questions. Never
  claim success until its persisted status is succeeded.
- When the user asks which local models are available, call
  list_ollama_models.
- Switch models only when the user explicitly asks to switch or use a
  named model. Call switch_ollama_model and report whether it succeeded.
- A model without Ollama tool support is incompatible with CosmoAgent,
  even if it can answer ordinary chat questions.
- A CURRENT_STUDY_SNAPSHOT system message is authoritative and comes
  directly from the configured SQLite database. Use it for basic study
  facts and never contradict it.
- Never print textual placeholders such as [DATA]GET_STUDY_SUMMARY[/DATA].
  Use native tool calls. If a compatibility layer supplies a tool result,
  answer from that result.
- Do not repeatedly call the same tool with identical arguments unless
  new information makes it necessary.

Conversation rules:
- Remember results already obtained earlier in the conversation.
- A follow-up question may be answered from previous tool results.
- When the user asks for a new numerical fact, use the relevant tool.
- Keep final answers precise and scientifically conservative.
"""


# ============================================================
# 8. Persistent conversational Agent session
# ============================================================

class CosmoAgentSession:
    """
    Persistent conversational Agent.

    The message history stores:
    - system instructions;
    - user messages;
    - assistant tool requests;
    - tool results;
    - final assistant answers.
    """

    def __init__(self) -> None:
        self.messages: list[Any] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

    def ask(self, user_message: str) -> str:
        """
        Send one user message and run the multi-turn Agent loop.

        The model may call zero, one, or several tools before
        producing its final answer.
        """
        grounding_snapshot = _current_study_grounding_snapshot()
        self.messages.append({
            "role": "system",
            "content": (
                "CURRENT_STUDY_SNAPSHOT (authoritative SQLite data; do not "
                "invent or contradict numerical study facts):\n"
                + grounding_snapshot
            ),
        })
        self.messages.append({
            "role": "user",
            "content": user_message,
        })
        direct_answer = _direct_grounded_study_answer(
            user_message,
            grounding_snapshot,
        )
        if direct_answer is not None:
            self.messages.append({
                "role": "assistant",
                "content": direct_answer,
            })
            return direct_answer

        for agent_turn in range(
            1,
            MAX_AGENT_TURNS + 1,
        ):
            model_name = get_active_model_name()
            capabilities = _get_model_capabilities(model_name)
            response: ollama.ChatResponse = ollama.chat(
                model=model_name,
                messages=self.messages,
                tools=TOOLS,
                think=_model_think_setting(model_name, capabilities),
                options={
                    "temperature": 0,
                    "num_ctx": NUM_CONTEXT_TOKENS,
                },
            )

            # Preserve the assistant message, including any tool calls.
            self.messages.append(response.message)

            tool_calls = (
                response.message.tool_calls or []
            )

            # No tool call means the model is ready to answer.
            if not tool_calls:
                final_content = (
                    response.message.content or ""
                ).strip()

                if not final_content:
                    return (
                        "The model returned an empty response "
                        "without requesting a tool."
                    )

                pseudo_call = _extract_pseudo_tool_call(final_content)
                if pseudo_call is not None:
                    tool_name, arguments = pseudo_call
                    tool_result = _invoke_registered_tool(tool_name, arguments)
                    if TRACE_TOOL_CALLS:
                        print(f"\n[Compatibility tool turn {agent_turn}]")
                        print(f"[Text tool request] {tool_name}")
                        print("[Tool result]\n" + tool_result[:1500])
                    self.messages.append({
                        "role": "system",
                        "content": (
                            f"AUTHORITATIVE TOOL RESULT for {tool_name}:\n"
                            f"{tool_result}\n"
                            "Now answer the user's question directly. Do not "
                            "repeat a [DATA] placeholder."
                        ),
                    })
                    continue

                return final_content

            if TRACE_TOOL_CALLS:
                print(
                    f"\n[Agent tool turn {agent_turn}]"
                )

            # One assistant message may contain several tool calls.
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = dict(
                    tool_call.function.arguments or {}
                )

                if TRACE_TOOL_CALLS:
                    print(f"[Tool] {tool_name}")
                    print(
                        "[Arguments] "
                        + to_json(arguments)
                    )

                tool_result = _invoke_registered_tool(tool_name, arguments)

                if TRACE_TOOL_CALLS:
                    preview = tool_result

                    if len(preview) > 1500:
                        preview = (
                            preview[:1500]
                            + "\n... [truncated]"
                        )

                    print(
                        "[Tool result]\n"
                        + preview
                    )

                # Add the real tool result for the model's next turn.
                self.messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": tool_result,
                })

        return (
            "CosmoAgent reached the maximum number "
            f"of tool turns ({MAX_AGENT_TURNS}) "
            "without producing a final answer."
        )

    def reset(self) -> None:
        """Clear the conversation while preserving system rules."""
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

    def message_count(self) -> int:
        """Return the number of stored conversation messages."""
        return len(self.messages)


# ============================================================
# 9. Startup checks
# ============================================================

def check_ollama_configuration(model_name: str) -> tuple[str, set[str]]:
    """Verify and activate one installed tool-capable Ollama model."""
    try:
        resolved, capabilities, _ = _activate_compatible_model(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Unable to activate the requested Ollama model.\n"
            f"MODEL = {model_name}\n"
            f"Original error: {exc}\n"
            "Make sure Ollama is running, then use `ollama list` to inspect "
            "installed models."
        ) from exc
    return resolved, capabilities


def check_study_configuration() -> None:
    """
    Verify that the configured Optuna study can be loaded.
    """
    study = load_current_study()
    completed = get_completed_trials(study)

    print("Optuna study loaded successfully.")
    print(f"Database path: {DATABASE_PATH}")
    print(f"Study name: {study.study_name}")
    print(f"Direction: {get_direction_name(study)}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len(completed)}")

    if completed:
        print(
            "Best trial: "
            f"{study.best_trial.number}"
        )
        print(
            "Best objective value: "
            f"{study.best_value}"
        )


# ============================================================
# 10. Command-line chat interface
# ============================================================

HELP_TEXT = """
Available commands:

  /help
      Show this help message.

  /reset
      Clear the current conversation history.

  /status
      Show the active model and number of messages in the current session.

  /models
      List installed Ollama models and CosmoAgent compatibility.

  /study
      Print the current SQLite study summary without using the language model.

  /model MODEL_NAME
      Switch models while preserving the conversation, for example:
      /model qwen3:1.7b

  /exit
      Exit CosmoAgent.

Example questions:

  How many completed, pruned, and failed trials are in this study?

  Which trial is currently best, and what parameters did it use?

  Compare the top five trials. Is the best trial clearly ahead?

  Compare trial 13 with trial 7.

  Which additional metrics were recorded for trial 13?

  Which parameter has the strongest association with objective variation?

  Does the current evidence show that the best parameters are robust?

  Summarize what the study establishes and what evidence is still missing.

  Preview the configuration that would train Phase 0 with the best trial.

  Start Phase 0 training with the best Optuna parameters.

  What is the status of the latest training job?

  Switch to qwen3:1.7b for the rest of this conversation.
"""


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CosmoPINNs Optuna analysis and training agent."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Installed tool-capable Ollama model "
            f"(default: {DEFAULT_MODEL_NAME!r})."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List installed Ollama models and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if args.list_models:
        try:
            print(list_ollama_models())
        except Exception as exc:
            print(f"Unable to list Ollama models: {exc}")
            sys.exit(1)
        return

    print("=" * 64)
    print("CosmoAgent: Optuna Analysis and Training Agent")
    print("=" * 64)

    try:
        resolved_model, capabilities = check_ollama_configuration(args.model)
        print(f"Ollama model ready: {resolved_model}")
        print(
            "Model capabilities: "
            + ", ".join(sorted(capabilities))
        )
        check_study_configuration()
    except Exception as exc:
        print("\nStartup error:")
        print(exc)
        print(
            "\nEdit STORAGE and STUDY_NAME at the "
            "top of cosmo_agent.py."
        )
        sys.exit(1)

    print(f"\nOllama context window: {NUM_CONTEXT_TOKENS} tokens")
    print(
        "Training jobs are launched only after an explicit conversational "
        "request and are saved below agent/training_runs/."
    )
    print("\nType /help for commands.")

    agent = CosmoAgentSession()

    while True:
        try:
            user_message = input(
                "\nYou> "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CosmoAgent.")
            break

        if not user_message:
            continue

        normalized = user_message.lower()

        if normalized in {
            "/exit",
            "/quit",
            "exit",
            "quit",
        }:
            print("Exiting CosmoAgent.")
            break

        if normalized == "/help":
            print(HELP_TEXT)
            continue

        if normalized == "/reset":
            agent.reset()
            print(
                "Conversation history has been cleared."
            )
            continue

        if normalized == "/status":
            print(
                f"Active model: {get_active_model_name()}\n"
                "Stored conversation messages: "
                f"{agent.message_count()}"
            )
            continue

        if normalized == "/models":
            try:
                print(list_ollama_models())
            except Exception as exc:
                print(f"Unable to list Ollama models: {exc}")
            continue

        if normalized == "/study":
            try:
                print(get_study_summary())
            except Exception as exc:
                print(f"Unable to read the Optuna study: {exc}")
            continue

        if normalized == "/model":
            print(
                f"Active model: {get_active_model_name()}\n"
                "Usage: /model MODEL_NAME"
            )
            continue

        if normalized.startswith("/model "):
            requested_model = user_message.split(maxsplit=1)[1].strip()
            try:
                print(switch_ollama_model(requested_model))
            except Exception as exc:
                print(f"Unable to switch Ollama models: {exc}")
            continue

        try:
            answer = agent.ask(user_message)
        except ollama.ResponseError as exc:
            print("\nOllama error:")
            print(exc)

            if getattr(exc, "status_code", None) == 404:
                print(
                    f"\nMake sure the model exists:\n"
                    f"ollama pull {get_active_model_name()}"
                )

            continue
        except ConnectionError as exc:
            print("\nUnable to connect to Ollama:")
            print(exc)
            print(
                "\nMake sure the Ollama App is running."
            )
            continue
        except Exception as exc:
            print("\nUnexpected Agent error:")
            print(exc)
            continue

        print("\nCosmoAgent>")
        print(answer)


if __name__ == "__main__":
    main()

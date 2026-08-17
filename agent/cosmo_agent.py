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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASE_CONFIG_PATH = REPO_ROOT / "config.json"
PHASE0_OPTUNA_CONFIG_PATH = SCRIPT_DIR / "phase0_optuna_config.json"
TRAINING_RUNS_DIR = SCRIPT_DIR / "training_runs"
TRAINING_RUNNER_PATH = SCRIPT_DIR / "run_training_job.py"

EXPECTED_OBJECTIVE_METRIC = "mean_per_output_relative_l2"
OBJECTIVE_DESCRIPTION = (
    "mean_j(||prediction_j-target_j||_2 / ||target_j||_2)"
)

DEFAULT_MODEL_NAME = (
    os.environ.get("COSMO_AGENT_MODEL", "phi4-mini:latest").strip()
    or "phi4-mini:latest"
)
ACTIVE_MODEL_NAME = DEFAULT_MODEL_NAME
MODEL_CAPABILITIES_CACHE: dict[str, set[str]] = {}

MAX_AGENT_TURNS = 8
MAX_TOP_TRIALS = 10
NUM_CONTEXT_TOKENS = 16384
TRACE_TOOL_CALLS = True
ACTIVE_TRAINING_STATES = {"queued", "running"}


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _normalise_output_part(value: Any) -> str:
    text = str(value if value is not None else "re").strip().lower()
    if text in {"re", "real"}:
        return "re"
    if text in {"im", "imag", "imaginary"}:
        return "im"
    raise ValueError(f"Unsupported phase0 output part: {value!r}")


def _eps_tag(value: Any) -> str:
    eps = float(value)
    if abs(eps) < 1e-12:
        return "0"
    magnitude = f"{abs(eps):.6g}".replace(".", "_")
    return f"m{magnitude}" if eps < 0 else f"p{magnitude}"


def _resolve_phase0_study() -> tuple[str, str, Path | None]:
    """Resolve the same Phase-0 study name/storage used by phase0_optuna.py."""
    base = _read_json_object(BASE_CONFIG_PATH)
    tune = _read_json_object(PHASE0_OPTUNA_CONFIG_PATH)
    format_values = {
        "eps": float(base["eps_global"]),
        "eps_tag": _eps_tag(base["eps_global"]),
        "output_part": _normalise_output_part(base.get("phase0_output_part", "re")),
    }
    study_name = str(tune["study_name"]).format(**format_values)
    output_dir_text = str(
        tune.get("output_dir", "agent/phase0_optuna_results")
    ).format(**format_values)
    output_dir_config = Path(output_dir_text)
    output_dir = (
        output_dir_config
        if output_dir_config.is_absolute()
        else REPO_ROOT / output_dir_config
    )

    configured_storage = tune.get("storage")
    if configured_storage:
        return study_name, str(configured_storage), None

    database_path = output_dir / "study.sqlite3"
    storage = f"sqlite:///{database_path.as_posix()}"
    return study_name, storage, database_path


STUDY_NAME, STORAGE, DATABASE_PATH = _resolve_phase0_study()


# ============================================================
# 2. JSON utilities
# ============================================================

def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


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
    if "thinking" in capabilities and model_name.lower().startswith("gpt-oss:"):
        return "low"
    return False


def list_ollama_models() -> str:
    """List installed Ollama models and CosmoAgent compatibility."""
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
    """Switch to another installed tool-capable Ollama model."""
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
# 4. Optuna study utilities and read-only tools
# ============================================================

def load_current_study() -> optuna.Study:
    """Load the Phase-0 Optuna study resolved from the current configs."""
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the Phase 0 Optuna study.\n"
            f"STORAGE = {STORAGE}\n"
            f"STUDY_NAME = {STUDY_NAME}\n"
            f"Original error: {exc}"
        ) from exc

    if len(study.directions) != 1:
        raise RuntimeError("CosmoAgent supports only single-objective studies.")
    return study


def get_direction_name(study: optuna.Study) -> str:
    return study.directions[0].name


def get_completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]


def find_trial(study: optuna.Study, trial_number: int) -> FrozenTrial:
    for trial in study.trials:
        if trial.number == trial_number:
            return trial
    raise ValueError(f"Trial number {trial_number} does not exist.")


def duration_seconds(duration: timedelta | None) -> float | None:
    return None if duration is None else duration.total_seconds()


def trial_to_dict(
    trial: FrozenTrial,
    include_intermediate_values: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "trial_number": trial.number,
        "state": trial.state.name,
        "objective_value": float(trial.value) if trial.value is not None else None,
        "parameters": trial.params,
        "user_attributes": trial.user_attrs,
        "system_attributes": trial.system_attrs,
        "datetime_start": (
            trial.datetime_start.isoformat() if trial.datetime_start else None
        ),
        "datetime_complete": (
            trial.datetime_complete.isoformat() if trial.datetime_complete else None
        ),
        "duration_seconds": duration_seconds(trial.duration),
    }
    if include_intermediate_values:
        items = sorted(trial.intermediate_values.items())
        data["intermediate_values"] = {
            str(step): float(value) for step, value in items[-50:]
        }
    return data


def get_study_summary() -> str:
    """Return the current study, trial states, objective, and best trial."""
    study = load_current_study()
    completed = get_completed_trials(study)
    signature = study.user_attrs.get("phase0_signature")
    if not isinstance(signature, dict):
        signature = {}

    state_counts = {
        state.name: sum(trial.state == state for trial in study.trials)
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

    return to_json({
        "study_name": study.study_name,
        "storage": STORAGE,
        "database_path": str(DATABASE_PATH) if DATABASE_PATH else None,
        "optimization_direction": get_direction_name(study),
        "objective_metric": signature.get("objective_metric"),
        "objective_definition": OBJECTIVE_DESCRIPTION,
        "total_trials": len(study.trials),
        "trial_state_counts": state_counts,
        "completed_trials_with_objective": len(completed),
        "best_trial": best_result,
        "recorded_user_attribute_keys": metric_counts,
        "fixed_phase0_settings": {
            "learning_rate_p0": signature.get("learning_rate_p0"),
            "lambda2": signature.get("lambda2"),
        },
        "warnings": [
            "The best single trial is not automatically a robust configuration.",
            "Optuna parameter importance is study-dependent association, not causality.",
        ],
    })


def _current_study_grounding_snapshot() -> str:
    summary = json.loads(get_study_summary())
    compact = {
        "data_source": str(DATABASE_PATH) if DATABASE_PATH else STORAGE,
        "study_name": summary["study_name"],
        "optimization_direction": summary["optimization_direction"],
        "objective_metric": summary["objective_metric"],
        "objective_definition": summary["objective_definition"],
        "total_trials": summary["total_trials"],
        "trial_state_counts": summary["trial_state_counts"],
        "completed_trials_with_objective": summary["completed_trials_with_objective"],
        "best_trial": summary["best_trial"],
        "fixed_phase0_settings": summary["fixed_phase0_settings"],
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _direct_grounded_study_answer(
    user_message: str,
    grounding_snapshot: str,
) -> str | None:
    text = str(user_message).strip()
    lowered = text.lower()
    if any(term in lowered for term in ("robust", "reliable", "repeat", "稳健", "可靠", "重复")):
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
    parameter_text = ", ".join(f"{name}={value}" for name, value in parameters.items())
    is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)
    if is_chinese:
        criterion = "最低" if direction == "MINIMIZE" else "最高"
        return (
            f"当前最佳的已完成 trial 是 Trial {number}。"
            f"该 study 的方向是 {direction}，{criterion}目标值为 {objective}；"
            f"参数为：{parameter_text}。"
        )
    criterion = "lowest" if direction == "MINIMIZE" else "highest"
    return (
        f"The best completed trial is Trial {number}. The study direction is "
        f"{direction}, with the {criterion} objective value {objective}. "
        f"Parameters: {parameter_text}."
    )


def get_top_trials(n: int = 5) -> str:
    """Return top completed trials ranked by the study objective."""
    study = load_current_study()
    completed = get_completed_trials(study)
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 5
    n = max(1, min(n, MAX_TOP_TRIALS))
    if not completed:
        return to_json({"status": "no_completed_trials"})

    direction = get_direction_name(study)
    reverse = direction == "MAXIMIZE"
    ranked = sorted(completed, key=lambda trial: float(trial.value), reverse=reverse)[:n]
    best_value = float(ranked[0].value)
    output: list[dict[str, Any]] = []
    for rank, trial in enumerate(ranked, start=1):
        value = float(trial.value)
        difference = value - best_value if direction == "MINIMIZE" else best_value - value
        percentage = 100.0 * difference / abs(best_value) if best_value != 0 else None
        output.append({
            "rank": rank,
            "trial_number": trial.number,
            "objective_value": value,
            "absolute_difference_from_best": difference,
            "percentage_difference_from_best": percentage,
            "parameters": trial.params,
            "user_attributes": trial.user_attrs,
            "duration_seconds": duration_seconds(trial.duration),
        })
    return to_json({
        "optimization_direction": direction,
        "objective_definition": OBJECTIVE_DESCRIPTION,
        "number_requested": n,
        "trials": output,
    })


def get_trial_details(trial_number: int) -> str:
    """Return details and intermediate values for one Optuna trial."""
    study = load_current_study()
    try:
        number = int(trial_number)
    except (TypeError, ValueError) as exc:
        return to_json({"status": "error", "reason": "trial_number must be an integer.", "details": str(exc)})
    try:
        trial = find_trial(study, number)
    except ValueError as exc:
        return to_json({"status": "not_found", "reason": str(exc)})
    return to_json(trial_to_dict(trial, include_intermediate_values=True))


def compare_trials(trial_numbers_csv: str) -> str:
    """Compare selected Optuna trial numbers."""
    study = load_current_study()
    try:
        numbers = [int(item.strip()) for item in trial_numbers_csv.split(",") if item.strip()]
    except (AttributeError, TypeError, ValueError) as exc:
        return to_json({"status": "error", "reason": "Provide comma-separated integer trial numbers.", "details": str(exc)})
    numbers = list(dict.fromkeys(numbers))
    if not numbers:
        return to_json({"status": "error", "reason": "No trial numbers were provided."})
    if len(numbers) > 10:
        return to_json({"status": "error", "reason": "At most 10 trials can be compared at once."})

    selected: list[FrozenTrial] = []
    missing: list[int] = []
    for number in numbers:
        try:
            selected.append(find_trial(study, number))
        except ValueError:
            missing.append(number)

    complete = [trial for trial in selected if trial.state == TrialState.COMPLETE and trial.value is not None]
    direction = get_direction_name(study)
    ranked = sorted(complete, key=lambda trial: float(trial.value), reverse=direction == "MAXIMIZE")
    return to_json({
        "optimization_direction": direction,
        "objective_definition": OBJECTIVE_DESCRIPTION,
        "requested_trial_numbers": numbers,
        "missing_trial_numbers": missing,
        "ranked_completed_trials": [trial_to_dict(trial) for trial in ranked],
        "noncompleted_trials": [
            trial_to_dict(trial)
            for trial in selected
            if trial.state != TrialState.COMPLETE or trial.value is None
        ],
    })


def get_parameter_importance() -> str:
    """Estimate Optuna parameter importance for the current study."""
    study = load_current_study()
    completed = get_completed_trials(study)
    if len(completed) < 2:
        return to_json({"status": "unavailable", "reason": "At least two completed trials are required."})
    try:
        importance = get_param_importances(study)
    except Exception as exc:
        return to_json({"status": "unavailable", "reason": str(exc)})
    return to_json({
        "status": "available",
        "importance_scores": importance,
        "interpretation": "Higher scores indicate stronger association with objective variation inside this study.",
        "warning": "These values do not establish physical causality.",
    })


# ============================================================
# 5. CosmoPINNs training tools
# ============================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise ValueError("phase must be one of: phase0, phase0_phase1, phase0_phase2")
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
    if "lambda1" not in trial.params:
        raise RuntimeError("The best trial is missing the required Phase 0 parameter: lambda1")
    return study, trial


def _validate_study_compatibility(study: optuna.Study, config: dict[str, Any]) -> None:
    signature = study.user_attrs.get("phase0_signature")
    if not isinstance(signature, dict):
        raise RuntimeError(
            "The Optuna study has no phase0_signature, so its setup cannot be "
            "verified against config.json."
        )

    expected = {
        "eps_global": float(config["eps_global"]),
        "output_part": _normalise_output_part(config.get("phase0_output_part", "re")),
        "domain": [
            float(config["x1_min"]),
            float(config["x1_max"]),
            float(config["x2_min"]),
            float(config["x2_max"]),
        ],
        "hidden_size": int(config["hidden_size"]),
        "n_hidden_layers": int(config["n_hidden_layers"]),
        "learning_rate_p0": float(config["learning_rate_p0"]),
        "lambda2": float(config.get("lambda2", 1.0)),
        "objective_metric": EXPECTED_OBJECTIVE_METRIC,
    }
    try:
        observed = {
            "eps_global": float(signature["eps_global"]),
            "output_part": _normalise_output_part(signature["output_part"]),
            "domain": [float(value) for value in signature["domain"]],
            "hidden_size": int(signature["hidden_size"]),
            "n_hidden_layers": int(signature["n_hidden_layers"]),
            "learning_rate_p0": float(signature["learning_rate_p0"]),
            "lambda2": float(signature["lambda2"]),
            "objective_metric": str(signature["objective_metric"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "The Optuna study signature is incomplete for the current lambda-only workflow."
        ) from exc

    if observed != expected:
        raise RuntimeError(
            "The Optuna study is incompatible with the current Phase 0 config. "
            f"Study signature={observed}; config signature={expected}"
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

    lambda1 = float(trial.params["lambda1"])
    learning_rate_p0 = float(config["learning_rate_p0"])
    lambda2 = float(config.get("lambda2", 1.0))

    overrides: dict[str, Any] = {
        "lambda1": lambda1,
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
        "objective_definition": OBJECTIVE_DESCRIPTION,
        "optuna_parameters": {"lambda1": lambda1},
        "fixed_config_parameters": {
            "learning_rate_p0": learning_rate_p0,
            "lambda2": lambda2,
        },
        "config_mapping": {
            "lambda1": lambda1,
            "learning_rate_p0": learning_rate_p0,
            "lambda2": lambda2,
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
    """Preview a training config using the best lambda1 without starting a run."""
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
            "learning_rate_p0": config.get("learning_rate_p0"),
            "lambda1": config.get("lambda1"),
            "lambda2": config.get("lambda2"),
            "phase0_output_part": config.get("phase0_output_part"),
            "eps_global": config.get("eps_global"),
            "enable_phase1": config.get("enable_phase1"),
            "enable_phase2": config.get("enable_phase2"),
        },
        "note": (
            "Final training starts from a new model. Only lambda1 comes from "
            "Optuna; learning_rate_p0 and lambda2 remain fixed by config.json."
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
            "The recorded runner process is no longer active and did not write "
            "a final exit status."
        )
        _write_json_atomic(metadata_path, payload)
    return payload


def _active_training_jobs() -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for path in _job_metadata_paths():
        payload = _refresh_job(_read_json_object(path), path)
        if payload.get("status") == "queued":
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
    """Start a background CosmoPINNs run using the best Optuna lambda1."""
    active = _active_training_jobs()
    if active:
        return to_json({
            "status": "not_started",
            "reason": "Another CosmoAgent training job is already active.",
            "active_jobs": [
                {"job_id": item.get("job_id"), "status": item.get("status"), "pid": item.get("pid")}
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
        "next_action": "Use get_training_status with this job_id to monitor the run.",
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


def get_training_status(job_id: str = "latest", tail_lines: int = 30) -> str:
    """Return persisted status and recent log lines for one training job."""
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
            log_tail = [line.rstrip("\n") for line in handle.readlines()[-tail_lines:]]
    payload["log_tail"] = log_tail
    return to_json(payload)


def list_training_jobs(limit: int = 10) -> str:
    """List recent CosmoAgent-launched training jobs."""
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

AVAILABLE_FUNCTIONS: dict[str, Callable[..., str]] = {
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
TOOLS = list(AVAILABLE_FUNCTIONS.values())

PSEUDO_TOOL_PATTERN = re.compile(
    r"\[DATA\]\s*(.*?)\s*\[/DATA\]",
    flags=re.IGNORECASE | re.DOTALL,
)


def _invoke_registered_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    function = AVAILABLE_FUNCTIONS.get(tool_name)
    if function is None:
        return to_json({"status": "error", "reason": f"Unknown tool: {tool_name}"})
    try:
        return function(**dict(arguments or {}))
    except TypeError as exc:
        return to_json({"status": "error", "tool": tool_name, "reason": "Invalid tool arguments.", "details": str(exc)})
    except Exception as exc:
        return to_json({"status": "error", "tool": tool_name, "reason": str(exc)})


def _extract_pseudo_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    match = PSEUDO_TOOL_PATTERN.search(content or "")
    if match is None:
        return None

    body = match.group(1).strip()
    arguments: dict[str, Any] = {}
    if body.startswith("{"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return "__invalid_pseudo_tool__", {}
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
            return "__invalid_pseudo_tool__", {}
        name = name_match.group(1)
        if name_match.group(2):
            try:
                parsed = json.loads(name_match.group(2))
            except json.JSONDecodeError:
                return "__invalid_pseudo_tool__", {}
            if isinstance(parsed, dict):
                arguments = parsed
    return name.strip().lower(), arguments


# ============================================================
# 7. Agent instructions
# ============================================================

SYSTEM_PROMPT = """
You are Observer, a scientific machine-learning experiment-analysis and
training-operations agent for CosmoPINNs.

You have tools for the configured Phase-0 Optuna study and tools that can launch
and monitor isolated CosmoPINNs training jobs.

Scientific rules:
- Obtain every numerical statement about the Optuna study from tools.
- Never invent trials, objective values, parameters, metrics, or physical explanations.
- Respect the study's MINIMIZE/MAXIMIZE direction.
- The current Phase-0 tuner searches only lambda1. learning_rate_p0 and lambda2
  are fixed by the root config.json and are not Optuna parameters.
- The Phase-0 objective is the equal-weight mean over outputs of
  ||prediction_j-target_j||_2 / ||target_j||_2. Lower is better.
- A best trial is not automatically robust; do not claim robustness without
  repeated-seed evidence.
- Parameter importance indicates association, not causality.
- Clearly distinguish observations from hypotheses.
- Never modify Optuna trials or the Optuna database.
- Start training only when the user explicitly asks to start, run, or launch it.
- When explicitly asked to train with the best Optuna result, call
  start_cosmopinns_training. Do not merely describe shell commands.
- The default training pipeline is phase0. Use phase0_phase1 or phase0_phase2
  only when the user explicitly requests that transfer stage.
- phase0_epochs=0 means keep the production value from config.json.
- Use preview_cosmopinns_training when the user asks to inspect the effective
  configuration without launching work.
- Training runs in the background; use get_training_status for progress or
  completion questions and never claim success until the persisted status is succeeded.
- Switch Ollama models only when explicitly requested.
- A CURRENT_STUDY_SNAPSHOT system message is authoritative SQLite data.
- Never print textual placeholders such as [DATA]GET_STUDY_SUMMARY[/DATA].
- Answer in the same language as the user and keep answers precise.
"""


# ============================================================
# 8. Persistent conversational Agent session
# ============================================================

class CosmoAgentSession:
    def __init__(self) -> None:
        self.messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, user_message: str) -> str:
        grounding_snapshot = _current_study_grounding_snapshot()
        self.messages.append({
            "role": "system",
            "content": (
                "CURRENT_STUDY_SNAPSHOT (authoritative SQLite data; do not invent "
                "or contradict numerical study facts):\n" + grounding_snapshot
            ),
        })
        self.messages.append({"role": "user", "content": user_message})

        direct_answer = _direct_grounded_study_answer(user_message, grounding_snapshot)
        if direct_answer is not None:
            self.messages.append({"role": "assistant", "content": direct_answer})
            return direct_answer

        for agent_turn in range(1, MAX_AGENT_TURNS + 1):
            model_name = get_active_model_name()
            capabilities = _get_model_capabilities(model_name)
            response: ollama.ChatResponse = ollama.chat(
                model=model_name,
                messages=self.messages,
                tools=TOOLS,
                think=_model_think_setting(model_name, capabilities),
                options={"temperature": 0, "num_ctx": NUM_CONTEXT_TOKENS},
            )
            self.messages.append(response.message)
            tool_calls = response.message.tool_calls or []

            if not tool_calls:
                final_content = (response.message.content or "").strip()
                if not final_content:
                    return "The model returned an empty response without requesting a tool."

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
                            f"AUTHORITATIVE TOOL RESULT for {tool_name}:\n{tool_result}\n"
                            "Now answer the user's question directly. Do not repeat a [DATA] placeholder."
                        ),
                    })
                    continue
                return final_content

            if TRACE_TOOL_CALLS:
                print(f"\n[Agent tool turn {agent_turn}]")

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = dict(tool_call.function.arguments or {})
                if TRACE_TOOL_CALLS:
                    print(f"[Tool] {tool_name}")
                    print("[Arguments] " + to_json(arguments))
                tool_result = _invoke_registered_tool(tool_name, arguments)
                if TRACE_TOOL_CALLS:
                    preview = tool_result if len(tool_result) <= 1500 else tool_result[:1500] + "\n... [truncated]"
                    print("[Tool result]\n" + preview)
                self.messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": tool_result,
                })

        return (
            "CosmoAgent reached the maximum number of tool turns "
            f"({MAX_AGENT_TURNS}) without producing a final answer."
        )

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def message_count(self) -> int:
        return len(self.messages)


# ============================================================
# 9. Startup checks
# ============================================================

def check_ollama_configuration(model_name: str) -> tuple[str, set[str]]:
    try:
        resolved, capabilities, _ = _activate_compatible_model(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Unable to activate the requested Ollama model.\n"
            f"MODEL = {model_name}\nOriginal error: {exc}\n"
            "Make sure Ollama is running and inspect installed models with `ollama list`."
        ) from exc
    return resolved, capabilities


def check_study_configuration() -> None:
    study = load_current_study()
    completed = get_completed_trials(study)
    print("Optuna study loaded successfully.")
    print(f"Database path: {DATABASE_PATH if DATABASE_PATH else STORAGE}")
    print(f"Study name: {study.study_name}")
    print(f"Direction: {get_direction_name(study)}")
    print(f"Objective: {OBJECTIVE_DESCRIPTION}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len(completed)}")
    if completed:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best objective value: {study.best_value}")
        print(f"Best lambda1: {study.best_trial.params.get('lambda1')}")


# ============================================================
# 10. Command-line chat interface
# ============================================================

HELP_TEXT = """
Available commands:

  /help       Show this help message.
  /reset      Clear the conversation history.
  /status     Show active model and message count.
  /models     List installed Ollama models.
  /study      Print the current SQLite study summary directly.
  /model NAME Switch Ollama model while preserving the conversation.
  /exit       Exit CosmoAgent.

Example questions:

  How many completed, pruned, and failed trials are in this study?
  Which trial is currently best, and what lambda1 did it use?
  Compare the top five trials. Is the best trial clearly ahead?
  What are the four per-output relative L2 values for the best trial?
  Preview the configuration that would train Phase 0 with the best lambda1.
  Start Phase 0 training with the best Optuna lambda1.
  What is the status of the latest training job?
"""


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CosmoPINNs Optuna analysis and training agent."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Installed tool-capable Ollama model (default: {DEFAULT_MODEL_NAME!r}).",
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
        print("Model capabilities: " + ", ".join(sorted(capabilities)))
        check_study_configuration()
    except Exception as exc:
        print("\nStartup error:")
        print(exc)
        print(
            "\nCheck config.json and agent/phase0_optuna_config.json, and make "
            "sure the matching Phase-0 Optuna study has been created."
        )
        sys.exit(1)

    print(f"\nOllama context window: {NUM_CONTEXT_TOKENS} tokens")
    print(
        "Training jobs are launched only after an explicit conversational request "
        "and are saved below agent/training_runs/."
    )
    print("\nType /help for commands.")

    agent = CosmoAgentSession()
    while True:
        try:
            user_message = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CosmoAgent.")
            break

        if not user_message:
            continue
        normalized = user_message.lower()

        if normalized in {"/exit", "/quit", "exit", "quit"}:
            print("Exiting CosmoAgent.")
            break
        if normalized == "/help":
            print(HELP_TEXT)
            continue
        if normalized == "/reset":
            agent.reset()
            print("Conversation history has been cleared.")
            continue
        if normalized == "/status":
            print(
                f"Active model: {get_active_model_name()}\n"
                f"Stored conversation messages: {agent.message_count()}\n"
                f"Study: {STUDY_NAME}"
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
            print(f"Active model: {get_active_model_name()}\nUsage: /model MODEL_NAME")
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
                print(f"\nMake sure the model exists:\nollama pull {get_active_model_name()}")
            continue
        except ConnectionError as exc:
            print("\nUnable to connect to Ollama:")
            print(exc)
            print("\nMake sure the Ollama App is running.")
            continue
        except Exception as exc:
            print("\nUnexpected Agent error:")
            print(exc)
            continue

        print("\nCosmoAgent>")
        print(answer)


if __name__ == "__main__":
    main()
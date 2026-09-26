"""保存済みモデルの実パラメータ・学習ログを読み、診断結果に添付する。"""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean

from .model_registry import ModelRegistry


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for directory in ("domain", "agents", "rl"):
        for path in sorted((root / directory).glob("*.py")):
            digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _scalar_logs(directory: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    wanted = ("train/entropy_loss", "train/policy_gradient_loss", "train/value_loss",
              "train/approx_kl", "train/clip_fraction", "train/explained_variance",
              "rollout/ep_rew_mean", "rollout/ep_len_mean")
    values = {tag: {} for tag in wanted}
    for path in sorted(directory.rglob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        for tag in wanted:
            if tag in events.Tags()["scalars"]:
                for event in events.Scalars(tag):
                    previous = values[tag].get(event.step)
                    if previous is None or event.wall_time >= previous.wall_time:
                        values[tag][event.step] = event
    result = {}
    for tag, series in values.items():
        samples = [series[step] for step in sorted(series)]
        if samples:
            width = min(100, len(samples))
            result[tag] = {"samples": len(samples), "start_step": samples[0].step,
                           "end_step": samples[-1].step,
                           "first_100_mean": fmean(e.value for e in samples[:width]),
                           "last_100_mean": fmean(e.value for e in samples[-width:])}
    return result


def model_audit(model_id: str, model, registry: ModelRegistry) -> dict:
    directory = registry.models_root / model_id
    manifest = registry.load(model_id)
    attributes = ("gamma", "n_steps", "batch_size", "n_epochs", "gae_lambda", "ent_coef",
                  "vf_coef", "max_grad_norm", "normalize_advantage", "target_kl", "n_envs")
    parameters = {name: getattr(model, name) for name in attributes}
    parameters.update({"learning_rate_start": model.lr_schedule(1.0),
                       "learning_rate_end": model.lr_schedule(0.0),
                       "clip_range_start": model.clip_range(1.0),
                       "clip_range_end": model.clip_range(0.0),
                       "policy_network": model.policy.net_arch,
                       "activation": model.policy.activation_fn.__name__,
                       "optimizer": type(model.policy.optimizer).__name__,
                       "num_timesteps": model.num_timesteps})
    return {"model_id": model_id, "training_seed": manifest.training_seed,
            "training_steps": manifest.training_steps,
            "model_sha256": sha256((directory / manifest.artifact_filename).read_bytes()).hexdigest(),
            "parameters_from_saved_model": parameters,
            "training_config": json.loads((directory / "config.json").read_text(encoding="utf-8")),
            "tensorboard": _scalar_logs(directory / "tensorboard")}


def runtime_versions() -> dict[str, str]:
    return {package: version(package) for package in
            ("numpy", "gymnasium", "torch", "stable-baselines3", "sb3-contrib", "tensorboard")}

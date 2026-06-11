
from __future__ import annotations

from copy import deepcopy
import io
import sys
import torch

from ultralytics import YOLO
from ultralytics.engine import trainer as trainer_mod


def parse_value(value: str):
    if value == "True":
        return True
    if value == "False":
        return False
    if value in {"None", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def module_has_nonfinite(module) -> bool:
    model = trainer_mod.unwrap_model(module)
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
            if not torch.isfinite(tensor.detach()).all():
                return True
    return False


def safe_save_model(self):
    candidates = []
    if getattr(self, "ema", None) is not None and getattr(self.ema, "ema", None) is not None:
        candidates.append(("ema", self.ema.ema))
    candidates.append(("model", self.model))

    selected_name = ""
    selected_model = None
    for name, module in candidates:
        if module is not None and not module_has_nonfinite(module):
            selected_name = name
            selected_model = module
            break

    if selected_model is None:
        raise RuntimeError("Both EMA and raw model contain NaN/Inf; refusing to save unusable checkpoint.")

    if selected_name != "ema":
        trainer_mod.LOGGER.warning("EMA contains NaN/Inf; saving raw model fallback checkpoint instead.")

    buffer = io.BytesIO()
    scaler_state = self.scaler.state_dict() if getattr(self, "scaler", None) is not None else {}
    ckpt = {
        "epoch": self.epoch,
        "best_fitness": self.best_fitness,
        "model": None,
        "ema": deepcopy(trainer_mod.unwrap_model(selected_model)).half(),
        "updates": getattr(getattr(self, "ema", None), "updates", 0),
        "optimizer": trainer_mod.convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
        "scaler": scaler_state,
        "train_args": vars(self.args),
        "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
        "train_results": self.read_results_csv(),
        "date": trainer_mod.datetime.now().isoformat(),
        "version": trainer_mod.__version__,
        "git": {
            "root": str(trainer_mod.GIT.root),
            "branch": trainer_mod.GIT.branch,
            "commit": trainer_mod.GIT.commit,
            "origin": trainer_mod.GIT.origin,
        },
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "checkpoint_source": selected_name,
    }
    torch.save(ckpt, buffer)
    serialized_ckpt = buffer.getvalue()

    self.wdir.mkdir(parents=True, exist_ok=True)
    self.last.write_bytes(serialized_ckpt)
    if self.best_fitness == self.fitness:
        self.best.write_bytes(serialized_ckpt)
    if (self.save_period > 0) and (self.epoch % self.save_period == 0):
        (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)


trainer_mod.BaseTrainer.save_model = safe_save_model

overrides = {}
for raw in sys.argv[1:]:
    if "=" not in raw:
        continue
    key, value = raw.split("=", 1)
    overrides[key] = parse_value(value)

model_path = overrides.pop("model")
model = YOLO(model_path)
model.train(**overrides)

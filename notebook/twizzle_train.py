"""
Twizzle training library (Person 2). Imported by model_x3d.ipynb, model_r2plus1d.ipynb and model_compare.ipynb.

Lives at the repo root next to preprocess.py. It lives in a module (not in the notebooks) so that
  * DataLoader workers can import the Dataset on Windows,
  * both model notebooks run exactly the same training code, and
  * the notebooks stay short enough to read.

Data contract (Team Technical Spec, Section 3): <data_root>/labels.csv with columns
clip_id, split, label, ... and <data_root>/<split>/<clip_id>.pt = uint8 tensor (16,160,160,3).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["Toeloop", "Salchow", "Flip", "Loop", "Lutz", "Axel",
               "CamelSpin", "SitSpin", "UprightSpin", "NoBasic", "Sequence"]
NUM_CLASSES = len(CLASS_NAMES)
JUMP_LABELS = list(range(6))          # the six jump classes: headline metric is macro-F1 over these

KINETICS = {"mean": (0.43216, 0.394666, 0.37645), "std": (0.22803, 0.22145, 0.216989)}   # torchvision video models
PYTORCHVIDEO = {"mean": (0.45, 0.45, 0.45), "std": (0.225, 0.225, 0.225)}                   # pytorchvideo X3D
MODEL_STATS = {"x3d_xs": PYTORCHVIDEO, "x3d_s": PYTORCHVIDEO, "x3d_m": PYTORCHVIDEO,
               "r2plus1d_18": KINETICS, "r3d_18": KINETICS, "mc3_18": KINETICS}


# ============================================================================ config
@dataclass
class Config:
    run_name: str = "baseline"
    model: str = "x3d_s"                  # x3d_xs | x3d_s | x3d_m | r2plus1d_18 | r3d_18 | mc3_18
    data_root: str = "dataset/preprocessed"
    runs_dir: str = "runs/x3d_s"
    pretrained: bool = True
    finetune_mode: str = "frozen"         # frozen = new classifier only | partial = + last stage | full = everything
    optimizer: str = "adamw"              # adamw | sgd
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 8
    warmup_epochs: int = 1
    batch_size: int = 8
    accum_steps: int = 1                  # effective batch = batch_size * accum_steps
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    weight_cap: float = 5.0
    dropout: float = 0.3
    augmentation: str = "off"             # off | basic | basic_flip
    amp: bool = True                      # mixed precision (GPU only)
    num_workers: int = 4
    subset_frac: float = 1.0              # fraction of the TRAIN split (stratified) used for quick sweeps
    early_stop_patience: int = 4
    seed: int = 42

    def fingerprint(self) -> str:
        """Identical settings -> identical fingerprint, so repeated configs are never trained twice."""
        d = asdict(self)
        for k in ("run_name", "runs_dir", "num_workers"):
            d.pop(k)
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================ data
def stratified_subset(lab: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed) for _, g in lab.groupby("label")]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def augment_clip(x: torch.Tensor, aug: str) -> torch.Tensor:
    """x: float (3,T,H,W) in [0,1]. One random crop / colour change / temporal window for the whole clip."""
    C, T, H, W = x.shape
    if random.random() < 0.7:                                    # temporal jitter: random window, resampled back to T
        win = random.randint(max(8, int(T * 0.75)), T)
        start = random.randint(0, T - win)
        idx = torch.linspace(start, start + win - 1, T).round().long()
        x = x[:, idx]
    area = random.uniform(0.7, 1.0)                              # random resized crop
    ratio = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
    ch = min(H, int(round(math.sqrt(area / ratio) * H)))
    cw = min(W, int(round(math.sqrt(area * ratio) * W)))
    top, left = random.randint(0, H - ch), random.randint(0, W - cw)
    x = x[:, :, top:top + ch, left:left + cw]
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(H, W), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    b, c = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)    # brightness / contrast
    m = x.mean()
    x = (((x - m) * c + m) * b).clamp(0, 1)
    if aug == "basic_flip" and random.random() < 0.5:            # flipping reverses rotation direction: ablation only
        x = x.flip(-1)
    return x


class ClipDataset(Dataset):
    def __init__(self, data_root, split: str, cfg: Config):
        self.root = Path(data_root)
        lab = pd.read_csv(self.root / "labels.csv")
        lab = lab[lab.split == split].reset_index(drop=True)
        if split == "train" and cfg.subset_frac < 1.0:
            lab = stratified_subset(lab, cfg.subset_frac, cfg.seed)
        self.lab, self.split = lab, split
        self.aug = cfg.augmentation if split == "train" else "off"
        stats = MODEL_STATS[cfg.model]
        self.mean = torch.tensor(stats["mean"]).view(3, 1, 1, 1)
        self.std = torch.tensor(stats["std"]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        r = self.lab.iloc[i]
        x = torch.load(self.root / r["split"] / f"{r['clip_id']}.pt")          # (T,H,W,3) uint8
        x = x.permute(3, 0, 1, 2).float().div_(255.0)                           # (3,T,H,W)
        if self.aug != "off":
            x = augment_clip(x, self.aug)
        return (x - self.mean) / self.std, int(r["label"])


def make_loader(ds, cfg: Config, shuffle: bool, batch_size: int | None = None):
    bs = batch_size or cfg.batch_size
    nw = cfg.num_workers
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle and len(ds) > bs, persistent_workers=nw > 0)


def class_weights_from(ds: ClipDataset, cap: float) -> torch.Tensor:
    counts = np.bincount(ds.lab["label"].values, minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1
    w = np.minimum(counts.sum() / (NUM_CLASSES * counts), cap)
    return torch.tensor(w, dtype=torch.float32)


# ============================================================================ models
def _load_x3d(name: str, pretrained: bool):
    def _import():
        from pytorchvideo.models.hub import x3d_m, x3d_s, x3d_xs
        return {"x3d_xs": x3d_xs, "x3d_s": x3d_s, "x3d_m": x3d_m}
    try:
        fns = _import()
    except ModuleNotFoundError as e:
        if "functional_tensor" in str(e):             # known pytorchvideo / torchvision>=0.17 incompatibility
            import torchvision.transforms.functional as _tf
            sys.modules["torchvision.transforms.functional_tensor"] = _tf
            fns = _import()
        else:
            raise ImportError("X3D needs pytorchvideo: pip install pytorchvideo fvcore iopath  "
                              "(fallback: set model='r3d_18', which only needs torchvision)") from e
    return fns[name](pretrained=pretrained)


def _replace_last_linear(model: nn.Module, new_linear: nn.Module):
    """Replaces the final nn.Linear anywhere in the model; returns (parent_module, attribute_name)."""
    last = None
    for mod_name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            if isinstance(child, nn.Linear):
                last = (mod, child_name)
    assert last is not None, "no nn.Linear found to replace"
    setattr(last[0], last[1], new_linear)
    return last


def build_model(cfg: Config):
    """Returns (model, head_module). The head module holds only the NEW 11-way classifier."""
    name = cfg.model
    if name.startswith("x3d"):
        model = _load_x3d(name, cfg.pretrained)
        in_f = [m for m in model.modules() if isinstance(m, nn.Linear)][-1].in_features
        head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_f, NUM_CLASSES))
        _replace_last_linear(model, head)
        for m in model.modules():                     # pytorchvideo heads may apply softmax / their own dropout
            if "Head" in type(m).__name__:
                if getattr(m, "activation", None) is not None:
                    m.activation = None               # we need raw logits for the loss, calibration and ONNX
                if getattr(m, "dropout", None) is not None:
                    m.dropout = nn.Identity()         # dropout is applied once, by our own classifier head
    else:
        import torchvision.models.video as tvv
        builders = {"r2plus1d_18": (tvv.r2plus1d_18, "R2Plus1D_18_Weights"),
                    "r3d_18": (tvv.r3d_18, "R3D_18_Weights"), "mc3_18": (tvv.mc3_18, "MC3_18_Weights")}
        fn, wname = builders[name]
        weights = getattr(tvv, wname).KINETICS400_V1 if cfg.pretrained else None
        model = fn(weights=weights)
        head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(model.fc.in_features, NUM_CLASSES))
        model.fc = head
    return model, head


def _last_stage_modules(model: nn.Module, name: str):
    if name.startswith("x3d"):
        return [model.blocks[-2], model.blocks[-1]]       # last residual stage + the (pretrained part of the) head
    return [model.layer4]


def apply_finetune_mode(model, head, cfg: Config):
    for p in model.parameters():
        p.requires_grad = False
    if cfg.finetune_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in head.parameters():
        p.requires_grad = True
    if cfg.finetune_mode == "partial":
        for m in _last_stage_modules(model, cfg.model):
            for p in m.parameters():
                p.requires_grad = True
    elif cfg.finetune_mode != "frozen":
        raise ValueError(f"finetune_mode must be frozen|partial|full, got {cfg.finetune_mode}")


def set_train_mode(model: nn.Module):
    """model.train(), but BatchNorm layers whose parameters are frozen stay in eval mode so their
    running statistics are not silently overwritten by our (very different) data."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================ metrics / prediction
def compute_metrics(labels, preds) -> dict:
    labels, preds = np.asarray(labels), np.asarray(preds)
    per_class = f1_score(labels, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
        "jump_macro_f1": float(f1_score(labels, preds, labels=JUMP_LABELS, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in per_class],
    }


@torch.no_grad()
def predict(model, loader, device, amp: bool = False):
    """Returns logits (N,11) float32 and labels (N,), in dataset order (loader must not shuffle)."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x)
        logits.append(out.float().cpu()); labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


# ============================================================================ training
def _lr_lambda(warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def find_existing(cfg: Config):
    root = Path(cfg.runs_dir)
    if not root.exists():
        return None
    fp = cfg.fingerprint()
    for f in root.glob("*/result.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r.get("fingerprint") == fp:
            return r
    return None


def train_run(cfg: Config, force: bool = False, verbose: bool = True) -> dict:
    """Trains one configuration. Resumable: a config that was already trained (same fingerprint) is
    returned from disk instead of being trained again."""
    
    if not force:
        done = find_existing(cfg)

        if done is not None:
            source_name = done["config"]["run_name"]
            source_dir = Path(cfg.runs_dir) / source_name
            target_dir = Path(cfg.runs_dir) / cfg.run_name

            # Make sure the source checkpoint actually exists.
            source_ckpt = source_dir / "best.pt"
            if not source_ckpt.is_file():
                raise FileNotFoundError(
                    f"Matching run '{source_name}' was found, "
                    f"but its checkpoint is missing: {source_ckpt}"
                )

            # Materialize the checkpoint under the requested run name.
            target_dir.mkdir(parents=True, exist_ok=True)
            target_ckpt = target_dir / "best.pt"

            if source_ckpt.resolve() != target_ckpt.resolve():
                shutil.copy2(source_ckpt, target_ckpt)

            # Copy validation logits if available.
            source_logits = source_dir / "val_logits.npz"
            if source_logits.is_file():
                shutil.copy2(source_logits, target_dir / "val_logits.npz")

            # Save a result record under the requested run name.
            reused = copy.deepcopy(done)
            reused["config"] = asdict(cfg)
            reused["fingerprint"] = cfg.fingerprint()

            (target_dir / "result.json").write_text(
                json.dumps(reused, indent=2)
            )

            if verbose:
                print(
                    f"[reuse] '{cfg.run_name}' uses identical run "
                    f"'{source_name}' | "
                    f"val macro-F1 {reused['best_val_macro_f1']:.3f}"
                )
                print(f"Checkpoint available: {target_ckpt}")

            return reused


    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = get_device()
    use_amp = cfg.amp and device.type == "cuda"

    train_ds, val_ds = ClipDataset(cfg.data_root, "train", cfg), ClipDataset(cfg.data_root, "val", cfg)
    train_dl, val_dl = make_loader(train_ds, cfg, True), make_loader(val_ds, cfg, False)

    model, head = build_model(cfg)
    apply_finetune_mode(model, head, cfg)
    model.to(device)
    total_p, train_p = count_params(model)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    else:
        raise ValueError("optimizer must be adamw|sgd")
    steps_per_epoch = max(1, math.ceil(len(train_dl) / cfg.accum_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch))
    scaler = _make_scaler(use_amp)
    weights = class_weights_from(train_ds, cfg.weight_cap).to(device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    if verbose:
        print(f"[{cfg.run_name}] {cfg.model} | mode={cfg.finetune_mode} opt={cfg.optimizer} lr={cfg.lr:g} aug={cfg.augmentation} "
              f"| train {len(train_ds)} val {len(val_ds)} | params {total_p/1e6:.1f}M (trainable {train_p/1e6:.2f}M) | "
              f"device={device.type} amp={use_amp}")

    history, best_f1, best_epoch, bad = [], -1.0, -1, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(cfg.epochs):
        t0 = time.time()
        set_train_mode(model)
        run_loss, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss / cfg.accum_steps).backward()
            if (step + 1) % cfg.accum_steps == 0 or step + 1 == len(train_dl):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            run_loss += loss.item() * x.size(0); n_seen += x.size(0)
        train_loss = run_loss / max(1, n_seen)

        logits, labels = predict(model, val_dl, device, use_amp)
        val_loss = float(val_criterion(torch.from_numpy(logits), torch.from_numpy(labels)))
        m = compute_metrics(labels, logits.argmax(1))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": m["acc"],
                        "val_macro_f1": m["macro_f1"], "val_jump_f1": m["jump_macro_f1"],
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        if verbose:
            print(f"  epoch {epoch+1:>2}/{cfg.epochs} train_loss {train_loss:.3f} val_loss {val_loss:.3f} "
                  f"acc {m['acc']:.3f} macroF1 {m['macro_f1']:.3f} jumpF1 {m['jump_macro_f1']:.3f} ({history[-1]['seconds']:.0f}s)")
        if m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = m["macro_f1"], epoch + 1, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                if verbose:
                    print(f"  early stop (no val macro-F1 gain for {bad} epochs)")
                break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    logits, labels = predict(model, val_dl, device, use_amp)
    m = compute_metrics(labels, logits.argmax(1))
    np.savez(run_dir / "val_logits.npz", logits=logits, labels=labels, clip_ids=val_ds.lab["clip_id"].values.astype(str))
    result = {
        "config": asdict(cfg), "fingerprint": cfg.fingerprint(), "history": history, "best_epoch": best_epoch,
        "best_val_macro_f1": m["macro_f1"], "best_val_jump_f1": m["jump_macro_f1"], "best_val_acc": m["acc"],
        "val_per_class_f1": m["per_class_f1"], "params_total": total_p, "params_trainable": train_p,
        "train_seconds": time.time() - t_start, "sec_per_epoch": float(np.mean([h["seconds"] for h in history])),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "n_train": len(train_ds), "n_val": len(val_ds),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    if verbose:
        print(f"[{cfg.run_name}] best epoch {best_epoch}: val macro-F1 {m['macro_f1']:.3f}, jump macro-F1 {m['jump_macro_f1']:.3f}, acc {m['acc']:.3f}")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def apply_overrides(cfg: Config, overrides: dict, run_name: str) -> Config:
    c = copy.deepcopy(cfg)
    for k, v in overrides.items():
        assert hasattr(c, k), f"unknown config field {k}"
        setattr(c, k, v)
    c.run_name = run_name
    return c


def run_stage(base_cfg: Config, variants: dict, prefix: str, metric: str = "best_val_macro_f1"):
    """Runs one experiment stage: one training run per variant, returns (best_variant_name, best_overrides, table)."""
    rows, results = [], {}
    for name, overrides in variants.items():
        cfg = apply_overrides(base_cfg, overrides, f"{prefix}_{name}")
        res = train_run(cfg)
        results[name] = res
        rows.append({"variant": name, **{k: overrides[k] for k in overrides},
                     "val_macro_f1": res["best_val_macro_f1"], "val_jump_f1": res["best_val_jump_f1"],
                     "val_acc": res["best_val_acc"], "sec_per_epoch": round(res["sec_per_epoch"], 1)})
    table = pd.DataFrame(rows)
    best = max(results, key=lambda n: results[n][metric])
    print(f"\n>>> best of stage '{prefix}': {best}  ({metric} = {results[best][metric]:.3f})")
    return best, variants[best], table


def summarize_runs(runs_dir) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(runs_dir).glob("*/result.json")):
        r = json.loads(f.read_text()); c = r["config"]
        rows.append({"run": c["run_name"], "mode": c["finetune_mode"], "opt": c["optimizer"], "lr": c["lr"],
                     "aug": c["augmentation"], "subset": c["subset_frac"], "epochs_cfg": c["epochs"],
                     "best_epoch": r["best_epoch"], "val_macro_f1": round(r["best_val_macro_f1"], 4),
                     "val_jump_f1": round(r["best_val_jump_f1"], 4), "val_acc": round(r["best_val_acc"], 4),
                     "sec/epoch": round(r["sec_per_epoch"], 1), "trainable_M": round(r["params_trainable"] / 1e6, 2)})
    return pd.DataFrame(rows)


# ============================================================================ batch size / speed probes
def probe_batch_size(cfg: Config, candidates=(4, 8, 12, 16, 24, 32)):
    """Largest batch size that fits in GPU memory for FULL fine-tuning (worst case). Run once per model."""
    device = get_device()
    if device.type != "cuda":
        print("No GPU: batch-size probe skipped."); return None
    c = copy.deepcopy(cfg); c.finetune_mode = "full"
    model, head = build_model(c); apply_finetune_mode(model, head, c); model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = _make_scaler(c.amp)
    ok = None
    for bs in candidates:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            set_train_mode(model)
            for _ in range(2):
                x = torch.randn(bs, 3, 16, 160, 160, device=device); y = torch.randint(0, NUM_CLASSES, (bs,), device=device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=c.amp):
                    loss = F.cross_entropy(model(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            print(f"  batch {bs:>2}: fits, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
            ok = bs
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:>2}: out of memory"); break
    del model, opt
    torch.cuda.empty_cache()
    return ok


def benchmark_cpu_latency(cfg: Config, n_runs: int = 12, threads: int = 2) -> dict:
    """Median/P90 CPU latency (ms) for ONE clip, 2 threads (the size of a free Hugging Face CPU Space)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    c = copy.deepcopy(cfg); c.pretrained = False
    model, _ = build_model(c); model.eval()
    x = torch.randn(1, 3, 16, 160, 160)
    times = []
    with torch.no_grad():
        for i in range(n_runs + 3):
            t0 = time.time(); model(x); dt = time.time() - t0
            if i >= 3:
                times.append(dt * 1000)
    torch.set_num_threads(prev)
    return {"cpu_ms_median": float(np.median(times)), "cpu_ms_p90": float(np.percentile(times, 90)), "threads": threads}


# ============================================================================ evaluation, calibration, export
def load_trained(cfg: Config, ckpt: Path, device=None):
    c = copy.deepcopy(cfg); c.pretrained = False
    model, _ = build_model(c)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device or torch.device("cpu")).eval()


def evaluate_split(cfg: Config, ckpt: Path, split: str) -> dict:
    device = get_device()
    ds = ClipDataset(cfg.data_root, split, cfg)
    dl = make_loader(ds, cfg, False, batch_size=max(1, cfg.batch_size))
    model = load_trained(cfg, ckpt, device)
    logits, labels = predict(model, dl, device, cfg.amp)
    m = compute_metrics(labels, logits.argmax(1))
    return {"logits": logits, "labels": labels, "clip_ids": ds.lab["clip_id"].values.astype(str), "metrics": m,
            "confusion": confusion_matrix(labels, logits.argmax(1), labels=list(range(NUM_CLASSES)))}


def softmax_np(logits, temperature: float = 1.0):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels) -> float:
    """Temperature scaling on the VALIDATION set only (minimises NLL)."""
    z = torch.from_numpy(logits).float(); y = torch.from_numpy(labels).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(z / log_t.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(log_t.exp().item())


def choose_ambiguity_margin(probs, labels, target_acc: float = 0.95):
    """Margin = top-1 minus top-2 probability. Returns (margin, table). Predictions with margin >= m are
    'confident'; we pick the smallest m whose confident predictions reach target_acc (on VAL)."""
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    correct = probs.argmax(1) == labels
    rows = []
    for m in np.round(np.arange(0.0, 0.96, 0.05), 2):
        conf = margin >= m
        rows.append({"margin": m, "coverage": float(conf.mean()),
                     "acc_confident": float(correct[conf].mean()) if conf.any() else float("nan")})
    table = pd.DataFrame(rows)
    ok = table[table.acc_confident >= target_acc]
    chosen = float(ok.margin.iloc[0]) if len(ok) else float(table.margin.iloc[-1])
    return chosen, table


def _replace_swish(model: nn.Module):
    """pytorchvideo's Swish is a custom autograd Function that ONNX cannot export. nn.SiLU computes the
    same x * sigmoid(x), so swapping it in for export changes nothing numerically."""
    n = 0
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if type(child).__name__ == "Swish":
                setattr(parent, child_name, nn.SiLU()); n += 1
    return n


def export_onnx(cfg: Config, ckpt: Path, out_path: Path, opset: int = 17, check: bool = True) -> dict:
    model = load_trained(cfg, ckpt)
    n_swish = _replace_swish(model)
    dummy = torch.randn(1, 3, 16, 160, 160)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = dict(input_names=["clip"], output_names=["logits"], opset_version=opset, do_constant_folding=True)
    dyn = {"dynamic_axes": {"clip": {0: "batch"}, "logits": {0: "batch"}}}
    attempts = [("classic exporter, dynamic batch", {**base, **dyn, "dynamo": False}),
                ("default exporter, dynamic batch", {**base, **dyn}),
                ("default exporter, fixed batch 1", dict(base))]
    used, errors = None, []
    for label, kw in attempts:
        try:
            torch.onnx.export(model, dummy, str(out_path), **kw)
            used = label
            break
        except Exception as e:                                   # exporter API differs between torch versions
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:160]}")
    if used is None:
        raise RuntimeError("ONNX export failed with every strategy:\n  " + "\n  ".join(errors))
    dynamic = "dynamic" in used
    info = {"onnx_path": str(out_path), "onnx_mb": round(out_path.stat().st_size / 1e6, 1), "opset": opset,
            "export_strategy": used, "dynamic_batch": dynamic, "swish_replaced": n_swish}
    if check:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            x = torch.randn(3 if dynamic else 1, 3, 16, 160, 160)
            with torch.no_grad():
                ref = model(x).numpy()
            got = sess.run(None, {"clip": x.numpy()})[0]
            info["max_abs_diff_vs_pytorch"] = float(np.abs(ref - got).max())
            info["parity_ok"] = info["max_abs_diff_vs_pytorch"] < 1e-3
        except ImportError:
            info["parity_ok"] = None
            print("onnxruntime not installed: parity check skipped (pip install onnxruntime)")
    return info


def write_deployment_meta(path: Path, model_name: str, cfg: Config, temperature: float, margin: float,
                          val_macro_f1: float, val_jump_f1: float, extra: dict | None = None):
    stats = MODEL_STATS[cfg.model]
    meta = {"model_name": model_name, "torch_model": cfg.model, "class_names": CLASS_NAMES, "num_frames": 16,
            "img_size": 160, "mean": list(stats["mean"]), "std": list(stats["std"]), "input_layout": "BCTHW",
            "model_file": "model.onnx", "temperature": temperature, "ambiguity_margin": margin,
            "val_macro_f1": val_macro_f1, "val_jump_macro_f1": val_jump_f1}
    meta.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
    return meta


# ============================================================================ late fusion
# def load_predictions(path):
#     d = np.load(path, allow_pickle=False)
#     return {k: d[k] for k in d.files}

def load_predictions(path):
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}

def fuse_probs(probs_a, probs_b, w: float):
    """Weighted average of two probability matrices (w = weight of model A)."""
    return w * probs_a + (1 - w) * probs_b


def tune_fusion_weight(probs_a, probs_b, labels, grid=None):
    """Chooses w on the VALIDATION set by macro-F1. Returns (best_w, table)."""
    rows = []
    for w in (grid if grid is not None else np.round(np.arange(0.0, 1.01, 0.1), 2)):
        m = compute_metrics(labels, fuse_probs(probs_a, probs_b, w).argmax(1))
        rows.append({"w_model_A": float(w), "macro_f1": m["macro_f1"], "jump_macro_f1": m["jump_macro_f1"], "acc": m["acc"]})
    table = pd.DataFrame(rows)
    return float(table.loc[table.macro_f1.idxmax(), "w_model_A"]), table


# ============================================================================ plots
def plot_history(history, title=""):
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train (smoothed, weighted)")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val (plain CE)")
    ax[0].set_title(f"{title} loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1")
    ax[1].plot(ep, [h["val_jump_f1"] for h in history], label="val jump macro-F1")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val accuracy", alpha=0.6)
    ax[1].set_title(f"{title} validation metrics"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.show()


def plot_confusion(cm, title="Confusion matrix", normalize=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(cmn, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, cbar=False)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(title)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()


def top_confusions(cm, k=6):
    pairs = [(CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j])) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
             if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda t: -t[2])[:k]

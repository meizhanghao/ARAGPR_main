"""Paired residual controls; fitting and selection never receive test labels."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


def standardize_splits(arrays):
    """Use the training distribution for every split, including constant axes."""
    scaler = StandardScaler().fit(np.asarray(arrays["train"], dtype=np.float32))
    scaled = {s: scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
              for s, x in arrays.items()}
    if not all(np.isfinite(x).all() for x in scaled.values()):
        raise ValueError("Nonfinite residual inputs.")
    return scaled, scaler


def random_projection(input_dim, output_dim, seed):
    """Fixed orthonormal rows; no geometry, class labels, or fitting."""
    if not 1 <= output_dim <= input_dim:
        raise ValueError("Projection dimension must be between 1 and input dimension.")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((input_dim, output_dim)))
    return q.T.astype(np.float32)


def frozen_semantic_logits(features, checkpoint, device="cpu", batch_size=512):
    """Reproduce GlobalCBM, using its saved training scaler and semantic head."""
    concepts = torch.as_tensor(checkpoint["concepts"], dtype=torch.float32, device=device)
    norm2 = torch.linalg.vector_norm(concepts, dim=1).square()
    weight = checkpoint["model_state"]["weight"].to(device)
    bias = checkpoint["model_state"]["bias"].to(device)
    outputs = []
    with torch.no_grad():
        for first in range(0, len(features), batch_size):
            h = torch.as_tensor(features[first:first+batch_size], dtype=torch.float32,
                                device=device)
            raw = h @ concepts.T / norm2[None, :]
            if torch.isnan(raw).any():
                raw = torch.nan_to_num(raw, nan=float("-inf"))
            a = torch.softmax(raw, dim=1).cpu().numpy()
            a = checkpoint["scaler"].transform(a).astype(np.float32)
            z = F.linear(torch.from_numpy(a).to(device), weight, bias)
            outputs.append(z.cpu().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Frozen semantic scores are nonfinite.")
    return result


def prediction_metrics(logits, labels, frozen_logits):
    labels = np.asarray(labels)
    pred, base = np.asarray(logits).argmax(1), np.asarray(frozen_logits).argmax(1)
    if len(labels) == 0 or len(pred) != len(labels) or len(base) != len(labels):
        raise ValueError("Empty or misaligned evaluation split.")
    base_correct, correct = base == labels, pred == labels
    corrected = int((~base_correct & correct).sum())
    harmed = int((base_correct & ~correct).sum())
    return {"n": len(labels), "n_correct": int(correct.sum()),
            "accuracy_pct": 100.0 * float(correct.mean()),
            "delta_frozen_pp": 100.0 * float(correct.mean() - base_correct.mean()),
            "n_corrected": corrected, "n_harmed": harmed,
            "correction_pct": 100.0 * corrected / max(1, int((~base_correct).sum())),
            "harm_pct": 100.0 * harmed / max(1, int(base_correct.sum()))}


def fit_teacher_decoder(train_x, train_targets, val_x, val_targets, *, seed,
                        device="cpu", epochs=300, patience=30, batch_size=64,
                        lr=1e-3, weight_decay=1e-4, progress=None):
    """Original affine SmoothL1/AdamW decoder, with train/validation inputs only."""
    if not len(train_x) or not len(val_x):
        raise ValueError("Decoder needs covered training AND validation images.")
    if not all(np.isfinite(x).all() for x in (train_x, train_targets, val_x, val_targets)):
        raise ValueError("Nonfinite teacher fitting arrays.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    decoder = nn.Linear(train_x.shape[1], train_targets.shape[1], bias=True)
    nn.init.xavier_uniform_(decoder.weight, gain=1.0)
    nn.init.zeros_(decoder.bias)
    decoder = decoder.to(device)
    tx, ty, vx, vy = [torch.as_tensor(x, dtype=torch.float32, device=device)
                      for x in (train_x, train_targets, val_x, val_targets)]
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batches = torch.Generator(device="cpu").manual_seed(seed + 2000003)
    best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    for epoch in range(1, epochs + 1):
        decoder.train()
        order = torch.randperm(len(tx), generator=batches).to(device)
        total = torch.zeros((), device=device)
        for first in range(0, len(tx), batch_size):
            idx = order[first:first+batch_size]
            loss = F.smooth_l1_loss(decoder(tx[idx]), ty[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(idx)
        scheduler.step()
        decoder.eval()
        with torch.no_grad():
            val_loss = float(F.smooth_l1_loss(decoder(vx), vy).item())
        if not math.isfinite(val_loss):
            raise ValueError("Teacher decoder training diverged.")
        history.append({"epoch": epoch, "train_loss": float(total.item()) / len(tx),
                        "val_loss": val_loss})
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
        else:
            stale += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            progress(f"decoder epoch={epoch}/{epochs} val_loss={val_loss:.6f} best={best:.6f}")
        if patience and stale >= patience:
            break
    return {"state": best_state, "best_epoch": best_epoch, "val_loss": best,
            "history": history, "epochs_ran": epoch}


def apply_affine_map(arrays, state, device="cpu", batch_size=512):
    weight, bias = state["weight"].to(device), state["bias"].to(device)
    result = {}
    with torch.no_grad():
        for split, x in arrays.items():
            pieces = []
            for first in range(0, len(x), batch_size):
                h = torch.as_tensor(x[first:first+batch_size], dtype=torch.float32, device=device)
                pieces.append(F.linear(h, weight, bias).cpu().numpy())
            result[split] = np.concatenate(pieces)
    return result


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    patience: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    l1: float = 1e-4
    eta_values: tuple = (0.1, 0.25, 0.5, 0.75, 1.0)

    def validate(self):
        if self.epochs < 1 or self.patience < 0 or self.batch_size < 1:
            raise ValueError("Invalid epoch, patience, or batch-size setting.")
        if self.lr <= 0 or self.l1 < 0 or not math.isfinite(self.lr + self.l1):
            raise ValueError("Invalid optimizer setting.")
        if (not self.eta_values or len(set(self.eta_values)) != len(self.eta_values)
                or any(not math.isfinite(v) or v <= 0 for v in self.eta_values)):
            raise ValueError("Eta candidates must be unique, finite and positive.")


def fit_residual(train_x, train_semantic, train_labels,
                 val_x, val_semantic, val_labels, *, config, seed, device="cpu",
                 progress=None):
    """Fit CE + L1 heads. Select by validation correct count; first tie wins.

    Only train/validation arrays enter this function. Test evaluation is a
    separate operation after the globally selected eta/head has been restored.
    Independent minibatch RNG gives every branch identical sample orders.
    """
    config.validate()
    arrays = (train_x, train_semantic, val_x, val_semantic)
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Nonfinite fitting arrays.")
    if (len(train_x) != len(train_labels) or len(train_x) != len(train_semantic)
            or len(val_x) != len(val_labels) or len(val_x) != len(val_semantic)
            or not len(train_x) or not len(val_x)
            or train_x.shape[1] != val_x.shape[1]
            or train_semantic.shape[1] != val_semantic.shape[1]):
        raise ValueError("Misaligned fitting arrays.")
    tx, ts, vx, vs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays]
    ty = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    vy = torch.as_tensor(val_labels, dtype=torch.long, device=device)
    history, candidates, selected = [], [], None
    for eta in config.eta_values:
        # Equal-dimensional controls start with exactly the same head.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        head = nn.Linear(tx.shape[1], ts.shape[1], bias=True)
        nn.init.xavier_uniform_(head.weight, gain=1.0)
        nn.init.zeros_(head.bias)
        head = head.to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        batches = torch.Generator(device="cpu").manual_seed(seed + 1000003)
        best, best_state, best_epoch, stale = -1, None, 0, 0
        for epoch in range(1, config.epochs + 1):
            head.train()
            order = torch.randperm(len(tx), generator=batches).to(device)
            # Accumulate on-device, avoiding one GPU synchronization per batch.
            loss_sum = torch.zeros((), device=device)
            train_correct = torch.zeros((), dtype=torch.long, device=device)
            for first in range(0, len(tx), config.batch_size):
                idx = order[first:first+config.batch_size]
                logits = ts[idx] + eta * head(tx[idx])
                loss = F.cross_entropy(logits, ty[idx]) + config.l1 * head.weight.abs().sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach() * len(idx)
                train_correct += (logits.detach().argmax(1) == ty[idx]).sum()
            head.eval()
            with torch.no_grad():
                val_logits = vs + eta * head(vx)
                val_correct = int((val_logits.argmax(1) == vy).sum().item())
                val_loss = float(F.cross_entropy(val_logits, vy).item())
            if not math.isfinite(val_loss):
                raise ValueError("Residual training diverged.")
            row = {"eta": eta, "epoch": epoch,
                   "train_loss": float(loss_sum.item()) / len(tx),
                   "train_accuracy_pct": 100.0 * int(train_correct.item()) / len(tx),
                   "val_correct": val_correct, "val_accuracy_pct": 100.0 * val_correct / len(vx),
                   "val_loss": val_loss}
            history.append(row)
            if val_correct > best:
                best, best_epoch, stale = val_correct, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                stale += 1
            scheduler.step()
            if progress and (epoch == 1 or epoch % 10 == 0):
                progress(f"eta={eta:g} epoch={epoch}/{config.epochs} "
                         f"val={row['val_accuracy_pct']:.4f}% best={100*best/len(vx):.4f}%")
            if config.patience and stale >= config.patience:
                break
        candidate = {"eta": eta, "best_epoch": best_epoch, "epochs_ran": epoch,
                     "val_correct": best, "val_accuracy_pct": 100.0 * best / len(vx)}
        candidates.append(candidate)
        if selected is None or best > selected["val_correct"]:
            selected = {**candidate, "state": best_state}
    return {"selected": selected, "candidates": candidates, "history": history,
            "trainable_parameters": tx.shape[1] * ts.shape[1] + ts.shape[1]}


def apply_residual(x, semantic, fitted, device="cpu", batch_size=512):
    """Apply a validation-selected affine head; no selection occurs here."""
    selected = fitted["selected"]
    weight = selected["state"]["weight"].to(device)
    bias = selected["state"]["bias"].to(device)
    outputs = []
    with torch.no_grad():
        for first in range(0, len(x), batch_size):
            features = torch.as_tensor(x[first:first+batch_size], dtype=torch.float32, device=device)
            correction = F.linear(features, weight, bias).cpu().numpy()
            outputs.append(semantic[first:first+batch_size] + selected["eta"] * correction)
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite final logits.")
    return result

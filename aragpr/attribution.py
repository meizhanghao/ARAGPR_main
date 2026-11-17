"""Exact coordinate attribution for an additive affine residual head."""
import numpy as np


def class_margin_attribution(base_logits, coordinates, weight, bias, eta=1.0,
                             *, reference_classes=None, target_classes=None):
    """Decompose a selected class margin into base, coordinates and head bias.

    Rows are images; weight has shape (number of classes, number of coordinates).
    Coordinates must already include any training-fitted standardization and
    optional reliability weights used by the fitted head. Default contrasts are
    the final predicted class against the original base predicted class. This
    algebra does not localize evidence in the test image or establish causality.
    """
    base, u, w, b = [np.asarray(x, dtype=np.float64)
                     for x in (base_logits, coordinates, weight, bias)]
    if base.ndim != 2 or u.ndim != 2 or w.ndim != 2:
        raise ValueError("Expected matrix logits, coordinates and weights.")
    n, classes = base.shape
    if u.shape[0] != n or w.shape != (classes, u.shape[1]) or b.shape != (classes,):
        raise ValueError("Incompatible image, class or coordinate dimensions.")
    if not all(np.isfinite(x).all() for x in [base, u, w, b]) or not np.isfinite(eta):
        raise ValueError("Nonfinite model inputs.")
    final = base + float(eta) * (u @ w.T + b)

    def class_indices(value, default):
        if value is None:
            return default
        raw = np.asarray(value)
        if raw.shape != (n,) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Each image requires one integer class index.")
        if np.any((raw < 0) | (raw >= classes)):
            raise ValueError("Class index outside the classifier's class order.")
        return raw.astype(np.int64)

    reference = class_indices(reference_classes, base.argmax(1))
    target = class_indices(target_classes, final.argmax(1))
    ix = np.arange(n)
    q = float(eta) * u * (w[target] - w[reference])
    qb = float(eta) * (b[target] - b[reference])
    base_margin = base[ix, target] - base[ix, reference]
    final_margin = final[ix, target] - final[ix, reference]
    reconstructed = base_margin + q.sum(1) + qb
    if not np.allclose(reconstructed, final_margin, atol=1e-9, rtol=1e-9):
        raise ArithmeticError("Coordinate contributions do not close the margin.")
    return dict(base_logits=base, final_logits=final,
                reference_classes=reference, target_classes=target,
                base_margin=base_margin, coordinate_contributions=q,
                bias_contribution=qb, final_margin=final_margin,
                reconstructed_margin=reconstructed)


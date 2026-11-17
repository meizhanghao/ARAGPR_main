# Protocols and interpretation

## Defaults and controls

The primary shape configuration is geo11, 128 coordinates, training-fitted
descriptor standardization, ordinary KMeans, and a median-nearest observed
training-region representative. FD/curvature families use closed-contour
resampling; raw-contour descriptors preserve their own area/moment definitions.
Neither ordinary KMeans nor an observed representative implies whole-cluster
semantic purity.

The teacher is the vector of coordinate-wise maximum similarities over available
regions, standardized with training statistics. Different coordinates can draw
their maximum from different regions. A missing teacher is excluded from decoder
supervision/recovery metrics; decoded classification can still operate on that
image's embedding.

The experiment runners record their actual settings. CLI overrides take
precedence over defaults. Keep the base checkpoint, members and class order fixed
within a paired study. Validation selection is separate from test evaluation.

| Study | What changes | What remains fixed | Scope |
|---|---|---|---|
| Ordinary affine residual | Head input is the image embedding | Base logits, members, training/selection rules | Extra embedding capacity comparator |
| Random projection | A fixed random coordinate map | Coordinate dimension and fitted head protocol | Coordinate-map comparator |
| Decoder baselines | Mean, paired or shuffled regression targets | Held-out recovery targets and metric convention | Partial target accessibility |
| Training-pair control | Whole teacher vectors paired globally or within class | Labels, decoder architecture, head protocol | Usefulness of image-target pairing |
| Complete constant term | Remove input-dependent affine terms post hoc | Saved head, decoder offsets, scalers and scale | Input dependence of the fitted branch |
| Fixed-head Teacher | Replace decoded head inputs with true teacher inputs | Head, scale, transforms, image subset and contrast | Decision-level correspondence |
| Shuffled Teacher at evaluation | Reassign available target vectors | Same already-fitted models and contrasts | Correspondence reference distribution |
| Ordinary linear base | Base classifier and its own training protocol | Compatible embedding/coordinate interface | Interface check outside a CBM base |

Direct teacher retrains a classification head on mask-derived target responses.
Fixed-head Teacher replacement reuses the decoded head. Their accuracies answer
different questions and need not coincide.

## Attribution and metrics

For fixed target/reference classes `f` and `b`, each coordinate contributes
`q_k = eta * u_k * (W[f,k] - W[b,k])`. The bias is
`eta * (bias[f] - bias[b])`. Summing all coordinates, bias and base margin gives
the final margin before rounding. When `f == b`, this contrast is identically
zero. A positive contribution supports the selected class contrast; it does not
by itself mean high raw similarity or a semantically correct prototype name.

Top-k references are ranked by absolute signed contribution, not activation.
Keep the full contribution vector when displaying a subset. Correction and harm
are different events: a correction turns an originally wrong prediction right;
a harm turns an originally right prediction wrong. Conditional rates use their
respective initially-wrong/initially-right denominators.

Teacher replacement and alignment comparisons use the same available-target test
subset. Contribution direction and Top-k identity overlap are different metrics.
Use the saved contrast rather than choosing a new contrast after each replacement.
The complete-constant control also includes propagated decoder/scaler offsets;
it is different from keeping only the fitted head's bias.

## Aggregation

The manuscript's principal accuracy rows use the arithmetic mean of per-epoch
test accuracy over the final 15 training epochs. The Frozen comparison averages
the corresponding sequence of original base checkpoints.

The added diagnostic runners select saved models using validation data and
report those fixed-checkpoint results. Pairing studies can repeat branch fits
with seeds 42, 43 and 44 while holding the base/bank/members fixed. Their sample
standard deviation is between those fits; it is not final-epoch oscillation or
uncertainty over independent base-training runs.

Do not substitute one convention for another. Exact reproduction of a published
epoch mean requires the corresponding experiment configuration and checkpoint
sequence. This source snapshot does not contain the server's full run artifacts.

## Interpretation boundaries

The methods provide score accounting, training-reference traceability and
measured correspondence. They do not establish test-image localization, pure
contour causality, medical deployment validity, or statistically significant
improvements solely from a small-seed mean. Review each control in its actual
cohort and protocol. Auxiliary names remain outside the classifier arithmetic.


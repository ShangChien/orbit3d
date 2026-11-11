# Uni-QSAR prediction and fold dispersion

The bridge preserves the trained model's point-prediction definition: aggregate
in normalized target space, then apply the fitted inverse transform. Changing
that order for a nonlinear scaler would change the model's predictions. Both
ordinary and coordinate-differentiable entry points use this definition.

Dispersion is a separate diagnostic. Each fold prediction is inverse-transformed
before calculating its physical-unit spread. Average ensembles assign equal
weight to each selected model and equal weight to folds within a model; unequal
fold counts do not give one model more weight. Stacking uses its meta-model folds.
Vote mode reports the spread of the selected model predictions. A nonlinear
scaler cannot inverse-transform a standard deviation by multiplying `scale_`.
This spread is not a calibrated prediction interval or a quantum error estimate.

The shared inverse implementation respects the fitted scaler flags and function,
including identity FunctionTransformer, MinMax, Robust, MaxAbs, Standard and
Box-Cox/Yeo-Johnson transforms. It retains invalid inverse domains rather than
clipping them into finite predictions. Unknown inverse functions require an
explicit differentiable implementation; ordinary NumPy prediction remains usable.
Single multivariate scalers and per-target scaler lists are both supported.

Regression tests compare actual fitted sklearn inverse transforms, ordinary and
differentiable ensemble values, physical fold dispersion with unequal fold
counts, and finite-difference coordinate derivatives. Nonlinear stacking remains
a black-box predictor and is not replaced with an averaging surrogate.

The saved configuration, model-selection order, target scaler and complete fold
sets are required for reloading. A missing selected model or stacking file is an
incomplete checkpoint, not permission to average a different subset. An explicit
`infer_choices` subset retains native Uni-QSAR's averaging semantics, including
when the saved method was vote. Target names are never invented to hide a shape
mismatch.

Only verified linear regression estimator types use extracted coefficients.
Classification stacks retain their native probability predictions: even a linear
SVC's calibrated probabilities are not the sigmoid of its decision function.
LightGBM checkpoints require the real compatible LightGBM runtime; the former
partial tree parser has been removed. An auxiliary head cannot supply gradients
for another model's reported value. Use it as a separately declared Oracle if
needed. Tests cover actual serialized ensembles, missing folds/scalers, a fitted
probability-calibrated linear classifier and native Ridge prediction parity.

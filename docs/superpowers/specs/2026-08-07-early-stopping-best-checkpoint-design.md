# EMA Early Stopping and Best Checkpoint Design

## Goal

Improve the stability of the final delivered model without changing the current
150-epoch learning-rate or image-augmentation schedules in the first
experiment. Training may stop early when validation performance has stopped
improving, but the final model must always restore the complete checkpoint from
the epoch with the highest observed EMA `mAP50_95`.

The first experiment is an early-stopping baseline (`ES-Base`). It must isolate
checkpoint selection and restoration from later schedule experiments. In
particular, it does not change loss formulas, precision settings, optimizer
parameters, learning-rate boundaries, image augmentations, or stage behavior.

## Success Criteria

- The final delivered EMA model achieves at least 95% of the highest observed
  validation `mAP50_95`. With deterministic evaluation and successful restore,
  the expected result is equality with the observed peak within floating-point
  tolerance.
- `epoches: 150` remains the only hard training limit.
- The FP16 and BF16 experiments use identical early-stopping parameters while
  independently selecting their own best epoch.
- `last.pth` remains a true interruption-resume checkpoint and is not
  overwritten by the restored best state.

## Configuration Contract

The training configuration accepts an optional `early_stopping` mapping:

```yaml
early_stopping:
  enabled: true
  metric: mAP50_95
  mode: max
  min_epochs: 100
  patience: 12
  min_delta: 0.001
  restore_best: true
```

The initial experiment keeps the existing schedule unchanged:

```yaml
epoches: 150
warmup_iter: 15
flat_epoch: 75
no_aug_epoch: 15
```

`epoches` is both the scheduler horizon and the hard loop limit. No separate
`max_epochs` or `schedule_epochs` setting is introduced.

In the current `FlatCosineLRScheduler`, `no_aug_epoch` controls the number of
epochs spent at the minimum learning rate. It does not disable image
augmentation. Image-augmentation stages remain controlled by the collate
configuration, including `policy.epoch`, `mixup_epochs`, and `stop_epoch`.

Missing `early_stopping` configuration preserves the current training behavior.

## Monitored Model and Metric

Validation already evaluates:

```python
module = self.ema.module if self.ema else self.model
```

Early stopping therefore monitors the EMA model when EMA is enabled. For OBB
training, the only monitored metric is `test_stats["mAP50_95"]`. The state
machine must not reuse the existing cross-metric `top1` behavior because an
unrelated validation metric must never reset patience or replace the global
best checkpoint.

Two best values are maintained:

- `best_observed_metric`: the highest numeric `mAP50_95` seen so far. Any
  strict improvement updates this value and saves `best.pth`.
- `best_significant_metric`: the reference value used for patience. It is
  updated only when `current_metric > best_significant_metric + min_delta`.

This distinction ensures that small improvements still become the delivered
best checkpoint while validation noise below `min_delta` cannot indefinitely
extend training.

## Early-Stopping State

The complete persistent state contains:

- `best_observed_metric`
- `best_significant_metric`
- `best_epoch`
- `epochs_without_improvement`

At the end of every completed validation epoch:

1. Record whether `current_metric > best_observed_metric`. If so, update the
   observed best and record the epoch.
2. If `current_metric > best_significant_metric + min_delta`, update the
   significant best and set `epochs_without_improvement` to zero.
3. Otherwise increment `epochs_without_improvement`.
4. After both best values and the wait counter are current, save the complete
   training state as `best.pth` if step 1 recorded an observed improvement.
5. Do not stop before `min_epochs` completed epochs.
6. Once the minimum has been reached, stop when
   `epochs_without_improvement >= patience`.
7. If patience is not exhausted, training ends normally when the 150-epoch hard
   limit is reached.

`min_epochs` prevents early exit only. Best metrics and the wait counter are
tracked from the first validation epoch. Consequently, if the last significant
improvement happened more than `patience` epochs earlier, training may stop as
soon as the minimum training requirement is met.

## Checkpoint Responsibilities

Checkpoint files have separate contracts:

- `best.pth`: complete state from the highest observed EMA `mAP50_95`; used for
  final delivery and reproducible best-model evaluation.
- `last.pth`: complete state from the most recently completed training and
  validation epoch; used for interruption recovery.
- `best_stg1.pth` and `best_stg2.pth`: retain their current stage-transition
  responsibilities.

`best.pth` contains the same normal solver state as `state_dict()`, including
the model, EMA, optimizer, scheduler, GradScaler, and `last_epoch`, plus the
early-stopping state listed above.

Because the early-stopping wait counter changes after validation, `last.pth`
must be written after validation and early-stopping state updates. Periodic
checkpoints follow the same ordering. This ensures a resumed run continues with
the exact latest patience state.

The final restore must never overwrite `last.pth`; otherwise interruption
history would be replaced by the best epoch and resuming would repeat work.

## Stage Compatibility

The current OBB stage logic remains intact:

- `best_stg1.pth` and `best_stg2.pth` continue to select stage checkpoints.
- Reaching the collate `stop_epoch` may still restore `best_stg1.pth` and
  restart EMA as currently implemented.
- A stage transition resets `epochs_without_improvement` to zero so the new
  stage receives a full patience window.
- A stage transition does not clear `best_observed_metric`,
  `best_significant_metric`, `best_epoch`, or `best.pth`; the final checkpoint
  remains the global best across stages.

The early-stopping mechanism is therefore additive and does not redefine the
existing stage algorithm.

## Distributed Coordination

Rank 0 reads the validation metric, updates the early-stopping state, and saves
checkpoints. It then broadcasts the updated state and `should_stop` decision to
all ranks. Every process exits the epoch loop together.

The stop condition is computed from the updated state, but the loop exits only
after the epoch's log entry and checkpoint writes have completed. Rank 0 then
broadcasts the decision. This guarantees that an early-stopping epoch has
complete records and that no rank enters the next training epoch alone.

## Resume Compatibility

When resuming from a new `last.pth`, the solver restores the early-stopping
state and continues the same patience window.

When resuming from an older checkpoint that does not contain early-stopping
state, the existing pre-resume EMA validation initializes both best metrics and
`best_epoch`; `epochs_without_improvement` starts at zero. This preserves
backward compatibility without inventing historical patience data.

The standalone validation command loads the requested checkpoint and evaluates
it normally. It does not run the early-stopping state machine.

## Training Exit and Final Verification

Training has three exit paths:

- `early_stopping`: patience was exhausted after `min_epochs`.
- `max_epochs`: the 150-epoch hard limit was reached.
- `diagnostic`: `max_optimizer_steps` stopped a diagnostic run.

For `early_stopping` and `max_epochs`, if `restore_best` is enabled:

1. Capture the real `stop_epoch` and `stop_reason` before loading any earlier
   checkpoint.
2. Synchronize all ranks after leaving the loop.
3. Load the complete `best.pth` state. This intentionally restores
   `last_epoch`, model, EMA, optimizer, scheduler, and scaler to the best epoch.
4. Evaluate the restored EMA model once more.
5. Verify that restored `mAP50_95` matches `best_observed_metric` within the
   configured evaluation tolerance.
6. Record the captured `stop_reason` and `stop_epoch` together with
   `best_epoch`, `best_mAP50_95`, `restored_mAP50_95`, and
   `epochs_after_best` in final run metadata. This metadata is not derived from
   the restored checkpoint's `last_epoch`.

The `diagnostic` path does not restore `best.pth`; diagnostic step caps must
remain fast and must not silently alter the state under inspection.

## First Experiment and Follow-Up Experiments

The first experiment is `ES-Base`:

- Enable early stopping with the approved configuration.
- Keep `epoches: 150`.
- Keep `flat_epoch: 75` and `no_aug_epoch: 15`.
- Keep the current image-augmentation policy unchanged.
- Always restore `best.pth` after a normal training exit.

If the delivered checkpoint is stable but the validation curve still peaks too
late, later experiments change one schedule variable at a time:

1. `ES-LR`: reduce `flat_epoch` from 75 to 60 or 65 to start cosine decay
   earlier and move the convergence window forward.
2. `ES-Aug`: after selecting the better LR boundary, extend the light
   augmentation stage by changing the relevant collate policy boundary.
3. Combine the LR and augmentation changes only after their individual effects
   are established.

The first implementation does not include these schedule changes.

## Tests

Tests must verify:

1. Missing or disabled configuration preserves current training behavior.
2. A strict numeric improvement saves a new complete `best.pth`.
3. An improvement smaller than `min_delta` updates the observed best checkpoint
   but does not reset patience.
4. A significant improvement resets patience.
5. Exhausted patience cannot stop training before `min_epochs`.
6. Exhausted patience stops all distributed ranks after `min_epochs`.
7. Reaching the 150-epoch limit restores and validates `best.pth` even if
   patience was not exhausted.
8. A new `last.pth` restores the complete early-stopping state.
9. An old checkpoint initializes state from the current EMA validation.
10. A stage transition resets patience but preserves the global best checkpoint.
11. Final restored evaluation matches the saved best metric within tolerance.
12. `last.pth` remains the actual final training epoch after best restoration.
13. A `max_optimizer_steps` diagnostic exit does not restore `best.pth`.

## Non-Goals

- Guaranteeing that the last five raw training epochs have a flat validation
  curve; this design guarantees the quality of the delivered checkpoint.
- Changing the 150-epoch hard limit in the first experiment.
- Changing LR, image augmentation, regularization, loss weights, precision, or
  model architecture in `ES-Base`.
- Removing or redesigning `best_stg1.pth` and `best_stg2.pth`.
- Replacing EMA with raw-model validation.

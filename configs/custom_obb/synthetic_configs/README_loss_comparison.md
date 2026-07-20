# Synthetic OBB loss comparison

The three active configs form a controlled criterion-geometry matrix. They
inherit the same density-020 dataset, model, augmentation, optimizer, schedule,
parent config, and 80-epoch budget from
`synthetic_exp_020_anrep1_offset_per.yml`. They also use identical matcher
weights, including `cost_chamfer: 2`. Apart from distinct output directories,
only the listed criterion geometry switches and weights differ.

| Active config | ProbIoU | Angle | Active geometry weights |
| --- | --- | --- | --- |
| `synthetic_exp_020_loss_kld.yml` | off | off | `loss_bbox: 5`, `loss_kld: 2` |
| `synthetic_exp_020_loss_prob_kld.yml` | on | off | `loss_bbox: 5`, `loss_probiou: 5`, `loss_kld: 2` |
| `synthetic_exp_020_loss_prob_angle_kld.yml` | on | on | `loss_bbox: 5`, `loss_probiou: 5`, `loss_angle: 3`, `loss_kld: 2`; `angle_lambda: 3.0` |

## Historical provenance

The completed run named `synthetic_exp_020_loss_prob_kld` was historically
mislabeled: its launch config enabled both ProbIoU and angle, with
`loss_angle: 3` and `angle_lambda: 3.0`. The active file now represents the
name truthfully as ProbIoU without angle. The original launch-config bytes are
preserved separately in
`provenance/synthetic_exp_020_loss_prob_kld.completed.yml`; the completed KLD
launch config is preserved alongside it.

`provenance/completed_runs.yml` records only facts supported by those two
snapshots and their SHA256 digests. Resolved historical configs and output
artifacts are unavailable, so the provenance does not claim metrics,
checkpoints, resolved inheritance, timestamps, or other run outputs. The
snapshots document launch configuration only and are not active experiment
configs.

The active `synthetic_exp_020_loss_prob_angle_kld.yml` is intended only for
bounded diagnostic reproduction of the problematic ProbIoU-plus-angle case,
not as a general training recommendation. The two already completed 80-epoch
runs should not be rerun merely to reconstruct unavailable provenance.

# GRPO + Entropy Bonus Ablation Report (Experiment A)

## Entropy bonus runs (final state)

| run | val_sr_end3 | val_sr_last | train_sr_end3 | train_sr_last | ent_end3 |
|---|---:|---:|---:|---:|---:|
| entropy_bn0.005_s0 | nan | nan | nan | nan | nan |
| entropy_bn0.01_s0 | nan | nan | nan | nan | nan |
| entropy_bn0.02_s0 | nan | nan | nan | nan | nan |

## Baseline comparison (dual-token vs observe, from prior logs)

| run | val_sr_end3 | val_sr_last | train_sr_end3 | ent_end3 |
|---|---:|---:|---:|---:|
| dual-token (l49ikuco) | nan | nan | nan | nan |
| observe (lmlyvpa6) | nan | nan | nan | nan |

## Interpretation hook

If entropy_bonus runs reach val_sr_end3 >= dual-token (~0.81-0.88 end5 range),
framing (b) 'entropy regularizer' is supported. If they stay near observe baseline
(~0.79-0.81), framing (a) 'observation-grounded LM' is supported. See EXPERIMENT_LOG
§4.2 for the three framings.

# Entropy × Surprise Joint Analysis (3 wandb runs)

Inputs:
- ALFW observe: `ocar/analysis_results/webshop/alfworld_observe_history.csv`
- Webshop observe: `ocar/analysis_results/webshop/history_full.csv`
- ALFW dual-token: `ocar/analysis_results/wandb_dualtoken_l49ikuco_full.csv`

## Q1. Training-level Spearman
| run | n | rho(ent,s_th) | p(ent,s_th) | rho(ent,dS) | p(ent,dS) | rho(ent,SR) | p(ent,SR) | rho(s_th,SR) | p(s_th,SR) | rho(dS,SR) | p(dS,SR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ALFW observe | 30 | 0.135 | 0.477 | 0.011 | 0.954 | 0.118 | 0.534 | 0.320 | 0.085 | 0.722 | 0.000 |
| Webshop observe | 31 | 0.177 | 0.341 | 0.470 | 0.008 | 0.285 | 0.120 | -0.057 | 0.760 | 0.139 | 0.457 |
| ALFW dual | 150 | 0.045 | 0.584 | 0.132 | 0.108 | -0.013 | 0.874 | -0.187 | 0.022 | -0.138 | 0.092 |

## Q2. Within-batch succ/fail gaps (fail minus succ)
| run | n | ent_gap_mean | ent_P(f>s) | ent_t | ent_p | s_th_gap_mean | s_th_P(f>s) | s_th_t | s_th_p |
|---|---|---|---|---|---|---|---|---|---|
| Webshop observe | 31 | 0.0927 | 0.8710 | 6.9516 | 0.0000 | 0.0828 | 0.6129 | 1.9590 | 0.0595 |
| ALFW dual | 150 | 0.4523 | 1.0000 | 17.2443 | 0.0000 | 0.0209 | 0.5133 | 2.0612 | 0.0410 |

## Q3. Aligned trajectory: dual-token vs GRPO+observe on ALFW
| metric | obs_mean | dt_mean | obs_end3 | dt_end3 | delta_end3 |
|---|---|---|---|---|---|
| s_th | 1.9295 | 1.8394 | 2.1929 | 1.9418 | -0.2512 |
| dS | -0.2230 | -0.3286 | -0.0054 | -0.3559 | -0.3505 |
| ent | 0.7253 | 0.6502 | 0.8821 | 0.7583 | -0.1238 |
| SR | 0.6648 | 0.6112 | 0.8359 | 0.7630 | -0.0729 |
| wm_s | 3.3336 | 3.1376 | 3.8220 | 3.0997 | -0.7223 |
| wm_s_B | 4.0960 | 4.0590 | 4.6842 | 3.9654 | -0.7188 |

## Q5. Detrended Spearman (linear _step trend removed)
| run | n | rhodet(ent,SR) | p(ent,SR) | rhodet(s_th,SR) | rhodet(dS,SR) | rhodet(ent,s_th) | rhodet(ent,dS) |
|---|---|---|---|---|---|---|---|
| ALFW observe | 30 | -0.539 | 0.002 | -0.171 | -0.175 | 0.106 | -0.223 |
| Webshop observe | 31 | -0.192 | 0.302 | -0.178 | -0.073 | 0.206 | 0.231 |
| ALFW dual | 150 | -0.668 | 0.000 | -0.221 | -0.319 | 0.215 | 0.199 |

## Q6. Gap quartiles — dual-token (ALFW)
| quartile | step_lo | step_hi | n | ent_gap_mean | ent_gap_std | s_th_gap_mean | s_th_gap_std | SR_mean |
|---|---|---|---|---|---|---|---|---|
| Q1 | 1 | 38 | 38 | 0.0987 | 0.0366 | 0.0155 | 0.1308 | 0.4350 |
| Q2 | 39 | 76 | 38 | 0.2399 | 0.0677 | 0.0025 | 0.0869 | 0.6287 |
| Q3 | 77 | 113 | 37 | 0.6382 | 0.1531 | 0.0307 | 0.1368 | 0.6590 |
| Q4 | 114 | 150 | 37 | 0.8479 | 0.1473 | 0.0357 | 0.1388 | 0.7107 |

## Q6. Gap quartiles — webshop observe
| quartile | step_lo | step_hi | n | ent_gap_mean | ent_gap_std | s_th_gap_mean | s_th_gap_std | SR_mean |
|---|---|---|---|---|---|---|---|---|
| Q1 | 20 | 160 | 8 | 0.1017 | 0.0681 | -0.0123 | 0.1473 | 0.3281 |
| Q2 | 180 | 320 | 8 | 0.0913 | 0.0791 | 0.2063 | 0.2364 | 0.4805 |
| Q3 | 360 | 500 | 8 | 0.0846 | 0.0904 | 0.0058 | 0.2818 | 0.6133 |
| Q4 | 520 | 640 | 7 | 0.0933 | 0.0715 | 0.1383 | 0.2227 | 0.6562 |

## Q8. Entropy phase means: dual-token vs observe (ALFW)
| step_lo | step_hi | n | ent_obs | ent_dt | delta_dt_minus_obs | actor_ent_loss_dt |
|---|---|---|---|---|---|---|
| 5.0000 | 50.0000 | 10.0000 | 0.5905 | 0.5579 | -0.0326 | 0.5859 |
| 55.0000 | 100.0000 | 10.0000 | 0.8011 | 0.6425 | -0.1586 | 0.7730 |
| 105.0000 | 150.0000 | 10.0000 | 0.7844 | 0.7501 | -0.0343 | 1.0462 |

## Q10. Cumulative t-stat: entropy succ/fail gap (dual-token)
| first_k | mean_gap | t | p |
|---|---|---|---|
| 5.0000 | 0.0600 | 5.8046 | 0.0044 |
| 10.0000 | 0.0633 | 11.0923 | 0.0000 |
| 15.0000 | 0.0700 | 12.7571 | 0.0000 |
| 20.0000 | 0.0757 | 13.4837 | 0.0000 |
| 30.0000 | 0.0874 | 16.8657 | 0.0000 |
| 50.0000 | 0.1160 | 17.2834 | 0.0000 |
| 75.0000 | 0.1673 | 16.4306 | 0.0000 |
| 100.0000 | 0.2662 | 13.1688 | 0.0000 |
| 150.0000 | 0.4523 | 17.2443 | 0.0000 |

## Per-task end-of-training (dual-token, last 5 logged steps)
| task | n | ent_end5 | s_th_end5 | SR_end5 | rho(ent,SR) | p(ent,SR) | rho(s_th,SR) | rho(ent,s_th) |
|---|---|---|---|---|---|---|---|---|
| heat | 116 | 0.1096 | 1.7025 | 1.0000 | -0.8059 | 0.0000 | -0.0580 | 0.0359 |
| examine | 133 | 0.4991 | 2.0126 | 0.9004 | -0.7915 | 0.0000 | -0.1701 | 0.1131 |
| pick_place | 150 | 0.6456 | 2.0211 | 0.8370 | -0.6813 | 0.0000 | -0.2658 | 0.2481 |
| cool | 147 | 0.6323 | 1.6232 | 0.8112 | -0.7269 | 0.0000 | -0.0612 | 0.0599 |
| clean | 150 | 0.7733 | 1.7175 | 0.7960 | -0.4889 | 0.0000 | -0.1610 | 0.0177 |
| other | 150 | 0.8915 | 1.9620 | 0.6789 | 0.1244 | 0.1292 | -0.1877 | 0.0241 |

## Lead-lag (argmax |CCF|)
| run | lag(ent->SR) | cc(ent,SR) | lag(dS->SR) | cc(dS,SR) | lag(ent->dS) | cc(ent,dS) |
|---|---|---|---|---|---|---|
| ALFW observe | -8 | 0.636 | 9 | 0.769 | -10 | 0.730 |
| Webshop observe | 10 | 0.588 | 5 | 0.525 | 1 | 0.543 |
| ALFW dual | 4 | 0.448 | -6 | -0.466 | -12 | 0.271 |

## Residual hardness: SR vs ent_gap (dual-token)
rho(SR, fail_ent - succ_ent) = +0.655  p = 9.6e-20  n = 150
mean gap = +0.4523, mean SR = 0.607

## Val vs train SR contrast (ALFW)
observe train_end3 = 0.8359, val_end3 = 0.8073, val_last = 0.7969
dual    train_end3 = 0.7135, val_end3 = 0.8125, val_last = 0.8828

Figure: `overview.png`
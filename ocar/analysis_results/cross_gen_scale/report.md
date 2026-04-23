# Cross-Generation × Scale Analysis (obs-NLL)

Models loaded: 9/10

## Per-model summary

| model | gen | scale (B) | obs_nll_mean | succ | fail | gap | AUC | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | Qwen2.5 | 0.5 | 2.013 | 2.053 | 1.976 | +0.077 | 0.495 | 0.942 |
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 1.5 | 1.240 | 1.254 | 1.227 | +0.027 | 0.488 | 0.847 |
| Qwen2.5-3B-Instruct | Qwen2.5 | 3.0 | 1.752 | 1.899 | 1.614 | +0.285 | 0.547 | 0.435 |
| Qwen2.5-7B-Instruct | Qwen2.5 | 7.0 | 1.793 | 1.854 | 1.736 | +0.119 | 0.498 | 0.975 |
| Qwen2.5-14B-Instruct | Qwen2.5 | 14.0 | 1.904 | 2.118 | 1.703 | +0.414 | 0.545 | 0.458 |
| Qwen3-1.7B | Qwen3 | 1.7 | 3.277 | 3.294 | 3.261 | +0.034 | 0.491 | 0.878 |
| Qwen3-4B | Qwen3 | 4.0 | 2.708 | 2.721 | 2.695 | +0.026 | 0.484 | 0.797 |
| Qwen3-8B | Qwen3 | 8.0 | 2.584 | 2.699 | 2.476 | +0.222 | 0.483 | 0.776 |
| Qwen3-14B | Qwen3 | 14.0 | 2.361 | 2.484 | 2.245 | +0.240 | 0.506 | 0.923 |

## Spearman corr of per-step obs_nll across models

| model | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Qwen2.5-0.5B-Instruct | 1.000 | 0.882 | 0.896 | 0.892 | 0.945 | 0.917 | 0.916 | 0.960 | 0.964 |
| Qwen2.5-1.5B-Instruct | 0.882 | 1.000 | 0.904 | 0.898 | 0.874 | 0.931 | 0.933 | 0.897 | 0.876 |
| Qwen2.5-3B-Instruct | 0.896 | 0.904 | 1.000 | 0.867 | 0.914 | 0.890 | 0.910 | 0.879 | 0.890 |
| Qwen2.5-7B-Instruct | 0.892 | 0.898 | 0.867 | 1.000 | 0.902 | 0.921 | 0.921 | 0.895 | 0.895 |
| Qwen2.5-14B-Instruct | 0.945 | 0.874 | 0.914 | 0.902 | 1.000 | 0.901 | 0.914 | 0.939 | 0.951 |
| Qwen3-1.7B | 0.917 | 0.931 | 0.890 | 0.921 | 0.901 | 1.000 | 0.941 | 0.939 | 0.915 |
| Qwen3-4B | 0.916 | 0.933 | 0.910 | 0.921 | 0.914 | 0.941 | 1.000 | 0.937 | 0.927 |
| Qwen3-8B | 0.960 | 0.897 | 0.879 | 0.895 | 0.939 | 0.939 | 0.937 | 1.000 | 0.969 |
| Qwen3-14B | 0.964 | 0.876 | 0.890 | 0.895 | 0.951 | 0.915 | 0.927 | 0.969 | 1.000 |

## Variance decomposition

| source | SS | % of total |
|---|---:|---:|
| total | 3395.6 | 100.0 |
| between-model (model id) | 278.7 | 8.2 |
|   → attributable to generation | 203.3 | 6.0 |
|   → attributable to scale | 269.0 | 7.9 |
| between-step (text intrinsic) | 2667.2 | 78.5 |
| model × step residual | 449.7 | 13.2 |

## AUC trend by scale within each generation

| generation | scale | AUC |
|---|---:|---:|
| Qwen2.5 | 0.5 | 0.495 |
| Qwen2.5 | 1.5 | 0.488 |
| Qwen2.5 | 3.0 | 0.547 |
| Qwen2.5 | 7.0 | 0.498 |
| Qwen2.5 | 14.0 | 0.545 |
| Qwen3 | 1.7 | 0.491 |
| Qwen3 | 4.0 | 0.484 |
| Qwen3 | 8.0 | 0.483 |
| Qwen3 | 14.0 | 0.506 |

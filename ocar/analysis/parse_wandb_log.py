"""Parse local wandb output.log to extract step-wise metrics as a DataFrame."""
import re
from pathlib import Path
import pandas as pd


STEP_RE = re.compile(r"^step:(\d+)\s+-\s+(.*)$")
KV_RE = re.compile(r"([^\s:]+):([-\d.eE+]+|nan|inf|-inf)")


def parse_output_log(path):
    """Parse a wandb output.log and return a DataFrame indexed by step."""
    rows = {}
    with open(path) as f:
        for line in f:
            m = STEP_RE.match(line.strip())
            if not m:
                continue
            step = int(m.group(1))
            kv_str = m.group(2)
            row = rows.setdefault(step, {})
            for km in KV_RE.finditer(kv_str):
                key = km.group(1)
                try:
                    val = float(km.group(2))
                except ValueError:
                    continue
                row[key] = val
    if not rows:
        return pd.DataFrame()
    steps = sorted(rows.keys())
    df = pd.DataFrame([rows[s] for s in steps], index=pd.Index(steps, name="step"))
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/local_nvme/guanyiming/project/verl-agent/wandb/run-20260415_105029-lmlyvpa6/files/output.log"
    df = parse_output_log(path)
    print(f"shape={df.shape}, steps=[{df.index.min()}..{df.index.max()}]")
    print(f"columns ({len(df.columns)}):")
    for c in sorted(df.columns):
        if any(x in c for x in ["obs_s_theta", "wm_", "success_rate", "entropy", "delta_s"]):
            print(f"  {c}")

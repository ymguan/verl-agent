#!/usr/bin/env python3
"""
Checkpoint Auto-Uploader

监控 checkpoint 目录，自动将 FSDP checkpoints 合并为 HuggingFace 格式并上传。

Pipeline:
  1. 检测 global_step_* 目录
  2. 验证 FSDP rank 分片完整性
  3. 合并分片为 HuggingFace 模型（via verl FSDPModelMerger）
  4. 上传到 HuggingFace Hub
  5. 添加到 HuggingFace Collection

命名规范:
  - Collection: {model_prefix}-{date_prefix}     e.g. ocar-alfworld-0410
  - Model repo: {model_prefix}-{date_prefix}-step-{step}  e.g. ocar-alfworld-0410-step-5

用法:
    # 持续轮询模式（训练时后台运行）
    python tools/checkpoint_uploader.py \\
        --checkpoint_dir outputs/checkpoints \\
        --hf_user Ricardo-H \\
        --model_prefix ocar-alfworld \\
        --poll_interval 60

    # 一次性模式（处理所有待上传 checkpoint 后退出）
    python tools/checkpoint_uploader.py \\
        --checkpoint_dir outputs/checkpoints \\
        --hf_user Ricardo-H \\
        --model_prefix ocar-alfworld \\
        --poll_interval 0
"""

import argparse
import json
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for verl imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint Auto-Uploader")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="监控目录，检测 global_step_* checkpoints",
    )
    parser.add_argument(
        "--hf_user",
        type=str,
        default="Ricardo-H",
        help="HuggingFace 用户名或组织",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="ocar",
        help="模型命名前缀 (e.g. ocar-alfworld, ocar-webshop)",
    )
    parser.add_argument(
        "--date_prefix",
        type=str,
        default=None,
        help="日期前缀 (e.g. 0410)，默认当天 mmdd",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=60,
        help="轮询间隔（秒）。0 = 一次性模式",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="上传为私有仓库（默认公开）",
    )
    parser.add_argument(
        "--merge_dir",
        type=str,
        default=None,
        help="临时合并目录（默认在 checkpoint_dir 内）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery & readiness
# ---------------------------------------------------------------------------

def find_checkpoint_steps(checkpoint_dir: str) -> list:
    """Find all global_step_* directories and return sorted step numbers."""
    steps = []
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return steps

    for entry in checkpoint_path.iterdir():
        if entry.is_dir():
            match = re.match(r"global_step_(\d+)$", entry.name)
            if match:
                steps.append(int(match.group(1)))

    return sorted(steps)


def is_checkpoint_ready(checkpoint_dir: str, step: int) -> bool:
    """Check if a checkpoint is fully written and ready for merge."""
    actor_dir = Path(checkpoint_dir) / f"global_step_{step}" / "actor"
    if not actor_dir.exists():
        return False

    # Must have config.json (HF model config)
    if not (actor_dir / "config.json").exists():
        return False

    # Detect world_size from model shard files
    model_files = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_files:
        return False

    # Extract world_size and verify all ranks present
    match = re.match(r"model_world_size_(\d+)_rank_\d+\.pt", model_files[0].name)
    if not match:
        return False
    world_size = int(match.group(1))

    for rank in range(world_size):
        rank_file = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not rank_file.exists():
            return False

    return True


# ---------------------------------------------------------------------------
# Merge FSDP shards -> HuggingFace model
# ---------------------------------------------------------------------------

def merge_checkpoint(checkpoint_dir: str, step: int, target_dir: str) -> bool:
    """Merge FSDP sharded checkpoint into a single HuggingFace model directory."""
    import subprocess

    actor_dir = os.path.join(checkpoint_dir, f"global_step_{step}", "actor")
    merger_script = os.path.join(str(PROJECT_ROOT), "scripts", "model_merger.py")

    print(f"[Uploader] Merging FSDP shards for step {step} ...")
    print(f"[Uploader]   Source: {actor_dir}")
    print(f"[Uploader]   Target: {target_dir}")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable, merger_script, "merge",
                "--backend", "fsdp",
                "--local_dir", actor_dir,
                "--target_dir", target_dir,
            ],
            capture_output=True, text=True, timeout=600, env=env,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"[Uploader] ERROR merging step {step}:\n{result.stderr}")
            return False
        print(f"[Uploader] Merge completed for step {step}")
        return True
    except Exception as e:
        print(f"[Uploader] ERROR merging step {step}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# HuggingFace upload & collection
# ---------------------------------------------------------------------------

def upload_to_hf(target_dir: str, repo_id: str, private: bool = False) -> bool:
    """Upload a local model directory to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"[Uploader] Uploading to {repo_id} ...")
        api.upload_folder(
            folder_path=target_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[Uploader] Upload completed: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"[Uploader] ERROR uploading to {repo_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_collection(
    hf_user: str, collection_title: str, checkpoint_dir: str, private: bool = False
) -> Optional[str]:
    """Create or retrieve a HuggingFace Collection. Returns its slug."""
    from huggingface_hub import HfApi

    api = HfApi()

    slug_file = os.path.join(checkpoint_dir, ".collection_slug")
    if os.path.exists(slug_file):
        try:
            with open(slug_file) as f:
                saved_slug = f.read().strip()
            if saved_slug:
                try:
                    api.get_collection(saved_slug)
                    print(f"[Uploader] Reusing existing collection: {saved_slug}")
                    return saved_slug
                except Exception:
                    print(f"[Uploader] Saved slug '{saved_slug}' no longer valid, creating new collection")
        except IOError:
            pass

    try:
        collections = api.list_collections(owner=hf_user)
        for col in collections:
            if col.title == collection_title:
                print(f"[Uploader] Found existing collection by title: {col.slug}")
                _save_collection_slug(slug_file, col.slug)
                return col.slug
    except Exception as e:
        print(f"[Uploader] WARNING: Could not search collections: {e}")

    try:
        collection = api.create_collection(
            title=collection_title,
            namespace=hf_user,
            private=private,
            exists_ok=True,
        )
        print(f"[Uploader] Collection created: {collection.slug}")
        _save_collection_slug(slug_file, collection.slug)
        return collection.slug
    except Exception as e:
        print(f"[Uploader] WARNING: Could not create collection "
              f"'{collection_title}': {e}")
        return None


def _save_collection_slug(slug_file: str, slug: str):
    """Persist the collection slug to disk."""
    try:
        os.makedirs(os.path.dirname(slug_file), exist_ok=True)
        with open(slug_file, "w") as f:
            f.write(slug)
    except IOError:
        pass


def add_to_collection(collection_slug: str, repo_id: str) -> bool:
    """Add a model repo to a HuggingFace Collection."""
    from huggingface_hub import HfApi

    api = HfApi()

    try:
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        print(f"[Uploader] Added {repo_id} to collection {collection_slug}")
        return True
    except Exception as e:
        print(f"[Uploader] WARNING: Could not add {repo_id} to collection: {e}")
        return False


# ---------------------------------------------------------------------------
# End-to-end pipeline for a single checkpoint
# ---------------------------------------------------------------------------

def process_checkpoint(
    checkpoint_dir: str,
    step: int,
    hf_user: str,
    model_prefix: str,
    date_prefix: str,
    collection_slug: Optional[str],
    private: bool = False,
    merge_base_dir: Optional[str] = None,
) -> bool:
    """Merge -> Upload -> Add to Collection for one checkpoint."""

    model_name = f"{model_prefix}-{date_prefix}-step-{step}"
    repo_id = f"{hf_user}/{model_name}"

    base = merge_base_dir if merge_base_dir else checkpoint_dir
    merge_dir = os.path.join(base, f"_merged_step_{step}")
    os.makedirs(merge_dir, exist_ok=True)

    try:
        if not merge_checkpoint(checkpoint_dir, step, merge_dir):
            return False

        if not upload_to_hf(merge_dir, repo_id, private=private):
            return False

        if collection_slug:
            add_to_collection(collection_slug, repo_id)

        return True
    finally:
        if os.path.exists(merge_dir):
            print(f"[Uploader] Cleaning up temp dir: {merge_dir}")
            shutil.rmtree(merge_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

def load_processed_steps(checkpoint_dir: str) -> set:
    processed_file = os.path.join(checkpoint_dir, ".uploaded_steps")
    if os.path.exists(processed_file):
        try:
            with open(processed_file) as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return set()


def save_processed_steps(checkpoint_dir: str, steps: set):
    processed_file = os.path.join(checkpoint_dir, ".uploaded_steps")
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(processed_file, "w") as f:
            json.dump(sorted(steps), f)
    except IOError:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    date_prefix = args.date_prefix or datetime.now().strftime("%m%d")
    collection_title = f"{args.model_prefix}-{date_prefix}"

    print("=" * 60)
    print("Checkpoint Auto-Uploader")
    print(f"  Checkpoint Dir : {args.checkpoint_dir}")
    print(f"  HF User        : {args.hf_user}")
    print(f"  Model Prefix   : {args.model_prefix}")
    print(f"  Date Prefix    : {date_prefix}")
    print(f"  Collection     : {args.hf_user}/{collection_title}")
    print(f"  Model Pattern  : {args.model_prefix}-{date_prefix}-step-{{N}}")
    print(f"  Private        : {args.private}")
    print(f"  Poll Interval  : {args.poll_interval}s "
          f"{'(one-shot)' if args.poll_interval == 0 else '(continuous)'}")
    print("=" * 60)

    processed_steps = load_processed_steps(args.checkpoint_dir)
    if processed_steps:
        print(f"[Uploader] Resuming - already processed steps: "
              f"{sorted(processed_steps)}")

    collection_slug = ensure_collection(
        args.hf_user, collection_title,
        checkpoint_dir=args.checkpoint_dir,
        private=args.private,
    )

    running = True

    def signal_handler(signum, frame):
        nonlocal running
        print(f"\n[Uploader] Signal {signum} received, shutting down ...")
        running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    while running:
        all_steps = find_checkpoint_steps(args.checkpoint_dir)
        new_steps = [s for s in all_steps if s not in processed_steps]

        for step in new_steps:
            if not running:
                break

            if not is_checkpoint_ready(args.checkpoint_dir, step):
                print(f"[Uploader] Step {step} not ready yet, skipping ...")
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[Uploader] === Processing step {step}  ({ts}) ===")

            success = process_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                step=step,
                hf_user=args.hf_user,
                model_prefix=args.model_prefix,
                date_prefix=date_prefix,
                collection_slug=collection_slug,
                private=args.private,
                merge_base_dir=args.merge_dir,
            )

            if success:
                processed_steps.add(step)
                save_processed_steps(args.checkpoint_dir, processed_steps)
                print(f"[Uploader] Step {step} done")
            else:
                print(f"[Uploader] Step {step} failed, will retry next poll")

        if args.poll_interval == 0:
            break

        time.sleep(args.poll_interval)

    print("[Uploader] Exiting.")


if __name__ == "__main__":
    main()

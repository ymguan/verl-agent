"""
Merge FSDP DTensor sharded checkpoint into HuggingFace format and upload.

Usage:
    python ocar/merge_fsdp_to_hf.py \
        --ckpt_dir checkpoints/.../actor \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir /local_nvme/guanyiming/merged_models/grpo_observe_7b_step150 \
        [--push_to_hub Ricardo-H/grpo-observe-alfworld-7b-step150]
"""
import argparse
import os
import re
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_dtensor_shards(ckpt_dir: str) -> OrderedDict:
    shard_files = sorted(
        f for f in os.listdir(ckpt_dir)
        if re.match(r"model_world_size_\d+_rank_\d+\.pt", f)
    )
    m = re.search(r"world_size_(\d+)_rank_(\d+)", shard_files[0])
    world_size = int(m.group(1))
    print(f"Found {len(shard_files)} shards (world_size={world_size})")

    shards = []
    for f in shard_files:
        print(f"  Loading {f}...")
        s = torch.load(os.path.join(ckpt_dir, f), map_location="cpu", weights_only=False)
        converted = OrderedDict()
        for k, v in s.items():
            if hasattr(v, '_local_tensor'):
                converted[k] = v._local_tensor
            else:
                converted[k] = v
        shards.append(converted)
        del s

    merged = OrderedDict()
    for key in shards[0].keys():
        locals_ = [s[key] for s in shards]
        if all(t.shape == locals_[0].shape for t in locals_) and torch.equal(locals_[0], locals_[1]):
            merged[key] = locals_[0]
        else:
            merged[key] = torch.cat(locals_, dim=0)

    del shards
    print(f"Merged {len(merged)} parameters")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_hub", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Merging DTensor shards...")
    merged = merge_dtensor_shards(args.ckpt_dir)

    # Verify shapes match expected model
    print("Sample merged shapes:")
    for k in list(merged.keys())[:3] + list(merged.keys())[-2:]:
        print(f"  {k}: {merged[k].shape}")

    state_bf16 = {k: v.to(torch.bfloat16) for k, v in merged.items()}
    del merged

    print(f"Loading base model from {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    missing, unexpected = model.load_state_dict(state_bf16, strict=False)
    if missing:
        print(f"WARNING: Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    del state_bf16

    print(f"Saving to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing to {args.push_to_hub}...")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Done! https://huggingface.co/{args.push_to_hub}")
    else:
        print("Done! Use --push_to_hub to upload.")


if __name__ == "__main__":
    main()

"""Build ALFWorld canary probe set for data-contamination detection.

Strategy:
- original: real ALFWorld observations (verbatim templates).
- shuffled: same items, random order within each obs (breaks memorized order).
- swapped: items replaced with plausible-but-wrong ones from other rooms (breaks item-location co-occurrence).
- nonsense: same grammar, random objects from a fixed pool (breaks semantic prior).
- generic: matched-length generic English (wiki-style) control.

Metric: for a memorized model, NLL(original) << NLL(shuffled) ≈ NLL(swapped) ≈ NLL(nonsense).
For a non-memorized model, gaps should be small and explained only by natural item co-occurrence.
"""
import json
import random
import re
from pathlib import Path

random.seed(0)

OBS_FILE = Path("/tmp/alf_obs_samples.txt")
OUT_FILE = Path(__file__).parent / "probes.jsonl"

# Pool of common ALFWorld objects (used for swap / nonsense).
OBJECT_POOL = [
    "apple", "bread", "butterknife", "cd", "cellphone", "cloth", "creditcard",
    "cup", "dishsponge", "egg", "fork", "glassbottle", "kettle", "keychain",
    "knife", "laptop", "lettuce", "mug", "pan", "pen", "pencil", "pillow",
    "plate", "pot", "potato", "saltshaker", "soapbar", "soapbottle", "spatula",
    "spoon", "spraybottle", "statue", "tomato", "towel", "vase", "watch",
]

ITEM_RE = re.compile(r"a ([a-z]+) (\d+)")

# 20 generic English sentences (matched approximately to obs length in tokens).
GENERIC = [
    "In the library, you will find a book, a lamp, a desk, and a chair.",
    "The museum exhibit contains three paintings, two sculptures, and a bronze statue.",
    "On the table, he placed a cup, a plate, a spoon, and a napkin.",
    "She opened the drawer. The drawer was empty.",
    "The small garden had roses, tulips, daisies, and sunflowers blooming together.",
    "He walked into the kitchen. On the counter, he saw bread, butter, jam, and tea.",
    "Nothing unusual happened during the meeting that afternoon.",
    "The box was closed. Inside, there were letters, photographs, and old postcards.",
    "At the park, children played with balls, kites, frisbees, and jump ropes.",
    "She arrived at the station. The platform was crowded with passengers and luggage.",
    "The classroom contained desks, chairs, a whiteboard, and several bookshelves.",
    "He checked the fridge. Inside were milk, cheese, eggs, and some vegetables.",
    "The closet was open. In it, you see coats, shirts, and folded sweaters.",
    "On the windowsill, there were flowerpots, a small cactus, and a watering can.",
    "She walked to the desk. On the desk, she saw a laptop, a notebook, and a pen.",
    "The cupboard is closed. The cupboard has a wooden handle and brass hinges.",
    "He placed the keys, wallet, phone, and glasses on the entryway table.",
    "The bathroom had towels, soap, a toothbrush, shampoo, and a hair dryer.",
    "Inside the attic, you could see dust, cobwebs, old boxes, and forgotten toys.",
    "Nothing was left in the room except a single chair near the window.",
]


def parse_items(text: str):
    """Extract (item_name, idx) pairs from an ALFWorld observation."""
    return ITEM_RE.findall(text)


def shuffle_items(text: str) -> str:
    """Keep the template, shuffle the item order within each 'you see ...' list."""
    items = ITEM_RE.findall(text)
    if len(items) < 2:
        return text
    shuffled = items[:]
    random.shuffle(shuffled)
    # Replace items in-order
    out = text
    # Remove all item substrings first via a placeholder approach:
    parts = ITEM_RE.split(text)  # [pre, name1, idx1, mid, name2, idx2, ..., tail]
    rebuilt = [parts[0]]
    # parts[1::3], parts[2::3] are the original names/indices; replace with shuffled order
    for i, (name, idx) in enumerate(shuffled):
        rebuilt.append(f"a {name} {idx}")
        tail_idx = 3 + i * 3
        if tail_idx < len(parts):
            rebuilt.append(parts[tail_idx])
    out = "".join(rebuilt)
    return out


def swap_items(text: str) -> str:
    """Replace each item with a random object from the pool (keep indices)."""
    items = ITEM_RE.findall(text)
    if not items:
        return text
    used = set()
    parts = ITEM_RE.split(text)
    rebuilt = [parts[0]]
    for i, (name, idx) in enumerate(items):
        # pick a replacement different from original
        candidates = [o for o in OBJECT_POOL if o != name and o not in used]
        if not candidates:
            candidates = [o for o in OBJECT_POOL if o != name]
        repl = random.choice(candidates)
        used.add(repl)
        rebuilt.append(f"a {repl} {idx}")
        tail_idx = 3 + i * 3
        if tail_idx < len(parts):
            rebuilt.append(parts[tail_idx])
    return "".join(rebuilt)


def nonsense_items(text: str) -> str:
    """Fully randomize both item names and indices."""
    items = ITEM_RE.findall(text)
    if not items:
        return text
    parts = ITEM_RE.split(text)
    rebuilt = [parts[0]]
    for i, _ in enumerate(items):
        name = random.choice(OBJECT_POOL)
        idx = random.randint(1, 4)
        rebuilt.append(f"a {name} {idx}")
        tail_idx = 3 + i * 3
        if tail_idx < len(parts):
            rebuilt.append(parts[tail_idx])
    return "".join(rebuilt)


def main():
    obs_list = [ln.strip() for ln in OBS_FILE.read_text().splitlines() if ln.strip()]
    records = []
    for i, obs in enumerate(obs_list):
        records.append({"id": f"orig_{i}", "split": "original", "text": obs})
        records.append({"id": f"shuf_{i}", "split": "shuffled", "text": shuffle_items(obs)})
        records.append({"id": f"swap_{i}", "split": "swapped",  "text": swap_items(obs)})
        records.append({"id": f"nons_{i}", "split": "nonsense", "text": nonsense_items(obs)})
    for i, s in enumerate(GENERIC):
        records.append({"id": f"gen_{i}", "split": "generic", "text": s})

    OUT_FILE.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"Wrote {len(records)} probes to {OUT_FILE}")
    # Sanity print
    for split in ["original", "shuffled", "swapped", "nonsense", "generic"]:
        ex = next(r for r in records if r["split"] == split)
        print(f"  [{split:9s}] {ex['text']}")


if __name__ == "__main__":
    main()

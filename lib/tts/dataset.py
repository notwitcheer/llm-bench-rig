"""Parse the Seed-TTS-eval EN manifest (meta.lst): pipe-delimited
name|prompt_text|prompt_audio|target_text[|gt_audio]. EN lines have 4 fields (no ground-truth
audio). Relative audio paths are joined to base_dir. Pure stdlib."""
import os


def parse_manifest_line(line, base_dir):
    parts = line.rstrip("\n").split("|")
    name, ref_text, ref_audio, target_text = parts[0], parts[1], parts[2], parts[3]
    return {
        "name": name,
        "ref_text": ref_text,
        "ref_audio": os.path.join(base_dir, ref_audio),
        "target_text": target_text,
    }


def load_manifest(path, base_dir, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(parse_manifest_line(line, base_dir))
            if limit is not None and len(rows) >= limit:
                break
    return rows

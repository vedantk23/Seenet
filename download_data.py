

import os
import zipfile
import urllib.request
from tqdm import tqdm
import config

ZENODO_URL  = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
ZIP_NAME    = "Audio_Speech_Actors_01-24.zip"


class _Progress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_ravdess(data_dir: str = config.DATA_DIR) -> str:
    """Download and extract RAVDESS speech audio to data_dir."""
    os.makedirs(data_dir, exist_ok=True)

    # Already extracted?
    actors = [d for d in os.listdir(data_dir) if d.startswith("Actor_")]
    if len(actors) >= 24:
        print(f"✓ RAVDESS already present ({len(actors)} actors in {data_dir})")
        return data_dir

    zip_path = os.path.join(data_dir, ZIP_NAME)
    if not os.path.exists(zip_path):
        print(f"\n📥 Downloading RAVDESS (~215 MB) from Zenodo ...")
        with _Progress(unit='B', unit_scale=True, miniters=1, desc="RAVDESS") as t:
            urllib.request.urlretrieve(ZENODO_URL, zip_path, reporthook=t.update_to)
        print("✓ Download complete.")

    print(f"\n📦 Extracting {ZIP_NAME} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in tqdm(zf.namelist(), desc="Extracting"):
            zf.extract(member, data_dir)
    print("✓ Extraction complete.\n")
    return data_dir


def scan_ravdess(data_dir: str = config.DATA_DIR):
    """
    Walk data_dir and return list of sample dicts.
    Each dict: {wav_path, emotion_code, emotion_name, actor_id, session_id}

    Session assignment (paper Section IV-A):
      actors 1-4  → session 1
      actors 5-8  → session 2  ... actors 21-24 → session 6
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data dir not found: {data_dir}\n"
            "Run download_ravdess() or set DATA_DIR to your Kaggle dataset path."
        )

    samples = []
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) != 7:
                continue
            modality, vocal, emotion_code = parts[0], parts[1], parts[2]
            if modality != "03" or vocal != "01":   # speech only
                continue
            actor_id   = int(parts[6])              # 1-24
            session_id = (actor_id - 1) // 4 + 1   # 1-6
            samples.append({
                "wav_path":     os.path.join(root, fname),
                "emotion_code": emotion_code,
                "emotion_name": config.RAVDESS_EMOTIONS.get(emotion_code, "?"),
                "actor_id":     actor_id,
                "session_id":   session_id,
            })

    print(f"✓ Found {len(samples)} speech samples across "
          f"{len(set(s['session_id'] for s in samples))} sessions.")
    return samples


if __name__ == "__main__":
    download_ravdess()
    samples = scan_ravdess()
    from collections import Counter
    dist = Counter(s["emotion_name"] for s in samples)
    print("\nEmotion distribution:")
    for emo, cnt in sorted(dist.items()):
        print(f"  {emo:12s}: {cnt}")

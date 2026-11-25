"""
Script utilitario para descargar los pesos preentrenados oficiales de TR-DETR
(entrenados sobre QVHighlights con características CLIP) desde el registro oficial
de Zenodo (https://zenodo.org/records/13960580, licencia MIT / Apache 2.0).
"""

import os
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm


ZENODO_RECORD_ID = "13960580"
CHECKPOINT_FILENAME = "clip_tr_detr_qvhighlight.ckpt"
ZENODO_DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{CHECKPOINT_FILENAME}?download=1"
)


def download_pretrained_weights(target_dir: Optional[str] = None) -> str:
    """
    Descarga el archivo `clip_tr_detr_qvhighlight.ckpt` en la carpeta `weights/`.
    """
    if target_dir is None:
        weights_dir = Path(__file__).resolve().parent / "weights"
    else:
        weights_dir = Path(target_dir)

    weights_dir.mkdir(parents=True, exist_ok=True)
    dest_path = weights_dir / CHECKPOINT_FILENAME

    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        print(f"[OK] El archivo de pesos ya existe en: {dest_path}")
        return str(dest_path)

    print(f"[INFO] Descargando {CHECKPOINT_FILENAME} desde Zenodo ({ZENODO_DOWNLOAD_URL})...")
    response = requests.get(ZENODO_DOWNLOAD_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=CHECKPOINT_FILENAME,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                written = f.write(chunk)
                bar.update(written)

    print(f"[OK] Descarga completada exitosamente en: {dest_path}")
    return str(dest_path)


if __name__ == "__main__":
    from typing import Optional
    download_pretrained_weights()

"""
Descarga y verificación de los pesos preentrenados oficiales de TR-DETR.

Los checkpoints están publicados por los autores en Zenodo (registro 13960580,
licencia CC-BY-4.0) y fueron entrenados sobre QVHighlights con características CLIP.

Cada archivo se comprueba contra el tamaño y el MD5 publicados por Zenodo, de modo
que un archivo corrupto o una descarga incompleta se detectan antes de intentar
cargar el modelo.

Uso:
    python download_weights.py                  # descarga la variante que usa la app (CLIP)
    python download_weights.py --check          # sólo verifica el archivo que ya está en disco
    python download_weights.py --feature clip_slowfast
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "13960580"

# nombre del archivo -> (tamaño en bytes, MD5 publicado por Zenodo)
CHECKPOINTS: Dict[str, Tuple[str, int, str]] = {
    "clip": ("clip_tr_detr_qvhighlight.ckpt", 92695215, "941148ee2fd53584ea3c4f4ca9e0761b"),
    "clip_slowfast": (
        "clip_slowfast_tr_detr_qvhighlight.ckpt",
        99828463,
        "6907c776a48c40551965b76b6d95e2c2",
    ),
    "clip_slowfast_pann": (
        "clip_slowfast_pann_tr_detr_qvhighlight.ckpt",
        106169199,
        "5dbc294288c32f28a9fed799509c0397",
    ),
}


def md5_archivo(path: Path) -> str:
    """Calcula el MD5 de un archivo leyéndolo por bloques (no lo carga en memoria)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for bloque in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def url_zenodo(filename: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{filename}?download=1"


def ruta_destino(target_dir: Optional[str], filename: str) -> Path:
    base = Path(target_dir) if target_dir else Path(__file__).resolve().parent / "weights"
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


def verificar_pesos(path: Path, filename: str, size: int, md5: str) -> Tuple[bool, str]:
    """
    Comprueba que `path` corresponde al checkpoint esperado.

    Devuelve (True, mensaje) si coincide en tamaño y MD5, o (False, motivo).
    """
    if not path.exists():
        return False, f"no existe {path}"
    real_size = path.stat().st_size
    if real_size != size:
        return False, f"tamaño {real_size} bytes, se esperaban {size} (descarga incompleta)"
    real_md5 = md5_archivo(path)
    if real_md5 != md5:
        return False, f"MD5 {real_md5} no coincide con el publicado ({md5})"
    return True, f"MD5 correcto ({md5})"


def download_pretrained_weights(target_dir: Optional[str] = None, feature: str = "clip") -> str:
    """
    Descarga el checkpoint oficial de TR-DETR en `weights/` y lo verifica por MD5.

    Parámetros:
        target_dir: carpeta destino (por defecto `weights/` junto a este script).
        feature:    variante de características -> 'clip', 'clip_slowfast' o 'clip_slowfast_pann'.

    Devuelve la ruta local del checkpoint verificado.
    """
    if feature not in CHECKPOINTS:
        raise ValueError(f"Variante desconocida: {feature}. Opciones: {', '.join(CHECKPOINTS)}")

    filename, size, md5 = CHECKPOINTS[feature]
    dest_path = ruta_destino(target_dir, filename)
    url = url_zenodo(filename)

    if dest_path.exists():
        ok, detalle = verificar_pesos(dest_path, filename, size, md5)
        if ok:
            print(f"[OK] {filename} ya está descargado y verificado ({size / 1e6:.1f} MB, {detalle}).")
            return str(dest_path)
        print(f"[AVISO] {filename} presente pero inconsistente: {detalle}. Se descarga de nuevo.")

    print(f"[INFO] Descargando {filename} (~{size / 1e6:.1f} MB) desde Zenodo...")
    print(f"       {url}")
    respuesta = requests.get(url, stream=True, timeout=120)
    respuesta.raise_for_status()

    total = int(respuesta.headers.get("content-length", 0)) or size
    parcial = dest_path.with_suffix(dest_path.suffix + ".part")
    with open(parcial, "wb") as handle, tqdm(
        desc=filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as barra:
        for bloque in respuesta.iter_content(chunk_size=1024 * 256):
            if bloque:
                barra.update(handle.write(bloque))

    ok, detalle = verificar_pesos(parcial, filename, size, md5)
    if not ok:
        parcial.unlink(missing_ok=True)
        raise RuntimeError(f"Descarga descartada: {detalle}. Intente de nuevo.")

    os.replace(parcial, dest_path)
    print(f"[OK] Descarga verificada: {dest_path} ({size / 1e6:.1f} MB, {detalle}).")
    return str(dest_path)


def verificar_sin_descargar(target_dir: Optional[str] = None, feature: str = "clip") -> bool:
    """Verifica el checkpoint local sin descargar nada. Devuelve True si es válido."""
    filename, size, md5 = CHECKPOINTS[feature]
    dest_path = ruta_destino(target_dir, filename)
    ok, detalle = verificar_pesos(dest_path, filename, size, md5)
    estado = "OK" if ok else "FALLA"
    print(f"[{estado}] {dest_path}: {detalle}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga los pesos oficiales de TR-DETR desde Zenodo")
    parser.add_argument(
        "--feature",
        default="clip",
        choices=list(CHECKPOINTS),
        help="Variante de características del checkpoint (por defecto 'clip', la que usa la app).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Sólo verifica el archivo local (tamaño y MD5); no descarga.",
    )
    args = parser.parse_args()

    if args.check:
        return 0 if verificar_sin_descargar(feature=args.feature) else 1

    download_pretrained_weights(feature=args.feature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

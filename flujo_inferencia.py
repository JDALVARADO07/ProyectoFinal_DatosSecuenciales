"""
Flujo de inferencia de TR-DETR con los pesos oficiales: `encode_video -> predict`.

Muestra, en el orden en que se expone:

  1. El checkpoint que se va a usar (tamaño y MD5 verificados contra Zenodo).
  2. La arquitectura que contiene, leída del propio archivo.
  3. La carga del modelo y su coste en tiempo.
  4. `encode_video`: el video se convierte UNA vez en una secuencia de features CLIP.
  5. `predict`: cada consulta se responde reutilizando esa codificación.
  6. Un resumen de cifras para la exposición.

Uso:
    python flujo_inferencia.py
    python flujo_inferencia.py --video sample_data/demo_video_eventos.mp4
    python flujo_inferencia.py --consultas "A man opens the door of a car" "Car driving on the road"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence

from download_weights import CHECKPOINTS, verificar_pesos

BASE_DIR = Path(__file__).resolve().parent
CKPT = BASE_DIR / "weights" / "clip_tr_detr_qvhighlight.ckpt"
VIDEO_REAL = BASE_DIR / "sample_data" / "demo_real_30s.mp4"

CONSULTAS_POR_DEFECTO = [
    "Car driving on the road",
    "A man opens the door of a car",
    "An old man walking in the park",
]


def cabecera(texto: str) -> None:
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def titulo(texto: str) -> None:
    print()
    print(texto)
    print("-" * 78)


def describir_checkpoint(ckpt: Path) -> bool:
    """Comprueba el checkpoint y muestra la arquitectura que declara el propio archivo."""
    titulo("1. CHECKPOINT")
    if not ckpt.exists():
        print(f"[FALTA] {ckpt}")
        print("        Descárguelo con: python download_weights.py")
        return False

    filename, size, md5 = CHECKPOINTS["clip"]
    ok, detalle = verificar_pesos(ckpt, filename, size, md5)
    print(f"archivo   : weights/{ckpt.name}")
    print(f"tamaño    : {ckpt.stat().st_size / 1e6:.1f} MB")
    print(f"integridad: {'OK' if ok else 'FALLA'} — {detalle}")
    if not ok:
        print("          Vuelva a descargarlo: python download_weights.py")
        return False

    import torch

    crudo = torch.load(ckpt, map_location="cpu", weights_only=False)
    opt = crudo.get("opt", {})
    print(
        f"arquitectura: d_model={opt.get('hidden_dim')} · cabezas={opt.get('nheads')} · "
        f"encoder={opt.get('enc_layers')} · decoder={opt.get('dec_layers')} · "
        f"queries={opt.get('num_queries')} · clip={opt.get('clip_length')} s"
    )
    modulos = sorted({clave.split(".")[0] for clave in crudo.get("model", {})})
    print(f"módulos   : {', '.join(modulos)}")
    return True


def cargar_motor(ckpt: Path) -> "tuple[object, float]":
    """Construye el motor del proyecto y comprueba que Lighthouse quedó activo."""
    titulo("2. CARGA DEL MODELO")
    from inference import TRDETRInferenceEngine

    t0 = time.perf_counter()
    engine = TRDETRInferenceEngine(ckpt_path=str(ckpt))
    t_carga = time.perf_counter() - t0

    predictor = engine.lighthouse_predictor
    if predictor is None:
        print("[ERROR] Lighthouse no está disponible; la app usaría la reimplementación didáctica.")
        print("        Instale el motor oficial:")
        print("          pip install -r requirements-lighthouse.txt")
        print("          pip install --no-deps git+https://github.com/line/lighthouse.git")
        raise SystemExit(1)

    n_params = sum(p.numel() for p in predictor._model.parameters())
    print(f"clase     : {type(predictor).__name__} (lighthouse)")
    print(f"estado    : {engine.engine_status}")
    print(f"parámetros: {n_params:,}".replace(",", "."))
    print(f"tiempo    : {t_carga:.1f} s (incluye cargar CLIP ViT-B/32 para las features)")
    return engine, t_carga


def codificar_video(predictor: "object", video: Path) -> "object":
    """`encode_video`: una sola vez por video."""
    titulo("3. ENTRADA — encode_video (una vez por video)")
    t0 = time.perf_counter()
    encoded = predictor.encode_video(str(video))
    t_encode = time.perf_counter() - t0

    feats = encoded["video_feats"]
    n_clips = feats.shape[1]
    clip_len = predictor._clip_len
    print(f"video     : {video.name} — {n_clips} clips de {clip_len:.0f} s = {n_clips * clip_len:.0f} s")
    print(f"features  : {tuple(feats.shape)}  (512 de CLIP + 2 de timestamp por clip)")
    print(f"tiempo    : {t_encode:.1f} s  ({t_encode / n_clips * 1000:.0f} ms por clip)")
    print("clave     : estas features NO dependen de la consulta, por eso se reutilizan.")
    return encoded


def responder_consultas(
    predictor: "object", encoded: "object", consultas: Sequence[str]
) -> List[float]:
    """`predict`: una consulta a la vez sobre la codificación ya calculada."""
    titulo("4. CONSULTA Y SALIDA — predict (reutiliza la codificación)")
    # Un primer llamado descartado: la primera inferencia incluye inicialización de
    # buffers y no representa el coste real por consulta.
    predictor.predict(consultas[0], encoded)

    tiempos: List[float] = []
    primera = True
    for i, consulta in enumerate(consultas, start=1):
        t0 = time.perf_counter()
        salida = predictor.predict(consulta, encoded)
        t_pred = time.perf_counter() - t0
        tiempos.append(t_pred)

        inicio, fin, confianza = salida["pred_relevant_windows"][0]
        n_sal = len(salida["pred_saliency_scores"])
        print(f"[{i}] \"{consulta}\"")
        print(
            f"     Top-1 [{inicio:.2f} s, {fin:.2f} s] · confianza {confianza:.3f} · "
            f"{n_sal} scores (uno por clip) · {t_pred * 1000:.0f} ms"
        )
        if primera:
            tokens, mascara = predictor._encode_text(consulta)
            print(
                f"     texto: {tuple(tokens.shape)} features + máscara con "
                f"{int(mascara.sum().item())} tokens reales"
            )
            print(f"     salidas: {sorted(salida.keys())}")
            primera = False
    return tiempos


def resumen(engine: "object", video: Path, t_carga: float, tiempos: Sequence[float]) -> None:
    predictor = engine.lighthouse_predictor
    titulo("5. RESUMEN")
    n_params = sum(p.numel() for p in predictor._model.parameters())
    print(f"· Pesos oficiales verificados por MD5 y cargados en {t_carga:.1f} s.")
    print(f"· {n_params / 1e6:.2f} M parámetros, d=256, 2+2 capas, 10 moment queries.")
    print(f"· `encode_video` se hace una sola vez por video ({video.name}).")
    media = sum(tiempos) / len(tiempos) if tiempos else 0.0
    print(f"· `predict` tarda {media * 1000:.0f} ms por consulta (media de {len(tiempos)}).")
    print("· Devuelve ventanas (inicio, fin, confianza) y un score de relevancia por clip.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Flujo encode_video -> predict con los pesos oficiales")
    parser.add_argument("--video", default=str(VIDEO_REAL), help="Ruta del video de entrada")
    parser.add_argument("--ckpt", default=str(CKPT), help="Ruta del checkpoint oficial")
    parser.add_argument(
        "--consultas",
        nargs="+",
        default=None,
        help="Una o más consultas en lenguaje natural",
    )
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    video = Path(args.video)
    consultas = args.consultas or CONSULTAS_POR_DEFECTO

    cabecera("FLUJO DE INFERENCIA — TR-DETR (AAAI 2024) CON PESOS OFICIALES")
    if not video.exists():
        print(f"[FALTA] No existe el video {video}")
        return 1

    if not describir_checkpoint(ckpt):
        return 1

    engine, t_carga = cargar_motor(ckpt)

    encoded = codificar_video(engine.lighthouse_predictor, video)
    tiempos = responder_consultas(engine.lighthouse_predictor, encoded, consultas)
    resumen(engine, video, t_carga, tiempos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

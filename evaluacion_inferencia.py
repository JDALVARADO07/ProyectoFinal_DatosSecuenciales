"""
Evaluación del modelo con los pesos oficiales sobre el video de demostración.

Mide tres cosas, en el orden en que se exponen:

  1. Localización (Moment Retrieval): IoU del intervalo Top-1 contra el intervalo
     anotado, y los aciertos R1@0.5 y R1@0.7 (mismas definiciones que el artículo).
  2. Detección de highlights: contraste entre la relevancia media de los clips que
     están dentro del intervalo real y la de los que quedan fuera.
  3. Control negativo: consultas que NO aparecen en el video. Se comparan su
     confianza top-1, su entropía normalizada y su desviación típica frente a las
     consultas que sí corresponden, para comprobar si la curva se aplana.

Incluye además la tabla de latencias (`encode_video` una vez, `predict` por consulta).

Uso:
    python evaluacion_inferencia.py
    python evaluacion_inferencia.py --json resultados/evaluacion.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
CKPT = BASE_DIR / "weights" / "clip_tr_detr_qvhighlight.ckpt"
VIDEO = BASE_DIR / "sample_data" / "demo_real_30s.mp4"

# Intervalos anotados a mano sobre demo_real_30s.mp4 (tres escenas de 10 s).
CASOS_POSITIVOS: List[Tuple[str, Tuple[float, float]]] = [
    ("Car driving on the road", (0.0, 10.0)),
    ("A man opens the door of a car", (10.0, 20.0)),
    ("An old man walking in the park", (20.0, 30.0)),
]

# Consultas que no describen nada del video (control negativo).
CASOS_NEGATIVOS: List[str] = [
    "A penguin skating on a frozen lake",
    "A cat playing the piano in a jazz bar",
    "The stock market crashes on a trading floor",
]


@dataclass
class Medida:
    """Resultado de una consulta."""

    consulta: str
    etiqueta: str
    gt: Optional[Tuple[float, float]]
    top1: Tuple[float, float]
    confianza: float
    iou: Optional[float]
    r1_05: Optional[bool]
    r1_07: Optional[bool]
    latencia_ms: float
    curva: Dict[str, float] = field(default_factory=dict)
    contraste: Optional[Dict[str, float]] = None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersección sobre unión de dos intervalos [inicio, fin]."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def normalizar(valores: Sequence[float]) -> List[float]:
    """Lleva una secuencia a [0, 1] (los saliency scores del modelo son logits)."""
    if not valores:
        return []
    lo, hi = min(valores), max(valores)
    if hi - lo < 1e-9:
        return [0.5] * len(valores)
    return [(v - lo) / (hi - lo) for v in valores]


def softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    tope = max(logits)
    exps = [math.exp(v - tope) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def estadisticos_curva(logits: Sequence[float]) -> Dict[str, float]:
    """Resume la forma de la curva: dispersión, pico y entropía normalizada."""
    n = len(logits)
    if n == 0:
        return {}
    media = sum(logits) / n
    varianza = sum((v - media) ** 2 for v in logits) / n
    probs = softmax(logits)
    entropia = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(n)
    return {
        "min": min(logits),
        "max": max(logits),
        "std": math.sqrt(varianza),
        "rango": max(logits) - min(logits),
        "p_max": max(probs),
        "entropia": entropia,
    }


def contraste_dentro_fuera(
    saliencia: Sequence[float], gt: Tuple[float, float], clip_len: float
) -> Dict[str, float]:
    """Relevancia media dentro del intervalo anotado frente a fuera de él."""
    dentro, fuera = [], []
    for i, valor in enumerate(normalizar(saliencia)):
        centro = (i + 0.5) * clip_len
        (dentro if gt[0] <= centro <= gt[1] else fuera).append(valor)
    media_dentro = sum(dentro) / len(dentro) if dentro else 0.0
    media_fuera = sum(fuera) / len(fuera) if fuera else 0.0
    return {
        "dentro": media_dentro,
        "fuera": media_fuera,
        "contraste": media_dentro - media_fuera,
    }


def evaluar(
    predictor: "object", encoded: "object", consulta: str, gt: Optional[Tuple[float, float]], etiqueta: str, clip_len: float
) -> Medida:
    t0 = time.perf_counter()
    salida = predictor.predict(consulta, encoded)
    latencia = (time.perf_counter() - t0) * 1000

    inicio, fin, confianza = salida["pred_relevant_windows"][0]
    saliencia = salida["pred_saliency_scores"]
    top1 = (float(inicio), float(fin))
    valor_iou = iou(top1, gt) if gt else None

    return Medida(
        consulta=consulta,
        etiqueta=etiqueta,
        gt=gt,
        top1=top1,
        confianza=float(confianza),
        iou=valor_iou,
        r1_05=(valor_iou >= 0.5) if valor_iou is not None else None,
        r1_07=(valor_iou >= 0.7) if valor_iou is not None else None,
        latencia_ms=latencia,
        curva=estadisticos_curva(saliencia),
        contraste=contraste_dentro_fuera(saliencia, gt, clip_len) if gt else None,
    )


def tabla(cabeceras: Sequence[str], filas: Sequence[Sequence[str]], anchos: Sequence[int]) -> None:
    print("  " + " ".join(h.ljust(a) for h, a in zip(cabeceras, anchos)))
    print("  " + " ".join("-" * a for a in anchos))
    for fila in filas:
        print("  " + " ".join(str(v).ljust(a) for v, a in zip(fila, anchos)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación del modelo con los pesos oficiales")
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--json", default=None, help="Ruta opcional para guardar los resultados")
    args = parser.parse_args()

    from inference import TRDETRInferenceEngine

    video = Path(args.video)
    if not video.exists():
        print(f"[FALTA] No existe el video {video}")
        return 1

    print("=" * 78)
    print("EVALUACIÓN DEL MODELO — INTERVALO, HIGHLIGHTS Y CONTROL NEGATIVO")
    print("=" * 78)

    print("\n1. CONFIGURACIÓN")
    t0 = time.perf_counter()
    try:
        engine = TRDETRInferenceEngine(ckpt_path=args.ckpt)
    except Exception as exc:  # el motor oficial es obligatorio: informa el motivo real
        print(f"  [ERROR] No se pudo cargar el motor oficial: {exc}")
        return 1
    t_carga = time.perf_counter() - t0
    predictor = engine.predictor
    clip_len = float(predictor._clip_len)
    print(f"  modelo    : {type(predictor).__name__} — {engine.engine_status}")
    print(f"  carga     : {t_carga:.1f} s")
    print(f"  video     : {video.name} — clips de {clip_len:.0f} s")
    print(f"  casos     : {len(CASOS_POSITIVOS)} con intervalo anotado, {len(CASOS_NEGATIVOS)} sin coincidencia")

    t0 = time.perf_counter()
    encoded = predictor.encode_video(str(video))
    t_encode = time.perf_counter() - t0
    n_clips = encoded["video_feats"].shape[1]
    print(f"  codificado: {n_clips} clips en {t_encode:.1f} s ({t_encode / n_clips * 1000:.0f} ms por clip)")

    # Un predict descartado para medir latencias en régimen (el primero inicializa buffers).
    predictor.predict(CASOS_POSITIVOS[0][0], encoded)

    positivos = [evaluar(predictor, encoded, q, gt, "presente", clip_len) for q, gt in CASOS_POSITIVOS]
    negativos = [evaluar(predictor, encoded, q, None, "ausente", clip_len) for q in CASOS_NEGATIVOS]

    print("\n2. INTERVALO REAL VS. PREDICHO (Moment Retrieval)")
    filas = []
    for m in positivos:
        filas.append(
            [
                f"[{m.gt[0]:.0f}, {m.gt[1]:.0f}] s",
                f"[{m.top1[0]:.2f}, {m.top1[1]:.2f}] s",
                f"{m.iou:.3f}",
                "sí" if m.r1_05 else "no",
                "sí" if m.r1_07 else "no",
                f"{m.confianza:.3f}",
                f"{m.latencia_ms:.0f} ms",
                m.consulta,
            ]
        )
    tabla(["GT", "Top-1", "IoU", "R1@0.5", "R1@0.7", "Conf.", "Latencia", "Consulta"], filas, [13, 18, 6, 7, 7, 7, 9, 40])

    iou_medio = sum(m.iou for m in positivos) / len(positivos)
    aciertos_05 = sum(1 for m in positivos if m.r1_05)
    aciertos_07 = sum(1 for m in positivos if m.r1_07)
    print(f"\n  IoU medio: {iou_medio:.3f} · R1@0.5: {aciertos_05}/{len(positivos)} · R1@0.7: {aciertos_07}/{len(positivos)}")

    print("\n3. HIGHLIGHT DETECTION — RELEVANCIA DENTRO VS. FUERA DEL INTERVALO")
    filas = []
    for m in positivos:
        c = m.contraste
        filas.append([f"{c['dentro']:.3f}", f"{c['fuera']:.3f}", f"{c['contraste']:+.3f}", m.consulta])
    tabla(["Dentro", "Fuera", "Contraste", "Consulta"], filas, [8, 8, 10, 45])
    contraste_medio = sum(m.contraste["contraste"] for m in positivos) / len(positivos)
    print(f"\n  Contraste medio dentro-fuera: {contraste_medio:+.3f}")

    print("\n4. CONTROL NEGATIVO — CONSULTAS QUE NO APARECEN EN EL VIDEO")
    filas = []
    for m in positivos + negativos:
        filas.append(
            [
                m.etiqueta,
                f"{m.confianza:.3f}",
                f"{m.curva['p_max']:.3f}",
                f"{m.curva['entropia']:.3f}",
                f"{m.curva['std']:.3f}",
                m.consulta,
            ]
        )
    tabla(["Tipo", "Confianza", "p_max", "Entropía", "std logits", "Consulta"], filas, [9, 10, 7, 10, 10, 42])

    conf_pos = sum(m.confianza for m in positivos) / len(positivos)
    conf_neg = sum(m.confianza for m in negativos) / len(negativos)
    ent_pos = sum(m.curva["entropia"] for m in positivos) / len(positivos)
    ent_neg = sum(m.curva["entropia"] for m in negativos) / len(negativos)
    print(f"\n  Confianza media : presentes {conf_pos:.3f} vs. ausentes {conf_neg:.3f} (brecha {conf_pos - conf_neg:+.3f})")
    print(f"  Entropía media  : presentes {ent_pos:.3f} vs. ausentes {ent_neg:.3f} (1.0 = curva plana)")

    aplana = (ent_neg - ent_pos) > 0.05 and (conf_pos - conf_neg) > 0.1
    print(
        "\n  LECTURA: "
        + (
            "las consultas ausentes SÍ producen una curva más plana y menor confianza."
            if aplana
            else "las consultas ausentes NO aplanan la curva de forma concluyente: el modelo "
            "siempre devuelve un intervalo con confianza alta. No existe una clase "
            "\"sin momento\", así que la confianza no es una probabilidad calibrada de "
            "que la frase aparezca en el video."
        )
    )

    print("\n5. LATENCIAS")
    latencias = [m.latencia_ms for m in positivos + negativos]
    print(f"  encode_video : {t_encode:.1f} s total ({t_encode / n_clips * 1000:.0f} ms por clip), una sola vez")
    print(f"  predict      : mín {min(latencias):.0f} ms · media {sum(latencias) / len(latencias):.0f} ms · máx {max(latencias):.0f} ms")
    print(f"  consultas    : {len(latencias)} sobre una única codificación")

    if args.json:
        destino = Path(args.json)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(
                {
                    "video": video.name,
                    "clips": n_clips,
                    "clip_len": clip_len,
                    "carga_s": round(t_carga, 2),
                    "encode_s": round(t_encode, 2),
                    "iou_medio": round(iou_medio, 4),
                    "r1_05": f"{aciertos_05}/{len(positivos)}",
                    "r1_07": f"{aciertos_07}/{len(positivos)}",
                    "contraste_medio": round(contraste_medio, 4),
                    "confianza_presentes": round(conf_pos, 4),
                    "confianza_ausentes": round(conf_neg, 4),
                    "entropia_presentes": round(ent_pos, 4),
                    "entropia_ausentes": round(ent_neg, 4),
                    "medidas": [asdict(m) for m in positivos + negativos],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n  Resultados guardados en {destino}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

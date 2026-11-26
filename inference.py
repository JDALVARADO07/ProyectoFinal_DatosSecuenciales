"""
Pipeline de inferencia para TR-DETR (AAAI 2024) sobre el modelo oficial de Lighthouse.

Todo lo que devuelve este módulo sale del checkpoint preentrenado de QVHighlights
(clip_tr_detr_qvhighlight.ckpt) ejecutado con `lighthouse.models.TRDETRPredictor`:
  - intervalos (pred_relevant_windows) y curva de relevancia (pred_saliency_scores);
  - mapas de atención reales, capturados con forward hooks sobre los módulos de
    atención del modelo durante `predictor.predict()`:
      * fusión clip←palabra : transformer.t2v_encoder.layers[i].self_attn   (L x N)
      * self-attention       : transformer.encoder.layers[i].self_attn       (L x L)
      * cross-attention      : transformer.decoder.layers[i].cross_attn      (N_q x L)
"""

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def get_ffmpeg_binary() -> str:
    """Localiza el ejecutable de FFmpeg en el entorno virtual o en el sistema."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        venv_ffmpeg = Path(__file__).resolve().parents[3] / ".venv" / "Scripts" / "ffmpeg.exe"
        if venv_ffmpeg.exists():
            return str(venv_ffmpeg)
        return "ffmpeg"


def _prepare_lighthouse_windows() -> None:
    """
    Adapta Lighthouse para Windows antes de usarlo:
    1. ffmpeg-python busca un ejecutable `ffmpeg` en el PATH; creamos un shim
       a partir del binario que ya trae imageio-ffmpeg.
    2. El loader de Lighthouse usa `ffprobe`, que no viene con imageio-ffmpeg.
       Sustituimos `_video_info` por una lectura de metadatos con OpenCV o,
       si no está disponible, con imageio-ffmpeg.
    Si Lighthouse no está instalado, la excepción de importación se propaga.
    """
    # Evita que transformers/msclap carguen TensorFlow (más rápido y silencioso).
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg

        shim_dir = Path(tempfile.gettempdir()) / "tr_detr_ffmpeg"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_exe = shim_dir / "ffmpeg.exe"
        if not shim_exe.exists():
            try:
                os.link(imageio_ffmpeg.get_ffmpeg_exe(), shim_exe)
            except OSError:
                shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), shim_exe)
        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")

    from lighthouse.frame_loaders.base_loader import BaseLoader

    if getattr(BaseLoader, "_tr_detr_windows_patch", False):
        return

    def _video_info(self, video_path: str):
        try:
            import cv2
        except ImportError:
            cv2 = None

        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                duration = n_frames / fps if fps > 0 and n_frames > 0 else 0.0
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap.release()
                if duration <= 0:
                    duration = get_video_duration(str(video_path))
                return {
                    "duration": float(duration),
                    "frames_length": n_frames,
                    "fps": max(1, int(round(fps))),
                    "height": height,
                    "width": width,
                }

        import imageio_ffmpeg

        gen = imageio_ffmpeg.read_frames(str(video_path))
        meta = next(gen)
        with contextlib.suppress(Exception):
            gen.close()
        fps = float(meta.get("fps") or 30.0)
        duration = float(meta.get("duration") or 0.0)
        width, height = meta.get("size") or (224, 224)
        if duration <= 0:
            duration = get_video_duration(str(video_path))
        return {
            "duration": duration,
            "frames_length": int(duration * fps),
            "fps": max(1, int(round(fps))),
            "height": int(height),
            "width": int(width),
        }

    BaseLoader._video_info = _video_info
    BaseLoader._tr_detr_windows_patch = True


def get_video_duration(video_path: str) -> float:
    """Obtiene la duración total de un archivo de video en segundos usando FFmpeg."""
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [ffmpeg_bin, "-i", str(video_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = proc.stderr
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", output)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return max(2.0, hours * 3600 + minutes * 60 + seconds)
    return 30.0


def generate_demo_video(output_path: str) -> str:
    """
    Genera un video MP4 sintético de 30 segundos con FFmpeg, con 3 escenas de 10 s
    de colores distintos. Útil solo para comprobar que el pipeline corre de punta a punta.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 1000:
        return str(out_file)

    ffmpeg_bin = get_ffmpeg_binary()
    # Creamos 3 segmentos de 10 segundos con patrones visuales distintos y los concatenamos
    filter_complex = (
        "color=c=0x1e3a8a:s=320x240:d=10,drawbox=x=40+t*15:y=90:w=80:h=50:color=0x93c5fd@0.9:t=fill[v0];"
        "color=c=0x7c2d12:s=320x240:d=10,drawbox=x=110:y=60:w=100:h=120:color=0xfacc15@0.95:t=fill[v1];"
        "color=c=0x14532d:s=320x240:d=10,drawbox=x=220-t*12:y=80:w=60:h=90:color=0x86efac@0.9:t=fill[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[outv]"
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-nostdin",
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "15",
        str(out_file),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return str(out_file)


def clip_query_tokens(query: str) -> List[str]:
    """
    Tokens BPE reales que CLIP (y por tanto TR-DETR) recibe para la consulta,
    incluyendo los marcadores de inicio y fin de secuencia.
    """
    import clip
    from clip.simple_tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    ids = clip.tokenize(query)[0].tolist()
    tokens = []
    for tok_id in ids:
        if tok_id == 0:
            break
        piece = tokenizer.decoder[tok_id]
        if piece == "<|startoftext|>":
            tokens.append("<sos>")
        elif piece == "<|endoftext|>":
            tokens.append("<eos>")
        else:
            tokens.append(piece.replace("</w>", ""))
    return tokens


class AttentionRecorder:
    """
    Registra forward hooks sobre los módulos de atención de TR-DETR (Lighthouse) y guarda
    las matrices de pesos (promedio de las cabezas) de cada capa. Los hooks se retiran al
    salir del bloque `with`, sin tocar el código de Lighthouse.
    """

    def __init__(self, model: torch.nn.Module):
        transformer = model.transformer
        self.blocks = {
            "fusion": [layer.self_attn for layer in transformer.t2v_encoder.layers],
            "encoder": [layer.self_attn for layer in transformer.encoder.layers],
            "decoder_cross": [layer.cross_attn for layer in transformer.decoder.layers],
        }
        self.records: Dict[str, List[List[torch.Tensor]]] = {
            name: [[] for _ in modules] for name, modules in self.blocks.items()
        }
        self.model_outputs: Optional[Dict[str, torch.Tensor]] = None
        self._handles: List[Any] = []
        self._model = model

    @staticmethod
    def _force_weights(module, args, kwargs):
        # Garantiza que el módulo devuelva los pesos promediados sobre las cabezas.
        kwargs["need_weights"] = True
        if isinstance(module, torch.nn.MultiheadAttention):
            kwargs["average_attn_weights"] = True
        return args, kwargs

    def _make_hook(self, block: str, layer_idx: int):
        def hook(module, args, output):
            weights = output[1]
            if weights is None:
                raise RuntimeError(f"{block}[{layer_idx}] no devolvió pesos de atención.")
            self.records[block][layer_idx].append(weights.detach().cpu())
        return hook

    def _capture_outputs(self, module, args, output):
        self.model_outputs = {k: v.detach().cpu() for k, v in output.items() if torch.is_tensor(v)}

    def __enter__(self) -> "AttentionRecorder":
        for block, modules in self.blocks.items():
            for idx, module in enumerate(modules):
                self._handles.append(
                    module.register_forward_pre_hook(self._force_weights, with_kwargs=True)
                )
                self._handles.append(module.register_forward_hook(self._make_hook(block, idx)))
        self._handles.append(self._model.register_forward_hook(self._capture_outputs))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class TRDETRInferenceEngine:
    """
    Motor de inferencia de TR-DETR sobre el modelo oficial de Lighthouse.
    Procesa un video y una consulta en lenguaje natural y devuelve intervalos temporales,
    curva de relevancia por clip y los mapas de atención internos del modelo.
    Si Lighthouse o el checkpoint no están disponibles, lanza la excepción original.
    """

    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if ckpt_path is None:
            ckpt_path = str(Path(__file__).resolve().parent / "weights" / "clip_tr_detr_qvhighlight.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No se encontró el checkpoint de TR-DETR en '{ckpt_path}'. "
                "Ejecute `python download_weights.py`."
            )
        self.ckpt_path = ckpt_path

        _prepare_lighthouse_windows()
        from lighthouse.models import TRDETRPredictor

        self.predictor = TRDETRPredictor(self.ckpt_path, device=self.device, feature_name="clip")
        self.clip_duration = float(self.predictor._clip_len)
        self.engine_status = "Modelo TR-DETR oficial — pesos QVHighlights + CLIP ViT-B/32 (Lighthouse)"

    @torch.no_grad()
    def predict(self, video_path: str, query: str) -> Dict[str, Any]:
        """Ejecuta la inferencia completa sobre `video_path` para la consulta `query`."""
        video = self.predictor.encode_video(video_path)
        num_clips = int(video["video_feats"].shape[1])
        tokens = clip_query_tokens(query)
        num_tokens = len(tokens)

        with AttentionRecorder(self.predictor._model) as recorder:
            prediction = self.predictor.predict(query, video)
        if prediction is None:
            raise RuntimeError("Lighthouse no devolvió predicciones para este video.")

        # --- Mapas de atención: (1, Q, K) por llamada -> matrices recortadas a L clips y N tokens
        def layer_maps(block: str, call: int, rows: int, cols: int) -> List[np.ndarray]:
            maps = []
            for idx, calls in enumerate(recorder.records[block]):
                weights = calls[call][0]
                if weights.shape[0] < rows or weights.shape[1] < cols:
                    raise RuntimeError(
                        f"{block}[{idx}]: forma {tuple(weights.shape)} menor que la esperada ({rows}, {cols})."
                    )
                maps.append(weights[:rows, :cols].numpy())
            return maps

        num_queries = int(recorder.records["decoder_cross"][0][0].shape[1])
        attention = {
            # Q = clips, K/V = 77 posiciones de texto de CLIP; solo las N primeras son tokens reales.
            "fusion": layer_maps("fusion", 0, num_clips, num_tokens),
            # El encoder se ejecuta dos veces: tras la fusión y dentro de HD2MR (memoria del decoder).
            "encoder": layer_maps("encoder", -1, num_clips, num_clips),
            "encoder_first_pass": layer_maps("encoder", 0, num_clips, num_clips),
            "decoder_cross": layer_maps("decoder_cross", 0, num_queries, num_clips),
        }

        # --- Momentos: Lighthouse los entrega ordenados; recuperamos qué moment query generó cada uno
        outputs = recorder.model_outputs
        query_scores = torch.softmax(outputs["pred_logits"], -1)[0, :, 0]
        query_order = sorted(range(num_queries), key=lambda q: float(query_scores[q]), reverse=True)

        duration = num_clips * self.clip_duration
        ranked_moments: List[Dict[str, Any]] = []
        for rank, (st, ed, score) in enumerate(prediction["pred_relevant_windows"]):
            ranked_moments.append({
                "query_id": query_order[rank] + 1,
                "start_sec": round(float(st), 2),
                "end_sec": round(float(ed), 2),
                "duration_sec": round(float(ed - st), 2),
                "confidence": round(float(score), 4),
                "center_norm": round(float((st + ed) / (2.0 * duration)), 4),
                "width_norm": round(float((ed - st) / duration), 4),
            })

        # --- Curva de relevancia (HD) del modelo, reescalada a [0, 1] solo para visualizar
        saliency_raw = np.array(prediction["pred_saliency_scores"], dtype=np.float32)
        if len(saliency_raw) != num_clips:
            raise RuntimeError(
                f"La curva de relevancia tiene {len(saliency_raw)} valores y el video {num_clips} clips."
            )
        lo, hi = float(saliency_raw.min()), float(saliency_raw.max())
        saliency_final = (saliency_raw - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(saliency_raw)

        return {
            "engine_status": self.engine_status,
            "video_path": str(video_path),
            "video_duration_sec": round(duration, 2),
            "clip_duration_sec": self.clip_duration,
            "num_clips": num_clips,
            "query": query,
            "tokens": tokens,
            "num_tokens": num_tokens,
            "num_queries": num_queries,
            "clip_timestamps": [round((i + 0.5) * self.clip_duration, 2) for i in range(num_clips)],
            "saliency_raw": [round(float(x), 4) for x in saliency_raw],
            "saliency_final": [round(float(x), 4) for x in saliency_final],
            "ranked_moments": ranked_moments,
            "attention": attention,
            "attention_shapes": {
                block: [tuple(m.shape) for m in maps] for block, maps in attention.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia con TR-DETR: Moment Retrieval & Highlight Detection"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Ruta al archivo de video (.mp4). Si se omite, genera y usa el video de demostración.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="A man opens the door of a car",
        help="Consulta en lenguaje natural para localizar el momento en el video.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Ruta opcional al archivo clip_tr_detr_qvhighlight.ckpt.",
    )
    args = parser.parse_args()

    video_path = args.video
    if not video_path or not os.path.exists(video_path):
        demo_path = Path(__file__).resolve().parent / "sample_data" / "demo_video_eventos.mp4"
        print(f"[INFO] Generando video de prueba en: {demo_path}")
        video_path = generate_demo_video(str(demo_path))

    engine = TRDETRInferenceEngine(ckpt_path=args.ckpt if args.ckpt else None)
    result = engine.predict(video_path=video_path, query=args.query)

    print("\n" + "=" * 72)
    print("RESULTADOS DE INFERENCIA TR-DETR (Transformer Encoder-Decoder)")
    print("=" * 72)
    print(f"Motor activo       : {result['engine_status']}")
    print(f"Video analizado    : {result['video_path']}")
    print(f"Duración total     : {result['video_duration_sec']} s ({result['num_clips']} clips de {result['clip_duration_sec']} s)")
    print(f"Consulta (Query)   : {result['query']}")
    print(f"Tokens CLIP        : {result['tokens']}")
    print("-" * 72)
    print("MAPAS DE ATENCIÓN (promedio de cabezas, por capa):")
    for block, shapes in result["attention_shapes"].items():
        print(f"  - {block:<20}: {shapes}")
    print("-" * 72)
    print("TOP 3 MOMENTOS RECUPERADOS (Moment Retrieval):")
    for rank, m in enumerate(result["ranked_moments"][:3], start=1):
        print(
            f"  #{rank} [Query #{m['query_id']:02d}] -> Intervalo: [{m['start_sec']:05.2f}s - {m['end_sec']:05.2f}s] "
            f"| Duración: {m['duration_sec']:.2f}s | Confianza: {m['confidence']*100:.1f}%"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()

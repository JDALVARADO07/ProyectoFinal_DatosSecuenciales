"""
Interfaz Gráfica Interactiva en Streamlit para la Sustentación del Proyecto Final:
"TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection"
(Sun et al., AAAI 2024).

Ejecución local:
    streamlit run app.py
"""

import html
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st

from inference import TRDETRInferenceEngine, generate_demo_video
from slides import render_slides


st.set_page_config(
    page_title="TR-DETR · Moment Retrieval & Highlight Detection",
    page_icon=None,
    layout="wide",
)


def render_saliency_svg(
    timestamps: List[float],
    saliency_final: List[float],
    top_moment: Dict[str, Any],
    duration: float,
) -> str:
    width, height = 860, 260
    pad_l, pad_r, pad_t, pad_b = 55, 25, 25, 42
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_coord(t: float) -> float:
        return pad_l + (t / max(1.0, duration)) * plot_w

    def y_coord(s: float) -> float:
        return pad_t + (1.0 - max(0.0, min(1.0, s))) * plot_h

    m_start_x = x_coord(top_moment["start_sec"])
    m_end_x = x_coord(top_moment["end_sec"])
    span_w = max(4.0, m_end_x - m_start_x)

    pts_final = " ".join(f"{x_coord(t):.1f},{y_coord(s):.1f}" for t, s in zip(timestamps, saliency_final))

    area_pts = (
        f"{x_coord(timestamps[0]):.1f},{pad_t + plot_h:.1f} "
        + pts_final
        + f" {x_coord(timestamps[-1]):.1f},{pad_t + plot_h:.1f}"
    )

    circles = "".join(
        f'<circle cx="{x_coord(t):.1f}" cy="{y_coord(s):.1f}" r="4" fill="#2563eb">'
        f"<title>Clip #{idx+1} ({t}s): Relevancia MR2HD = {s:.3f}</title></circle>"
        for idx, (t, s) in enumerate(zip(timestamps, saliency_final))
    )

    grid_lines = ""
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = y_coord(val)
        grid_lines += (
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            f'stroke="#e2e8f0" stroke-dasharray="3,3" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{val:.2f}</text>'
        )

    time_ticks = ""
    num_ticks = min(10, len(timestamps))
    for i in range(num_ticks + 1):
        t_val = (i / max(1, num_ticks)) * duration
        tx = x_coord(t_val)
        time_ticks += (
            f'<line x1="{tx:.1f}" y1="{pad_t + plot_h}" x2="{tx:.1f}" y2="{pad_t + plot_h + 5}" stroke="#64748b"/>'
            f'<text x="{tx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-size="11" fill="#475569">{t_val:.1f}s</text>'
        )

    svg = f"""
    <svg viewBox="0 0 {width} {height}" width="100%" style="background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;padding:6px;">
        {grid_lines}
        <rect x="{m_start_x:.1f}" y="{pad_t}" width="{span_w:.1f}" height="{plot_h}"
              fill="#fef08a" fill-opacity="0.55" stroke="#eab308" stroke-width="1.5" stroke-dasharray="4,2"/>
        <text x="{m_start_x + span_w/2:.1f}" y="{pad_t + 14}" text-anchor="middle" font-size="11" font-weight="bold" fill="#854d0e">
            Top-1 Momento [{top_moment['start_sec']}s - {top_moment['end_sec']}s]
        </text>
        <polygon points="{area_pts}" fill="#93c5fd" fill-opacity="0.28"/>
        <polyline fill="none" stroke="#1d4ed8" stroke-width="3" points="{pts_final}"/>
        {circles}
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#334155" stroke-width="1.5"/>
        <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="#334155" stroke-width="1.5"/>
        {time_ticks}
        <text x="{width/2:.1f}" y="{height - 5}" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e293b">
            Tiempo del Video en Segundos (Clips de 2.0 s)
        </text>
    </svg>
    """
    return svg


def render_heatmap_html(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    subtitle: str,
) -> str:
    m_min = float(matrix.min())
    m_max = float(matrix.max())
    denom = max(1e-6, m_max - m_min)

    header_cells = "".join(
        f'<th style="padding:4px 6px;font-size:11px;background:#f1f5f9;color:#1e293b;border:1px solid #cbd5e1;">{html.escape(c)}</th>'
        for c in col_labels
    )
    rows_html = ""
    for r_idx, r_label in enumerate(row_labels):
        cells = f'<td style="padding:4px 8px;font-size:11px;font-weight:600;background:#f8fafc;color:#0f172a;border:1px solid #cbd5e1;white-space:nowrap;">{html.escape(r_label)}</td>'
        for c_idx in range(len(col_labels)):
            val = float(matrix[r_idx, c_idx])
            norm_v = (val - m_min) / denom
            alpha = 0.10 + 0.85 * norm_v
            text_color = "#ffffff" if norm_v > 0.58 else "#0f172a"
            cells += (
                f'<td style="padding:4px 6px;font-size:11px;text-align:center;'
                f'background:rgba(37, 99, 235, {alpha:.2f});color:{text_color};border:1px solid #e2e8f0;">'
                f"{val:.2f}</td>"
            )
        rows_html += f"<tr>{cells}</tr>"

    return f"""
    <div style="margin-top:8px;margin-bottom:14px;">
        <div style="font-weight:700;font-size:14px;color:#0f172a;">{title}</div>
        <div style="font-size:12px;color:#475569;margin-bottom:6px;">{subtitle}</div>
        <div style="overflow-x:auto;">
            <table style="border-collapse:collapse;width:100%;">
                <thead>
                    <tr>
                        <th style="padding:4px 8px;font-size:11px;background:#e2e8f0;color:#0f172a;border:1px solid #cbd5e1;">Q \\ K</th>
                        {header_cells}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """


@st.cache_resource
def load_engine(ckpt_path: str) -> TRDETRInferenceEngine:
    return TRDETRInferenceEngine(ckpt_path=ckpt_path or None)


def layer_selector(label: str, num_layers: int, key: str) -> int:
    """Selector de capa; por defecto la última del bloque."""
    return st.selectbox(
        label,
        options=list(range(num_layers)),
        index=num_layers - 1,
        format_func=lambda i: f"Capa {i + 1} de {num_layers}" + (" (última)" if i == num_layers - 1 else ""),
        key=key,
    )


def main():
    st.title("TR-DETR: Localización de Momentos y Detección de Highlights en Video")
    st.caption("Sun et al., AAAI 2024 — arXiv:2401.02309")

    base_dir = Path(__file__).resolve().parent
    default_ckpt = base_dir / "weights" / "clip_tr_detr_qvhighlight.ckpt"

    top_k = 5

    with st.sidebar:
        st.header("Configuración")
        ckpt_input = st.text_input(
            "Pesos preentrenados (.ckpt)",
            value=str(default_ckpt) if default_ckpt.exists() else "",
        )

        try:
            engine = load_engine(ckpt_input)
        except Exception as exc:
            engine = None
            st.error(f"No se pudo cargar el modelo oficial TR-DETR (Lighthouse): {type(exc).__name__}: {exc}")
        else:
            st.info(engine.engine_status)

        st.markdown("---")
        st.subheader("Video de entrada")
        uploaded = st.file_uploader("Subir video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi", "mkv"])

    active_video_path = None
    if uploaded is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "tr_detr_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        saved_path = tmp_dir / uploaded.name
        with open(saved_path, "wb") as f:
            f.write(uploaded.getbuffer())
        active_video_path = str(saved_path)

    tab_pres, tab_inf, tab_qkv = st.tabs([
        "Presentación",
        "Inferencia — Moment Retrieval & Highlight Detection",
        "Tensores Q, K, V y Mapas de Atención",
    ])

    with tab_pres:
        render_slides()

    with tab_inf:
        result = None
        if engine is None:
            st.error("La inferencia no está disponible: el modelo oficial no se pudo cargar (ver barra lateral).")
        elif active_video_path is None or not os.path.exists(active_video_path):
            st.info("Suba un video en la barra lateral para iniciar la inferencia.")
        else:
            col_q, col_preset = st.columns([3, 2])
            with col_preset:
                preset_query = st.selectbox(
                    "Consultas de ejemplo:",
                    options=[
                        "A man opens the door of a car",
                        "Car driving on the road",
                        "An old man walking in the park",
                        "A person is speaking in front of the camera",
                    ],
                )
            with col_q:
                query_text = st.text_input(
                    "Consulta en lenguaje natural (español o inglés):",
                    value=preset_query,
                )
            try:
                result = engine.predict(video_path=active_video_path, query=query_text)
            except Exception as exc:
                st.session_state.pop("_last_result", None)
                st.error(f"Falló la inferencia con el modelo oficial: {type(exc).__name__}: {exc}")
            else:
                st.session_state["_last_result"] = result

        if result is not None:
            c_vid, c_res = st.columns([1.15, 1.35])

            with c_vid:
                st.subheader("Reproductor")
                selected_rank = st.selectbox(
                    "Intervalo predicho:",
                    options=list(range(min(top_k, len(result["ranked_moments"])))),
                    format_func=lambda idx: (
                        f"#{idx+1} [{result['ranked_moments'][idx]['start_sec']}s - "
                        f"{result['ranked_moments'][idx]['end_sec']}s] "
                        f"({result['ranked_moments'][idx]['confidence']*100:.1f}%)"
                    ),
                )
                chosen_moment = result["ranked_moments"][selected_rank]
                st.video(active_video_path, start_time=int(chosen_moment["start_sec"]))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Inicio", f"{chosen_moment['start_sec']} s")
                m2.metric("Fin", f"{chosen_moment['end_sec']} s")
                m3.metric("Duración", f"{chosen_moment['duration_sec']} s")
                m4.metric("Confianza", f"{chosen_moment['confidence']*100:.1f} %")

            with c_res:
                st.subheader("Relevancia por clip")
                svg_chart = render_saliency_svg(
                    timestamps=result["clip_timestamps"],
                    saliency_final=result["saliency_final"],
                    top_moment=chosen_moment,
                    duration=result["video_duration_sec"],
                )
                st.markdown(svg_chart, unsafe_allow_html=True)
                st.caption(
                    "Relevancia por clip (pred_saliency_scores del modelo oficial), "
                    "reescalada a [0, 1] para visualizar."
                )

                st.subheader(f"Top-{top_k} propuestas del Decoder")
                table_rows = []
                for idx, m in enumerate(result["ranked_moments"][:top_k], start=1):
                    table_rows.append({
                        "Rank": f"#{idx}",
                        "Query": f"#{m['query_id']:02d}",
                        "Inicio (s)": m["start_sec"],
                        "Fin (s)": m["end_sec"],
                        "Duración (s)": m["duration_sec"],
                        "Centro norm.": m["center_norm"],
                        "Ancho norm.": m["width_norm"],
                        "Confianza": f"{m['confidence']*100:.1f}%",
                    })
                st.dataframe(table_rows, use_container_width=True, hide_index=True)

    with tab_qkv:
        qkv_result = st.session_state.get("_last_result")
        if qkv_result is None:
            st.info("Suba un video y ejecute la inferencia para ver los mapas de atención.")
        else:
            st.caption(
                "Matrices de atención reales del modelo oficial TR-DETR (Lighthouse), capturadas con "
                "forward hooks durante la predicción. Cada mapa es el promedio de las 8 cabezas."
            )
            attn = qkv_result["attention"]
            clip_len = qkv_result["clip_duration_sec"]
            clip_labels = [
                f"C{i+1} ({i*clip_len:g}-{(i+1)*clip_len:g}s)" for i in range(qkv_result["num_clips"])
            ]
            short_clip_labels = [f"C{i+1}" for i in range(qkv_result["num_clips"])]
            rank_of_query = {m["query_id"]: r for r, m in enumerate(qkv_result["ranked_moments"], start=1)}
            query_labels = [
                f"Query #{q:02d}" + (f" (top-{rank_of_query[q]})" if q in rank_of_query else "")
                for q in range(1, qkv_result["num_queries"] + 1)
            ]

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                li = layer_selector("Capa de fusión (t2v_encoder)", len(attn["fusion"]), "layer_fusion")
                fusion_html = render_heatmap_html(
                    matrix=attn["fusion"][li],
                    row_labels=clip_labels,
                    col_labels=qkv_result["tokens"],
                    title="Fusión clip ← palabra — softmax(Q_clips @ K_palabrasᵀ / √32)",
                    subtitle="Filas: clips de video (Q) | Columnas: tokens CLIP de la consulta (K/V)",
                )
                st.markdown(fusion_html, unsafe_allow_html=True)

            with col_h2:
                li = layer_selector("Capa del decoder", len(attn["decoder_cross"]), "layer_decoder")
                dec_html = render_heatmap_html(
                    matrix=attn["decoder_cross"][li],
                    row_labels=query_labels,
                    col_labels=short_clip_labels,
                    title="Decoder Cross-Attention — softmax(Q_dec @ K_encᵀ / √64)",
                    subtitle="Filas: Moment Queries (Q) | Columnas: memoria de clips del Encoder (K/V)",
                )
                st.markdown(dec_html, unsafe_allow_html=True)

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                li = layer_selector("Capa del encoder", len(attn["encoder"]), "layer_encoder")
            with col_e2:
                enc_pass = st.radio(
                    "Pasada del encoder",
                    options=["encoder", "encoder_first_pass"],
                    format_func=lambda k: {
                        "encoder": "HD2MR (memoria que usa el decoder)",
                        "encoder_first_pass": "Primera pasada (tras la fusión)",
                    }[k],
                    horizontal=True,
                )
            enc_html = render_heatmap_html(
                matrix=attn[enc_pass][li],
                row_labels=clip_labels,
                col_labels=short_clip_labels,
                title="Encoder Self-Attention — softmax(Q_clips @ K_clipsᵀ / √32)",
                subtitle="Filas: clips (Q) | Columnas: clips (K/V)",
            )
            st.markdown(enc_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

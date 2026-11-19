"""
Interfaz Gráfica Interactiva en Streamlit para la Sustentación del Proyecto Final:
"TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection"
(Sun et al., AAAI 2024).

Ejecución local:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st

from inference import TRDETRInferenceEngine, generate_demo_video


st.set_page_config(
    page_title="TR-DETR · Moment Retrieval & Highlight Detection",
    page_icon="🎬",
    layout="wide",
)


def render_saliency_svg(
    timestamps: List[float],
    saliency_init: List[float],
    saliency_final: List[float],
    top_moment: Dict[str, Any],
    duration: float,
) -> str:
    """
    Genera un gráfico vectorial SVG interactivo que muestra:
    - La franja sombreada del intervalo Top-1 recuperado por el Decoder [t_inicio, t_fin].
    - La curva preliminar de relevancia por clip (HD2MR, línea punteada).
    - La curva final refinada por reciprocidad (MR2HD con GRU, línea continua).
    """
    width, height = 860, 260
    pad_l, pad_r, pad_t, pad_b = 55, 25, 25, 42
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_coord(t: float) -> float:
        return pad_l + (t / max(1.0, duration)) * plot_w

    def y_coord(s: float) -> float:
        return pad_t + (1.0 - max(0.0, min(1.0, s))) * plot_h

    # Coordenadas de la franja del intervalo Top-1
    m_start_x = x_coord(top_moment["start_sec"])
    m_end_x = x_coord(top_moment["end_sec"])
    span_w = max(4.0, m_end_x - m_start_x)

    pts_init = " ".join(f"{x_coord(t):.1f},{y_coord(s):.1f}" for t, s in zip(timestamps, saliency_init))
    pts_final = " ".join(f"{x_coord(t):.1f},{y_coord(s):.1f}" for t, s in zip(timestamps, saliency_final))

    # Polígono bajo la curva final
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
        <!-- Franja del intervalo predicho Top-1 -->
        <rect x="{m_start_x:.1f}" y="{pad_t}" width="{span_w:.1f}" height="{plot_h}"
              fill="#fef08a" fill-opacity="0.55" stroke="#eab308" stroke-width="1.5" stroke-dasharray="4,2"/>
        <text x="{m_start_x + span_w/2:.1f}" y="{pad_t + 14}" text-anchor="middle" font-size="11" font-weight="bold" fill="#854d0e">
            Top-1 Momento [{top_moment['start_sec']}s - {top_moment['end_sec']}s]
        </text>
        <!-- Área y curvas de relevancia -->
        <polygon points="{area_pts}" fill="#93c5fd" fill-opacity="0.28"/>
        <polyline fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,4" points="{pts_init}"/>
        <polyline fill="none" stroke="#1d4ed8" stroke-width="3" points="{pts_final}"/>
        {circles}
        <!-- Ejes -->
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
    """
    Renderiza una matriz de atención Softmax(Q @ K^T / sqrt(d_k)) como tabla HTML coloreada
    para inspeccionar con precisión los pesos de atención en la sustentación.
    """
    m_min = float(matrix.min())
    m_max = float(matrix.max())
    denom = max(1e-6, m_max - m_min)

    header_cells = "".join(
        f'<th style="padding:4px 6px;font-size:11px;background:#f1f5f9;color:#1e293b;border:1px solid #cbd5e1;">{c}</th>'
        for c in col_labels
    )
    rows_html = ""
    for r_idx, r_label in enumerate(row_labels):
        cells = f'<td style="padding:4px 8px;font-size:11px;font-weight:600;background:#f8fafc;color:#0f172a;border:1px solid #cbd5e1;white-space:nowrap;">{r_label}</td>'
        for c_idx in range(len(col_labels)):
            val = float(matrix[r_idx, c_idx])
            norm_v = (val - m_min) / denom
            # Escala de color azul suave a azul intenso
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
def load_engine(ckpt_path: str, clip_duration: float) -> TRDETRInferenceEngine:
    return TRDETRInferenceEngine(
        ckpt_path=ckpt_path if ckpt_path and os.path.exists(ckpt_path) else None,
        clip_duration=clip_duration,
    )


def main():
    st.title("🎬 TR-DETR: Localización de Momentos y Detección de Highlights en Video")
    st.caption(
        "**Proyecto Final · Procesamiento de Datos Secuenciales con Deep Learning** | "
        "Arquitectura Transformer Encoder–Decoder basada en *Sun et al., AAAI 2024 (arXiv:2401.02309)*"
    )

    base_dir = Path(__file__).resolve().parent
    default_ckpt = base_dir / "weights" / "clip_tr_detr_qvhighlight.ckpt"
    demo_video_path = base_dir / "sample_data" / "demo_video_eventos.mp4"

    # Barra lateral de configuración
    with st.sidebar:
        st.header("⚙️ Configuración del Modelo")
        clip_duration = st.slider(
            "Duración por clip de video (Δt seg)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="En QVHighlights cada clip dura 2.0 segundos.",
        )
        top_k = st.slider("Número de propuestas a mostrar (Top-K)", 1, 10, 5)
        ckpt_input = st.text_input(
            "Ruta a pesos preentrenados (.ckpt)",
            value=str(default_ckpt) if default_ckpt.exists() else "",
            help="Archivo clip_tr_detr_qvhighlight.ckpt descargado de Zenodo (registro 13960580).",
        )

        engine = load_engine(ckpt_input, clip_duration)
        st.info(f"**Estado del motor:**\n{engine.engine_status}")

        st.markdown("---")
        st.subheader("📹 Fuente de Video")
        video_source = st.radio(
            "Seleccione el video de entrada:",
            options=[
                "Video de demostración integrado (3 escenas)",
                "Subir archivo de video propio (.mp4, .mov, .avi)",
            ],
        )

    # Selección o carga del archivo de video
    active_video_path = None
    if video_source.startswith("Video de demostración"):
        active_video_path = generate_demo_video(str(demo_video_path))
        st.sidebar.success(
            "**Escenas en el video de demostración (30 s):**\n"
            "- `[00s - 10s]`: Auto conduciendo en la vía\n"
            "- `[10s - 20s]`: El hombre abre la puerta del carro\n"
            "- `[20s - 30s]`: Persona caminando en el parque"
        )
    else:
        uploaded = st.file_uploader(
            "Suba un archivo de video para ejecutar la inferencia multimodal",
            type=["mp4", "mov", "avi", "mkv"],
        )
        if uploaded is not None:
            tmp_dir = Path(tempfile.gettempdir()) / "tr_detr_uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            saved_path = tmp_dir / uploaded.name
            with open(saved_path, "wb") as f:
                f.write(uploaded.getbuffer())
            active_video_path = str(saved_path)

    # Entrada de consulta en lenguaje natural
    col_q, col_preset = st.columns([3, 2])
    with col_preset:
        preset_query = st.selectbox(
            "Consultas de ejemplo rápido:",
            options=[
                "El hombre abre la puerta del carro",
                "The man opens the car door",
                "Auto conduciendo en la carretera",
                "Persona caminando en el parque",
            ],
        )
    with col_q:
        query_text = st.text_input(
            "🔎 Consulta en lenguaje natural (Español o Inglés):",
            value=preset_query,
        )

    if not active_video_path or not os.path.exists(active_video_path):
        st.warning("Por favor seleccione o suba un archivo de video para iniciar la inferencia.")
        return

    # Ejecución de la inferencia en el Transformer Encoder-Decoder
    result = engine.predict(video_path=active_video_path, query=query_text)
    top_moment = result["ranked_moments"][0]

    tab_inf, tab_qkv, tab_theory = st.tabs([
        "🎯 1. Inferencia: Moment Retrieval & Highlight Detection",
        "🧠 2. Inspección en Vivo de Tensores Q, K, V y Cross-Attention",
        "📖 3. Guía de Arquitectura para la Sustentación",
    ])

    # =========================================================================
    # PESTAÑA 1: RESULTADOS DE INFERENCIA Y REPRODUCCIÓN DE INTERVALOS
    # =========================================================================
    with tab_inf:
        c_vid, c_res = st.columns([1.15, 1.35])

        with c_vid:
            st.subheader("Reproductor de Video sincronizado al Momento")
            selected_rank = st.selectbox(
                "Saltar al intervalo predicho:",
                options=list(range(min(top_k, len(result["ranked_moments"])))),
                format_func=lambda idx: (
                    f"Top #{idx+1} (Query #{result['ranked_moments'][idx]['query_id']}) -> "
                    f"[{result['ranked_moments'][idx]['start_sec']}s - {result['ranked_moments'][idx]['end_sec']}s] "
                    f"(Confianza: {result['ranked_moments'][idx]['confidence']*100:.1f}%)"
                ),
            )
            chosen_moment = result["ranked_moments"][selected_rank]
            st.video(active_video_path, start_time=int(chosen_moment["start_sec"]))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Inicio predicho", f"{chosen_moment['start_sec']} s")
            m2.metric("Fin predicho", f"{chosen_moment['end_sec']} s")
            m3.metric("Duración", f"{chosen_moment['duration_sec']} s")
            m4.metric("Confianza", f"{chosen_moment['confidence']*100:.1f} %")

        with c_res:
            st.subheader("Curva de Relevancia por Clip (Highlight Detection)")
            st.caption(
                "Línea continua azul: puntaje final $S_{\\text{final}}$ refinado por **MR2HD (GRU)**. "
                "Línea punteada gris: puntaje inicial $S_{\\text{init}}$ de **HD2MR**. "
                "Franja amarilla: intervalo Top-1 localizado por el Decoder."
            )
            svg_chart = render_saliency_svg(
                timestamps=result["clip_timestamps"],
                saliency_init=result["saliency_init"],
                saliency_final=result["saliency_final"],
                top_moment=chosen_moment,
                duration=result["video_duration_sec"],
            )
            st.markdown(svg_chart, unsafe_allow_html=True)

            st.subheader(f"Top-{top_k} Propuestas de Momentos del Decoder ($N_q = 10$)")
            table_rows = []
            for idx, m in enumerate(result["ranked_moments"][:top_k], start=1):
                table_rows.append({
                    "Ranking": f"#{idx}",
                    "Moment Query": f"Query #{m['query_id']:02d}",
                    "Inicio (s)": m["start_sec"],
                    "Fin (s)": m["end_sec"],
                    "Duración (s)": m["duration_sec"],
                    "Centro norm. (ĉ)": m["center_norm"],
                    "Ancho norm. (ŵ)": m["width_norm"],
                    "Confianza": f"{m['confidence']*100:.1f}%",
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # =========================================================================
    # PESTAÑA 2: INSPECCIÓN DE TENSORES Q, K, V Y MAPAS DE ATENCIÓN
    # =========================================================================
    with tab_qkv:
        st.subheader("Flujo de Dimensiones y Creación de Tensores $Q$, $K$ y $V$ en esta Inferencia")
        shapes = result["tensor_shapes"]
        norms = result["tensor_norms"]

        qkv_summary = [
            {
                "Etapa de la Arquitectura": "1. Entradas Multimodales (CLIP)",
                "Quién es Query (Q)": f"Video Clips F_v: {shapes['visual_input_Fv']}",
                "Quién es Key (K)": f"Tokens Texto F_t: {shapes['text_input_Ft']}",
                "Quién es Value (V)": f"Tokens Texto F_t: {shapes['text_input_Ft']}",
                "Función en TR-DETR": "Extracción de features visuales cada 2s y embeddings de tokens.",
            },
            {
                "Etapa de la Arquitectura": "2. Cross-Attention Multimodal (V2T)",
                "Quién es Query (Q)": f"Clips alineados Q_v: {shapes['v2t_Q']} -> 8 cabezas {shapes['v2t_Q_heads']} (||Q||={norms['v2t_Q_l2']})",
                "Quién es Key (K)": f"Tokens consulta K_t: {shapes['v2t_K']} (||K||={norms['v2t_K_l2']})",
                "Quién es Value (V)": f"Tokens consulta V_t: {shapes['v2t_V']} (||V||={norms['v2t_V_l2']})",
                "Función en TR-DETR": "Cada clip pregunta qué palabras de la consulta coinciden con su contenido visual.",
            },
            {
                "Etapa de la Arquitectura": "3. Transformer Encoder (Self-Attention)",
                "Quién es Query (Q)": f"Secuencia multimodal Z_mm: {shapes['encoder_Q']}",
                "Quién es Key (K)": f"Secuencia multimodal Z_mm: {shapes['encoder_K']}",
                "Quién es Value (V)": f"Secuencia multimodal Z_mm: {shapes['encoder_V']}",
                "Función en TR-DETR": "Contextualiza cada clip respecto a los clips vecinos del video.",
            },
            {
                "Etapa de la Arquitectura": "4. Transformer Decoder (Cross-Attention)",
                "Quién es Query (Q)": f"10 Moment Queries Q_dec: {shapes['decoder_cross_Q']} -> 8 cabezas {shapes['decoder_cross_Q_heads']} (||Q||={norms['dec_Q_l2']})",
                "Quién es Key (K)": f"Memoria Encoder + HD2MR K_enc: {shapes['decoder_cross_K']} (||K||={norms['dec_K_l2']})",
                "Quién es Value (V)": f"Memoria Encoder + HD2MR V_enc: {shapes['decoder_cross_V']} (||V||={norms['dec_V_l2']})",
                "Función en TR-DETR": "Las 10 Moment Queries consultan la memoria del Encoder para regresar (centro, ancho) del intervalo.",
            },
        ]
        st.dataframe(qkv_summary, use_container_width=True, hide_index=True)

        col_h1, col_h2 = st.columns(2)
        clip_labels = [f"C{i+1} ({int(i*clip_duration)}-{int((i+1)*clip_duration)}s)" for i in range(result["num_clips"])]

        with col_h1:
            v2t_html = render_heatmap_html(
                matrix=result["v2t_attention_matrix"],
                row_labels=clip_labels,
                col_labels=result["tokens"],
                title="Mapa 1: Cross-Attention V2T — A_V2T = softmax(Q_v @ K_t^T / √32)",
                subtitle="Filas (Queries Q_v): Clips de video | Columnas (Keys K_t): Tokens de la consulta",
            )
            st.markdown(v2t_html, unsafe_allow_html=True)

        with col_h2:
            query_labels = [f"Moment Query #{i+1:02d}" for i in range(len(result["raw_query_proposals"]))]
            dec_html = render_heatmap_html(
                matrix=result["decoder_cross_attention_matrix"],
                row_labels=query_labels,
                col_labels=clip_labels,
                title="Mapa 2: Decoder Cross-Attention — A_dec = softmax(Q_dec @ K_enc^T / √32)",
                subtitle="Filas (Queries Q_dec): 10 Moment Queries aprendibles | Columnas (Keys K_enc): Memoria del Encoder",
            )
            st.markdown(dec_html, unsafe_allow_html=True)

    # =========================================================================
    # PESTAÑA 3: GUÍA TEÓRICA PARA LA SUSTENTACIÓN
    # =========================================================================
    with tab_theory:
        st.markdown(
            """
            ### 1. ¿Cómo se generan y qué rol cumplen los tensores $Q$, $K$ y $V$ en TR-DETR?
            En **TR-DETR** ($d_{\\text{model}} = 256$, $h = 8$ cabezas, $d_k = d_{\\text{model}} / h = 32$), la ecuación fundamental de atención es:
            $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{32}}\\right) V$$

            Existen **dos bloques de Cross-Attention** con roles complementarios:
            1. **En el Refinamiento Visual Guiado por Consulta (V2T Cross-Attention, antes del Encoder):**
               - **$Q_v = (\\tilde{F}_v + PE_v) W^Q \\in \\mathbb{R}^{L_v \\times 256}$**: Proviene de los **clips de video**.
               - **$K_t = \\tilde{F}_t W^K \\in \\mathbb{R}^{L_t \\times 256}$** y **$V_t = \\tilde{F}_t W^V \\in \\mathbb{R}^{L_t \\times 256}$**: Provienen de los **tokens de texto**.
               - **Interpretación física:** Cada clip de video consulta al texto para absorber significado lingüístico y atenuar el ruido visual de fondo que no tiene relación con la frase del usuario.
            2. **En el Transformer Decoder (DETR Cross-Attention):**
               - **$Q_{\\text{dec}} = (H_{\\text{dec}} + P_q) W^Q \\in \\mathbb{R}^{N_q \\times 256}$**: Proviene de las **$N_q = 10$ Moment Queries aprendibles** (vectores entrenados que buscan inicios y finales de eventos).
               - **$K_{\\text{enc}} = (M_{\\text{HD2MR}} + PE_v) W^K \\in \\mathbb{R}^{L_v \\times 256}$** y **$V_{\\text{enc}} = M_{\\text{HD2MR}} W^V \\in \\mathbb{R}^{L_v \\times 256}$**: Provienen de la **salida del Encoder modulada por el puntaje de relevancia HD2MR**.
               - **Interpretación física:** Cada *Moment Query* actúa como un "detector de intervalos" que interroga a todos los clips del video para decidir dónde empieza y dónde termina la acción ($\hat{c}_i, \hat{w}_i$).

            ---
            ### 2. ¿Por qué es innovadora la arquitectura TR-DETR?
            - **Reciprocidad Bidireccional entre Tareas (`HD2MR` y `MR2HD`):** A diferencia de *Moment-DETR* o *QD-DETR*, que tratan la localización de intervalos (MR) y la detección de momentos destacados (HD) como dos cabezas separadas al final de la red, **TR-DETR** demuestra que ambas tareas son mutuamente complementarias:
              - **HD2MR:** Usa la predicción preliminar de relevancia por clip para filtrar los vectores del Encoder antes de que el Decoder busque el intervalo.
              - **MR2HD:** Usa los momentos localizados por el Decoder y los procesa con una **celda recurrente GRU** para refinar la curva final de relevancia por clip.
            - **Alineación Multimodal Local-Global (LGAM):** Reduce la brecha entre modalidades (video y texto) antes de fusionarlas.

            ---
            ### 3. ¿Qué limitaciones presenta?
            - **Granularidad temporal fija ($\Delta t = 2$ s):** Al discretizar el video en clips de 2 segundos, eventos micro-temporales que duren menos de 1 segundo pueden perder precisión en sus bordes.
            - **Dependencia de extractores congelados (CLIP / SlowFast):** El modelo no entrena el backbone visual extremo a extremo por costo de memoria GPU, por lo que si un concepto visual fino no es capturado por CLIP, el Transformer no puede recuperarlo.
            """
        )


if __name__ == "__main__":
    main()

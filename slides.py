"""
slides.py — Presentación de TR-DETR dentro de Streamlit.

Las diapositivas se dibujan como un componente HTML autónomo (iframe):
la navegación (botones y flechas del teclado) ocurre en JavaScript, sin
recargar Streamlit. Las ecuaciones se dibujan con KaTeX (requiere internet)
y las ilustraciones son SVG generados en Python.

Uso desde app.py:
    from slides import render_slides
    render_slides()
"""

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).resolve().parent / "assets"

# Colores: uno por fase (los mismos de la figura de fases) + generales
C = {
    1: "#d62728", 2: "#ff7f0e", 3: "#9467bd", 4: "#1f77b4", 5: "#2ca02c", 6: "#17becf",
    "ink": "#0f172a", "muted": "#64748b", "accent": "#4f46e5", "soft": "#eef2ff",
}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _img(name: str) -> str:
    """Devuelve una imagen de assets/ como data-URI (o '' si no existe)."""
    path = ASSETS / name
    if not path.exists():
        return ""
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _figure(name: str, max_h: int = 330) -> str:
    src = _img(name)
    if not src:
        return ""
    return f'<img class="fig" style="max-height:{max_h}px" src="{src}">'


def _card(title: str, body: str, color: str = C["accent"]) -> str:
    return (f'<div class="card" style="border-top:4px solid {color}">'
            f'<div class="card-t" style="color:{color}">{title}</div>{body}</div>')


_UID = [0]


def _uid(prefix: str) -> str:
    """Id único por SVG: los marcadores con id repetido no se dibujan si la primera copia está oculta."""
    _UID[0] += 1
    return f"{prefix}{_UID[0]}"


def _pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}1a;color:{color}">{text}</span>'


# ---------------------------------------------------------------------------
# Ilustraciones SVG
# ---------------------------------------------------------------------------

def svg_timeline(n=30, span=(12, 20), color=C[5], scores=None, label_left="0 s",
                 label_right="150 s", span_label="[34 s, 60 s]", w=560, h=120) -> str:
    """Tira de clips con un intervalo resaltado y, opcionalmente, una curva de puntajes."""
    pad, cw = 10, (w - 20) / n
    parts = [f'<svg viewBox="0 0 {w} {h}" class="svg">']
    for i in range(n):
        inside = span[0] <= i <= span[1]
        fill = color if inside else "#e2e8f0"
        parts.append(f'<rect x="{pad + i*cw:.1f}" y="62" width="{cw-2:.1f}" height="26" rx="3" fill="{fill}"/>')
    if scores:
        pts = " ".join(f"{pad + i*cw + cw/2:.1f},{56 - s*46:.1f}" for i, s in enumerate(scores))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{C["accent"]}" stroke-width="2.5"/>')
    x0, x1 = pad + span[0]*cw, pad + (span[1]+1)*cw - 2
    parts.append(f'<path d="M{x0:.1f} 96 L{x0:.1f} 102 L{x1:.1f} 102 L{x1:.1f} 96" '
                 f'stroke="{color}" stroke-width="2" fill="none"/>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="116" text-anchor="middle" class="t" fill="{color}" '
                 f'font-weight="700">{span_label}</text>')
    parts.append(f'<text x="{pad}" y="116" class="t s">{label_left}</text>')
    parts.append(f'<text x="{w-pad}" y="116" text-anchor="end" class="t s">{label_right}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_heatmap(matrix, rows, cols, color, title, w=300) -> str:
    """Matriz de atención pequeña con valores escritos."""
    cell_w, cell_h, left, top = 62, 38, 96, 46
    h = top + cell_h * len(rows) + 8
    w = left + cell_w * len(cols) + 6
    parts = [f'<svg viewBox="0 0 {w} {h}" class="svg" style="max-width:{w}px">',
             f'<text x="{w/2}" y="16" text-anchor="middle" class="t" font-weight="700" fill="{color}">{title}</text>']
    for j, c in enumerate(cols):
        parts.append(f'<text x="{left + j*cell_w + cell_w/2}" y="{top-8}" text-anchor="middle" class="t s">{c}</text>')
    for i, r in enumerate(rows):
        y = top + i * cell_h
        parts.append(f'<text x="{left-8}" y="{y + cell_h/2 + 5}" text-anchor="end" class="t s">{r}</text>')
        for j, v in enumerate(matrix[i]):
            x = left + j * cell_w
            parts.append(f'<rect x="{x+2}" y="{y+2}" width="{cell_w-4}" height="{cell_h-4}" rx="5" '
                         f'fill="{color}" fill-opacity="{0.08 + 0.85*v:.2f}"/>')
            txt_color = "#fff" if v > 0.55 else C["ink"]
            parts.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 5}" text-anchor="middle" '
                         f'class="t" fill="{txt_color}">{v:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_bars(before, after=None, labels=None, color=C[6], w=420, h=170, names=("antes", "después")) -> str:
    """Barras por clip (opcionalmente antes vs después)."""
    n = len(before)
    group = (w - 30) / n
    bw = group * (0.38 if after else 0.6)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="svg">',
             f'<line x1="20" y1="{h-30}" x2="{w-6}" y2="{h-30}" stroke="#cbd5e1"/>']
    for i in range(n):
        gx = 24 + i * group
        hb = before[i] * (h - 50)
        parts.append(f'<rect x="{gx:.1f}" y="{h-30-hb:.1f}" width="{bw:.1f}" height="{hb:.1f}" rx="3" '
                     f'fill="#94a3b8"/>')
        if after:
            ha = after[i] * (h - 50)
            parts.append(f'<rect x="{gx+bw+3:.1f}" y="{h-30-ha:.1f}" width="{bw:.1f}" height="{ha:.1f}" '
                         f'rx="3" fill="{color}"/>')
        if labels:
            parts.append(f'<text x="{gx + (bw*2 if after else bw)/2:.1f}" y="{h-12}" text-anchor="middle" '
                         f'class="t s">{labels[i]}</text>')
    if after:
        parts.append(f'<rect x="{w-150}" y="4" width="12" height="12" fill="#94a3b8" rx="2"/>'
                     f'<text x="{w-134}" y="14" class="t s">{names[0]}</text>'
                     f'<rect x="{w-80}" y="4" width="12" height="12" fill="{color}" rx="2"/>'
                     f'<text x="{w-64}" y="14" class="t s">{names[1]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_qkv(q_label, kv_label, q_items, kv_items, color, w=520, h=210) -> str:
    """Esquema de atención: elementos Q a la izquierda, K/V a la derecha, líneas de atención."""
    parts = [f'<svg viewBox="0 0 {w} {h}" class="svg">',
             f'<text x="80" y="18" text-anchor="middle" class="t" font-weight="700" fill="{color}">{q_label}</text>',
             f'<text x="{w-80}" y="18" text-anchor="middle" class="t" font-weight="700" fill="{C["muted"]}">{kv_label}</text>']
    qy = [40 + i * (h - 60) / max(1, len(q_items) - 1) for i in range(len(q_items))]
    ky = [40 + i * (h - 60) / max(1, len(kv_items) - 1) for i in range(len(kv_items))]
    for i, (_, strengths) in enumerate(q_items):
        for j, s in enumerate(strengths):
            if s > 0.05:
                parts.append(f'<line x1="150" y1="{qy[i]}" x2="{w-150}" y2="{ky[j]}" stroke="{color}" '
                             f'stroke-opacity="{0.15 + 0.8*s:.2f}" stroke-width="{1 + 5*s:.1f}"/>')
    for i, (name, _) in enumerate(q_items):
        parts.append(f'<rect x="10" y="{qy[i]-14}" width="140" height="28" rx="14" fill="{color}"/>'
                     f'<text x="80" y="{qy[i]+5}" text-anchor="middle" class="t" fill="#fff">{name}</text>')
    for j, name in enumerate(kv_items):
        parts.append(f'<rect x="{w-150}" y="{ky[j]-14}" width="140" height="28" rx="14" fill="#fff" '
                     f'stroke="#94a3b8"/><text x="{w-80}" y="{ky[j]+5}" text-anchor="middle" class="t">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_queries(w=560, h=230) -> str:
    """10 moment queries: cada una arranca en su ancla (gris); las que encuentran el momento se mueven (verde)."""
    mid = _uid("q")
    pad, L = 20, w - 40
    real = (0.23, 0.40)  # [34 s, 60 s] de 150 s
    parts = [f'<svg viewBox="0 0 {w} {h}" class="svg">',
             f'<defs><marker id="{mid}" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
             f'<path d="M0,0 L8,4 L0,8 z" fill="{C[5]}"/></marker></defs>',
             f'<rect x="{pad + real[0]*L:.1f}" y="6" width="{(real[1]-real[0])*L:.1f}" height="{h-46}" '
             f'fill="{C[5]}" fill-opacity="0.10" rx="6"/>']
    anchors = [0.05, 0.14, 0.30, 0.38, 0.47, 0.56, 0.66, 0.76, 0.86, 0.95]
    for k, a in enumerate(anchors):
        y = 18 + k * 17
        x0, x1 = pad + (a - 0.04) * L, pad + (a + 0.04) * L
        parts.append(f'<text x="{pad-4}" y="{y+4}" text-anchor="end" class="t s">q{k+1}</text>')
        parts.append(f'<rect x="{x0:.1f}" y="{y-4}" width="{x1-x0:.1f}" height="8" rx="4" fill="#cbd5e1"/>')
        if k in (2, 3):
            parts.append(f'<rect x="{pad + real[0]*L:.1f}" y="{y-5}" width="{(real[1]-real[0])*L:.1f}" '
                         f'height="10" rx="5" fill="{C[5]}" fill-opacity="{0.9 if k == 2 else 0.35}"/>')
            parts.append(f'<text x="{pad + real[1]*L + 6:.1f}" y="{y+4}" class="t s" fill="{C[5]}">'
                         f'{"92 %" if k == 2 else "20 %"}</text>')
    parts.append(f'<line x1="{pad}" y1="{h-36}" x2="{pad+L}" y2="{h-36}" stroke="#cbd5e1"/>'
                 f'<text x="{pad}" y="{h-20}" class="t s">0 s</text>'
                 f'<text x="{pad+L}" y="{h-20}" text-anchor="end" class="t s">150 s</text>'
                 f'<text x="{pad + (real[0]+real[1])/2*L:.1f}" y="{h-20}" text-anchor="middle" class="t" '
                 f'fill="{C[5]}" font-weight="700">momento: [34 s, 60 s]</text>'
                 f'<rect x="{w-170}" y="{h-12}" width="14" height="8" rx="4" fill="#cbd5e1"/>'
                 f'<text x="{w-152}" y="{h-4}" class="t s">ancla inicial</text>'
                 f'<rect x="{w-80}" y="{h-12}" width="14" height="8" rx="4" fill="{C[5]}"/>'
                 f'<text x="{w-62}" y="{h-4}" class="t s">ajustada</text></svg>')
    return "".join(parts)


def svg_cycle(w=520, h=200) -> str:
    """HD ⇄ MR: cooperación entre tareas."""
    a1, a2 = _uid("c"), _uid("c")
    l, r = 100, w - 100
    return (f'<svg viewBox="0 0 {w} {h}" class="svg"><defs>'
            f'<marker id="{a1}" markerWidth="10" markerHeight="10" refX="7" refY="5" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{C[5]}"/></marker>'
            f'<marker id="{a2}" markerWidth="10" markerHeight="10" refX="7" refY="5" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{C[6]}"/></marker></defs>'
            f'<circle cx="{l}" cy="100" r="58" fill="{C[4]}" fill-opacity="0.12" stroke="{C[4]}" stroke-width="2"/>'
            f'<text x="{l}" y="96" text-anchor="middle" class="t big" fill="{C[4]}">HD</text>'
            f'<text x="{l}" y="118" text-anchor="middle" class="t s">puntaje por clip</text>'
            f'<circle cx="{r}" cy="100" r="58" fill="{C[5]}" fill-opacity="0.12" stroke="{C[5]}" stroke-width="2"/>'
            f'<text x="{r}" y="96" text-anchor="middle" class="t big" fill="{C[5]}">MR</text>'
            f'<text x="{r}" y="118" text-anchor="middle" class="t s">intervalo</text>'
            f'<path d="M{l+40} 58 Q {w/2} 8 {r-46} 56" fill="none" stroke="{C[5]}" stroke-width="3" marker-end="url(#{a1})"/>'
            f'<text x="{w/2}" y="26" text-anchor="middle" class="t" font-weight="700" fill="{C[5]}">HD2MR: limpia el video</text>'
            f'<path d="M{r-40} 142 Q {w/2} 192 {l+46} 144" fill="none" stroke="{C[6]}" stroke-width="3" marker-end="url(#{a2})"/>'
            f'<text x="{w/2}" y="186" text-anchor="middle" class="t" font-weight="700" fill="{C[6]}">MR2HD: corrige los puntajes</text>'
            f'</svg>')


def svg_pipeline(steps, w=760, h=90) -> str:
    """Cadena de cajas con flechas."""
    n = len(steps)
    bw, gap = (w - (n - 1) * 34) / n, 34
    pid = _uid("p")
    parts = [f'<svg viewBox="0 0 {w} {h}" class="svg"><defs><marker id="{pid}" markerWidth="10" '
             f'markerHeight="10" refX="7" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/>'
             f'</marker></defs>']
    for i, (title, sub, color) in enumerate(steps):
        x = i * (bw + gap)
        parts.append(f'<rect x="{x:.1f}" y="10" width="{bw:.1f}" height="66" rx="12" fill="{color}" '
                     f'fill-opacity="0.12" stroke="{color}" stroke-width="2"/>'
                     f'<text x="{x + bw/2:.1f}" y="38" text-anchor="middle" class="t" font-weight="700" '
                     f'fill="{color}">{title}</text>'
                     f'<text x="{x + bw/2:.1f}" y="60" text-anchor="middle" class="t s">{sub}</text>')
        if i < n - 1:
            parts.append(f'<line x1="{x + bw + 4:.1f}" y1="43" x2="{x + bw + gap - 6:.1f}" y2="43" '
                         f'stroke="#94a3b8" stroke-width="2.5" marker-end="url(#{pid})"/>')
    parts.append("</svg>")
    return "".join(parts)


def svg_metric_bars(w=420, h=200) -> str:
    """R1@0.7 en QVHighlights (Tabla 1 del artículo)."""
    data = [("Moment-DETR", 33.0, "#94a3b8"), ("QD-DETR", 45.0, "#64748b"), ("TR-DETR", 49.0, C["accent"])]
    parts = [f'<svg viewBox="0 0 {w} {h}" class="svg">',
             f'<text x="{w/2}" y="16" text-anchor="middle" class="t" font-weight="700">R1@0.7 (QVHighlights test)</text>']
    for i, (name, v, col) in enumerate(data):
        y = 36 + i * 52
        bw = (w - 150) * v / 55
        parts.append(f'<text x="112" y="{y+24}" text-anchor="end" class="t">{name}</text>'
                     f'<rect x="120" y="{y+6}" width="{bw:.1f}" height="28" rx="6" fill="{col}"/>'
                     f'<text x="{126+bw:.1f}" y="{y+26}" class="t" font-weight="700" fill="{col}">{v:.1f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Contenido de las diapositivas
# ---------------------------------------------------------------------------

def _slide(title: str, body: str, kicker: str = "", color: str = C["accent"], phase: int = 0) -> dict:
    """phase > 0 agrega arriba a la derecha el mapa de la arquitectura con esa fase iluminada."""
    return {"title": title, "body": body, "kicker": kicker, "color": color, "phase": phase}


def build_slides() -> list:
    s = []

    # ------------------------------ A ------------------------------
    s.append(_slide(
        "Encontrar momentos en un video con una frase",
        f"""
<div class="cover">
  <div class="cover-l">
    <div class="big-title">TR-DETR</div>
    <div class="sub">Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection</div>
    <div class="muted">Sun et al. · AAAI 2024 · arXiv:2401.02309</div>
    <div class="team">Juliana Campuzano · Jorge Iván González · Juan José Mosquera · David Alvarado</div>
  </div>
  <div class="cover-r">
    {svg_pipeline([("Video + frase", "entrada", C[1]), ("Encoder", "contexto", C[4]),
                   ("Decoder", "intervalos", C[5])], w=420, h=90)}
    {svg_timeline(n=26, span=(8, 13), color=C[5], scores=[0.1,0.1,0.15,0.1,0.2,0.15,0.2,0.4,0.8,0.95,0.9,0.85,0.8,0.6,0.25,0.15,0.1,0.1,0.15,0.1,0.1,0.1,0.15,0.1,0.1,0.1],
                   w=420, h=124)}
  </div>
</div>""", kicker="Procesamiento de Datos Secuenciales"))

    s.append(_slide(
        "El problema",
        f"""
<p class="lead">¿En qué segundo del video ocurre lo que describo? Hoy se responde <b>revisando a mano</b>.</p>
<div class="grid3">
  {_card("Entrada", "<p>🎬 Un video (máx. 150 s)</p><p>💬 Una frase en inglés</p><p class='mono'>\"A woman walking in a market\"</p>", C[1])}
  {_card("Salida MR", "<p><b>Moment Retrieval</b></p><p>El intervalo <b>[inicio, fin]</b> donde ocurre.</p>", C[5])}
  {_card("Salida HD", "<p><b>Highlight Detection</b></p><p>Un <b>puntaje</b> por cada clip de 2 s.</p>", C[6])}
</div>
{svg_timeline(n=30, span=(11, 19), color=C[5], scores=[0.1,0.12,0.1,0.15,0.2,0.15,0.1,0.12,0.2,0.3,0.55,0.8,0.9,0.95,0.85,0.9,0.8,0.7,0.6,0.5,0.25,0.15,0.1,0.12,0.1,0.1,0.15,0.1,0.1,0.1])}
""", kicker="Introducción"))

    s.append(_slide(
        "Demostración",
        f"""
<div class="center-box">
  <div class="huge">▶</div>
  <p class="lead">Vamos a la pestaña <b>Inferencia</b></p>
  {svg_pipeline([("1 · Subir video", "máx. 150 s", C[1]), ("2 · Escribir frase", "en inglés", C[3]),
                 ("3 · Ver resultado", "intervalo + curva", C[5])], w=720)}
</div>""", kicker="Introducción"))

    s.append(_slide(
        "Metodología",
        f"""
{svg_pipeline([("Artículo", "AAAI 2024", C[3]), ("Lighthouse", "código oficial", C[4]),
               ("Zenodo", "pesos .ckpt", C[2]), ("Streamlit", "interfaz", C[5])])}
<div class="grid3">
  {_card("Solo inferencia", "<p>No se entrena: se usan pesos <b>ya entrenados</b> en QVHighlights.</p>", C[4])}
  {_card("Pesos", "<p class='mono'>clip_tr_detr_qvhighlight.ckpt</p><p>Features <b>CLIP</b> (sin SlowFast).</p>", C[2])}
  {_card("Hardware", "<p>Funciona en <b>CPU</b>.</p><p>Videos de hasta 75 clips (150 s).</p>", C[5])}
</div>""", kicker="Metodología"))

    # ------------------------------ B ------------------------------
    s.append(_slide(
        "El mecanismo de atención",
        f"""
<div class="eq">$$\\text{{Attention}}(Q,K,V) = \\text{{softmax}}\\!\\left(\\frac{{QK^T}}{{\\sqrt{{d}}}}\\right)V$$</div>
<div class="grid3">
  {_card("Q · Query", "<p><b>Lo que busca.</b></p><p>La salida tiene una fila por cada Q.</p>", C["accent"])}
  {_card("K · Key", "<p><b>Con qué se compara.</b></p><p>\\(QK^T\\) mide la similitud.</p>", C[3])}
  {_card("V · Value", "<p><b>Lo que se lleva.</b></p><p>Se mezcla según los pesos del softmax.</p>", C[5])}
</div>
<p class="note">Self-attention: Q, K y V de la misma secuencia · Cross-attention: Q de una, K y V de otra · 8 cabezas de 32 dimensiones</p>
""", kicker="Marco teórico I"))

    s.append(_slide(
        "La arquitectura en 6 fases",
        f'<div class="center-box">{_figure("fases_tr_detr.png", 440)}</div>',
        kicker="Marco teórico I"))

    s.append(_slide(
        "Fase 1 · CLIP: todo se vuelve vectores",
        f"""
<div class="two">
  <div>{_figure("foco1.jpg", 330)}</div>
  <div>
    <ul class="arrows">
      <li>Video → <b>1 foto cada 2 s</b> → 1 vector por clip</li>
      <li>Frase → <b>1 vector por palabra</b></li>
      <li>CLIP está <b>congelado ❄</b></li>
      <li>Imagen y texto en <b>el mismo espacio</b></li>
    </ul>
    <div class="eq">$$F_v \\in \\mathbb{{R}}^{{L\\times 512}} \\qquad F_t \\in \\mathbb{{R}}^{{N\\times 512}}$$</div>
    <p class="note">CLIP ve fotos sueltas: la secuencia la modela TR-DETR.</p>
  </div>
</div>""", kicker="Fase 1", color=C[1]))

    s.append(_slide(
        "Fase 2 · Alineación local-global",
        f"""
<div class="two">
  <div>{_figure("foco2.jpg", 300)}
    <div class="eq sm">$$\\hat F_v = \\text{{MLP}}_v(F_v) \\quad \\hat F_t = \\text{{MLP}}_t(F_t) \\quad (512 \\to 256)$$</div>
  </div>
  <div>
    {_card("Regulador local · clip ↔ palabra",
           "<div class='eq sm'>$$\\mathcal{L}_{local} = -\\sum_i \\big[C_i\\log\\hat S_i + (1-C_i)\\log(1-\\hat S_i)\\big]$$</div>"
           "<p>¿El clip <i>i</i> pertenece al momento? (C = 1 sí, 0 no)</p>", C[2])}
    {_card("Regulador global · video ↔ frase",
           "<div class='eq sm'>$$\\mathcal{L}_{global} = -\\frac{1}{B}\\sum_i \\log\\frac{e^{G_v^i\\cdot G_t^i}}{\\sum_j e^{G_v^i\\cdot G_t^j}}$$</div>"
           "<p>¿Cuál de las B frases es la de este video?</p>", C[2])}
    {_pill("solo en entrenamiento", C[2])}
  </div>
</div>""", kicker="Fase 2", color=C[2]))

    s.append(_slide(
        "Fase 3 · Refinamiento: matriz clip × palabra",
        f"""
<div class="eq sm">$$A = \\frac{{\\text{{Linear}}(\\hat F_v)\\,\\text{{Linear}}(\\hat F_t)^T}}{{\\sqrt d}} \\in \\mathbb{{R}}^{{L\\times N}}$$</div>
<div class="two">
  <div class="center-box">
    {svg_heatmap([[0.20,0.70,0.10],[0.40,0.10,0.50],[0.33,0.33,0.33]], ["calle","mercado","cielo"],
                 ["woman","walking","market"], C[3], "A_r · cada clip reparte")}
    <p class="note">\\(F_{{v2q}} = A_r\\hat F_t\\) → qué palabras describen al clip</p>
  </div>
  <div class="center-box">
    {svg_heatmap([[0.10,0.60,0.05],[0.80,0.30,0.90],[0.10,0.10,0.05]], ["calle","mercado","cielo"],
                 ["woman","walking","market"], C[1], "A_c · cada palabra busca")}
    <p class="note">\\(F_{{q2v}} = A_r A_c^T\\hat F_v\\) → cómo se ve la palabra <b>en este video</b></p>
  </div>
</div>
<p class="note">El cielo no se parece a nada → <b>se apaga</b> · "market" vive en el clip del mercado</p>
""", kicker="Fase 3", color=C[3], phase=3))

    s.append(_slide(
        "Fase 3 · Fusión: cross-attention",
        f"""
<div class="two">
  <div class="center-box">
    {svg_qkv("Q · clips", "K, V · palabras",
             [("calle", [0.1, 0.7, 0.05, 0.1]), ("mercado", [0.5, 0.1, 0.05, 0.8]), ("cielo", [0.1, 0.1, 0.05, 0.1])],
             ["woman", "walking", "in a", "market"], C[3])}
  </div>
  <div>
    <div class="eq">$$Z = \\text{{softmax}}\\!\\left(\\frac{{Q_vK_t^T}}{{\\sqrt d}}\\right)V_t$$</div>
    <ul class="arrows">
      <li><b>Q = clips</b>: "¿qué tengo que ver con la frase?"</li>
      <li><b>K, V = palabras</b>: con qué compara y qué se lleva</li>
      <li>Clips como Q → se conserva <b>el eje del tiempo</b></li>
      <li>Salida <b>Z (75 × 256)</b>: clips "teñidos" por la frase</li>
    </ul>
  </div>
</div>""", kicker="Fase 3", color=C[3], phase=3))

    # ------------------------------ C ------------------------------
    s.append(_slide(
        "Fase 4 · Encoder: contexto temporal",
        f"""
<div class="two">
  <div>{_figure("foco4.jpg", 300)}</div>
  <div>
    <ul class="arrows">
      <li>Self-attention: <b>Q = K = V = clips</b> (75 × 75)</li>
      <li>Cada clip mira a sus vecinos → <b>contexto</b></li>
      <li>2 capas: atención + FFN (256 → 1024 → 256)</li>
      <li>Cabeza lineal → <b>puntaje inicial</b> (borrador)</li>
    </ul>
    <div class="eq">$$H = \\text{{Linear}}(\\text{{MHA}}(Z)) \\in \\mathbb{{R}}^{{L}}$$</div>
  </div>
</div>""", kicker="Fase 4", color=C[4]))

    s.append(_slide(
        "Fase 5 · HD2MR: el highlight limpia el video",
        f"""
<div class="eq">$$\\bar Z = \\text{{softmax}}(H)\\odot Z \\qquad \\hat Z = \\text{{MHA}}(Z + \\bar Z)$$</div>
<div class="two">
  <div class="center-box">
    {svg_bars([0.30, 0.90, 0.80, 0.40, 0.10], [0.20, 0.95, 0.85, 0.35, 0.03],
              ["calle", "mercado", "mercado", "espaldas", "cielo"], C[5], names=("Z", "Z + Z̄"))}
  </div>
  <div>
    <ul class="arrows">
      <li><b>⊗</b>: el puntaje multiplica a los clips</li>
      <li>Los irrelevantes (el cielo) <b>se apagan</b></li>
      <li><b>+ Z</b> = residual: no se borra la información</li>
      <li>Pasa otra vez por <b>el mismo encoder</b></li>
    </ul>
  </div>
</div>""", kicker="Fase 5", color=C[5], phase=5))

    s.append(_slide(
        "Fase 5 · Decoder: 10 buscadores",
        f"""
<div class="two">
  <div class="center-box">{svg_queries()}</div>
  <div>
    <ul class="arrows">
      <li><b>10 moment queries</b> con un ancla fija (centro, ancho)</li>
      <li>Cross-attention: <b>Q = queries · K, V = memoria Ẑ</b></li>
      <li>El ancla dice <b>dónde</b> mirar; la memoria, <b>qué</b> es relevante</li>
      <li>2 capas: cada una corrige centro y ancho</li>
    </ul>
    <div class="grid2">
      {_card("Localización", "<p>MLP → (centro, ancho)</p><p><b>regresión</b></p>", C[5])}
      {_card("Clase", "<p>Lineal → momento / fondo</p><p><b>confianza</b></p>", C[5])}
    </div>
    <p class="note">0.31 y 0.17 × 150 s → <b>[34 s, 60 s] · 92 %</b></p>
  </div>
</div>""", kicker="Fase 5", color=C[5], phase=5))

    s.append(_slide(
        "Fase 6 · MR2HD: el intervalo corrige el highlight",
        f"""
<div class="eq sm">$$F^M_v = \\text{{GRU}}(m) \\qquad S_{{ref}} = \\frac{{F^M_v\\hat F_v^T}}{{\\|F^M_v\\|\\|\\hat F_v\\|}} \\qquad H = \\text{{Linear}}(Z + \\text{{softmax}}(S_{{ref}})\\odot\\hat Z)$$</div>
<div class="two">
  <div>
    {svg_pipeline([("Clips del intervalo", "en orden", C[5]), ("GRU", "resume", C[6]), ("«Foto»", "del momento", C[6])], w=460, h=90)}
    <ul class="arrows">
      <li>La foto se compara con <b>cada clip</b></li>
      <li>Los parecidos <b>suben</b> → puntaje final</li>
    </ul>
  </div>
  <div class="center-box">
    {svg_bars([0.30, 0.90, 0.40, 0.40], [0.20, 0.95, 0.80, 0.35], ["calle", "mercado", "espaldas", "cielo"],
              C[6], names=("inicial", "final"))}
    <p class="note">La mujer de espaldas: no se parece a la frase, pero <b>sí al momento</b></p>
  </div>
</div>""", kicker="Fase 6", color=C[6], phase=6))

    s.append(_slide(
        "La innovación: las tareas cooperan",
        f"""
<div class="two">
  <div class="center-box">{svg_cycle()}</div>
  <div>
    {_card("Antes · Moment-DETR, QD-DETR", "<p>MR y HD con <b>cabezas separadas</b> que no se comunican.</p>", "#94a3b8")}
    {_card("TR-DETR · Task-Reciprocal", "<p><b>HD → MR:</b> el puntaje limpia el video.</p><p><b>MR → HD:</b> el intervalo corrige el puntaje.</p>", C["accent"])}
  </div>
</div>""", kicker="Innovación"))

    s.append(_slide(
        "¿Cómo aprendió? Las pérdidas",
        f"""
<div class="eq">$$\\mathcal{{L}}_{{total}} = \\mathcal{{L}}_{{mom}} + \\mathcal{{L}}_{{high}} + 0.3\\,(\\mathcal{{L}}_{{local}} + \\mathcal{{L}}_{{global}})$$</div>
<div class="grid2">
  {_card("L_mom · intervalos",
         "<div class='eq sm'>$$\\lambda_{L1}\\|m-\\hat m\\|_1 + \\lambda_{gIoU}\\mathcal{L}_{gIoU} + \\lambda_{cls}\\mathcal{L}_{CE}$$</div>"
         "<p>L1 = distancia · gIoU = traslape · CE = momento o fondo</p>"
         "<p class='muted'>Antes: <i>Hungarian matching</i> (una query por momento real)</p>", C[5])}
  {_card("L_high · orden de los clips",
         "<div class='eq sm'>$$\\max(0,\\ \\Delta + S_{neg} - S_{pos})$$</div>"
         "<p>El clip bueno debe superar al malo por un margen</p>"
         "<p class='muted'>Aprende a <b>ordenar</b>, no valores absolutos</p>", C[6])}
</div>
<p class="note">Entrenamiento conjunto (end-to-end) · CLIP congelado · nosotros usamos los pesos ya entrenados</p>
""", kicker="Entrenamiento"))

    # ------------------------------ D ------------------------------
    s.append(_slide(
        "Implementación",
        f"""
<pre class="code"><span class="c"># 1. Cargar el modelo: torch.load → load_state_dict → eval()</span>
model = TRDETRPredictor(<span class="s">"weights/clip_tr_detr_qvhighlight.ckpt"</span>,
                        device=<span class="s">"cpu"</span>, feature_name=<span class="s">"clip"</span>)

<span class="c"># 2. Video → CLIP (1 foto cada 2 s) + marcas de tiempo</span>
video = model.encode_video(<span class="s">"video.mp4"</span>)

<span class="c"># 3. Frase → CLIP-texto → TR-DETR completo → posprocesamiento</span>
pred = model.predict(<span class="s">"A woman walking in a market"</span>, video)

pred[<span class="s">"pred_relevant_windows"</span>]  <span class="c"># [[inicio, fin, confianza], ...]</span>
pred[<span class="s">"pred_saliency_scores"</span>]   <span class="c"># un puntaje por clip</span></pre>
{svg_pipeline([("encode_video", "CLIP ❄", C[1]), ("predict", "fases 2–6", C[4]), ("post-proceso", "segundos", C[5])], w=720, h=90)}
""", kicker="Implementación"))

    s.append(_slide(
        "Métricas del artículo",
        f"""
<div class="two">
  <div class="center-box">{svg_metric_bars()}</div>
  <div>
    <table class="tbl"><tr><th>Modelo</th><th>R1@0.5</th><th>R1@0.7</th><th>HIT@1</th></tr>
      <tr><td>Moment-DETR</td><td>52.9</td><td>33.0</td><td>55.6</td></tr>
      <tr><td>QD-DETR</td><td>62.4</td><td>45.0</td><td>62.4</td></tr>
      <tr class="hl"><td>TR-DETR</td><td>64.7</td><td>49.0</td><td>63.4</td></tr></table>
    <ul class="arrows">
      <li><b>R1@0.7</b>: el mejor intervalo coincide ≥ 70 % con el real</li>
      <li>La mayor mejora está en las métricas <b>más exigentes</b></li>
    </ul>
  </div>
</div>""", kicker="Resultados"))

    s.append(_slide(
        "Resultados propios",
        f"""
<table class="tbl"><tr><th>Consulta</th><th>Top-1 predicho</th><th>Confianza</th><th>Escena real</th></tr>
  <tr><td class="mono">Car driving on the road</td><td>0.07–12.46 s</td><td>97.3 %</td><td>0–10 s</td></tr>
  <tr><td class="mono">A man opens the door of a car</td><td>8.29–19.91 s</td><td>97.2 %</td><td>10–20 s</td></tr>
  <tr><td class="mono">An old man walking in the park</td><td>21.05–30.0 s</td><td>99.5 %</td><td>20–30 s</td></tr></table>
<p class="note">Video real de 30 s (<span class="mono">demo_real_30s.mp4</span>) · modelo oficial Lighthouse + CLIP ViT-B/32</p>
<div class="grid2">
  {_card("✓ Funciona", "<p>Localiza las tres escenas y el pico de la curva cae en el intervalo.</p>", C[5])}
  {_card("⚠ Cuidado", "<p>Siempre devuelve un intervalo, aunque la frase no aparezca.</p>", C[2])}
</div>""", kicker="Resultados"))

    s.append(_slide(
        "Limitaciones",
        f"""
<div class="grid3">
  {_card("⏱ 150 s máximo", "<p>La codificación posicional solo cubre 75 clips.</p>", C[1])}
  {_card("✂ Clips de 2 s", "<p>Lo muy rápido se pierde entre muestras.</p>", C[2])}
  {_card("📷 Fotos sueltas", "<p>Con solo CLIP no se capta bien el movimiento.</p>", C[3])}
  {_card("🌐 Solo inglés", "<p>Entrenado con consultas en inglés.</p>", C[4])}
  {_card("🔇 Sin audio útil", "<p>Concatenarlo sin alinear empeora métricas.</p>", C[6])}
  {_card("🎯 Una palabra", "<p>Puede fijarse en un objeto y perder la acción.</p>", C[5])}
</div>""", kicker="Cierre"))

    s.append(_slide(
        "Trabajo futuro y conclusión",
        f"""
<div class="two">
  <div>
    {_card("Mejoras", "<p>➕ <b>SlowFast</b> para captar movimiento</p><p>➕ <b>Videos largos</b> por ventanas</p><p>➕ <b>Alinear el audio</b> antes de mezclarlo</p>", C[4])}
  </div>
  <div class="center-box">
    {svg_cycle(w=420, h=200)}
    <p class="lead">Que las tareas <b>cooperen</b> mejora a las dos.</p>
  </div>
</div>
<div class="thanks">¡Gracias! · ¿Preguntas?</div>""", kicker="Cierre"))

    return s


# ---------------------------------------------------------------------------
# Página HTML
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box}
body{margin:0;padding:8px 10px;font-family:"Segoe UI",system-ui,sans-serif;color:#0f172a;background:transparent}
.nav{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.nav button{padding:6px 18px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;cursor:pointer;font-size:14px}
.nav button:disabled{opacity:.35;cursor:default}
.nav button:hover:not(:disabled){background:#f1f5f9}
.ctr{flex:1;text-align:center;color:#64748b;font-size:13px;font-weight:600}
.bar{height:4px;background:#e2e8f0;border-radius:4px;overflow:hidden;margin-bottom:10px}
.bar>div{height:100%;background:#4f46e5;transition:width .25s}
.stage{background:#fff;border:1px solid #e2e8f0;border-radius:18px;min-height:640px;padding:26px 34px 20px;position:relative;
       box-shadow:0 6px 24px rgba(15,23,42,.06);border-left:8px solid var(--c)}
.slide{display:none;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.kicker{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--c);margin-bottom:4px}
h2{margin:0 0 16px;font-size:28px;line-height:1.2}
p{margin:6px 0;font-size:16px;line-height:1.5}
.lead{font-size:19px}
.muted{color:#64748b;font-size:14px}
.note{color:#475569;font-size:14px;margin-top:10px;text-align:center}
.mono{font-family:Consolas,monospace;font-size:14px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:center}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 0}
.card{background:#f8fafc;border-radius:14px;padding:12px 16px;margin:8px 0}
.card-t{font-weight:700;font-size:15px;margin-bottom:4px}
.card p{font-size:15px}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-size:13px;font-weight:700;margin-top:6px}
ul.arrows{list-style:none;padding:0;margin:6px 0}
ul.arrows li{font-size:17px;margin:9px 0;padding-left:26px;position:relative;line-height:1.4}
ul.arrows li:before{content:"→";position:absolute;left:0;color:var(--c);font-weight:700}
.eq{margin:8px 0;font-size:19px;text-align:center}
.eq.sm{font-size:15px}
.fig{max-width:100%;border-radius:10px;display:block;margin:0 auto}
.svg{width:100%;height:auto;display:block;margin:6px auto}
.svg .t{font-family:"Segoe UI",system-ui,sans-serif;font-size:13px;fill:#0f172a}
.svg .t.s{font-size:11px;fill:#64748b}
.svg .t.big{font-size:26px;font-weight:800}
.center-box{text-align:center}
.huge{font-size:64px;color:#4f46e5;line-height:1}
.cover{display:grid;grid-template-columns:1.1fr 1fr;gap:30px;align-items:center;margin-top:20px}
.big-title{font-size:64px;font-weight:800;letter-spacing:-.02em;color:#4f46e5;line-height:1}
.sub{font-size:20px;margin:12px 0;line-height:1.35}
.team{margin-top:26px;font-size:15px;color:#334155;border-top:1px solid #e2e8f0;padding-top:12px}
.tbl{border-collapse:collapse;width:100%;font-size:15px;margin:8px 0}
.tbl th{background:#f1f5f9;text-align:left;padding:7px 10px}
.tbl td{border-top:1px solid #e2e8f0;padding:7px 10px}
.tbl tr.hl td{background:#eef2ff;font-weight:700;color:#4f46e5}
pre.code{background:#0f172a;color:#e2e8f0;padding:16px 18px;border-radius:12px;font-size:14px;line-height:1.55;overflow-x:auto}
pre.code .c{color:#94a3b8}
pre.code .s{color:#86efac}
.thanks{text-align:center;font-size:30px;font-weight:800;color:#4f46e5;margin-top:22px}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}
.mini{width:280px;border-radius:10px;border:1px solid #e2e8f0;margin:-8px -10px 6px 0;flex-shrink:0}
.foot{position:absolute;right:22px;bottom:10px;font-size:11px;color:#94a3b8}
"""

JS = """
var cur=0, slides=document.querySelectorAll('.slide'), tot=slides.length;
var stage=document.getElementById('stage');
function show(i){
  slides[cur].style.display='none';
  cur=Math.max(0,Math.min(tot-1,i));
  slides[cur].style.display='block';
  stage.style.setProperty('--c', slides[cur].dataset.color);
  document.getElementById('ctr').textContent=(cur+1)+' / '+tot;
  document.getElementById('bp').disabled=cur===0;
  document.getElementById('bn').disabled=cur===tot-1;
  document.getElementById('prog').style.width=((cur+1)/tot*100)+'%';
}
document.addEventListener('keydown',function(e){
  if(e.key==='ArrowRight'||e.key===' '){show(cur+1);e.preventDefault();}
  if(e.key==='ArrowLeft'){show(cur-1);e.preventDefault();}
});
show(0);
"""


def build_page() -> str:
    """Arma el HTML completo con todas las diapositivas."""
    slides = build_slides()
    html_slides = []
    for i, s in enumerate(slides):
        kicker = f'<div class="kicker">{s["kicker"]}</div>' if s["kicker"] else ""
        mini = ""
        if s["phase"]:
            mini = f'<img class="mini" src="{_img(f"foco{s["phase"]}.jpg")}" title="Ubicación en la arquitectura">'
        html_slides.append(
            f'<div class="slide" data-color="{s["color"]}" style="display:{"block" if i == 0 else "none"}">'
            f'<div class="head"><div>{kicker}<h2>{s["title"]}</h2></div>{mini}</div>{s["body"]}</div>'
        )
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">'
        '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>'
        '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" '
        'onload="renderMathInElement(document.body,{delimiters:[{left:\'$$\',right:\'$$\',display:true},'
        '{left:\'\\\\(\',right:\'\\\\)\',display:false}]})"></script>'
        f'<style>{CSS}</style></head><body>'
        '<div class="nav"><button id="bp" onclick="show(cur-1)">&#8592; Anterior</button>'
        '<div class="ctr" id="ctr"></div>'
        '<button id="bn" onclick="show(cur+1)">Siguiente &#8594;</button></div>'
        '<div class="bar"><div id="prog"></div></div>'
        '<div class="stage" id="stage">' + "".join(html_slides) +
        '<div class="foot">TR-DETR · AAAI 2024 · ← → para navegar</div></div>'
        f'<script>{JS}</script></body></html>'
    )


def render_slides() -> None:
    """Muestra la presentación dentro de Streamlit."""
    components.html(build_page(), height=800, scrolling=True)


if __name__ == "__main__":
    st.set_page_config(page_title="TR-DETR · Presentación", layout="wide")
    render_slides()

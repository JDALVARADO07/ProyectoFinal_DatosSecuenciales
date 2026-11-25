"""
slides.py — Presentación de TR-DETR dentro de Streamlit.

Las slides se renderizan como un componente HTML puro (iframe) para evitar
errores de reconciliación de React al cambiar entre slides con distinta estructura.
"""

import base64
import html as _html
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).resolve().parent / "assets"
FULL_DIAGRAM = "fases_tr_detr.png"

SLIDES = [
    # --------------------------------- A ---------------------------------
    {
        "title": "TR-DETR: Localización de Momentos y Detección de Highlights en Video",
        "body": """
**Artículo:** *TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection* · AAAI 2024

Transformer **encoder–decoder** para recuperación de momentos y detección de highlights en video mediante lenguaje natural.

**Integrantes:** Juliana Campuzano · Jorge Iván González · Juan José Mosquera · David Alvarado
""",
    },
    {
        "title": "El problema",
        "body": """
Localizar manualmente un evento específico dentro de un video extenso es una tarea costosa y no escalable.

TR-DETR aborda este problema de forma conjunta:

- **Entrada:** un video y una descripción en lenguaje natural.
- **Salida MR (Moment Retrieval):** el intervalo temporal **[inicio, fin]** donde ocurre el evento.
- **Salida HD (Highlight Detection):** un puntaje de relevancia para cada segmento de 2 segundos del video.
""",
    },
    {
        "title": "Demostración del sistema",
        "body": """
Se presenta una consulta en lenguaje natural sobre un video real. El sistema identifica automáticamente el intervalo temporal donde ocurre el evento descrito y genera una curva de relevancia por segmento.

**Pasos:**
1. Se carga un video (máx. 150 s).
2. Se escribe la consulta en inglés.
3. El modelo devuelve el intervalo predicho y la curva de puntajes por clip.
""",
    },
    {
        "title": "Metodología",
        "body": """
**Artículo** → **Lighthouse** (librería de inferencia oficial) → **pesos de Zenodo** → **Streamlit**

- No se realiza entrenamiento: se utiliza **solo inferencia** con pesos preentrenados.
- Pesos: `clip_tr_detr_qvhighlight.ckpt`, entrenados sobre el dataset QVHighlights.
- Compatible con CPU, sin necesidad de GPU.
""",
    },
    # --------------------------------- B ---------------------------------
    {
        "title": "La arquitectura en 6 fases",
        "image": FULL_DIAGRAM,
        "body": "El video y la consulta de texto recorren seis etapas hasta producir el intervalo y los puntajes de relevancia.",
    },
    {
        "title": "Fase 1 · CLIP: representación vectorial",
        "image": FULL_DIAGRAM,
        "body": """
- El video se muestrea a **1 fotograma cada 2 s** → se obtiene un vector de 512 dimensiones por clip (máximo 75 clips).
- La consulta se tokeniza → un vector de 512 dimensiones por token.
- El encoder CLIP está **congelado** (no se actualiza durante el entrenamiento) y proyecta video y texto al mismo espacio semántico.
""",
    },
    {
        "title": "Fase 2 · Alineación local-global",
        "image": FULL_DIAGRAM,
        "body": """
- Un **MLP** proyecta los vectores de 512 a **256 dimensiones**.
- **Regulador local:** mide la similitud entre cada clip y cada token de la consulta.
- **Regulador global:** compara la representación global del video con la de la consulta completa.
- Ambos reguladores actúan únicamente durante el entrenamiento como funciones de pérdida adicionales.
""",
    },
    {
        "title": "Fase 3 · Refinamiento y fusión multimodal",
        "image": FULL_DIAGRAM,
        "body": """
Se construyen dos matrices de atención cruzada:

- **A_r:** cada clip distribuye su atención entre las palabras de la consulta.
- **A_c:** cada palabra identifica en qué clips del video aparece.

Resultado: los clips semánticamente no relacionados con la consulta reducen su representación, mientras que los relevantes la refuerzan.
""",
    },
    {
        "title": "Fase 3 · Cross-attention de fusión",
        "body": """
| Tensor | Origen | Rol |
|--------|--------|-----|
| **Q** | Clips de video | Consulta qué información aporta cada token de la frase |
| **K** | Tokens de la consulta | Base de comparación |
| **V** | Tokens de la consulta | Contenido que el clip incorpora a su representación |

**Salida:** representación multimodal de cada clip, enriquecida con el significado de la consulta.
""",
    },
    # --------------------------------- C ---------------------------------
    {
        "title": "Fase 4 · Encoder: contexto temporal",
        "image": FULL_DIAGRAM,
        "body": """
- **Self-attention:** Q, K y V provienen de los propios clips (matriz 75 × 75).
- Cada clip pondera la información de los clips adyacentes → el modelo adquiere **contexto temporal**.
- Una cabeza de predicción lineal genera un **puntaje preliminar de highlight** por clip.
""",
    },
    {
        "title": "Fase 5 · HD2MR + Decoder: predicción del intervalo",
        "image": FULL_DIAGRAM,
        "body": """
- Los puntajes de highlight **modulan** las representaciones del encoder: los clips irrelevantes se atenúan (operación ⊗).
- El decoder emplea **10 moment queries** (vectores aprendibles) que actúan como candidatos de intervalo.
- **Q = moment queries · K, V = memoria del encoder modulada**.
- Cada query produce una terna **(centro, ancho, confianza)** que define un intervalo candidato.
""",
    },
    {
        "title": "Fase 6 · MR2HD: refinamiento del puntaje",
        "image": FULL_DIAGRAM,
        "body": """
- Se extraen los clips dentro del intervalo de mayor confianza.
- Una **GRU** resume esos clips en un único vector representativo del momento.
- Ese vector se compara con cada clip del video.
- Los clips con alta similitud incrementan su puntaje → **puntaje final de highlight**.
""",
    },
    {
        "title": "La innovación: cooperación entre tareas",
        "body": """
Modelos anteriores trataban MR y HD como tareas independientes.

TR-DETR establece una cooperación bidireccional:

- **HD → MR (HD2MR):** los puntajes de highlight filtran el contexto antes de que el decoder busque el intervalo.
- **MR → HD (MR2HD):** el intervalo predicho retroalimenta y corrige los puntajes de highlight.

Este esquema **Task-Reciprocal** es la contribución principal del artículo.
""",
    },
    # --------------------------------- D ---------------------------------
    {
        "title": "Implementación",
        "code_lang": "python",
        "code": """model = TRDETRPredictor("weights/clip_tr_detr_qvhighlight.ckpt",
                        device="cpu", feature_name="clip")
video = model.encode_video("video.mp4")          # extrae 1 fotograma / 2 s con CLIP
pred  = model.predict("A woman walking in a market", video)
# pred["pred_relevant_windows"] → [[inicio, fin, confianza], ...]
# pred["pred_saliency_scores"]  → un puntaje por clip""",
        "body": """
- `torch.load` carga el checkpoint; `load_state_dict` asigna los pesos capa por capa; `eval()` desactiva dropout y batch norm.
""",
    },
    {
        "title": "Métricas en QVHighlights",
        "body": """
| Modelo | R1@0.5 | R1@0.7 | HIT@1 |
|--------|--------|--------|-------|
| Moment-DETR | 52.9 | 33.0 | 55.6 |
| QD-DETR | 62.4 | 45.0 | 62.4 |
| **TR-DETR** | **64.7** | **49.0** | **63.4** |

- **R1@0.7:** el intervalo predicho con mayor confianza presenta al menos un 70 % de solapamiento con el intervalo de referencia.
- TR-DETR supera a los modelos previos especialmente en las métricas de mayor exigencia.
""",
    },
    {
        "title": "Resultados propios",
        "body": """
*(Agregar capturas de la demostración)*

- ✓ Caso exitoso: intervalo predicho vs. intervalo real anotado.
- ✗ Caso límite: consulta que no corresponde a ningún segmento → curva de relevancia plana.
""",
    },
    {
        "title": "Limitaciones",
        "body": """
- Videos de **máximo 150 s** de duración.
- Resolución temporal de **2 s por clip**: eventos muy breves pueden no ser detectados.
- CLIP procesa **fotogramas individuales**: el movimiento entre frames no se captura directamente.
- El modelo fue entrenado en inglés: las consultas en otros idiomas pueden reducir la precisión.
- El canal de **audio no es utilizado** por el modelo (mencionado como trabajo futuro en el artículo).
""",
    },
    {
        "title": "Trabajo futuro y conclusiones",
        "body": """
**Posibles mejoras:**
- Incorporar **SlowFast** u otros encoders de video para captar dinámica temporal.
- Extender el soporte a **videos de mayor duración** mediante procesamiento por ventanas.
- Integrar el canal de **audio** como modalidad adicional.

**Conclusión:** la cooperación bidireccional entre Moment Retrieval y Highlight Detection permite que ambas tareas se beneficien mutuamente, logrando mejoras consistentes frente a arquitecturas que las tratan de forma independiente.

---
**¡Gracias! — ¿Preguntas?**
""",
    },
]


# ---------------------------------------------------------------------------
# Markdown → HTML helpers (solo para el contenido de las slides)
# ---------------------------------------------------------------------------

def _inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r'<code class="ic">\1</code>', text)
    return text


def _md_to_html(md: str) -> str:
    lines = md.strip().split("\n")
    out = []
    in_ul = in_ol = in_table = False
    table_head_done = False

    for raw in lines:
        s = raw.strip()

        # Table
        if s.startswith("|") and "|" in s[1:]:
            if not in_table:
                in_table = True
                table_head_done = False
                out.append('<table class="mt">')
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells if c.strip()):
                table_head_done = True
                out.append("<tbody>")
                continue
            tag = "th" if not table_head_done else "td"
            row = "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
            if not table_head_done:
                out.append(f"<thead><tr>{row}</tr></thead>")
            else:
                out.append(f"<tr>{row}</tr>")
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False
            table_head_done = False

        # Bullet list
        if s.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(s[2:])}</li>")
            continue
        if in_ul:
            out.append("</ul>")
            in_ul = False

        # Ordered list
        m = re.match(r"^\d+\.\s+(.+)$", s)
        if m:
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_inline(m.group(1))}</li>")
            continue
        if in_ol:
            out.append("</ol>")
            in_ol = False

        if s == "---":
            out.append('<hr class="sl-hr">')
        elif s.startswith("### "):
            out.append(f"<h3>{_inline(s[4:])}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{_inline(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h1>{_inline(s[2:])}</h1>")
        elif s:
            out.append(f"<p>{_inline(s)}</p>")

    if in_ul:
        out.append("</ul>")
    if in_ol:
        out.append("</ol>")
    if in_table:
        out.append("</tbody></table>")

    return "\n".join(out)


def _slide_to_html(slide: dict, img_b64: str) -> str:
    title = _html.escape(slide["title"])
    parts = [f'<h2 class="sl-title">{title}</h2>']

    for key in ("body", "code", "body2", "code2", "body3"):
        if key not in slide:
            continue
        if key.startswith("code"):
            code_esc = _html.escape(slide[key])
            parts.append(f'<pre class="cb"><code>{code_esc}</code></pre>')
        else:
            parts.append(_md_to_html(slide[key]))

    if "image" in slide and img_b64:
        parts.append(f'<img src="data:image/png;base64,{img_b64}" class="sl-img">')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Componente HTML autónomo (sin React, sin reruns de Python para navegar)
# ---------------------------------------------------------------------------

def render_slides() -> None:
    img_b64 = ""
    img_path = ASSETS / FULL_DIAGRAM
    if img_path.exists():
        img_b64 = base64.b64encode(img_path.read_bytes()).decode()

    total = len(SLIDES)
    slides_html = ""
    for i, slide in enumerate(SLIDES):
        b64 = img_b64 if "image" in slide else ""
        display = "block" if i == 0 else "none"
        content = _slide_to_html(slide, b64)
        slides_html += f'<div class="slide" id="sl{i}" style="display:{display}">{content}</div>\n'

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0; padding: 10px 14px;
    font-family: "Segoe UI", sans-serif;
    font-size: 15px; color: #1e293b; line-height: 1.65;
  }}
  .nav {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
  }}
  .nav button {{
    padding: 5px 20px; border: 1px solid #cbd5e1; border-radius: 5px;
    background: #fff; cursor: pointer; font-size: 14px; color: #1e293b;
    transition: background .15s;
  }}
  .nav button:disabled {{ opacity: .35; cursor: default; }}
  .nav button:hover:not(:disabled) {{ background: #f1f5f9; }}
  .ctr {{ flex: 1; text-align: center; color: #64748b; font-size: 14px; font-weight: 600; }}
  .div {{ border: none; border-top: 1px solid #e2e8f0; margin: 10px 0; }}
  h2.sl-title {{ margin-top: 0; color: #0f172a; font-size: 1.25rem; }}
  h3 {{ color: #0f172a; margin: 10px 0 4px; }}
  ul, ol {{ padding-left: 1.4rem; margin: 6px 0; }}
  li {{ margin: 3px 0; }}
  table.mt {{
    border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px;
  }}
  table.mt th, table.mt td {{ border: 1px solid #cbd5e1; padding: 5px 10px; }}
  table.mt th {{ background: #f1f5f9; font-weight: 600; }}
  pre.cb {{
    background: #1e293b; color: #e2e8f0;
    padding: 14px; border-radius: 7px;
    overflow-x: auto; font-size: 13px; line-height: 1.5; white-space: pre;
  }}
  code.ic {{
    background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 92%;
  }}
  .sl-img {{ max-width: 100%; margin-top: 12px; border-radius: 6px; }}
  hr.sl-hr {{ border: none; border-top: 1px solid #94a3b8; margin: 10px 0; }}
  p {{ margin: 5px 0; }}
</style>
</head>
<body>
<div class="nav">
  <button id="bp" onclick="go(-1)" disabled>&#8592; Anterior</button>
  <div class="ctr" id="ctr">1 / {total}</div>
  <button id="bn" onclick="go(1)">Siguiente &#8594;</button>
</div>
<hr class="div">
<div id="sc">
{slides_html}
</div>
<script>
var cur = 0, tot = {total};
function go(d) {{
  document.getElementById("sl" + cur).style.display = "none";
  cur = Math.max(0, Math.min(tot - 1, cur + d));
  document.getElementById("sl" + cur).style.display = "block";
  document.getElementById("ctr").textContent = (cur + 1) + " / " + tot;
  document.getElementById("bp").disabled = cur === 0;
  document.getElementById("bn").disabled = cur === tot - 1;
}}
</script>
</body>
</html>"""

    components.html(page, height=780, scrolling=True)


if __name__ == "__main__":
    st.set_page_config(page_title="TR-DETR · Presentación", layout="wide")
    render_slides()

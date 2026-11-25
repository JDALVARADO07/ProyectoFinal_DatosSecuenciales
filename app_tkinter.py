"""
Interfaz Gráfica Interactiva en Tkinter (Nativa de Python, sin dependencias externas de UI)
para la Sustentación del Proyecto Final:
"TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection"
(Sun et al., AAAI 2024).

Ejecución local:
    python app_tkinter.py
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Any, Dict

from inference import TRDETRInferenceEngine, generate_demo_video


class TRDETRTkinterApp:
    """Aplicación interactiva de escritorio en Tkinter para demostrar la inferencia de TR-DETR."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TR-DETR (AAAI 2024) · Moment Retrieval & Highlight Detection")
        self.root.geometry("1120x740")
        self.root.configure(bg="#f8fafc")

        base_dir = Path(__file__).resolve().parent
        self.demo_video_path = generate_demo_video(
            str(base_dir / "sample_data" / "demo_video_eventos.mp4")
        )
        self.current_video = self.demo_video_path
        self.engine = TRDETRInferenceEngine(clip_duration=2.0)
        self.last_result: Dict[str, Any] = {}

        self._build_ui()
        self.run_inference()

    def _build_ui(self):
        # Encabezado
        header = tk.Frame(self.root, bg="#1e3a8a", pady=10, padx=16)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval & Highlight Detection",
            font=("Segoe UI", 14, "bold"),
            fg="white",
            bg="#1e3a8a",
        ).pack(anchor="w")
        tk.Label(
            header,
            text=f"Motor activo: {self.engine.engine_status} | Arquitectura Transformer Encoder–Decoder (AAAI 2024)",
            font=("Segoe UI", 9),
            fg="#bfdbfe",
            bg="#1e3a8a",
        ).pack(anchor="w")

        # Panel de controles (Video y Consulta)
        ctrl = tk.LabelFrame(
            self.root,
            text=" 1. Selección de Video y Consulta en Lenguaje Natural ",
            font=("Segoe UI", 10, "bold"),
            bg="#f8fafc",
            padx=12,
            pady=8,
        )
        ctrl.pack(fill=tk.X, padx=14, pady=8)

        row1 = tk.Frame(ctrl, bg="#f8fafc")
        row1.pack(fill=tk.X, pady=3)
        tk.Label(row1, text="Archivo de Video:", font=("Segoe UI", 9, "bold"), bg="#f8fafc").pack(side=tk.LEFT)
        self.video_var = tk.StringVar(value=self.current_video)
        tk.Entry(row1, textvariable=self.video_var, width=82, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=8)
        tk.Button(
            row1,
            text="Examinar Video...",
            command=self.browse_video,
            bg="#e2e8f0",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            row1,
            text="Usar Video Demo",
            command=self.use_demo_video,
            bg="#dbeafe",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=4)

        row2 = tk.Frame(ctrl, bg="#f8fafc")
        row2.pack(fill=tk.X, pady=6)
        tk.Label(row2, text="Consulta (Query):", font=("Segoe UI", 9, "bold"), bg="#f8fafc").pack(side=tk.LEFT)
        self.query_var = tk.StringVar(value="El hombre abre la puerta del carro")
        query_combo = ttk.Combobox(
            row2,
            textvariable=self.query_var,
            width=60,
            font=("Segoe UI", 10),
            values=[
                "El hombre abre la puerta del carro",
                "The man opens the car door",
                "Auto conduciendo en la carretera",
                "Persona caminando en el parque",
            ],
        )
        query_combo.pack(side=tk.LEFT, padx=8)
        tk.Button(
            row2,
            text="▶ Ejecutar Inferencia TR-DETR",
            command=self.run_inference,
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=3,
        ).pack(side=tk.LEFT, padx=8)

        # Panel central dividido: Gráfica de Highlight Detection + Tabla de Moment Queries y Q, K, V
        center = tk.Frame(self.root, bg="#f8fafc")
        center.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)

        # Gráfica Canvas de Saliency Score por clip
        chart_frame = tk.LabelFrame(
            center,
            text=" 2. Curva Temporal de Relevancia (Highlight Detection: HD2MR vs MR2HD) e Intervalo Top-1 ",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            padx=8,
            pady=8,
        )
        chart_frame.pack(fill=tk.X, pady=4)
        self.canvas = tk.Canvas(chart_frame, width=1040, height=220, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.X)

        # Panel inferior dividido en dos columnas: Propuestas del Decoder y Dimensiones de Tensores Q, K, V
        bottom = tk.Frame(center, bg="#f8fafc")
        bottom.pack(fill=tk.BOTH, expand=True, pady=6)

        left_box = tk.LabelFrame(
            bottom,
            text=" 3. Propuestas de Intervalos del Decoder (10 Moment Queries) ",
            font=("Segoe UI", 10, "bold"),
            bg="#f8fafc",
            padx=6,
            pady=6,
        )
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        cols = ("rank", "query", "start", "end", "dur", "conf")
        self.tree = ttk.Treeview(left_box, columns=cols, show="headings", height=9)
        self.tree.heading("rank", text="Rank")
        self.tree.heading("query", text="Moment Query")
        self.tree.heading("start", text="Inicio (s)")
        self.tree.heading("end", text="Fin (s)")
        self.tree.heading("dur", text="Duración (s)")
        self.tree.heading("conf", text="Confianza")
        for c, w in zip(cols, (50, 110, 80, 80, 85, 85)):
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        right_box = tk.LabelFrame(
            bottom,
            text=" 4. Trazabilidad en Vivo de Tensores Query (Q), Key (K) y Value (V) ",
            font=("Segoe UI", 10, "bold"),
            bg="#f8fafc",
            padx=6,
            pady=6,
        )
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.qkv_text = tk.Text(right_box, font=("Consolas", 9), bg="#0f172a", fg="#e2e8f0", height=10)
        self.qkv_text.pack(fill=tk.BOTH, expand=True)

    def browse_video(self):
        chosen = filedialog.askopenfilename(
            title="Seleccionar archivo de video",
            filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")],
        )
        if chosen:
            self.video_var.set(chosen)
            self.run_inference()

    def use_demo_video(self):
        self.video_var.set(self.demo_video_path)
        self.run_inference()

    def run_inference(self):
        video_path = self.video_var.get().strip()
        query = self.query_var.get().strip()
        if not os.path.exists(video_path):
            messagebox.showwarning("Archivo no encontrado", f"No existe el video:\n{video_path}")
            return

        result = self.engine.predict(video_path=video_path, query=query)
        self.last_result = result
        self._update_table(result)
        self._update_qkv_panel(result)
        self._draw_saliency_chart(result)

    def _update_table(self, result: Dict[str, Any]):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, m in enumerate(result["ranked_moments"], start=1):
            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"#{idx}",
                    f"Query #{m['query_id']:02d}",
                    f"{m['start_sec']:.2f}s",
                    f"{m['end_sec']:.2f}s",
                    f"{m['duration_sec']:.2f}s",
                    f"{m['confidence']*100:.1f}%",
                ),
            )

    def _update_qkv_panel(self, result: Dict[str, Any]):
        s = result["tensor_shapes"]
        n = result["tensor_norms"]
        top1 = result["ranked_moments"][0]
        lines = [
            f"Tokens consulta ({result['num_tokens']}): {result['tokens']}",
            f"Clips de video  ({result['num_clips']} clips x {result['clip_duration_sec']}s = {result['video_duration_sec']}s)",
            "-" * 58,
            "[1] Cross-Attention V2T (Refinamiento Visual Guiado por Texto):",
            f"    Q_v (Clips Video) : {s['v2t_Q']} -> 8 cabezas {s['v2t_Q_heads']} (||Q||={n['v2t_Q_l2']})",
            f"    K_t (Tokens Texto): {s['v2t_K']} (||K||={n['v2t_K_l2']})",
            f"    V_t (Tokens Texto): {s['v2t_V']} (||V||={n['v2t_V_l2']})",
            "[2] Transformer Encoder (Self-Attention + Modulación HD2MR):",
            f"    Q = K = V (Z_mm)  : {s['encoder_Q']}",
            "[3] Transformer Decoder (Cross-Attention con Moment Queries):",
            f"    Q_dec (10 Queries): {s['decoder_cross_Q']} -> 8 cabezas {s['decoder_cross_Q_heads']} (||Q||={n['dec_Q_l2']})",
            f"    K_enc (Memoria)   : {s['decoder_cross_K']} -> 8 cabezas {s['decoder_cross_K_heads']} (||K||={n['dec_K_l2']})",
            f"    V_enc (Memoria)   : {s['decoder_cross_V']} (||V||={n['dec_V_l2']})",
            "-" * 58,
            f"RESULTADO TOP-1: [{top1['start_sec']:.2f}s - {top1['end_sec']:.2f}s] (Confianza: {top1['confidence']*100:.1f}%)",
        ]
        self.qkv_text.delete("1.0", tk.END)
        self.qkv_text.insert(tk.END, "\n".join(lines))

    def _draw_saliency_chart(self, result: Dict[str, Any]):
        self.canvas.delete("all")
        w = 1020
        h = 210
        pad_l, pad_r, pad_t, pad_b = 55, 25, 22, 35
        pw = w - pad_l - pad_r
        ph = h - pad_t - pad_b
        dur = max(1.0, result["video_duration_sec"])

        def x_c(t: float) -> float:
            return pad_l + (t / dur) * pw

        def y_c(val: float) -> float:
            return pad_t + (1.0 - max(0.0, min(1.0, val))) * ph

        # Franja amarilla del intervalo Top-1
        top1 = result["ranked_moments"][0]
        x1 = x_c(top1["start_sec"])
        x2 = x_c(top1["end_sec"])
        self.canvas.create_rectangle(x1, pad_t, x2, pad_t + ph, fill="#fef08a", outline="#eab308", width=2)
        self.canvas.create_text(
            (x1 + x2) / 2,
            pad_t + 12,
            text=f"Top-1: [{top1['start_sec']}s - {top1['end_sec']}s] ({top1['confidence']*100:.1f}%)",
            fill="#854d0e",
            font=("Segoe UI", 9, "bold"),
        )

        # Líneas horizontales de guía
        for val in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y_c(val)
            self.canvas.create_line(pad_l, yy, w - pad_r, yy, fill="#e2e8f0", dash=(3, 3))
            self.canvas.create_text(pad_l - 8, yy, text=f"{val:.2f}", anchor="e", fill="#475569", font=("Segoe UI", 8))

        # Curva preliminar HD2MR (gris punteada) y curva final MR2HD (azul sólida)
        ts = result["clip_timestamps"]
        s_init = result["saliency_init"]
        s_final = result["saliency_final"]

        if len(ts) > 1:
            pts_init = []
            pts_final = []
            for t, si, sf in zip(ts, s_init, s_final):
                pts_init.extend([x_c(t), y_c(si)])
                pts_final.extend([x_c(t), y_c(sf)])
            self.canvas.create_line(*pts_init, fill="#94a3b8", width=2, dash=(4, 3))
            self.canvas.create_line(*pts_final, fill="#1d4ed8", width=3)

        for t, sf in zip(ts, s_final):
            cx, cy = x_c(t), y_c(sf)
            self.canvas.create_oval(cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5, fill="#1d4ed8", outline="white")
            self.canvas.create_text(cx, pad_t + ph + 14, text=f"{int(t)}s", fill="#475569", font=("Segoe UI", 8))


if __name__ == "__main__":
    root = tk.Tk()
    app = TRDETRTkinterApp(root)
    root.mainloop()

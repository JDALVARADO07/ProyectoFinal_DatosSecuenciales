"""
Implementación en PyTorch de la arquitectura TR-DETR (AAAI 2024):
"TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection"
Autores originales: H. Sun, M. Zhou, W. Chen y W. Xie (arXiv:2401.02309).

Este módulo implementa de forma transparente y documentada cada componente del
Transformer Encoder-Decoder de TR-DETR, exponiendo explícitamente cómo se generan
y operan los tensores Query (Q), Key (K) y Value (V) en:
1. Alineación Multimodal Local-Global (LGAM).
2. Refinamiento Visual Guiado por Consulta (Cross-Attention V2T: Video-to-Text).
3. Transformer Encoder con Self-Attention temporal y módulo recíproco HD2MR.
4. Transformer Decoder estilo DETR con Moment Queries aprendibles y Cross-Attention.
5. Módulo recíproco MR2HD basado en GRU y cabezas de predicción de intervalos y relevancia.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TRDETRConfig:
    """Configuración de hiperparámetros de la arquitectura TR-DETR."""
    visual_dim: int = 512          # Dimensión de entrada de features visuales (CLIP ViT-B/32 = 512)
    text_dim: int = 512            # Dimensión de entrada de features textuales (CLIP Text = 512)
    hidden_dim: int = 256          # Dimensión latente del Transformer (d_model)
    nhead: int = 8                 # Número de cabezas de atención (d_k = 256 / 8 = 32)
    num_encoder_layers: int = 3    # Número de capas en el Transformer Encoder
    num_decoder_layers: int = 3    # Número de capas en el Transformer Decoder
    dim_feedforward: int = 1024    # Dimensión oculta de las redes Feed-Forward (4 * d_model)
    dropout: float = 0.1           # Tasa de dropout
    num_queries: int = 10          # Número de Moment Queries aprendibles en el Decoder (N_q)
    max_video_clips: int = 150     # Longitud máxima de secuencia de clips (ej. 150 clips * 2s = 300s)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal 1D (Vaswani et al., 2017).
    Inyecta información del orden temporal de los clips de video en el Transformer,
    dado que el mecanismo de atención por sí solo es invariante a permutaciones.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Se registra como buffer (no es un parámetro entrenable) con forma (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna la codificación posicional correspondiente a la longitud temporal L de x.
        Entrada x: (Batch, L, d_model) -> Salida pos: (1, L, d_model)
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class ExplicitMultiHeadAttention(nn.Module):
    """
    Implementación explícita de Multi-Head Attention (Scaled Dot-Product Attention)
    para inspeccionar y extraer directamente los tensores Q, K, V y los mapas de atención
    durante la inferencia.

    Fórmula canónica:
        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model debe ser exactamente divisible por nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # Proyecciones lineales aprendibles W^Q, W^K, W^V y proyección de salida W^O
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor,
        value_input: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcula Multi-Head Attention y devuelve:
        - output: Tensor de salida enriquecido (Batch, L_q, d_model)
        - attn_weights: Matriz de pesos de atención (Batch, nhead, L_q, L_k)
        - qkv_tensors: Diccionario con los tensores proyectados Q, K y V para visualización.
        """
        batch_size, len_q, _ = query_input.size()
        _, len_k, _ = key_input.size()

        # 1. Generación de los tensores Q, K y V mediante proyección lineal
        # Q: "¿Qué información estoy buscando?"
        # K: "¿Qué etiqueta o rasgo identifica a cada elemento?"
        # V: "¿Qué contenido semántico aporta cada elemento?"
        q_proj = self.w_q(query_input)  # (B, L_q, d_model)
        k_proj = self.w_k(key_input)    # (B, L_k, d_model)
        v_proj = self.w_v(value_input)  # (B, L_k, d_model)

        # 2. División del embedding en múltiples cabezas: (B, L, nhead, d_k) -> (B, nhead, L, d_k)
        q = q_proj.view(batch_size, len_q, self.nhead, self.d_k).transpose(1, 2)
        k = k_proj.view(batch_size, len_k, self.nhead, self.d_k).transpose(1, 2)
        v = v_proj.view(batch_size, len_k, self.nhead, self.d_k).transpose(1, 2)

        # 3. Producto punto escalado (similitud entre Queries y Keys)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, nhead, L_q, L_k)

        if key_padding_mask is not None:
            # key_padding_mask tiene True en las posiciones de <pad> que deben ignorarse (-inf)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)
            scores = scores.masked_fill(mask, float("-inf"))

        # 4. Normalización con Softmax (cada fila suma 100% de atención)
        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn_weights)

        # 5. Ponderación de los Values (V) por la matriz de atención
        context = torch.matmul(attn_dropped, v)  # (B, nhead, L_q, d_k)

        # 6. Concatenación de cabezas y proyección lineal final W^O
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, len_q, self.d_model)
        )
        output = self.w_o(context)

        qkv_tensors = {
            "Q": q_proj.detach(),
            "K": k_proj.detach(),
            "V": v_proj.detach(),
            "Q_heads": q.detach(),
            "K_heads": k.detach(),
            "V_heads": v.detach(),
        }
        return output, attn_weights.detach(), qkv_tensors


class LocalGlobalAlignmentModule(nn.Module):
    """
    Módulo 1 de TR-DETR: Alineación Multimodal Local-Global (LGAM).
    Proyecta las características visuales (clips de video) y textuales (tokens de la consulta)
    a un espacio latente compartido de dimensión d_model = 256 y calcula la afinidad semántica
    entre cada clip y el vector global de la oración.
    """

    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.visual_mlp = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.text_mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, visual_feats: torch.Tensor, text_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Entradas:
            visual_feats: (B, L_v, D_v) - Secuencia de embeddings de clips de video.
            text_feats:   (B, L_t, D_t) - Secuencia de embeddings de tokens de texto.
        Salidas:
            v_aligned: (B, L_v, d_model)
            t_aligned: (B, L_t, d_model)
            cosine_sim_matrix: (B, L_v, L_t) - Similitud coseno local clip-token.
        """
        v_aligned = self.visual_mlp(visual_feats)
        t_aligned = self.text_mlp(text_feats)

        v_norm = F.normalize(v_aligned, p=2, dim=-1)
        t_norm = F.normalize(t_aligned, p=2, dim=-1)
        cosine_sim_matrix = torch.matmul(v_norm, t_norm.transpose(-2, -1))
        return v_aligned, t_aligned, cosine_sim_matrix


class VisualFeatureRefinementV2T(nn.Module):
    """
    Módulo 2 de TR-DETR: Refinamiento de Características Visuales Guiado por Consulta (V2T).
    Utiliza Cross-Attention Multimodal donde:
      - Query (Q): Características visuales de los clips de video (v_aligned).
      - Key (K) y Value (V): Características textuales de la consulta (t_aligned).
    Objetivo: Filtrar el ruido de fondo en los clips de video e inyectar el contexto
    lingüístico relevante antes de ingresar al Transformer Encoder.
    """

    def __init__(self, hidden_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_v2t = ExplicitMultiHeadAttention(hidden_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        v_aligned: torch.Tensor,
        t_aligned: torch.Tensor,
        pos_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Ejecuta la atención cruzada Video-a-Texto (V2T) con compuerta de filtrado.
        """
        # En V2T Cross-Attention:
        # Q proviene del video + codificación posicional temporal
        # K y V provienen de los tokens de la consulta en lenguaje natural
        q_video = v_aligned + pos_v
        k_text = t_aligned
        v_text = t_aligned

        text_context, attn_v2t, qkv_v2t = self.cross_attn_v2t(
            query_input=q_video,
            key_input=k_text,
            value_input=v_text,
        )

        # Compuerta de refinamiento que pondera cuánto contexto textual absorbe cada clip
        gate = torch.sigmoid(self.gate_linear(torch.cat([v_aligned, text_context], dim=-1)))
        v_refined = self.norm1(v_aligned + self.dropout(gate * text_context))
        v_refined = self.norm2(v_refined + self.dropout(self.ffn(v_refined)))

        return v_refined, attn_v2t, qkv_v2t


class TransformerEncoderLayerExplicit(nn.Module):
    """
    Capa del Transformer Encoder con Self-Attention temporal entre todos los clips del video.
    Aquí Q, K y V provienen de la misma secuencia multimodal refinada (Z_mm).
    """

    def __init__(self, hidden_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = ExplicitMultiHeadAttention(hidden_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        q = k = x + pos
        v = x
        attn_out, attn_weights, qkv = self.self_attn(q, k, v)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights, qkv


class TransformerDecoderLayerExplicit(nn.Module):
    """
    Capa del Transformer Decoder estilo DETR en TR-DETR.
    Consta de dos mecanismos de atención consecutivos:
    1. Self-Attention entre las N_q Moment Queries (evita que dos queries predigan el mismo segmento).
    2. Cross-Attention Decoder-Encoder:
       - Query (Q): Moment Queries del Decoder (tgt + query_pos).
       - Key (K):   Memoria multimodal del Encoder modulada por HD2MR + pos_v.
       - Value (V): Memoria multimodal del Encoder modulada por HD2MR.
    """

    def __init__(self, hidden_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = ExplicitMultiHeadAttention(hidden_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.cross_attn = ExplicitMultiHeadAttention(hidden_dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Self-Attention entre Moment Queries
        q_self = k_self = tgt + query_pos
        v_self = tgt
        self_out, self_attn_w, _ = self.self_attn(q_self, k_self, v_self)
        tgt = self.norm1(tgt + self.dropout(self_out))

        # 2. Cross-Attention entre Moment Queries (Decoder) y Memoria Multimodal (Encoder)
        q_cross = tgt + query_pos       # (B, N_q, d_model) -> Queries del Decoder
        k_cross = memory + memory_pos   # (B, L_v, d_model) -> Keys del Encoder
        v_cross = memory                # (B, L_v, d_model) -> Values del Encoder

        cross_out, cross_attn_w, qkv_cross = self.cross_attn(
            query_input=q_cross,
            key_input=k_cross,
            value_input=v_cross,
        )
        tgt = self.norm2(tgt + self.dropout(cross_out))

        # 3. Red Feed-Forward por posición
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))

        return tgt, self_attn_w, cross_attn_w, qkv_cross


class TaskCooperationModule(nn.Module):
    """
    Módulo 3 de TR-DETR: Cooperación Recíproca entre Tareas (HD2MR y MR2HD).
    - HD2MR (Highlight Detection -> Moment Retrieval):
      Estima un puntaje de relevancia inicial por clip y lo utiliza para resaltar los clips
      más prometedores en la memoria del Encoder antes de alimentar al Decoder.
    - MR2HD (Moment Retrieval -> Highlight Detection):
      Usa una celda recurrente GRU sobre los momentos recuperados por el Decoder para
      actualizar la puntuación final de relevancia (saliency score) clip por clip.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Cabeza inicial de relevancia para HD2MR
        self.initial_saliency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Proyección de modulación HD2MR
        self.hd2mr_proj = nn.Linear(hidden_dim, hidden_dim)

        # GRU y cabeza de refinamiento para MR2HD
        self.mr2hd_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.final_saliency_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def apply_hd2mr(self, encoder_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica HD2MR: Calcula relevancia preliminar s_init y pondera la memoria del Encoder.
        """
        s_init_logits = self.initial_saliency_head(encoder_memory).squeeze(-1)  # (B, L_v)
        s_init = torch.sigmoid(s_init_logits)  # Probabilidad de relevancia por clip en [0, 1]

        # Modulación de características visuales-textuales mediante los puntajes de highlight
        modulated_memory = encoder_memory + s_init.unsqueeze(-1) * self.hd2mr_proj(encoder_memory)
        return modulated_memory, s_init

    def apply_mr2hd(
        self,
        modulated_memory: torch.Tensor,
        decoder_hidden: torch.Tensor,
        cross_attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aplica MR2HD: Agrega las representaciones de los momentos localizados con una GRU
        y las combina con la memoria por clip para producir el puntaje de highlight final.
        """
        # Procesamos la secuencia de consultas decodificadas con una GRU para obtener el contexto de momentos F_Mv
        gru_out, _ = self.mr2hd_gru(decoder_hidden)  # (B, N_q, d_model)

        # Proyectamos la información de los momentos de vuelta hacia los L_v clips usando la atención cruzada promedio
        # cross_attn_weights: (B, nhead, N_q, L_v) -> promedio sobre cabezas: (B, N_q, L_v)
        mean_cross_attn = cross_attn_weights.mean(dim=1)  # (B, N_q, L_v)
        moment_to_clip_context = torch.matmul(mean_cross_attn.transpose(1, 2), gru_out)  # (B, L_v, d_model)

        combined = torch.cat([modulated_memory, moment_to_clip_context], dim=-1)  # (B, L_v, 2*d_model)
        s_final = torch.sigmoid(self.final_saliency_head(combined).squeeze(-1))   # (B, L_v)
        return s_final


class TRDETR(nn.Module):
    """
    Arquitectura Completa TR-DETR (Transformer Encoder-Decoder para Moment Retrieval
    y Highlight Detection).
    """

    def __init__(self, config: Optional[TRDETRConfig] = None):
        super().__init__()
        self.config = config or TRDETRConfig()
        d_model = self.config.hidden_dim

        # 1. Codificación Posicional Sinusoidal para la secuencia de clips de video
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=self.config.max_video_clips)

        # 2. Alineación Multimodal Local-Global (LGAM)
        self.lgam = LocalGlobalAlignmentModule(
            visual_dim=self.config.visual_dim,
            text_dim=self.config.text_dim,
            hidden_dim=d_model,
            dropout=self.config.dropout,
        )

        # 3. Refinamiento de Características Visuales (Cross-Attention V2T)
        self.v2t_refinement = VisualFeatureRefinementV2T(
            hidden_dim=d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
        )

        # 4. Pila de capas del Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerExplicit(
                hidden_dim=d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_encoder_layers)
        ])

        # 5. Módulo de Cooperación Recíproca entre Tareas (HD2MR y MR2HD)
        self.task_cooperation = TaskCooperationModule(hidden_dim=d_model)

        # 6. Moment Queries Aprendibles del Decoder (N_q = 10 propuestas temporales)
        self.query_content_embed = nn.Embedding(self.config.num_queries, d_model)
        self.query_pos_embed = nn.Embedding(self.config.num_queries, d_model)

        # 7. Pila de capas del Transformer Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerExplicit(
                hidden_dim=d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_decoder_layers)
        ])

        # 8. Cabezas de Predicción del Decoder:
        # - class_head: Probabilidad de que la query i corresponda al momento buscado
        # - span_head: Coordenadas normalizadas (centro c_i, ancho w_i) en [0, 1]
        self.class_head = nn.Linear(d_model, 1)
        self.span_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )

    def forward(
        self, visual_feats: torch.Tensor, text_feats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Ejecuta el flujo completo de inferencia de TR-DETR y retorna tanto las
        predicciones (intervalos y relevancia por clip) como los tensores internos
        Q, K, V y mapas de atención para análisis académico.

        Parámetros:
            visual_feats: Tensor (Batch, L_v, 512) con los embeddings de los clips de video.
            text_feats:   Tensor (Batch, L_t, 512) con los embeddings de los tokens de la consulta.
        """
        batch_size, num_clips, _ = visual_feats.size()

        # Paso 1: Alineación Local-Global
        v_aligned, t_aligned, local_sim = self.lgam(visual_feats, text_feats)
        pos_v = self.pos_encoder(v_aligned)  # (1, L_v, d_model)

        # Paso 2: Refinamiento Visual Guiado por Texto (Cross-Attention V2T)
        v_refined, attn_v2t, qkv_v2t = self.v2t_refinement(v_aligned, t_aligned, pos_v)

        # Paso 3: Codificación Temporal en el Transformer Encoder (Self-Attention)
        enc_hidden = v_refined
        enc_attn_weights = None
        enc_qkv = None
        for layer in self.encoder_layers:
            enc_hidden, enc_attn_weights, enc_qkv = layer(enc_hidden, pos_v)

        # Paso 4: Cooperación HD2MR (Highlight Detection -> Moment Retrieval)
        modulated_memory, s_init = self.task_cooperation.apply_hd2mr(enc_hidden)

        # Paso 5: Decodificación de Propuestas Temporales con Moment Queries (Cross-Attention Decoder)
        tgt = self.query_content_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        dec_hidden = tgt
        dec_self_attn = None
        dec_cross_attn = None
        dec_qkv = None
        for layer in self.decoder_layers:
            dec_hidden, dec_self_attn, dec_cross_attn, dec_qkv = layer(
                tgt=dec_hidden,
                memory=modulated_memory,
                query_pos=query_pos,
                memory_pos=pos_v,
            )

        # Paso 6: Cooperación MR2HD (Moment Retrieval -> Highlight Detection con GRU)
        s_final = self.task_cooperation.apply_mr2hd(
            modulated_memory=modulated_memory,
            decoder_hidden=dec_hidden,
            cross_attn_weights=dec_cross_attn,
        )

        # Paso 7: Predicción de intervalos temporales (centro, ancho) y puntaje de confianza
        pred_logits = self.class_head(dec_hidden).squeeze(-1)  # (B, N_q)
        pred_scores = torch.sigmoid(pred_logits)               # (B, N_q)
        pred_spans_cw = self.span_head(dec_hidden)             # (B, N_q, 2) -> (centro, ancho)

        # Conversión de (centro, ancho) normalizados a [inicio, fin] en [0, 1]
        center = pred_spans_cw[:, :, 0]
        width = pred_spans_cw[:, :, 1]
        start_norm = torch.clamp(center - 0.5 * width, min=0.0, max=1.0)
        end_norm = torch.clamp(center + 0.5 * width, min=0.0, max=1.0)
        pred_intervals_norm = torch.stack([start_norm, end_norm], dim=-1)  # (B, N_q, 2)

        return {
            "pred_scores": pred_scores,                  # (B, N_q)
            "pred_spans_cw": pred_spans_cw,              # (B, N_q, 2)
            "pred_intervals_norm": pred_intervals_norm,  # (B, N_q, 2)
            "saliency_init": s_init,                     # (B, L_v) - Antes de MR2HD
            "saliency_final": s_final,                   # (B, L_v) - Después de MR2HD
            "local_sim_matrix": local_sim.detach(),      # (B, L_v, L_t)
            "attn_v2t": attn_v2t,                        # (B, nhead, L_v, L_t)
            "attn_encoder_self": enc_attn_weights,       # (B, nhead, L_v, L_v)
            "attn_decoder_self": dec_self_attn,          # (B, nhead, N_q, N_q)
            "attn_decoder_cross": dec_cross_attn,        # (B, nhead, N_q, L_v)
            "qkv_v2t": qkv_v2t,                          # Diccionario con Q, K, V de V2T
            "qkv_encoder": enc_qkv,                      # Diccionario con Q, K, V del Encoder
            "qkv_decoder_cross": dec_qkv,                # Diccionario con Q, K, V del Decoder Cross-Attention
            "v_aligned_shape": tuple(v_aligned.shape),
            "t_aligned_shape": tuple(t_aligned.shape),
            "encoder_memory_shape": tuple(modulated_memory.shape),
            "decoder_output_shape": tuple(dec_hidden.shape),
        }

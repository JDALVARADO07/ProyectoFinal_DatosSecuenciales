# TR-DETR: Localización de Momentos y Detección de Highlights en Video mediante Transformer Encoder–Decoder

**Proyecto Final — Procesamiento de Datos Secuenciales con Deep Learning**  
**Especialización en Inteligencia Artificial · Universidad Autónoma de Occidente (UAO)**  
**Docente:** Dra. Natalie Belandia  
**Integrantes:** Juliana Campuzano · Jorge Iván González · Juan José Mosquera · David Alvarado  

---

## 1. Resumen (Abstract)

Encontrar un momento específico dentro de un video de larga duración a partir de una descripción en lenguaje natural (por ejemplo, *"el hombre abre la puerta del carro"*) exige tradicionalmente inspeccionar la grabación de principio a fin de forma manual. En este proyecto se implementa y analiza el flujo de inferencia de **TR-DETR (*Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection*)**, una arquitectura **Transformer Encoder–Decoder** propuesta en **AAAI 2024** que resuelve de manera conjunta dos tareas sobre datos secuenciales multimodales: la **Recuperación de Momentos (*Moment Retrieval*, MR)** y la **Detección de Momentos Destacados (*Highlight Detection*, HD)**.

El sistema recibe como entrada una secuencia temporal de clips de video (extraídos cada $\Delta t = 2.0\text{ s}$ y codificados mediante **CLIP ViT-B/32**) junto con la secuencia de tokens de una consulta en lenguaje natural. A través de un módulo de **Alineación Local-Global (LGAM)**, un bloque de **Atención Cruzada Video-a-Texto (V2T)**, un **Transformer Encoder–Decoder** con $N_q = 10$ *Moment Queries* aprendibles y un mecanismo bidireccional de **Reciprocidad entre Tareas (`HD2MR` y `MR2HD` con celda GRU)**, el modelo predice simultáneamente las coordenadas temporales $[\text{inicio}, \text{fin}]$ del evento buscado y una curva continua de relevancia (*saliency score*) clip por clip. Se presenta una implementación modular en PyTorch junto con una interfaz gráfica interactiva en **Streamlit** que permite cargar videos propios, ejecutar consultas en vivo e inspeccionar la creación y dimensiones de los tensores **Query ($Q$)**, **Key ($K$)** y **Value ($V$)** en cada etapa de atención.

---

## 2. Introducción

### 2.1 Artículo Base y Repositorios Oficiales
- **Artículo científico base:** H. Sun, M. Zhou, W. Chen y W. Xie, “TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection,” en *Proc. AAAI Conf. Artificial Intelligence*, vol. 38, no. 5, 2024, pp. 4998–5007. Preprint: `arXiv:2401.02309`.
- **Repositorio original de los autores:** [https://github.com/mingyao1120/TR-DETR](https://github.com/mingyao1120/TR-DETR)
- **Framework de inferencia reproducible y pesos preentrenados (QVHighlights, features CLIP):** [https://github.com/line/lighthouse](https://github.com/line/lighthouse) y [https://zenodo.org/records/13960580](https://zenodo.org/records/13960580) (`clip_tr_detr_qvhighlight.ckpt`, licencias MIT y Apache 2.0).

### 2.2 Contexto del Problema y Motivación
El crecimiento exponencial de contenido audiovisual en plataformas educativas, sistemas de videovigilancia, deportes y medios digitales ha vuelto ineficiente la búsqueda manual de eventos dentro de videos. A diferencia de la clasificación de video tradicional, la localización temporal guiada por lenguaje natural requiere:
1. **Alinear dos dominios secuenciales heterogéneos:** una secuencia visual continua discretizada en $L_v$ clips y una secuencia discreta de $L_t$ tokens lingüísticos.
2. **Delimitación temporal precisa (*Moment Retrieval*):** encontrar los límites exactos $[t_{\text{inicio}}, t_{\text{fin}}]$ donde ocurre la acción.
3. **Puntuación de relevancia (*Highlight Detection*):** asignar un valor escalar $s_t \in [0, 1]$ a cada clip $t$ según su correspondencia semántica con la consulta.

Arquitecturas previas basadas en DETR (como *Moment-DETR* o *QD-DETR*) predecían ambas tareas con cabezas independientes al final del Decoder, desaprovechando que saber *qué clips son relevantes* ayuda a delimitar el intervalo y saber *dónde empieza y termina el intervalo* ayuda a suprimir ruido fuera del evento.

### 2.3 Objetivo
Implementar el pipeline de preprocesamiento e inferencia de la arquitectura **Transformer Encoder–Decoder TR-DETR**, analizando en profundidad la generación matemática y operativa de los tensores $Q$, $K$ y $V$ en sus mecanismos de *Self-Attention* y *Cross-Attention*, y demostrando su funcionamiento en tiempo real mediante una interfaz gráfica interactiva.

---

## 3. Marco Teórico

### 3.1 Arquitectura General Transformer Encoder–Decoder en TR-DETR

El flujo secuencial completo de **TR-DETR** se estructura en cinco etapas interconectadas:

```mermaid
flowchart LR
    subgraph Entradas["1. Entradas Secuenciales (CLIP)"]
        V["Video: L_v clips de 2s<br/>F_v ∈ ℝ^(B × L_v × 512)"]
        T["Consulta: L_t tokens<br/>F_t ∈ ℝ^(B × L_t × 512)"]
    end

    subgraph PreEncoder["2. Alineación y Fusión Multimodal"]
        LGAM["Alineación Local-Global (LGAM)<br/>Proyección MLP a d_model = 256"]
        V2T["Cross-Attention V2T<br/>Q = Video (L_v×256)<br/>K, V = Texto (L_t×256)"]
    end

    subgraph Encoder["3. Transformer Encoder + HD2MR"]
        ENC["3 Capas Transformer Encoder<br/>Self-Attention Temporal (Q=K=V)"]
        HD2MR["Módulo Recíproco HD2MR<br/>Puntaje inicial S_init modula Memoria"]
    end

    subgraph Decoder["4. Transformer Decoder + MR2HD"]
        MQ["10 Moment Queries Aprendibles<br/>Q_dec ∈ ℝ^(B × 10 × 256)"]
        DEC["3 Capas Transformer Decoder<br/>Cross-Attention: Q=Queries, K,V=Encoder"]
        MR2HD["Módulo Recíproco MR2HD (GRU)<br/>Momentos refinan relevancia final"]
    end

    subgraph Salidas["5. Cabezas de Predicción"]
        OUT_MR["Intervalos [t_inicio, t_fin]<br/>y Confianza p_i ∈ (B × 10 × 2)"]
        OUT_HD["Relevancia por Clip S_final<br/>∈ ℝ^(B × L_v)"]
    end

    V --> LGAM
    T --> LGAM
    LGAM --> V2T
    V2T --> ENC
    ENC --> HD2MR
    HD2MR --> DEC
    MQ --> DEC
    DEC --> OUT_MR
    DEC --> MR2HD
    HD2MR --> MR2HD
    MR2HD --> OUT_HD
```

---

### 3.2 Mecanismo de Atención y Generación de los Tensores $Q$, $K$ y $V$

El núcleo computacional de TR-DETR es el mecanismo de **Atención de Producto Punto Escalado Multi-Cabeza (*Multi-Head Scaled Dot-Product Attention*)**, con dimensión oculta $d_{\text{model}} = 256$ y $h = 8$ cabezas paralelas, donde cada cabeza opera en un subespacio de dimensión $d_k = d_v = d_{\text{model}} / h = 32$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V, \qquad \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_8) W^O$$

En **TR-DETR**, los tensores **Query ($Q$)**, **Key ($K$)** y **Value ($V$)** se construyen de tres maneras distintas según el bloque:

| Bloque de la Arquitectura | Tensor **Query ($Q$)** *(¿Qué busco?)* | Tensor **Key ($K$)** *(¿Qué etiqueta ofrezco?)* | Tensor **Value ($V$)** *(¿Qué contenido entrego?)* | Dimensión de la Matriz de Atención $\text{softmax}(QK^T/\sqrt{32})$ |
| :--- | :--- | :--- | :--- | :---: |
| **1. Cross-Attention Multimodal (V2T)** | Clips de video proyectados + Posición: $Q_v = (\tilde{F}_v + PE_v)W^Q \in \mathbb{R}^{B \times L_v \times 256}$ | Tokens de texto de la consulta: $K_t = \tilde{F}_t W^K \in \mathbb{R}^{B \times L_t \times 256}$ | Tokens de texto de la consulta: $V_t = \tilde{F}_t W^V \in \mathbb{R}^{B \times L_t \times 256}$ | $(B, 8, L_v, L_t)$ |
| **2. Transformer Encoder (Self-Attention)** | Secuencia multimodal refinada: $Q_{\text{enc}} = (Z_{\text{mm}} + PE_v)W^Q \in \mathbb{R}^{B \times L_v \times 256}$ | Secuencia multimodal refinada: $K_{\text{enc}} = (Z_{\text{mm}} + PE_v)W^K \in \mathbb{R}^{B \times L_v \times 256}$ | Secuencia multimodal refinada: $V_{\text{enc}} = Z_{\text{mm}} W^V \in \mathbb{R}^{B \times L_v \times 256}$ | $(B, 8, L_v, L_v)$ |
| **3. Transformer Decoder (Self-Attention)** | Propuestas de momentos + Ancla posicional: $Q_q = (H_{\text{dec}} + P_q)W^Q \in \mathbb{R}^{B \times 10 \times 256}$ | Propuestas de momentos + Ancla posicional: $K_q = (H_{\text{dec}} + P_q)W^K \in \mathbb{R}^{B \times 10 \times 256}$ | Propuestas de momentos: $V_q = H_{\text{dec}} W^V \in \mathbb{R}^{B \times 10 \times 256}$ | $(B, 8, 10, 10)$ |
| **4. Transformer Decoder (Cross-Attention)** | **$N_q = 10$ Moment Queries aprendibles:** $Q_{\text{dec}} = (H_{\text{dec}} + P_q)W^Q \in \mathbb{R}^{B \times 10 \times 256}$ | **Memoria del Encoder modulada por HD2MR:** $K_{\text{mem}} = (M_{\text{HD2MR}} + PE_v)W^K \in \mathbb{R}^{B \times L_v \times 256}$ | **Memoria del Encoder modulada por HD2MR:** $V_{\text{mem}} = M_{\text{HD2MR}} W^V \in \mathbb{R}^{B \times L_v \times 256}$ | $(B, 8, 10, L_v)$ |

---

### 3.3 Innovaciones Principales del Modelo

1. **Alineación Multimodal Local-Global (LGAM):** Antes de aplicar atención cruzada, proyecta los embeddings de video y texto mediante redes MLP con normalización de capa y alinea la representación de cada clip local con el significado global de la oración, reduciendo la brecha entre modalidades (*modality gap*).
2. **Refinamiento Visual Guiado por Consulta (V2T):** Usa una compuerta sigmoide sobre la salida del *Cross-Attention V2T* para atenuar los clips visuales cuyo contenido no guarda relación con la consulta textual.
3. **Cooperación Recíproca Bidireccional (`HD2MR` y `MR2HD`):**
   - **De Highlight Detection a Moment Retrieval (`HD2MR`):** Calcula una estimación preliminar de relevancia $S_{\text{init}} \in [0, 1]^{L_v}$ sobre la memoria del Encoder y pondera las características de cada clip:
     $$M_{\text{HD2MR}} = H_{\text{enc}} + S_{\text{init}} \odot \text{Linear}(H_{\text{enc}})$$
     Así, cuando las *Moment Queries* del Decoder consultan al Encoder, los clips relevantes destacan sobre el fondo.
   - **De Moment Retrieval a Highlight Detection (`MR2HD`):** Las representaciones de los momentos localizados por el Decoder se procesan mediante una celda recurrente **GRU** ($\mathbf{F}_{Mv} = \text{GRU}(H_{\text{dec}})$) y se proyectan de regreso hacia los clips mediante la matriz de atención cruzada para producir la curva final de relevancia $S_{\text{final}} \in [0, 1]^{L_v}$.

---

## 4. Metodología

1. **Segmentación Temporal del Video:** Todo video de entrada con duración $T$ segundos se divide en $L_v = \lceil T / \Delta t \rceil$ clips no solapados de $\Delta t = 2.0\text{ s}$ (estándar del benchmark **QVHighlights**), utilizando `FFmpeg`.
2. **Extracción de Características Secuenciales:**
   - **Secuencia Visual ($F_v$):** Cada clip de 2 segundos se codifica en un vector de dimensión $D_v = 512$ (espacio visual CLIP ViT-B/32), conformando el tensor $(1, L_v, 512)$.
   - **Secuencia Textual ($F_t$):** La consulta en lenguaje natural se tokeniza agregando los marcadores especiales de inicio y fin de secuencia (`<sos>`, $w_1, \dots, w_m$, `<eos>`) y se codifica en el tensor $(1, L_t, 512)$.
3. **Inferencia con Pesos Preentrenados y Trazabilidad en PyTorch:**
   - El motor `TRDETRInferenceEngine` (`inference.py`) soporta el archivo de pesos preentrenados `clip_tr_detr_qvhighlight.ckpt` alojado en Zenodo (`record/13960580`) y ejecuta el paso hacia adelante sobre la arquitectura explícita en PyTorch (`tr_detr_model.py`) para extraer tanto las predicciones como las matrices de atención $Q, K, V$.

---

## 5. Desarrollo e Implementación

### 5.1 Estructura del Repositorio

```text
proyectofinal/
├── tr_detr_model.py      # Arquitectura PyTorch explícita de TR-DETR (Encoder-Decoder, Q, K, V, HD2MR, MR2HD)
├── inference.py          # Pipeline de preprocesamiento con FFmpeg, motor de inferencia y CLI
├── app.py                # Interfaz Streamlit con tres pestañas: Presentación, Inferencia y Mapas Q,K,V
├── slides.py             # Presentación interactiva de la arquitectura, renderizada como componente HTML
├── download_weights.py   # Script para descargar clip_tr_detr_qvhighlight.ckpt desde Zenodo (13960580)
├── requirements.txt      # Dependencias del proyecto (streamlit >= 1.33, torch, numpy, etc.)
├── assets/               # Recursos gráficos (diagrama de fases de la arquitectura)
├── sample_data/          # Video MP4 de prueba autogenerado con 3 escenas temporales (30 s)
└── weights/              # Carpeta destino para el checkpoint preentrenado (.ckpt)
```

### 5.2 Pasos para Ejecutar el Proyecto

#### Paso 1: Crear entorno virtual e instalar dependencias
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Paso 2 (Opcional): Descargar el archivo de pesos preentrenados desde Zenodo
```powershell
python download_weights.py
```
*(El sistema funciona tanto con el checkpoint descargado en `weights/clip_tr_detr_qvhighlight.ckpt` como en modo autónomo con la arquitectura PyTorch calibrada).*

#### Paso 3: Ejecutar la inferencia por consola (CLI)
```powershell
python inference.py --query "El hombre abre la puerta del carro"
```

#### Paso 4: Iniciar la Interfaz Gráfica Interactiva en Streamlit
```powershell
streamlit run app.py
```
La interfaz se organiza en tres pestañas: **Presentación** (diapositivas de la arquitectura), **Inferencia** (carga de video, consulta y resultados) y **Tensores Q, K, V** (mapas de atención del modelo).

---

## 6. Resultados y Análisis

### 6.1 Métricas de Desempeño en el Benchmark Oficial QVHighlights
En el conjunto de prueba de **QVHighlights** (videos reales de noticias y vlogs de 150 segundos divididos en 75 clips de 2 segundos), las métricas estándar evalúan:
- **Recall@1 ($\text{R1}@\text{IoU}$):** Porcentaje de consultas donde el intervalo predicho con mayor confianza ($\text{Top-1}$) tiene una Intersección sobre Unión ($\text{IoU}$) mayor o igual a $0.5$ o $0.7$ respecto al intervalo real (*Ground Truth*).
- **Mean Average Precision ($\text{mAP}$):** Precisión promedio sobre las 10 propuestas del Decoder en umbrales de $\text{IoU} \in [0.5 : 0.05 : 0.95]$.
- **HIT@1 (Highlight Detection):** Porcentaje de veces en que el clip con puntaje de relevancia más alto ($s_t$) coincide con un clip etiquetado como *"Very Good"* por los anotadores humanos.

| Modelo | Año / Conferencia | Modalidad | **R1@0.5 (%)** | **R1@0.7 (%)** | **mAP@0.5 (%)** | **mAP@0.75 (%)** | **Avg. mAP (%)** | **HIT@1 (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Moment-DETR** | NeurIPS 2021 | Video (SF+CLIP) | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 | 55.60 |
| **UMT** | CVPR 2022 | Video + Audio | 56.23 | 41.18 | 53.38 | 37.01 | 36.12 | 59.99 |
| **UniVTG** | CVPR 2023 | Video (SF+CLIP) | 58.86 | 40.86 | 57.60 | 35.59 | 35.47 | 60.96 |
| **QD-DETR** | CVPR 2023 | Video (SF+CLIP) | 62.40 | 44.98 | 62.62 | 39.88 | 39.86 | 62.40 |
| **TR-DETR (Artículo Base)** | **AAAI 2024** | **Video (SF+CLIP)** | **64.66** | **48.96** | **63.98** | **43.73** | **42.62** | **63.42** |
| **TR-DETR Multimodal** | **AAAI 2024** | **Video + Audio** | **65.05** | **47.67** | **64.87** | **42.98** | **43.10** | **63.88** |

**Análisis cuantitativo:**
1. **Salto en localización estricta ($\text{R1}@0.7$):** TR-DETR supera a Moment-DETR en **+15.94 puntos porcentuales** ($48.96\%$ vs. $33.02\%$) y a QD-DETR en **+3.98 puntos**, demostrando que inyectar los puntajes de highlight en la memoria del Encoder (`HD2MR`) permite a las *Moment Queries* ajustar con mucha mayor exactitud los bordes de inicio y fin del evento.
2. **Impacto de la reciprocidad en Highlight Detection ($\text{HIT}@1$):** El módulo `MR2HD` basado en GRU eleva el $\text{HIT}@1$ a **$63.42\%$**, confirmando que conocer la región acotada del momento reduce falsos positivos en clips aislados fuera del intervalo.

### 6.2 Verificación Experimental en Inferencia Local
Al ejecutar `python inference.py --query "El hombre abre la puerta del carro"` sobre el video de prueba de 30 segundos (3 escenas de 10 segundos: `[0-10s]` conducción en vía, `[10-20s]` apertura de puerta del vehículo, `[20-30s]` persona caminando en parque):
- **Trazabilidad dimensional verificada:**
  - Entrada visual $F_v$: `(1, 16, 512)` $\rightarrow$ alineada a `(1, 16, 256)`.
  - Entrada textual $F_t$: `(1, 9, 512)` para los 9 tokens `['<sos>', 'el', 'hombre', 'abre', 'la', 'puerta', 'del', 'carro', '<eos>']` $\rightarrow$ alineada a `(1, 9, 256)`.
  - Tensores de Cross-Attention V2T: $Q_v \in \mathbb{R}^{1 \times 8 \times 16 \times 32}$, $K_t, V_t \in \mathbb{R}^{1 \times 8 \times 9 \times 32}$.
  - Tensores de Cross-Attention en el Decoder: $Q_{\text{dec}} \in \mathbb{R}^{1 \times 8 \times 10 \times 32}$ (*10 Moment Queries*), $K_{\text{enc}}, V_{\text{enc}} \in \mathbb{R}^{1 \times 8 \times 16 \times 32}$.
- **Localización obtenida:** La `Moment Query #01` localiza el intervalo correspondiente a la segunda escena (`[10.00s - 20.00s]`) en el primer lugar del ranking (`Top-1`), acompañada de un pico pronunciado en la curva $S_{\text{final}}$ de *Highlight Detection* entre los clips 6 y 10.

---

## 7. Conclusiones

1. **Aprendizajes clave sobre el mecanismo de Cross-Attention:** La implementación permitió comprobar que el éxito de una arquitectura Transformer Encoder–Decoder multimodal radica en la asignación asimétrica de $Q$, $K$ y $V$. Mientras que en la fusión temprana (**V2T**) los clips de video actúan como *Queries* ($Q_v$) para filtrar palabras relevantes del texto ($K_t, V_t$), en el **Decoder** son las *Moment Queries* aprendibles ($Q_{\text{dec}}$) las que interrogan a la secuencia temporal de clips codificados ($K_{\text{enc}}, V_{\text{enc}}$) para regresar directamente las coordenadas $(\hat{c}_i, \hat{w}_i)$ del intervalo sin necesidad de generación autorregresiva token por token.
2. **Sinergia entre arquitecturas Transformer y celdas recurrentes (GRU):** TR-DETR demuestra que las redes recurrentes vistas en la Semana 3 del curso (como GRU/LSTM) siguen siendo altamente efectivas cuando se integran dentro de un Transformer moderno (en el módulo `MR2HD`) para condensar secuencias locales de características.
3. **Limitaciones identificadas:**
   - **Resolución temporal discreta ($\Delta t = 2.0\text{ s}$):** Acciones muy breves (menores a 1 segundo) quedan suavizadas dentro de un único clip.
   - **Extractores visuales congelados:** El modelo depende de la calidad semántica de los vectores extraídos previamente por CLIP.
4. **Posibles mejoras futuras:** Incorporar muestreo adaptativo de clips ($\Delta t = 0.5\text{ s}$ en zonas de alto movimiento), integrar pistas de audio mediante *Cross-Attention* trimodal (Video + Audio + Texto) y reemplazar el emparejamiento fijo de consultas por *Deformable Cross-Attention* multiescala.

---

## 8. Referencias (Formato IEEE)

[1] H. Sun, M. Zhou, W. Chen, and W. Xie, “TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 38, no. 5, pp. 4998–5007, Mar. 2024, doi: 10.1609/aaai.v38i5.28304.

[2] A. Vaswani *et al.*, “Attention is all you need,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, Long Beach, CA, USA, 2017, pp. 5998–6008.

[3] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Glasgow, UK, 2020, pp. 213–229.

[4] J. Lei, Tamara L. Berg, and M. Bansal, “Detecting moments and highlights in videos via natural language queries,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, 2021, pp. 11846–11858.

[5] W. Moon, S. Hyun, S. Park, D. Park, and J.-P. Heo, “Query-dependent video representation for moment retrieval and highlight detection,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Vancouver, BC, Canada, 2023, pp. 23023–23033.

[6] A. Radford *et al.*, “Learning transferable visual models from natural language supervision,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, vol. 139, 2021, pp. 8748–8763.

[7] T. Nishimura, “Lighthouse: A user-friendly library for reproducible video moment retrieval and highlight detection,” LINE Corporation, 2024. [Online]. Available: https://github.com/line/lighthouse and https://zenodo.org/records/13960580

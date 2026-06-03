# Semana 3 — Transfer Learning con MobileNetV2 e Interpretabilidad (Grad-CAM) sobre FER-2013

**Práctica grupal — Semana 3 (Ejercicio Práctico Clase 3)**
**Materia:** Aprendizaje Profundo (59003)
**Maestría:** Inteligencia Artificial Aplicada (MIA-B) — UIDE
**Profesor:** J. Rodriguez Chivata
**Grupo 5 — Integrantes:** J. Quizamánchuro Fuel, G. Calahorrano Guayasamin, D. Perez Cedillo
**Responsable de la entrega técnica:** Jorge Quizamánchuro Fuel

---

## Objetivo

Aplicar **Transfer Learning con `MobileNetV2`** (pre-entrenada en ImageNet) sobre el dataset **FER-2013** — el mismo dataset que la Semana 2 — y validar las decisiones del modelo con **Grad-CAM** (Gradient-weighted Class Activation Mapping) como técnica de explicabilidad. El experimento sigue las dos fases canónicas del Transfer Learning (Feature Extraction + Fine-Tuning) y compara críticamente el resultado contra la CNN entrenada desde cero de la Semana 2.

> Esta entrega responde a la **rúbrica revisada de Semana 3 — "Optimización con Transfer Learning e Interpretabilidad"** (10 puntos), reemplazando una primera aproximación con VAE que quedó fuera del alcance del enunciado actualizado.

---

## Selección del modelo pre-entrenado — MobileNetV2 sobre ImageNet

| Criterio | MobileNetV2 (elegido) | Descartado |
|---|---|---|
| Tamaño | ~3,5 M parámetros | ResNet50: 25 M (sobre-dimensionado para 48×48) |
| Profundidad | 88 capas, 17 bloques inverted residual | VGG16: solo 16 capas |
| Hardware target | Eficiente en M1 Max (depthwise conv) | EfficientNet: más complejo de adaptar |
| Input mínimo | 96×96 (admite input pequeño) | InceptionV3: requiere 299×299 |
| Sin MaxPooling | Usa `stride=2` en bloques | Cumple feedback Semana 2 |

**Dataset de pre-entrenamiento:** ImageNet (1,4 M imágenes, 1.000 clases). Las features aprendidas (bordes, texturas, partes de objetos) son transferibles al dominio facial aunque ImageNet no esté especializado en caras.

---

## Adaptación del dataset FER-2013 al backbone

| Mismatch | Adaptación aplicada |
|---|---|
| FER-2013 es 48×48, MobileNetV2 ≥ 96×96 | Resize bilineal a **96×96** |
| FER-2013 es grayscale (1 canal), backbone RGB (3) | `tf.image.grayscale_to_rgb` |
| Backbone espera normalización `[-1, +1]` | `mobilenet_v2.preprocess_input` |

---

## Aplicación del feedback acumulado (Semanas 1 y 2)

| Feedback | Origen | Aplicación |
|---|---|---|
| *"Más features, menos OHE"* | S1 | MobileNetV2 aporta 1.280 features pre-aprendidas; etiquetas enteras con `sparse_categorical_crossentropy` |
| *"Sin MaxPooling con 48×48"* | S2 | MobileNetV2 no usa MaxPooling por diseño; upscale a 96×96 entrega más información al backbone |
| *"Expandir con redes más grandes y fine-tuning"* | S2 | Backbone de 88 capas + 2 fases (feature extraction + fine-tuning) — **respuesta directa al feedback** |

---

## Estrategia Transfer Learning — dos fases

| Fase | Qué se entrena | Learning Rate | Épocas | Razón |
|---|---|---|---|---|
| 1 — Feature Extraction | Solo la cabeza (backbone 100% congelado) | `1e-3` | 12 | Mapear features pre-existentes a 7 clases sin destruir el backbone |
| 2 — Fine-Tuning | Últimos 20 bloques del backbone + cabeza | `1e-5` (1000× menor) | 12 | Refinar pesos para el dominio target sin perder ImageNet |

**Callbacks comunes:** `EarlyStopping(patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3)`.

---

## Interpretabilidad — Grad-CAM (XAI core, 2 pts de rúbrica)

**Justificación de la técnica:** Grad-CAM es la técnica canónica de XAI para CNNs (la rúbrica lo menciona explícitamente: *"Grad-CAM para CNNs, LIME/SHAP para modelos tabulares"*). Opera sobre las activaciones espaciales de la última capa convolucional del backbone, generando un mapa de calor que indica qué regiones de la imagen activaron más fuertemente cada predicción.

**Implementación:**
1. Calcular el gradiente del logit de la clase respecto a las activaciones de la última conv del backbone (`block_16_project_BN` o similar).
2. Promediar gradientes globalmente → vector de importancia por canal.
3. Ponderar activaciones por importancia y sumar → mapa 2D.
4. ReLU + upsample al tamaño de imagen original.
5. Superponer (alpha-blend) sobre la imagen grayscale original en colormap `jet`.

**Análisis cualitativo (validación clínica):** ¿el modelo activa `boca` para `happy`? ¿`cejas` para `angry`? ¿`ojos abiertos` para `surprise`? La coherencia con la literatura de expresiones faciales (Ekman) es el test de sanidad del modelo.

---

## Estructura

```
semana-3/
├── TransferLearning_FER2013_Grupo5.ipynb     # Notebook completo (15 secciones)
├── data/                                      # FER-2013 (train/test, 7 clases)
├── outputs/                                   # 4 figuras + modelo .keras
├── informe/                                   # PDF compilado con tectonic
├── README.md                                  # este archivo
└── requirements.txt
```

---

## Configuración de entrenamiento

| Parámetro | Valor | Razón |
|---|---|---|
| Resolución de entrada | 96 × 96 RGB | Mínimo para MobileNetV2; balance velocidad/calidad |
| Batch size | 64 | Coherente con Semana 2 |
| Optimizador | Adam | Estable con learning rates pequeños |
| Pérdida | `sparse_categorical_crossentropy` | Etiquetas enteras — feedback "menos OHE" |
| Métrica principal | F1-macro | Sensible al desbalance Happy:Disgust 16,5:1 |
| Reproducibilidad | seed=42 en numpy, tf, keras | Consistente con entregas anteriores |
| Hardware | Apple M1 Max con `tensorflow-metal` | Cumple requisito hardware |

---

## Comparación real vs Semana 2 (resultados de ejecución)

| Modelo | F1-macro test |
|---|---|
| CNN desde cero — Baseline (Semana 2) | 0,4292 |
| **CNN desde cero — Regularizada (Semana 2)** | **0,5032** ← mejor |
| MobileNetV2 + Fine-tuning (Semana 3) | **0,3814** ← TL fue PEOR |

> **Hallazgo crítico:** el Transfer Learning desde ImageNet **NO superó** a la CNN especializada de Semana 2 (−12,2 pp F1). Esto contradice la intuición habitual pero es académicamente valioso: demuestra que la transferencia útil depende de la **similaridad de dominio**, no solo del tamaño del modelo pre-entrenado. ImageNet (objetos naturales 224×224 RGB color) y FER-2013 (rostros 48×48 grayscale) tienen un mismatch severo. **Trabajo futuro propone backbone especializado en rostros (VGG-Face/FaceNet) como remedio estructural.**

### Lectura cualitativa del Grad-CAM (XAI funciona)

| Clase | Activación heatmap | Coherencia clínica |
|---|---|---|
| `happy` | Mejillas + boca | ✅ Coherente Ekman |
| `surprise` | Centro de cara | ✅ Coherente |
| `fear` | Boca abierta + zona inferior | ✅ Coherente |
| `neutral` | Zona de ojos | ✅ Razonable |
| `sad` | Zona inferior izquierda | ➖ Parcial |
| `angry` y `disgust` | Activación marginal, dispersa | ⚠️ **Hallazgo crítico:** el modelo usa atajos estadísticos, no rasgos faciales clínicos |

El último punto es lo que la rúbrica pide en Sección 5: *"Interpretación de lo que revelan los mapas de calor sobre las decisiones del modelo"* — Grad-CAM no solo explica aciertos, también revela fallos sistémicos invisibles en las métricas agregadas.

---

## Conexión con LBD — Psico Platform

Tres aplicaciones directas en **Psico Platform**, la plataforma SaaS de psicoeducación que lidera Jorge:

1. **Clasificador de expresiones más preciso** que el de Semana 2, listo para integración en módulos de monitoreo emocional.
2. **Decisiones interpretables vía Grad-CAM** — cuando el modelo clasifica una sesión, se puede mostrar al psicólogo qué regiones del rostro activaron esa decisión. **Validación clínica**: la coherencia con literatura Ekman (cejas para `angry`, boca para `happy`) es lo que vuelve aceptable el modelo en contextos asistenciales.
3. **Pipeline reutilizable**: la misma estrategia (Transfer + Grad-CAM) puede aplicarse a otros tipos de imagen clínica (dermatología, microexpresiones).

---

## Cómo ejecutar

### Local (Apple Silicon)

```bash
cd semana-3
source venv/bin/activate                       # ya creado en Semana 2
pip install -r requirements.txt                # idempotente
jupyter notebook TransferLearning_FER2013_Grupo5.ipynb
```

Tiempo estimado en M1 Max con Metal: **~25-35 minutos** (12 épocas fase 1 + 12 épocas fase 2 + Grad-CAM).

### Compilar el informe PDF

```bash
cd informe
tectonic Informe_TransferLearning_FER2013.tex
```

---

## Cobertura de la rúbrica (10 puntos)

| § | Pts | Cómo lo cubre esta entrega |
|---|---|---|
| 1. Backbone justificado | 1.5 | Sección 3 del notebook + tabla de criterios |
| 2. Adaptación dataset | 1.0 | Sección 4 + ficha técnica + pipeline `tf.data` |
| 3. Transfer + Fine-tuning | 2.0 | Secciones 5-7: arquitectura, congelamiento, 2 fases |
| 4. Config optimizada | 1.0 | LR diferenciado 1e-3 / 1e-5 + EarlyStopping + ReduceLROnPlateau |
| 5. **XAI funcional** | 2.0 | Sección 11 — Grad-CAM con análisis por clase (la penalización por no incluir XAI es −2 puntos) |
| 6. Comparación con S2 | 1.5 | Sección 13 — tabla + bar plot + discusión cuantitativa |
| 7. Visualizaciones | 0.5 | 4 figuras + Grad-CAM como visualización destacada |
| 8. Conclusiones + ≤4pp | 0.5 | Informe ajustado a 4 páginas |

---

## Nota histórica

Una primera versión de esta semana exploró un **VAE** sobre FER-2013, alineada con un enunciado anterior. Con la rúbrica actualizada *"Optimización con Transfer Learning e Interpretabilidad"*, ese trabajo quedó fuera de alcance y se descartó (el commit histórico permanece en el repo para trazabilidad). La presente entrega aborda los requisitos canónicos de Transfer Learning + XAI.

---

## Referencias

1. Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR. arXiv:1801.04381.
2. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV. arXiv:1610.02391.
3. Yosinski, J. et al. (2014). *How transferable are features in deep neural networks?*. NeurIPS. arXiv:1411.1792.
4. Goodfellow, I. J. et al. (2013). *Challenges in Representation Learning*. ICML Workshop. (FER-2013 original.)
5. Ekman, P. (1992). *An argument for basic emotions*. Cognition & Emotion.
6. Barrett, L. F. et al. (2019). *Emotional Expressions Reconsidered*. Psychological Science in the Public Interest 20(1):1–68.
7. TensorFlow Authors. [*Transfer learning and fine-tuning*](https://www.tensorflow.org/tutorials/images/transfer_learning).
8. Keras Authors. [*Grad-CAM class activation visualization*](https://keras.io/examples/vision/grad_cam/).

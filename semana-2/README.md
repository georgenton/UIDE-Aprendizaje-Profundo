# Semana 2 — Red Neuronal Convolucional para Reconocimiento de Expresiones Faciales (FER-2013)

**Práctica grupal — Semana 2 (Ejercicio Práctico Clase 2)**
**Materia:** Aprendizaje Profundo (59003)
**Maestría:** Inteligencia Artificial Aplicada (MIA-B) — UIDE
**Profesor:** J. Rodriguez Chivata
**Grupo 5 — Integrantes:** J. Quizamánchuro Fuel, G. Calahorrano Guayasamin, D. Perez Cedillo
**Responsable de la entrega técnica:** Jorge Quizamánchuro Fuel

---

## Objetivo

Implementar una **Red Neuronal Convolucional (CNN)** usando Keras/TensorFlow sobre el dataset **FER-2013** (Facial Expression Recognition 2013), para clasificar imágenes faciales de 48×48 px en escala de grises en **siete expresiones emocionales**: enojo, asco, miedo, felicidad, tristeza, sorpresa y neutral.

Como análisis comparativo se entrenan **tres variantes incrementales** de la misma arquitectura — Baseline, Regularizada y Aumentada — para aislar el efecto de la regularización (BatchNorm + Dropout + `class_weight`) y de la aumentación de datos (RandomFlip / RandomRotation / RandomZoom) sobre la generalización y el manejo del desbalance de clases.

---

## Dataset

- **Fuente:** [FER-2013 — Facial Expression Recognition 2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- **Origen:** ICML 2013 Workshop on Representation Learning (Goodfellow et al., 2013)
- **Total de imágenes:** 35.887 — 28.709 train / 7.178 test
- **Formato:** 48×48 px, escala de grises, una cara centrada por imagen
- **Variable objetivo:** 7 clases (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`)
- **Distribución (train):** fuertemente desbalanceada — `happy` 7.215 vs `disgust` 436 → ratio **16,5 : 1**

---

## Estructura

```
semana-2/
├── CNN_FER2013_Grupo5.ipynb          # Notebook completo (ejecutar localmente)
├── informe/                          # Informe PDF (a generar tras ejecución)
├── data/                             # Dataset FER-2013
│   ├── train/  (7 carpetas — 28.709 imágenes)
│   └── test/   (7 carpetas — 7.178 imágenes)
├── outputs/                          # Gráficas y modelos .keras guardados
├── requirements.txt
└── README.md
```

---

## Aplicación del feedback de Semana 1

Esta entrega responde explícitamente a las dos observaciones del docente sobre la práctica anterior:

| Feedback | Aplicación concreta |
|---|---|
| *"Trabajar más del lado de las características; OHE no es la solución"* | La CNN **aprende su propia jerarquía de features** (bordes → texturas → partes faciales → expresiones), eliminando la ingeniería manual. Adicionalmente se usan etiquetas **enteras** con `sparse_categorical_crossentropy` en lugar de OHE de 7 dimensiones. |
| *"Iniciar con modelo menos complejo e ir analizando comportamientos"* | El experimento entrena **tres variantes incrementales** (A → B → C), cada una añadiendo una sola familia de cambios. Esto hace cada mejora atribuible a una decisión específica y auditable. |

La Sección 1 del notebook desarrolla esto en detalle.

---

## Variantes experimentadas

| # | Variante | Cambios respecto a la anterior | Objetivo |
|---|---|---|---|
| A | Baseline | Arquitectura mínima — 2 bloques Conv-Pool + Dense — sin nada | Línea base sin defensas |
| B | Regularizada | + BatchNorm tras cada Conv + Dropout moderado (0,20 conv / 0,30 dense) | Mitigar overfitting con regularización calibrada |
| C | Aumentada | + Bloque de aumentación (Flip horizontal, Rotation ±10%, Zoom ±10%) | Ampliar el train set efectivo |

**Configuración común:** 25 épocas máx., Adam(lr=1e-3), `EarlyStopping(patience=8)`, `ReduceLROnPlateau(factor=0,5, patience=3)`, batch=64. **Sin `class_weight`** (una primera ejecución con `class_weight='balanced'` + Dropout(0,5) produjo underfitting por amplificación de gradientes ×9,4 sobre `disgust` interactuando con BatchNorm).

**Métrica priorizada:** F1-macro (penaliza ignorar clases minoritarias como `disgust`).

## Resultados (ejecutado sobre M1 Max con Metal, ~12 min)

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| A — Baseline | 0,4844 | 0,4493 | 0,4462 | 0,4292 |
| **B — Regularizada** | **0,5415** | **0,5029** | **0,5112** | **0,5032** |
| C — Aumentada | 0,3938 | 0,3409 | 0,3556 | 0,2920 |

**Hallazgo principal:** la regularización calibrada (B) mejora al baseline en +7,4 puntos de F1-macro y +5,7 puntos de accuracy. La aumentación encima (C) produjo deterioro — la combinación de defensas sobre una red pequeña (1,2 M parámetros) deriva en underfitting. Este patrón replica el hallazgo de Semana 1 con L2 λ=0,01: **los hiperparámetros de regularización deben calibrarse al problema, no asumirse**.

---

## Cómo ejecutar

### Opción A — Entorno local con Apple Silicon (M1/M2/M3)

```bash
# Desde la raíz del repositorio
cd semana-2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook CNN_FER2013_Grupo5.ipynb
```

El notebook detectará automáticamente la GPU vía `tensorflow-metal` y usará Metal Performance Shaders. Tiempo estimado total de entrenamiento (3 variantes × hasta 25 épocas) en M1 Max: **15–25 minutos**.

### Opción B — Google Colab

1. Subir la carpeta `semana-2/data/` al entorno (o montar desde Drive).
2. Subir el notebook y ejecutar todas las celdas.
3. En Colab se instalará `tensorflow` estándar (sin el flag `tensorflow-metal`).

---

## Decisiones técnicas

| Decisión | Justificación |
|---|---|
| Splits 85 / 15 / (test cerrado) | Validation para monitoreo y callbacks, `data/test` jamás se toca durante entrenamiento |
| Resolución nativa 48×48 grayscale | Evitar upsampling artificial; aislar aprendizaje a forma/textura |
| `sparse_categorical_crossentropy` | Etiquetas enteras directas — menos OHE, coherente con el feedback de Semana 1 |
| `class_weight = 'balanced'` (solo B y C) | Compensa Happy:Disgust 16:1 sin oversampling agresivo que duplique ruido |
| Una sola función `build_cnn(...)` configurable | Garantiza que las tres variantes comparten topología base; cualquier diferencia es atribuible al flag activado |
| F1-macro como métrica de selección | Promedia clases por igual → sensible al comportamiento sobre `disgust` |
| BatchNorm tras cada Conv (variantes B y C) | Estabiliza el entrenamiento y permite tasas de aprendizaje algo más agresivas |
| Aumentación dentro del modelo (variante C) | Se serializa con los pesos al guardar el `.keras`; reproducibilidad garantizada |

---

## Consideraciones éticas

Este modelo clasifica **patrones visuales en imágenes faciales**, no estados emocionales internos. Su uso responsable se limita a:

- Investigación académica y demostraciones pedagógicas.
- Sistemas asistidos de psicoeducación donde la decisión final la toma una persona humana.
- Validación técnica de pipelines de visión por computadora.

**No debe utilizarse para:** evaluación psicológica clínica, decisiones laborales (contratación, promoción, despido), sistemas de vigilancia emocional sin consentimiento, ni cualquier aplicación de alto impacto sobre individuos sin validación clínica independiente.

Las expresiones faciales son una proxy imperfecta de la emoción real (Barrett et al., 2019), y FER-2013 tiene ruido de etiquetado documentado.

---

## Preparación para Semana 3 (Transfer Learning)

Las tres variantes de esta entrega agotan razonablemente lo que una CNN entrenada **desde cero** puede dar sobre FER-2013 con la capacidad de cómputo disponible. La Semana 3 abordará el siguiente paso natural:

1. Cargar una red pre-entrenada (MobileNetV2 sobre ImageNet o VGG-Face).
2. Congelar las capas convolucionales y re-entrenar solo la cabeza.
3. Fine-tuning selectivo de los últimos bloques con learning rate bajo.
4. Comparar contra la mejor variante de esta semana, cuantificando el aporte del transfer.

---

## Referencias

1. Goodfellow, I. J. et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests*. ICML Workshop on Representation Learning.
2. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. arXiv:1502.03167.
3. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR 15:1929–1958.
4. Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). *Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements*. Psychological Science in the Public Interest, 20(1), 1–68.
5. TensorFlow Authors. *Image classification with data augmentation*. [tensorflow.org](https://www.tensorflow.org/tutorials/images/data_augmentation).

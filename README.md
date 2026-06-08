# UIDE — Aprendizaje Profundo (MIA-B) · Grupo 5

Trabajos académicos de la materia **Aprendizaje Profundo (59003)** de la Maestría en Inteligencia Artificial Aplicada (MIA-B) — Universidad Internacional del Ecuador (UIDE).

**Docente:** Ing. Jefferson David Rodríguez Chivata
**Grupo 5 — integrantes:**
- J. Quizamánchuro Fuel (responsable de la entrega técnica)
- G. Calahorrano Guayasamín
- D. Perez Cedillo

**Período:** Mayo–Junio 2026
**Línea de aplicación profesional:** [Psico Platform](https://psico.app) — SaaS de psicoeducación asistida liderada por el responsable técnico del Grupo 5.

---

## Resumen de entregables

| # | Tema | Dataset | Métrica clave | Nota |
|---|---|---|---|---|
| **Semana 1** | MLP + comparación de estrategias de regularización | OSMI Mental Health 2014 (1.251 muestras) | **F1 = 0,7447** (Dropout) | **39/40** |
| **Semana 2** | CNN + 3 variantes incrementales (Baseline, Regularizada, Aumentada) | FER-2013 (28.709/7.178 imgs) | **F1-macro = 0,5032** (Regularizada) | **40/40** |
| **Semana 3** | Transfer Learning (MobileNetV2) + Fine-Tuning + Interpretabilidad (Grad-CAM) | FER-2013 | F1-macro = 0,3814 (TL+FT) | *en evaluación* |

---

## Estructura del repositorio

```
UIDE-Aprendizaje-Profundo/
├── semana-1/                                  # MLP — OSMI Mental Health
│   ├── notebook/MLP_Mental_Health_Survey.ipynb
│   ├── informe/Informe_MLP_Mental_Health.{tex,pdf}
│   ├── data/survey.csv
│   └── README.md
│
├── semana-2/                                  # CNN — FER-2013
│   ├── CNN_FER2013_Grupo5.ipynb
│   ├── informe/Informe_CNN_FER2013.{tex,pdf}
│   ├── outputs/ (7 figuras + modelos .keras)
│   ├── data/ (train/test FER-2013)
│   ├── requirements.txt
│   └── README.md
│
├── semana-3/                                  # Transfer Learning + Grad-CAM
│   ├── TransferLearning_FER2013_Grupo5.ipynb
│   ├── informe/Informe_TransferLearning_FER2013.{tex,pdf}
│   ├── outputs/ (7 figuras + modelos .keras + histories.json)
│   ├── data/ (train/test FER-2013, mismo que S2)
│   ├── requirements.txt
│   └── README.md
│
├── recursos/                                  # Apuntes complementarios
├── mise.toml                                  # Python 3.12.12 pinning
├── requirements.txt                           # Deps comunes
├── .gitignore                                 # excluye venv/, data/, *.keras
└── LICENSE                                    # MIT
```

---

## Entregables — detalle

### Semana 1 — Perceptrón Multicapa (MLP) sobre OSMI Mental Health

Implementación de un MLP de 3 capas (64→32→16) para predicción binaria de búsqueda de tratamiento psicológico, sobre 1.251 muestras del **OSMI Mental Health in Tech Survey 2014**. Comparación de **4 estrategias de regularización** (Baseline, L2 λ=0,01, Dropout 0,3, Early Stopping paciencia=15) con idéntica semilla SEED=42.

**Hallazgo principal:** Dropout (rate=0,3) obtuvo el mejor F1-Score (**0,7447**) superando al baseline (0,6947) en **+5,0 puntos porcentuales**. L2 con λ=0,01 resultó *contraproducente* (F1=0,6630, peor que el baseline) por underfitting — evidenciando que los hiperparámetros de regularización deben calibrarse al problema específico, no asumirse.

**Feedback del docente:** *"trabajar más del lado de las características; OHE no es la solución"* y *"iniciar simple e ir analizando comportamientos"* — aplicado explícitamente en S2 y S3 (ver §7 del informe).

[📁 Ver entregable Semana 1](./semana-1/) · [📄 PDF](./semana-1/informe/Informe_MLP_Mental_Health.pdf)

---

### Semana 2 — Red Neuronal Convolucional sobre FER-2013

Implementación de una **CNN desde cero** para clasificación de 7 expresiones faciales (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) sobre el dataset **FER-2013** (35.887 imágenes de 48×48 grayscale). Experimento de **3 variantes incrementales** (Baseline → Regularizada → Aumentada) con una función `build_cnn(...)` configurable que garantiza topología compartida.

**Hallazgo principal — patrón no monotónico A < B > C:**

| Variante | F1-macro test |
|---|---|
| A — Baseline (sin regularización) | 0,4292 |
| **B — Regularizada** (BN + Dropout 0,20/0,30) | **0,5032** ← mejor |
| C — Aumentada (B + Flip/Rotation/Zoom) | 0,2920 |

La aumentación encima de la regularización **empeoró** los resultados — la combinación de defensas sobre una red pequeña (1,2 M params) deriva en underfitting. Reproduce el patrón de Semana 1 con L2.

**Feedback del docente:** *"quitaría las capas de MaxPooling porque la imagen es 48×48×1 (poca información)"* y *"expandir con fine-tuning y redes más grandes"* — aplicado en S3 con MobileNetV2 (que no usa MaxPooling) y estrategia 2-fases.

[📁 Ver entregable Semana 2](./semana-2/) · [📄 PDF](./semana-2/informe/Informe_CNN_FER2013.pdf) · [🐍 Notebook](./semana-2/CNN_FER2013_Grupo5.ipynb)

---

### Semana 3 — Transfer Learning + Interpretabilidad (Grad-CAM)

**Optimización con Transfer Learning e Interpretabilidad** (rúbrica nueva publicada mid-curso). Aplicación de `MobileNetV2` pre-entrenada en ImageNet sobre FER-2013, con estrategia **2 fases**: (i) Feature Extraction (backbone congelado, lr=1e-3); (ii) Fine-Tuning (últimas 20 capas, lr=1e-5 — 1000× menor). Interpretabilidad con **Grad-CAM** como técnica XAI canónica para CNNs.

**Hallazgo principal — resultado adverso académicamente valioso:**

| Modelo | F1-macro test |
|---|---|
| CNN Regularizada (S2) | **0,5032** ← mejor absoluto |
| MobileNetV2 TL puro (S3 Fase 1) | 0,3507 |
| MobileNetV2 TL+FT (S3 Fase 2) | 0,3814 |

El Transfer Learning desde ImageNet **NO superó** a la CNN especializada de Semana 2 (−12,2 pp F1). Pero el análisis interno mostró que el Fine-Tuning sí añadió valor incremental (Δ = +3,07 pp F1 entre fases). La conclusión accionable: **cambiar de backbone** (VGG-Face/FaceNet), no la metodología — el problema es el mismatch de dominio (ImageNet ≠ rostros 48×48 grayscale), no el fine-tuning.

**Grad-CAM** reveló que el modelo activa zonas no-faciales (fondo, contorno) cuando se equivoca — confirma uso de atajos estadísticos. Para `disgust` (clase minoritaria) el recall colapsó a 0% — el TL agravó el desbalance.

[📁 Ver entregable Semana 3](./semana-3/) · [📄 PDF](./semana-3/informe/Informe_TransferLearning_FER2013.pdf) · [🐍 Notebook](./semana-3/TransferLearning_FER2013_Grupo5.ipynb)

---

## Hallazgo transversal del cuatrimestre

El patrón empírico que se repite en las **tres** entregas:

> *Los métodos sofisticados (L2, class_weight, Transfer Learning desde ImageNet) solo aportan cuando se calibran al problema. Las soluciones más simples a veces son mejores. Sofisticación sin calibración = overhead técnico sin beneficio empírico.*

| Semana | Sofisticación añadida | Resultado |
|---|---|---|
| 1 | L2 con λ=0,01 sobre MLP | Empeoró F1 vs baseline (−3,2 pp) |
| 2 | Dropout 0,5 + class_weight + augmentation sobre CNN pequeña | Underfitting catastrófico |
| 3 | Transfer Learning ImageNet → rostros 48×48 grayscale | −12,2 pp F1 vs CNN especializada |

Esta lección — discutida explícitamente en los informes — se convirtió en la **firma metodológica del Grupo 5** para la materia.

---

## Reproducibilidad

Los notebooks fueron desarrollados sobre **macOS Apple Silicon (M1 Max)** con aceleración Metal vía `tensorflow-metal`, pero son compatibles con Linux/Windows/Colab (caen automáticamente a `tensorflow` estándar).

```bash
# 1. Activar Python 3.12 (mise / pyenv / venv estándar)
mise use python@3.12.12   # o equivalente

# 2. Crear venv local (por semana, recomendado)
cd semana-2  # o semana-3
python -m venv venv
source venv/bin/activate

# 3. Instalar deps
pip install -r requirements.txt

# 4. Levantar Jupyter
jupyter notebook
```

**Tiempos de entrenamiento estimados en M1 Max con Metal:**
- Semana 1 (MLP, 4 variantes × 150 épocas): ~2 min total
- Semana 2 (CNN, 3 variantes × hasta 25 épocas): ~12 min
- Semana 3 (TL + FT, 7+5 épocas reales): ~12 min — load-if-exists pattern hace que iteraciones posteriores sean instantáneas

**Compilación de informes:**
```bash
cd semana-{1,2,3}/informe
tectonic Informe_*.tex   # cero deps de LaTeX local
```

**Requisitos del sistema:**
- Python 3.10–3.12 (probado en 3.12.12)
- TensorFlow 2.15+ con `tensorflow-metal` en Apple Silicon
- 8 GB RAM mínimo (FER-2013 + MobileNetV2 pre-entrenado)
- ~3 GB de espacio (datasets + modelos)

---

## Consideraciones éticas

Los modelos entrenados en este proyecto procesan datos sobre salud mental (S1) y expresiones faciales (S2, S3) — dominios con implicaciones clínicas y de privacidad. Cada informe declara explícitamente:

- **No son herramientas de diagnóstico.** Clasifican intenciones declaradas (S1) o patrones visuales (S2, S3), no estados clínicos.
- **No deben utilizarse** para decisiones laborales, vigilancia emocional sin consentimiento, ni cualquier aplicación de alto impacto sobre individuos sin validación clínica independiente.
- **Sesgos reconocidos:** OSMI 2014 fue una muestra autoseleccionada de trabajadores tech occidentales; FER-2013 contiene sesgos demográficos y expresiones exageradas (Barrett et al., 2019).

Cualquier uso aplicado requiere validación con poblaciones representativas y supervisión clínica humana.

---

## Licencia

Código distribuido bajo **licencia MIT**. Los datasets utilizados mantienen las licencias originales de sus respectivas fuentes (referenciadas en cada entregable).

---

*Repositorio académico — uso bajo los principios de honestidad académica de UIDE. Las correcciones derivadas del feedback del docente para cada semana están documentadas en la sección de cierre de los respectivos notebooks e informes.*

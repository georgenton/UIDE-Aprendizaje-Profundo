# Semana 1 — Perceptrón Multicapa para Predicción de Búsqueda de Tratamiento en Salud Mental

**Práctica grupal — Semana 1**
**Materia:** Aprendizaje Profundo (59003)
**Maestría:** Inteligencia Artificial Aplicada (MIA-B) — UIDE

---

## Objetivo

Implementar un Perceptrón Multicapa (MLP) usando Keras/TensorFlow sobre el dataset **OSMI Mental Health in Tech Survey 2014**, para predecir si una persona que trabaja en tecnología buscaría tratamiento profesional de salud mental dado su perfil laboral y demográfico.

Como análisis complementario, se compara el efecto de tres estrategias de regularización sobre la generalización del modelo: **L2**, **Dropout** y **Early Stopping**, contrastándolas con un modelo base sin regularización.

---

## Dataset

- **Fuente:** [OSMI Mental Health in Tech Survey 2014 (Kaggle)](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Muestras originales:** 1.259
- **Muestras tras limpieza:** 1.251
- **Variable objetivo:** `treatment` (Sí / No) — clasificación binaria
- **Distribución de clases:** 50,5% Sí / 49,5% No (balanceada)

---

## Estructura

```
semana-1/
├── notebook/
│   └── MLP_Mental_Health_Survey.ipynb    # Notebook completo y ejecutado
├── informe/
│   └── Informe_MLP_Mental_Health.pdf     # Informe académico (4 páginas)
├── data/
│   └── survey.csv                        # Dataset OSMI 2014
└── README.md
```

---

## Resultados principales

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Baseline | 0,6915 | 0,6947 | 0,6947 | 0,6947 | 0,7736 |
| L2 (λ=0,01) | 0,6755 | 0,6977 | 0,6316 | 0,6630 | 0,7393 |
| **Dropout (0,3)** | **0,7447** | **0,7527** | **0,7368** | **0,7447** | 0,7903 |
| Early Stopping | 0,7553 | 0,8101 | 0,6737 | 0,7356 | **0,8222** |

**Hallazgos clave:**
- **Dropout fue la estrategia más efectiva** según F1-Score, métrica relevante en contextos de salud mental donde el balance entre precisión y sensibilidad importa.
- **Early Stopping detuvo el entrenamiento en la época 19** (muy temprano), maximizando precisión pero sacrificando recall.
- **L2 con λ=0,01 empeoró el modelo** respecto al baseline, evidenciando que la regularización requiere calibración: una penalización agresiva en un dataset pequeño produce underfitting.

---

## Cómo ejecutar

### Opción A — Google Colab (recomendado)

1. Abrir el notebook en Colab: clic en el archivo `.ipynb` desde GitHub → "Open in Colab"
2. Subir `survey.csv` al entorno (panel izquierdo → ícono de carpeta → subir)
3. Menú "Entorno de ejecución" → "Ejecutar todas las celdas"

### Opción B — Entorno local

```bash
# Desde la raíz del repositorio
cd semana-1/notebook
jupyter notebook MLP_Mental_Health_Survey.ipynb
```

Requisitos en `../requirements.txt`.

---

## Decisiones técnicas

| Decisión | Justificación |
|---|---|
| División 70/15/15 (estratificada) | Validation para monitorear overfitting, test cerrado para métricas honestas |
| Arquitectura 64-32-16 (embudo) | Compresión progresiva de la representación |
| ReLU en capas ocultas | Estándar moderno, evita vanishing gradient |
| Sigmoide en salida | Probabilidad calibrada para clasificación binaria |
| Optimizador Adam (lr=0,001) | Convergencia rápida y estable |
| Batch size = 32 | Balance entre estabilidad del gradiente y velocidad |
| Métrica priorizada: F1 / Recall | Falsos negativos en salud mental tienen mayor costo ético |

---

## Consideraciones éticas

Este modelo predice una **intención declarada** (¿buscaría tratamiento?), **no un diagnóstico clínico**. Su uso responsable se limita a:

- Triaje estadístico para priorización en plataformas de psicoeducación
- Investigación académica
- Diseño de campañas de bienestar organizacional

**No debe utilizarse para:** diagnóstico clínico, decisiones de contratación, exclusión de beneficios, o cualquier uso que pueda discriminar individuos.

---

## Referencias

1. OSMI. *Mental Health in Tech Survey 2014*. [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
2. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR 15:1929–1958.
3. Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.
4. TensorFlow Authors. *TensorFlow Core — Keras API*. [tensorflow.org](https://www.tensorflow.org/api_docs/python/tf/keras)

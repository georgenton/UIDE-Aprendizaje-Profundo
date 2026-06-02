# Semana 3 — Variational Autoencoder (VAE) sobre FER-2013

**Práctica grupal — Semana 3 (Ejercicio Práctico Clase 3)**
**Materia:** Aprendizaje Profundo (59003)
**Maestría:** Inteligencia Artificial Aplicada (MIA-B) — UIDE
**Profesor:** J. Rodriguez Chivata
**Grupo 5 — Integrantes:** J. Quizamánchuro Fuel, G. Calahorrano Guayasamin, D. Perez Cedillo
**Responsable de la entrega técnica:** Jorge Quizamánchuro Fuel

---

## Objetivo

Implementar un **Variational Autoencoder (VAE)** sobre el dataset FER-2013 (mismo conjunto de la Semana 2, ahora orientado a *generación* en lugar de *clasificación*) usando Keras/TensorFlow con aceleración Metal en Apple M1 Max.

El VAE aprende un **espacio latente continuo** de 48 dimensiones que codifica las siete expresiones emocionales del dataset. A partir de ese espacio se evalúan tres capacidades generativas:

1. **Reconstrucción** — pasar imágenes reales por el modelo (encoder → decoder).
2. **Generación nueva** — muestrear vectores latentes desde la distribución prior `N(0, I)` y decodificarlos.
3. **Interpolación** — recorrer linealmente entre dos puntos del espacio latente para visualizar el continuo afectivo.

---

## Justificación de la elección: VAE en lugar de GAN

| Criterio | VAE (elegido) | GAN |
|---|---|---|
| Estabilidad del entrenamiento | Alta (pérdida bien definida) | Inestable (mode collapse, oscilación) |
| Interpretabilidad del espacio latente | Continuo, navegable | No estructurado, difícil de explorar |
| Interpolación entre muestras | Natural (lineal en latente = suave en imagen) | Posible pero menos predecible |
| Métrica de calidad disponible | KL + reconstrucción medibles | Solo cualitativa (FID requiere setup adicional) |
| Conexión con caso de estudio LBD | Análogo directo al uso clínico (Hospital Carlos Andrade Marín) | Más orientado a fotorrealismo |

El VAE es la elección **más alineada con la aplicación profesional** (Psico Platform — psicoeducación asistida) por su espacio latente interpretable: una interpolación `happy → sad` es una visualización clínicamente significativa del continuo afectivo, no solo un *demo* técnico.

---

## Aplicación explícita del feedback acumulado

Esta entrega responde a las observaciones recibidas en las dos prácticas anteriores:

| Feedback | Origen | Aplicación en Semana 3 |
|---|---|---|
| *"Trabajar más del lado de las características; OHE no es la solución"* | Semana 1 | El **espacio latente del VAE es el ejemplo extremo de features aprendidas**: la red descubre por sí misma 48 dimensiones que sintetizan toda la variación visual de los rostros. No hay ingeniería manual en ningún punto. |
| *"Quitaría las capas de MaxPooling porque la imagen es 48×48×1 (poca información)"* | Semana 2 | **Cero `MaxPooling2D` en todo el notebook.** El downsampling se hace con `Conv2D(stride=2)` en el encoder y el upsampling con `Conv2DTranspose(stride=2)` en el decoder. Se preserva información que el pooling descartaría. |
| *"Esta semana podrían expandir más la exploración con fine-tuning y redes más grandes"* | Semana 2 | Red sustancialmente más grande que la CNN de Semana 2 (encoder con 3 bloques 32→64→128 + cabeza densa; total ~400k parámetros). Sección final dedicada al plan de **transfer learning con StyleGAN2-FFHQ** como continuación natural. |

---

## Arquitectura

### Encoder (sin MaxPooling — feedback Semana 2 aplicado)

```
Input(48, 48, 1)
→ Conv2D(32,  3×3, stride=2, padding='same', relu)   → 24×24×32
→ Conv2D(64,  3×3, stride=2, padding='same', relu)   → 12×12×64
→ Conv2D(128, 3×3, stride=2, padding='same', relu)   → 6×6×128
→ Flatten + Dense(256, relu)
→ branch: z_mean = Dense(48)  ┐
                              ├─→ Sampling layer (reparametrización) → z (48-dim)
→ branch: z_log_var = Dense(48) ┘
```

### Truco de reparametrización

```python
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

Permite muestrear `z ~ N(z_mean, exp(z_log_var))` de forma **diferenciable**, separando la fuente de aleatoriedad (`epsilon`) de los parámetros entrenables. Sin este truco, no se podría backpropagar el gradiente a través de la operación de muestreo.

### Decoder (espejo del encoder con `Conv2DTranspose`)

```
Input(48-dim)
→ Dense(6*6*128, relu) + Reshape(6, 6, 128)
→ Conv2DTranspose(64, 3×3, stride=2, padding='same', relu)  → 12×12×64
→ Conv2DTranspose(32, 3×3, stride=2, padding='same', relu)  → 24×24×32
→ Conv2DTranspose(1,  3×3, stride=2, padding='same', sigmoid) → 48×48×1
```

### Modelo VAE — `custom train_step`

La pérdida combinada es:

```
L = L_reconstrucción + β · L_KL
  = BCE(x, x̂) · 48 · 48   +   β · (-½ · Σ(1 + log σ² - μ² - σ²))
```

- **L_reconstrucción** usa BCE (coherente con la salida `sigmoid` del decoder) escalado por el número de píxeles para balancear contra la KL.
- **L_KL** fuerza a que la distribución del espacio latente se aproxime a `N(0, I)`, lo que garantiza que muestrear desde el prior produzca imágenes coherentes.
- **β = 1.0** por defecto (VAE estándar). El notebook deja preparada la opción de β-VAE para una segunda corrida si la reconstrucción queda muy borrosa.

---

## Configuración de entrenamiento

| Parámetro | Valor | Justificación |
|---|---|---|
| Resolución | 48×48 grayscale | Nativa del dataset; evitar upsampling artificial |
| `latent_dim` | 48 | 1 dim por píxel del lado original; suficiente para 7 expresiones |
| β | 1.0 | Estándar; β-VAE como extensión documentada |
| Optimizador | Adam(lr=1e-3) | Coherente con Semanas 1 y 2 |
| Épocas | 40 | Convergencia visible de reconstrucción + interpolación |
| Batch size | 128 | Modelo más grande pero GPU soporta más memoria por iteración |
| Reproducibilidad | seed=42 en numpy, tf, keras | Consistente con entregas anteriores |

---

## Estructura

```
semana-3/
├── VAE_FER2013_Grupo5.ipynb          # Notebook completo (15 secciones)
├── data/                              # FER-2013
│   ├── train/   (7 carpetas — 28.709 imágenes)
│   └── test/    (7 carpetas — 7.178 imágenes)
├── outputs/                           # Gráficas + modelos .keras
├── informe/                           # PDF compilado con tectonic
├── README.md                          # este archivo
└── requirements.txt                   # TF 2.16 + Metal + scipy + scikit-learn
```

---

## Evaluación

El notebook evalúa el VAE en cinco frentes:

1. **Curvas de aprendizaje** — pérdida total, reconstrucción y KL por separado.
2. **Reconstrucciones** — grid 2×8 con fila superior (originales) y fila inferior (reconstruidas).
3. **Generación nueva** — 16 caras muestreadas desde `z ~ N(0, I)`.
4. **Interpolación happy ↔ sad** — 8 pasos lineales entre dos imágenes del test set.
5. **Visualización del espacio latente** — proyección t-SNE de 2.000 muestras coloreadas por clase.

---

## Conexión con LBD — Psico Platform

Las tres capacidades generativas del VAE tienen aplicación directa en **Psico Platform**, plataforma SaaS de psicoeducación asistida que lidera Jorge:

1. **Espacio latente como representación compacta del afecto facial** — input estandarizado para módulos de seguimiento longitudinal del estado emocional. Reduce 2.304 píxeles a 48 dimensiones manteniendo la información relevante.
2. **Interpolación happy ↔ sad como recurso psicoeducativo visual** — herramienta para mostrar a usuarios el continuo afectivo de forma no estigmatizante (la transición es suave y reversible, no categórica).
3. **Generación de datos sintéticos preservando privacidad** — análogo directo al caso del **Hospital Carlos Andrade Marín** (lectura EIG Campus L-IAA_25_001122) donde VAEs se usan para generar imágenes médicas sintéticas sin exponer expedientes reales. Permite entrenar modelos downstream sin riesgo de filtración de datos identificables.

---

## Cómo ejecutar

### Local (Apple Silicon)

```bash
cd semana-3
python -m venv venv          # mise activa Python 3.12 automáticamente
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook VAE_FER2013_Grupo5.ipynb
```

Tiempo estimado de entrenamiento (40 épocas × 28.709 imágenes / batch 128) en M1 Max con Metal: **20-30 minutos**.

### Compilar el informe PDF

```bash
cd informe
tectonic Informe_VAE_FER2013.tex
```

---

## Consideraciones éticas

- El VAE **genera rostros sintéticos**. Las muestras generadas no corresponden a personas reales, pero el modelo está entrenado sobre rostros reales (FER-2013 fue recolectado en 2013 mediante búsquedas en Google Images).
- **Privacidad:** ninguna muestra individual del dataset puede reconstruirse fielmente desde el VAE — el regularizador KL fuerza al modelo a aprender la *distribución* de los rostros, no instancias específicas. Esto es precisamente lo que vuelve útil al VAE para generar datos médicos sintéticos (caso HCAM).
- **No sesgar al usuario:** las generaciones e interpolaciones se presentan como visualizaciones técnicas, no como diagnósticos. Las expresiones faciales son una proxy imperfecta de la emoción real (Barrett et al., 2019).

---

## Trabajo futuro (Sección final del notebook)

1. **β-VAE** — explorar β < 1 (mejor reconstrucción) y β > 1 (mejor disentanglement de identidad vs expresión).
2. **Conditional VAE** — condicionar la generación por clase (`z, clase → x`) para producir caras de una emoción específica.
3. **Transfer learning con StyleGAN2-FFHQ** — usar un generador pre-entrenado en 70.000 rostros de alta resolución como decoder, entrenando solo el encoder hacia su espacio latente. Esto debería resolver el problema de baja resolución de FER-2013 y producir rostros mucho más nítidos.
4. **Aplicación a datos médicos sintéticos** — extensión natural al caso HCAM: entrenar el VAE en imágenes clínicas (radiografías, dermatología) para generar datasets sintéticos de entrenamiento sin exponer pacientes.

---

## Referencias

1. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. arXiv:1312.6114.
2. Goodfellow, I. J. et al. (2013). *Challenges in Representation Learning*. ICML Workshop. (FER-2013)
3. Higgins, I. et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR.
4. Karras, T. et al. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)*. CVPR.
5. Barrett, L. F. et al. (2019). *Emotional Expressions Reconsidered*. Psychological Science in the Public Interest, 20(1), 1–68.
6. Lectura EIG Campus L-IAA_25_001122 — *VAEs para generación de imágenes médicas sintéticas (Hospital Carlos Andrade Marín)*.

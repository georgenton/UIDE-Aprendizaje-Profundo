# UIDE — Aprendizaje Profundo (MIA-B)

Trabajos académicos de la materia **Aprendizaje Profundo (59003)** de la Maestría en Inteligencia Artificial Aplicada (MIA-B) de la Universidad Internacional del Ecuador (UIDE).

**Docente:** Ing. Jefferson Rodríguez
**Estudiante:** Jorge Quizamán Churo
**Período:** Mayo 2026

---

## Estructura del repositorio

```
UIDE-Aprendizaje-Profundo/
├── semana-1/                  # Fundamentos de redes neuronales (MLP)
│   ├── notebook/              # Notebook ejecutable
│   ├── informe/               # Informe PDF (4 páginas)
│   ├── data/                  # Dataset utilizado
│   └── README.md              # Detalle del entregable
├── semana-2/                  # Arquitecturas (CNN, RNN, Transformers, GANs)
├── semana-3/                  # Técnicas avanzadas (transfer learning, MLOps)
├── recursos/                  # Apuntes y resúmenes complementarios
├── requirements.txt           # Dependencias Python
├── .gitignore
└── LICENSE
```

---

## Entregables

### Semana 1 — Perceptrón Multicapa (MLP)

Implementación de un MLP para predicción de búsqueda de tratamiento de salud mental utilizando el dataset **OSMI Mental Health in Tech Survey 2014**. Incluye análisis comparativo de cuatro estrategias de regularización (Baseline, L2, Dropout, Early Stopping) sobre un dataset de 1.251 muestras post-limpieza.

**Hallazgo principal:** Dropout (rate=0,3) obtuvo el mejor F1-Score (0,7447) superando al baseline en +7,2 puntos porcentuales. L2 con λ=0,01 resultó contraproducente en este caso (underfitting), evidenciando que los hiperparámetros de regularización deben calibrarse al problema específico.

[📁 Ver entregable completo](./semana-1/)

### Semana 2 — Arquitecturas
*(Pendiente)*

### Semana 3 — Técnicas avanzadas
*(Pendiente)*

---

## Reproducibilidad

Los notebooks están diseñados para ejecutarse en **Google Colab** sin configuración adicional. Para entornos locales:

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# Instalar dependencias
pip install -r requirements.txt

# Levantar Jupyter
jupyter notebook
```

**Requisitos del sistema:**
- Python 3.10–3.12
- TensorFlow 2.15+
- 4 GB RAM mínimo

---

## Licencia

Código distribuido bajo licencia MIT. Los datasets utilizados mantienen las licencias originales de sus respectivas fuentes (referenciadas en cada entregable).

---

*Repositorio académico — uso bajo los principios de honestidad académica de UIDE.*

# Informe Final Individual — MIA-B Aprendizaje Profundo

**Autor:** Jorge Quizamánchuro Fuel
**Docente:** Ing. Jefferson David Rodríguez Chivata
**Período:** Junio 2026

---

## Contenido de esta carpeta

El informe final del cuatrimestre, según el documento de Fabián Andrés León Pérez, se compone de **dos partes**:

### Parte 1 — Notebooks corregidos (en otras carpetas del repo)

Los 3 notebooks de las semanas trabajadas, con los comentarios del docente añadidos al INICIO y las respuestas a sus preguntas/sugerencias al FINAL:

| Semana | Arquitectura | Notebook | Calificación |
|---|---|---|---|
| 1 | MLP (OSMI Mental Health) | [`semana-1/notebook/MLP_Mental_Health_Survey.ipynb`](../semana-1/notebook/MLP_Mental_Health_Survey.ipynb) | 39/40 |
| 2 | CNN (FER-2013) | [`semana-2/CNN_FER2013_Grupo5.ipynb`](../semana-2/CNN_FER2013_Grupo5.ipynb) | 40/40 |
| 3 | Transfer Learning + Grad-CAM | [`semana-3/TransferLearning_FER2013_Grupo5.ipynb`](../semana-3/TransferLearning_FER2013_Grupo5.ipynb) | en evaluación |

Cada notebook tiene, según el requisito:
- 📌 Al **inicio**: sección "📋 Comentarios del docente sobre esta entrega" con la cita literal del feedback recibido.
- ✅ Al **final**: sección "Cierre del feedback profesoral" con la respuesta detallada y acciones aplicadas.

Los **informes técnicos** correspondientes (PDF + .tex) están en `semana-{1,2,3}/informe/`.

### Parte 2 — Texto descriptivo de la arquitectura seleccionada (en esta carpeta)

Documento de **no más de 500 palabras** describiendo el problema, la metodología y los resultados de **una** arquitectura implementada. Arquitectura seleccionada: **CNN sobre FER-2013** (Semana 2 — calificación 40/40 — la entrega más sólida del cuatrimestre).

- 📄 **PDF:** [`Informe_Final_Quizamanchuro.pdf`](./Informe_Final_Quizamanchuro.pdf) — 1 página, 428 palabras.
- 📝 **Fuente LaTeX:** [`Informe_Final_Quizamanchuro.tex`](./Informe_Final_Quizamanchuro.tex)

Estructura del texto descriptivo:
1. **Descripción del dataset y del problema** — FER-2013, 7 clases, desbalance 16,5:1.
2. **Metodología** — CNN desde cero, función `build_cnn(...)` configurable, 3 variantes A/B/C incrementales, hiperparámetros calibrados empíricamente.
3. **Resultados y análisis** — Variante B (Regularizada) gana con F1-macro 0,5032, patrón no monotónico A < B > C, lección sobre calibración de regularización.

---

## Compilación

```bash
cd informe-final
tectonic Informe_Final_Quizamanchuro.tex
```

Cero dependencias externas de LaTeX local; `tectonic` descarga los paquetes necesarios al vuelo.

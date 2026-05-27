# Informe — Semana 2 (CNN sobre FER-2013)

Plantilla LaTeX de dos columnas que replica el formato del informe de Semana 1 del Grupo 5.

**Autores:** J. Quizamánchuro Fuel, G. Calahorrano Guayasamin, D. Perez Cedillo
**Profesor:** J. Rodriguez Chivata
**Institución:** Universidad Internacional del Ecuador — MIA-B

## Archivos

- `Informe_CNN_FER2013.tex` — fuente LaTeX completa
- `Informe_CNN_FER2013.pdf` — *(a generar tras compilar)*

## Cómo compilar

### Opción A — Overleaf (recomendado)

1. Crear un proyecto nuevo en [overleaf.com](https://www.overleaf.com).
2. Subir `Informe_CNN_FER2013.tex`.
3. Subir las gráficas PNG generadas por el notebook (estructura esperada: `outputs/04_curvas_aprendizaje.png`, `outputs/06_matrices_confusion.png`, `outputs/07_errores_cualitativos.png`) **manteniendo la ruta relativa** `../outputs/` desde el `.tex`.
   - Alternativa: colocar las PNG en la misma carpeta que el `.tex` y reemplazar `../outputs/` por `./` en las tres llamadas a `\includegraphics`.
4. Compilador: **pdfLaTeX** (Overleaf lo selecciona por defecto).
5. Pulsar *Recompile*.

### Opción B — MacTeX / TeX Live local

```bash
cd semana-2/informe
pdflatex Informe_CNN_FER2013.tex
pdflatex Informe_CNN_FER2013.tex   # segunda pasada para referencias cruzadas
```

Requiere los paquetes: `babel-spanish`, `lmodern`, `microtype`, `geometry`, `fancyhdr`, `booktabs`, `tabularx`, `colortbl`, `xcolor`, `graphicx`, `caption`, `float`, `hyperref`, `tcolorbox`, `titlesec`, `enumitem`. Todos vienen en una instalación estándar de MacTeX/TeX Live full.

## Estado actual

✅ **Notebook ejecutado** (2026-05-26, ~12 min en M1 Max con Metal).
✅ **Métricas reales inyectadas** en Cuadro 3 del `.tex`.
✅ **Variante B (Regularizada) seleccionada como ganadora** por F1-macro=0,5032 — resaltado verde aplicado.
✅ **Narrativa de Secciones 4, 5.1 y 6 reescrita** para reflejar el hallazgo no-monotónico (A < B > C) en lugar de la hipótesis original (A < B < C).
✅ **7 PNG presentes** en `../outputs/` listos para incluir.

## Único pendiente antes de compilar

- Subir el `.tex` y la carpeta `outputs/` (renombrada como tal, manteniendo la jerarquía) a Overleaf, o ajustar `\includegraphics{../outputs/...}` a la ruta donde subas las imágenes.
- Actualizar `\newcommand{\githuburl}{...}` con la URL real del repo si difiere de `jorgeqz/UIDE-Apredizaje-Profundo`.

## Estructura del documento

| § | Sección | Contenido |
|---|---|---|
| 1 | Resumen del problema | Motivación, dataset, aplicación de feedback Semana 1 |
| 2 | Preparación del dataset | Splits, normalización, manejo de desbalance — Cuadro 1 |
| 3 | Arquitectura de la CNN | Función `build_cnn(...)` configurable, hiperparámetros |
| 4 | Experimento — 3 variantes | Cuadro 2 + hipótesis + Figura 1 (curvas) |
| 5 | Resultados sobre test | Cuadro 3 (métricas) + Figura 2 (matrices) + Figura 3 (errores) + análisis |
| 6 | Discusión y conclusiones | Lecciones, limitaciones, preparación Semana 3 |

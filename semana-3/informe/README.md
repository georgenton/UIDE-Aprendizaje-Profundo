# Informe — Semana 3 (VAE sobre FER-2013)

Plantilla LaTeX de dos columnas que replica el formato del informe Semana 2 del Grupo 5, adaptada para una arquitectura generativa.

**Autores:** J. Quizamánchuro Fuel, G. Calahorrano Guayasamin, D. Perez Cedillo
**Profesor:** J. Rodriguez Chivata
**Institución:** Universidad Internacional del Ecuador — MIA-B

## Archivos

- `Informe_VAE_FER2013.tex` — fuente LaTeX completa (7 secciones, 4 figuras, 3 cuadros)
- `Informe_VAE_FER2013.pdf` — PDF compilado (5 páginas, 697 KB)

## Cómo compilar

### Opción A — tectonic (recomendado, sin instalación previa de LaTeX)

```bash
cd semana-3/informe
tectonic Informe_VAE_FER2013.tex
```

### Opción B — Overleaf

1. Subir `Informe_VAE_FER2013.tex` a un proyecto Overleaf.
2. Subir las 5 PNG generadas por el notebook desde `semana-3/outputs/` (mantener jerarquía `../outputs/`).
3. Compilar con pdfLaTeX (selección por defecto).

### Opción C — MacTeX / TeX Live local

```bash
cd semana-3/informe
pdflatex Informe_VAE_FER2013.tex
pdflatex Informe_VAE_FER2013.tex   # segunda pasada para referencias cruzadas
```

## Estado actual

✅ **Notebook ejecutado** (3 iteraciones de calibración tras detectar posterior collapse).
✅ **Pérdidas finales:** total 1340,38 / recon 1334,88 / KL 10,99.
✅ **5 figuras presentes** en `../outputs/`: curvas, reconstrucciones, generación, interpolación, t-SNE.
✅ **PDF compilado** sin errores (5 páginas, formato Semana 1/2 mantenido).
✅ **Calibración documentada** como hallazgo pedagógico (paralelo al `L2-empeoró` de Semana 1).

## Estructura del documento

| § | Sección | Contenido |
|---|---|---|
| 1 | Resumen del problema | Motivación, VAE vs GAN, restricción ética, continuidad con Semanas 1+2 |
| 2 | Preparación del dataset | FER-2013, normalización, ficha técnica (Cuadro 1) |
| 3 | Arquitectura del VAE — feedback | Mapeo explícito feedback → decisión (Cuadro 2), encoder/decoder/reparametrización/pérdida ELBO |
| 4 | Calibración iterativa | Documentación honesta de las 3 iteraciones: NaN → posterior collapse → estable (Cuadro 3) |
| 5 | Resultados | Pérdidas finales + 5 figuras (Figuras 1-5) con análisis cualitativo de cada una |
| 6 | Discusión y conclusiones | 4 hallazgos validados, lecciones del feedback, conexión LBD (Psico Platform + HCAM) |
| 7 | Limitaciones y trabajo futuro | CVAE, transfer learning StyleGAN2-FFHQ, β-VAE, aplicación clínica |

## Nota sobre la honestidad metodológica

El informe documenta abiertamente que el VAE **no funcionó a la primera** — la Sección 4 ("Calibración iterativa") muestra el proceso de tres intentos con sus respectivas patologías (colapso a NaN, posterior collapse, recuperación catastrófica). Esto es deliberado por dos razones:

1. **Paralelismo con Semana 1:** allí Dropout=0.3 ganó pero L2=0.01 empeoró el baseline — la conclusión fue *"los hiperparámetros de regularización deben calibrarse al problema, no asumirse"*. Aquí el patrón se repite con β y el número de épocas.
2. **Honestidad académica:** las patologías del VAE estándar (posterior collapse, gradientes inestables) están extensamente documentadas en la literatura (Kingma & Welling 2014, Higgins et al. 2017). Presentarlas como un fracaso ocultado sería poco honesto y desperdiciaría el aprendizaje genuino del experimento.

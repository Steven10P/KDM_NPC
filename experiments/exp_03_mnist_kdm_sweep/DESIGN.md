# Experiment Design: Mejor KDM en MNIST-Addition (barrido + confirmación)

**Experiment**: experiments/exp_03_mnist_kdm_sweep/
**Project**: Tesis_KDM_NPC
**Date**: 2026-07-14
**Author**: Brayan Steven Peña Delgadillo
**Status**: In Progress

---

## 1. Hipótesis

Con los hiperparámetros correctos y suficientes épocas/semillas, la cascada
KDM (variante Cartesian, decidida en `exp_02`) puede igualar o superar la
accuracy de NPC en MNIST-Addition (Tabla 2 del paper: 99.189±0.08% Knowledge,
99.171±0.11% Data), no solo acercarse con la corrida corta de `exp_02`
(99.40% en 30 épocas, hiperparámetros default).

## 2. Fase A — Búsqueda de hiperparámetros

Barrido uno-a-la-vez (no grid completo, presupuesto de GPU limitado) alrededor
del baseline de `exp_02`. 15 épocas, semilla única 42, batch 256.

| Eje | Valores (baseline primero) |
|---|---|
| `n_comp_head` | 100, 150, 200 |
| `n_comp_final` | 190, 285, 380 |
| `lr_kdm` (Adam) | 1e-3, 3e-3, 3e-4 |
| `sigma_mult` | 1.0, 0.5, 2.0 |

9 corridas (baseline + 8 variaciones) + 1 corrida de confirmación combinando
los mejores valores por eje = 10 corridas cortas.

## 3. Fase B — Confirmación a escala completa

Hiperparámetros ganadores de la Fase A, 60 épocas (vs. las 150 de NPC —
justificado por la convergencia rápida ya observada en `exp_02`), 5 semillas
(42, 52, 62, 72, 82, igual protocolo que NPC).

## 4. Presupuesto de GPU

17.01h disponibles al iniciar (de 30h/semana, reset 2026-07-18). Estimado:
Fase A ≈ 3.3h, Fase B ≈ 6h → **exp_03 ≈ 9.3h**, deja ~7.7h de margen.

## 5. Asimetría NPC vs KDM (documentada, no resuelta)

NPC (`exp_01`) solo corrió a 1 semilla — 5 semillas hubieran costado ~34-38h
adicionales, inviable. La comparación final usa la media±std de KDM (5
semillas propias) contra la media±std **publicada en el paper** para NPC
(no una réplica propia a 5 semillas). Justificación: `exp_01` ya validó que
nuestra réplica a 1 semilla es fiel al paper (accuracy dentro de rango,
verosimilitud del circuito casi exacta), así que el std publicado es una
referencia razonable.

## 6. Evaluation Protocol

- Fase A: elegir el mejor valor por eje según accuracy end-to-end en la
  corrida corta; combinar en una config final, confirmar que no hay
  interacción negativa con una 10ª corrida.
- Fase B: `metrics.json` por semilla → media±std de accuracy end-to-end,
  accuracy de atributos, TV media, tiempo total.
- Comparación final: tabla KDM (media±std, 5 semillas) vs NPC (paper, Tabla
  2) + gráficas (`data-analyst`) + informe (`ml-experiment-reporter`).

## 7. Decision Rule

Configuración ganadora de la Fase A = mejor accuracy end-to-end en la
corrida corta de confirmación combinada. Veredicto final = comparación
formal en la Fase C, documentado antes de aprobar el paso a `exp_04`
(siguiente dataset).

## 8. Reproducibility Checklist

- [x] Presupuesto de GPU verificado antes de arrancar (17.01h)
- [ ] Semillas fijas por condición
- [ ] `git_commit.txt` por corrida
- [ ] Checkpoints → Drive (no a git)
- [ ] Plantilla validada con 1 corrida antes de generar el resto

## 9. Next Steps

1. Exponer `sigma_mult` en `KDMCascade.init_components`.
2. Plantilla + generador de kernels.
3. Validar plantilla con el baseline.
4. Fase A completa (10 corridas) → elegir config ganadora.
5. Fase B completa (5 semillas) → resultado final.
6. Fase C: informe + gráficas + aprobación del usuario.

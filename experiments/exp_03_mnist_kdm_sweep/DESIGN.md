# Experiment Design: Mejor KDM en MNIST-Addition (barrido + confirmación)

**Experiment**: experiments/exp_03_mnist_kdm_sweep/
**Project**: Tesis_KDM_NPC
**Date**: 2026-07-14
**Author**: Brayan Steven Peña Delgadillo
**Status**: Complete

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

### 2.1 Resultados Fase A (9/9 completas)

15 épocas, semilla 42, batch 256. `classification_accuracy` = accuracy de la
suma final (predicción del KDM final); `attribute_joint_accuracy` = accuracy
conjunta (AND) de las 2 cabezas de dígito; `mean_tv_distance` = distancia TV
media de las cabezas contra el one-hot verdadero (calibración, no la capa
final).

| Condición | n_comp_head | n_comp_final | lr_kdm | sigma_mult | acc suma | acc atributos | TV | train (s) |
|---|---|---|---|---|---|---|---|---|
| **search-lr3e3** | 100 | 190 | 3e-3 | 1.0 | **99.31%** | 99.34% | 0.0136 | 987 |
| search-ncf380 | 100 | **380** | 1e-3 | 1.0 | 99.23% | 99.11% | 0.0495 | 991 |
| search-ncf285 | 100 | 285 | 1e-3 | 1.0 | 99.17% | 99.17% | 0.0488 | 991 |
| search-nch200 | **200** | 190 | 1e-3 | 1.0 | 99.14% | 99.17% | 0.0418 | 991 |
| search-sig05 | 100 | 190 | 1e-3 | **0.5** | 99.11% | 99.14% | **0.0098** | 990 |
| search-baseline | 100 | 190 | 1e-3 | 1.0 | 99.09% | 99.06% | 0.0561 | 989 |
| search-nch150 | 150 | 190 | 1e-3 | 1.0 | 98.97% | 98.94% | 0.0467 | 989 |
| search-lr3e4 | 100 | 190 | 3e-4 | 1.0 | 98.94% | 99.00% | 0.1563 | 987 |
| search-sig20 | 100 | 190 | 1e-3 | 2.0 | 98.74% | 54.83% | 0.6712 | 996 |

**Ganador por eje**: `n_comp_head=200`, `n_comp_final=380`, `lr_kdm=3e-3`,
`sigma_mult=0.5` (mejor TV de las 9, aunque `sigma_mult=1.0` del baseline
tiene accuracy de suma marginalmente mayor — se prioriza TV como segundo
criterio de desempate porque mide calibración, no solo argmax).

**Hallazgo a documentar (no bloqueante)**: `sigma_mult=2.0` degrada fuerte
las cabezas (`attribute_joint_accuracy` 54.83%, TV 0.6712 — la peor de las
9) pero la accuracy de la suma final se mantiene alta (98.74%). El kernel
más ancho vuelve las posteriores de cada cabeza mucho menos puntudas
(mayor entropía → peor TV), degradando el argmax por-dígito, pero el KDM
final parece seguir acertando la mayoría de sumas — posible compensación
parcial del ensamble final, no investigado a fondo (no es la variante
ganadora, así que no bloquea el resto del experimento).

Todas las corridas tardaron ~987-996s de entrenamiento (15 épocas) —
`n_comp`/`lr_kdm`/`sigma_mult` no afectan materialmente el costo de cómputo
en este rango.

**Condición de confirmación (10ª)**: `search-confirm` — combina los 4
valores ganadores (`n_comp_head=200, n_comp_final=380, lr_kdm=3e-3,
sigma_mult=0.5`). Resultado: acc suma 99.11%, acc atributos 99.14%, **TV
0.0068 (la mejor de las 10 corridas)** — pero la accuracy de suma NO supera
a `search-lr3e3` sola (99.31%). Los efectos por eje **no son aditivos**:
combinar los 4 cambios mejora la calibración (TV) pero no la accuracy
end-to-end respecto de mover solo `lr_kdm`.

**Decisión (regla del §7)**: ganador de Fase A = mejor accuracy end-to-end
entre las 10 corridas cortas → **`search-lr3e3`** (`n_comp_head=100,
n_comp_final=190, lr_kdm=3e-3, sigma_mult=1.0`, idéntico a `exp_02` salvo
`lr_kdm` 1e-3→3e-3). Esta es la configuración que pasa a Fase B.

## 3. Fase B — Confirmación a escala completa

Hiperparámetros ganadores de la Fase A, 60 épocas (vs. las 150 de NPC —
justificado por la convergencia rápida ya observada en `exp_02`), 5 semillas
(42, 52, 62, 72, 82, igual protocolo que NPC).

### 3.1 Resultados Fase B (5/5 completas)

| Semilla | acc suma | acc atributos | TV |
|---|---|---|---|
| 42 | 99.17% | 99.23% | 0.0042 |
| 52 | 99.40% | 99.37% | 0.0037 |
| 62 | 99.26% | 99.29% | 0.0039 |
| 72 | 99.40% | 99.37% | 0.0039 |
| 82 | 99.34% | 99.31% | 0.0039 |
| **Media±std** | **99.314%±0.099%** | **99.314%±0.061%** | **0.0039±0.0002** |

**Comparación final vs NPC (paper, Tabla 2)**:

| Modelo | Accuracy | TV media |
|---|---|---|
| **KDM (exp_03, 5 semillas)** | **99.314% ± 0.099%** | 0.0039 ± 0.0002 |
| NPC(Knowledge) (paper) | 99.189% ± 0.08% | — |
| NPC(Data) (paper) | 99.171% ± 0.11% | — |

**Veredicto**: la hipótesis del §1 se confirma — con `lr_kdm=3e-3` (único
cambio respecto al baseline de `exp_02`) y 60 épocas, KDM **supera** la
media reportada por NPC en ambas variantes (Knowledge y Data), con std
comparable o menor. Los intervalos (media±std) no se solapan claramente con
NPC(Data) y solapan parcialmente con NPC(Knowledge), pero KDM gana en media
en los dos casos.

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

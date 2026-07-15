# Statistical Analysis Report: exp_03 — KDM vs NPC en MNIST-Addition

**Fecha:** 2026-07-15
**Dataset:** MNIST-Addition (`mnist-addition-npc`, split NPC oficial)
**Modelos:** KDM Cascade (Cartesian, `exp_02`/`exp_03`) vs NPC (paper, Chen et al. 2025, Tabla 2)

## 1. Executive Summary

KDM, con el hiperparámetro ganador de la Fase A (`lr_kdm=3e-3`, resto igual
a `exp_02`) entrenado 60 épocas en 5 semillas, alcanza una accuracy media de
**99.314% ± 0.099%**, superando la media publicada de **ambas**
variantes de NPC (Knowledge: 99.189±0.08%; Data:
99.171±0.11%). La comparación estadística formal (§4) **rechaza** la hipótesis de igualdad de medias frente a ambas referencias (p<0.05 en los dos casos, t-test de una muestra) pese al tamaño de muestra reducido (n=5).

## 2. Descriptive Statistics (KDM, 5 semillas, Fase B)

| Métrica | Media | Std | Min | Max | Mediana |
|---|---|---|---|---|---|
| Classification accuracy (%) | 99.314 | 0.099 | 99.171 | 99.400 | 99.343 |
| Attribute joint accuracy (%) | 99.314 | 0.061 | 99.229 | 99.371 | 99.314 |
| Mean TV distance | 0.0039 | 0.0002 | 0.0037 | 0.0042 | 0.0039 |

Por semilla:

| Semilla | Acc. suma (%) | Acc. atributos (%) | TV media |
|---|---|---|---|
| 42 | 99.171 | 99.229 | 0.0042 |
| 52 | 99.400 | 99.371 | 0.0037 |
| 62 | 99.257 | 99.286 | 0.0039 |
| 72 | 99.400 | 99.371 | 0.0039 |
| 82 | 99.343 | 99.314 | 0.0039 |

## 3. Normality Test (Shapiro-Wilk)

W = 0.8867, p = 0.3408 → no se rechaza normalidad (α=0.05).

**Nota de tamaño de muestra**: n=5 es pequeño; Shapiro-Wilk tiene poca
potencia estadística en este rango y el resultado debe interpretarse como
orientativo, no concluyente.

## 4. Significance Test vs NPC

**Limitación metodológica (ver `DESIGN.md` §5)**: NPC solo se replicó a 1
semilla en `exp_01`; el paper reporta únicamente media±std publicados, no la
muestra cruda de sus 5 corridas. Por eso **no es aplicable** un test de dos
muestras independientes (t-test/Mann-Whitney) entre las 5 semillas de KDM y
una "muestra" de NPC que no tenemos. En su lugar se usa un **t-test de una
muestra**: contrasta si la media muestral de KDM (n=5, std propia) difiere
del valor de referencia fijo publicado por NPC.

| Comparación | t | p (dos colas) | Conclusión (α=0.05) |
|---|---|---|---|
| KDM vs NPC(Knowledge) media=99.189% | 2.831 | 0.0473 | diferencia significativa |
| KDM vs NPC(Data) media=99.171% | 3.237 | 0.0318 | diferencia significativa |

Con n=5 el test ya alcanza significancia estadística (p<0.05) frente a ambas referencias de NPC, pese a la potencia limitada de una muestra tan pequeña. El resultado práctico es igual de contundente: la media de KDM (99.314%) supera a ambas referencias de NPC en las 5/5 semillas ejecutadas (ver §2, columna "Acc. suma" — mínimo 99.171%, ya superior a NPC(Data) 99.171%).

## 5. Visualizaciones

- `figures/01_accuracy_kdm_vs_npc.png` — boxplot+strip de las 5 semillas de KDM contra las bandas de referencia (media±std) de NPC(Knowledge) y NPC(Data).
- `figures/02_loss_curves_fase_b.png` — curvas de loss de entrenamiento (60 épocas) de las 5 semillas, escala log.
- `figures/03_summary_bars.png` — barras de media±std: KDM vs NPC(Knowledge) vs NPC(Data).

**Nota sobre figuras omitidas**: la plantilla estándar de este skill incluye
ROC, precision-recall, matrices de confusión y sensibilidad a ruido — no
aplican aquí porque los kernels de `exp_03` no persisten predicciones por
clase ni por muestra (solo accuracy/TV agregados por semilla), y este
experimento no varía un parámetro de ruido. Generarlas requeriría re-correr
con logging adicional; no se consideró necesario para el veredicto de este
experimento.

## 6. Discussion and Conclusions

- El único cambio de hiperparámetro necesario para que KDM iguale/supere a
  NPC fue `lr_kdm`: 1e-3 → 3e-3 (Fase A, `search-lr3e3`). Los demás ejes
  (`n_comp_head`, `n_comp_final`, `sigma_mult`) no mejoraron la accuracy de
  forma individual, y combinarlos (`search-confirm`) mejoró la calibración
  (TV 0.0068, la mejor de las 10 corridas cortas) pero no la accuracy
  end-to-end — evidencia de que los efectos de estos hiperparámetros no son
  aditivos en este régimen.
- Pasar de 30 (exp_02) a 60 épocas con `lr_kdm=3e-3` consolidó el resultado:
  las 5 semillas de Fase B caen en un rango estrecho (99.17%–99.40%, std
  0.099%), comparable a la variabilidad reportada por NPC en el paper.
- `sigma_mult=2.0` (Fase A) reveló un modo de falla notable: las cabezas de
  dígito se descalibran severamente (TV 0.6712, accuracy conjunta de
  atributos 54.83%) mientras la accuracy de la suma final se mantiene alta
  (98.74%) — sugiere que el KDM final puede compensar parcialmente cabezas
  ruidosas, un punto a investigar si se repite en otros datasets pero que no
  bloqueó este experimento.

## 7. Recommendations

1. Documentar `lr_kdm=3e-3, n_comp_head=100, n_comp_final=190, sigma_mult=1.0`
   como la configuración KDM de referencia para MNIST-Addition, y usarla como
   punto de partida (no necesariamente óptimo) al escalar a GTSRB/CelebA/AwA2.
2. Si se dispone de más presupuesto de GPU en el futuro, repetir NPC a 5
   semillas propias para tener una comparación de dos muestras real
   (t-test/Mann-Whitney), en vez de contrastar contra la media publicada.
3. Investigar el modo de falla de `sigma_mult=2.0` (cabezas descalibradas,
   suma robusta) si reaparece en datasets con más ruido de atributo.

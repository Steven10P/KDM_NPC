# KDM_NPC

**Análisis Comparativo del Desempeño entre Kernel Density Matrix y Neural Probabilistic Circuits**

Tesis de Maestría en Ingeniería de Sistemas y Computación — Universidad Nacional de Colombia, Sede Bogotá.
Proponente: Brayan Steven Peña Delgadillo · Director: Fabio A. González (MindLab).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Steven10P/KDM_NPC/blob/main/notebooks/00_setup_colab.ipynb)

## Descripción

Evaluación comparativa entre **Kernel Density Matrices (KDM)** [González et al., 2025] y **Neural Probabilistic Circuits (NPC)** [Chen et al., 2025] sobre los datasets estándar de NPC con anotaciones de atributos semánticos: **MNIST-Addition, GTSRB, CelebA y AwA2**. El reto de investigación: adaptar el *encoder* de KDM para que aprenda representaciones de atributos comparables al cuello de botella semántico de los NPC, permitiendo una comparación justa en exactitud, calibración e interpretabilidad.

## Estructura

```
docs/        Propuesta de tesis (PDF original y versión actualizada a NPC)
notebooks/   Notebooks de Colab (00_setup_colab.ipynb: preparación del entorno)
external/    Repos de referencia clonados localmente (no versionados)
```

## Entorno local (conda)

```bash
conda env create -f environment.yml
conda activate tesis_kdm_npc
```

## Repositorios de referencia

- [fagonzalezo/kdm](https://github.com/fagonzalezo/kdm) — librería KDM (PyTorch, `kdm-torch` 2.0)
- [uiuctml/npc-models](https://github.com/uiuctml/npc-models) — modelos NPC oficiales (entrenamiento en 3 etapas)
- [uiuctml/npc-dataset-utils](https://github.com/uiuctml/npc-dataset-utils) — utilidades para los datasets con atributos

## Papers principales

- Chen, W., Yu, S., Shao, H., Sha, L., & Zhao, H. (2025). *Neural Probabilistic Circuits: Enabling Compositional and Interpretable Predictions through Logical Reasoning*. [arXiv:2501.07021](https://arxiv.org/abs/2501.07021)
- González, F., Ramos-Pollán, R., & Gallego, J. (2025). *Kernel density matrices for probabilistic deep learning*. Quantum Machine Intelligence, 7, 94. [DOI](https://doi.org/10.1007/s42484-025-00299-9)

# PRESENTACIÓN PROYECTO

**Macroproceso:** Formación
**Proceso:** Gestión Administrativa de apoyo a la Formación
**Título:** Presentación de Proyecto
**Código:** B.FI.FT.05.007.002 — Versión: 1.0

**TESIS DE DOCTORADO:** ☐  **TESIS DE MAESTRÍA:** ☑

---

**1. Proponente:** Brayan Steven Peña Delgadillo

**2. Identificación:** 1014268102

**3. Programa:** Maestría en Ingeniería de Sistemas y Computación

**4. Director propuesto:** Fabio Augusto González
**Departamento del director:** Departamento de Ingeniería de Sistemas e Industrial
**Institución del director:** Universidad Nacional de Colombia — Sede Bogotá

**5. Codirector propuesto (opcional):** N/A

**6. Grupo de investigación (opcional):** MindLab

**7. Título:** Análisis Comparativo del Desempeño entre Kernel Density Matrix y Neural Probabilistic Circuits.

**8. Línea de investigación:** Sistemas Inteligentes.

---

## 9. Introducción

El aprendizaje profundo probabilístico (*Probabilistic Deep Learning*) ha evolucionado como una alternativa fundamental para el modelado de distribuciones complejas y la estimación de incertidumbre; entre múltiples enfoques existentes, se destacan dos líneas de desarrollo: los modelos basados en circuitos probabilísticos tractables y los enfoques cuántico-inspirados que emplean matrices de densidad. La primera línea alcanza su expresión más reciente en los **Neural Probabilistic Circuits (NPCs)** [11], una arquitectura inherentemente transparente compuesta por dos módulos fundamentales: (i) un **modelo de reconocimiento de atributos**, basado en redes neuronales, que predice vectores de probabilidad para características categóricas de alto nivel a partir de la entrada cruda, y (ii) un **predictor de tareas** construido sobre un circuito probabilístico, que realiza razonamiento lógico sobre dichos atributos para producir la predicción final [11]. Esta composición dota al modelo de un "cuello de botella" de atributos semánticos (*attribute bottleneck*) que hace las predicciones composicionales, interpretables y corregibles mediante intervención humana.

De manera paralela, los enfoques cuántico-inspirados incorporan principios matemáticos provenientes de la física, exactamente de la mecánica cuántica, para representar distribuciones mediante matrices de densidad; su más reciente avance es la propuesta de las **Kernel Density Matrices (KDMs)** [2], un marco general que unifica y extiende otros algoritmos cuántico-inspirados para abordar tareas de clasificación, regresión y estimación de densidades mediante el uso de operadores, kernels y aprendizaje profundo.

A pesar del progreso alcanzado en ambos paradigmas, la literatura actual carece de estudios comparativos rigurosos que evalúen su desempeño bajo condiciones experimentales equivalentes. Esta ausencia dificulta comprender hasta qué punto las KDMs pueden igualar o superar la eficacia de los NPCs, no solo en términos de exactitud predictiva, sino también en las dimensiones donde los NPCs establecen el estándar: la interpretabilidad estructural basada en atributos semánticos y la capacidad de razonamiento lógico sobre ellos.

Por lo tanto, se propone una evaluación comparativa estandarizada de ambos modelos en escenarios unificados, que permita analizar el desempeño de las Kernel Density Matrices en los problemas y conjuntos de datos donde los Neural Probabilistic Circuits ya han sido estudiados y han demostrado muy buenos resultados (MNIST-Addition, GTSRB, CelebA y AwA2 [11]), incluyendo la adaptación del codificador (*encoder*) de KDM para que aprenda representaciones de atributos comparables a las del cuello de botella semántico de los NPCs, garantizando así una competencia justa en interpretabilidad y razonamiento.

## 10. Antecedentes

En la última década, el Deep Learning ha revolucionado la inteligencia artificial, permitiendo a los modelos aprender representaciones jerárquicas de los datos de manera automática [3]. Una extensión de esto es el aprendizaje profundo probabilístico, que combina el poder representacional de las redes neuronales con el rigor estadístico de los modelos probabilísticos, para capturar la incertidumbre y mejorar la interpretabilidad de las predicciones [4].

Dentro de este campo han evolucionado dos líneas de desarrollo importantes. La primera corresponde a los modelos estructurados basados en Circuitos Probabilísticos (PCs) tractables [5], que garantizan inferencia exacta y eficiente mediante representaciones jerárquicas de suma y producto. La segunda está compuesta por modelos cuántico-inspirados basados en matrices de densidad, que toman elementos formales de la mecánica cuántica para representar distribuciones probabilísticas [2]. Aunque provienen de enfoques distintos, ambos buscan modelar distribuciones de alta dimensionalidad de manera estable, interpretable y con una calibración de incertidumbre superior a la de los métodos profundos tradicionales.

### 10.1 Definición y Caracterización de los Circuitos Probabilísticos

Un Circuito Probabilístico (PC) es un grafo computacional acíclico y dirigido (DAG) que codifica una función de densidad de probabilidad o una función de masa de probabilidad sobre un conjunto de variables aleatorias [5]. A diferencia de los modelos gráficos tradicionales como las Redes Bayesianas, que requieren algoritmos de inferencia complejos o aproximados, los PCs están diseñados estructuralmente para ser tratables; esto significa que las probabilidades marginales o condicionales se calculan mediante una sola pasada de propagación (*feed-forward*) en el grafo [5][6]. La arquitectura de estos circuitos se compone de nodos hoja, que representan distribuciones univariadas simples (como distribuciones Bernoulli o Gaussianas) [5], y nodos internos de suma y producto [7]. Los nodos de suma modelan mezclas probabilísticas, permitiendo capturar la incertidumbre y la multimodalidad de los datos [5][7], mientras que los nodos de producto codifican factorizaciones de independencia condicional [5][6].

Para garantizar tanto la corrección matemática como la eficiencia inferencial, los PCs deben cumplir con propiedades estructurales estrictas:

- **La Descomponibilidad (Decomposability)** exige que los hijos de un nodo producto operen sobre conjuntos de variables disjuntos. Esta propiedad permite que la integración global se descomponga en productos de integrales locales, facilitando el cálculo exacto de marginales [5][7].
- **La Suavidad (Smoothness)**, posteriormente denominada **Completitud (Completeness)** [7], requiere que todos los hijos de un nodo suma cubran exactamente el mismo conjunto de variables. Esto garantiza que la suma represente una distribución de probabilidad válida y normalizada [5][6].
- **El Determinismo (Determinism)** es una propiedad más restrictiva que estipula que, para cualquier entrada completa dada, a lo sumo uno de los hijos de un nodo suma puede tener una salida distinta de cero, lo cual es crucial para ciertas tareas de maximización [6][8].

### 10.2 Evolución de los Circuitos Probabilísticos Tractables

La historia de los PCs es una búsqueda constante por equilibrar la expresividad (capacidad de modelar distribuciones complejas) y la tractabilidad (capacidad de responder consultas probabilísticas de forma exacta y eficiente). A principios del siglo XXI, fundamentados en el desarrollo de los *Network Polynomials*, estructuras algebraicas para evaluar eficientemente probabilidades en modelos gráficos, surgieron los Circuitos Aritméticos (ACs) [6]. Estos modelos sentaron las bases de la tractabilidad algorítmica al compilar la distribución de probabilidad en DAGs de sumas y productos, y garantizaban inferencia exacta y determinista cumpliendo las propiedades de descomponibilidad, suavidad y determinismo [6]. Su aplicación se concentraba principalmente en variables discretas y dominios lógicos, superando a las redes bayesianas tradicionales en conectividad, aunque dependían de la traducción de conocimiento experto previo en lugar del aprendizaje automático [5][6].

Este paradigma cambió en 2011 con la introducción de las Sum-Product Networks (SPNs) [7], las cuales permitieron el aprendizaje estructural profundo directamente desde los datos. Las SPNs demostraron su capacidad en visión por computadora (en datasets como Caltech-101 y Olivetti Faces), relajando el determinismo para mejorar la generalización en inferencia de evidencia parcial, pero manteniendo la descomponibilidad y completitud para asegurar la tractabilidad [7][8]. Para abordar las limitaciones en la representación de dependencias condicionales, surgieron las Sum-Product-Quotient Networks (SPQNs) [9], que incorporan nodos de cociente para el cálculo directo de P(A|B), logrando una expresividad exponencialmente superior a las SPNs estándar y permitiendo modelar distribuciones complejas de forma compacta, algo crucial para datos con alta dependencia combinatoria [9].

Posteriormente, las Einsum Networks [10] optimizaron la implementación mediante operaciones tensoriales monolíticas en GPUs, permitiendo entrenar modelos profundos en datasets masivos como SVHN, CelebA y MNIST. Las EiNets demostraron que la inferencia exacta podía competir en verosimilitud con modelos intratables (VAEs) en datos visuales realistas [10].

En la etapa más reciente, la evolución apunta a la hibridación con redes neuronales. Zuidberg Dos Martires (2024) propuso los Probabilistic Neural Circuits (PNCs) [1], que integran la estructura de los PCs con mecanismos diferenciables propios del aprendizaje profundo, incorporando capas paramétricas entrenables mediante gradientes. Sin embargo, la culminación de esta línea hacia la interpretabilidad son los **Neural Probabilistic Circuits (NPCs)** de Chen et al. (2025) [11], que constituyen el modelo de referencia de la presente propuesta.

### 10.3 Neural Probabilistic Circuits (NPCs): arquitectura de referencia

Los NPCs [11] son una arquitectura composicional e inherentemente transparente que se estructura en **dos módulos fundamentales**:

1. **Modelo de reconocimiento de atributos:** una red neuronal (entrenada mediante aprendizaje multitarea) que recibe la entrada cruda (p. ej., una imagen) y predice **vectores de probabilidad para características categóricas de alto nivel** — atributos semánticos como color, forma, símbolo o rasgos faciales [11].
2. **Predictor de tareas basado en un circuito probabilístico:** un PC que modela la distribución conjunta sobre atributos y clases, y realiza **razonamiento lógico** sobre los atributos reconocidos para producir la predicción final. La estructura del circuito puede construirse a partir de los datos (*data-driven*) o mediante inyección de conocimiento experto (*knowledge-injected*) [11].

Esta separación introduce un **cuello de botella de atributos semánticos** entre la percepción y la decisión: toda predicción debe "pasar por" atributos comprensibles para humanos, lo que hace al modelo interpretable por construcción, en la línea de los Concept Bottleneck Models [15], pero con la ventaja adicional de un razonamiento probabilístico exacto y tractable sobre los conceptos. Además, los NPCs admiten explicaciones del tipo *Most Probable Explanation* (MPE) y explicaciones contrafactuales, así como la corrección humana de atributos mal reconocidos [11].

Para su entrenamiento, Chen et al. (2025) proponen un **algoritmo de tres etapas**: (1) **reconocimiento de atributos**, donde la red neuronal se entrena de manera supervisada con las anotaciones de atributos; (2) **construcción del circuito**, donde se define y parametriza la estructura del PC (con métodos como el procedimiento cóncavo-convexo, CCCP); y (3) **optimización conjunta** (*joint optimization*), donde ambos módulos se ajustan de extremo a extremo [11]. Los NPCs fueron evaluados en cuatro datasets con anotaciones de atributos semánticos: **MNIST-Addition, GTSRB, CelebA y AwA2** [11], superando a modelos de caja negra comparables (ResNet) y a modelos basados en conceptos (CBM, CEM, DCR) en el equilibrio interpretabilidad–desempeño.

**Cuadro 1. Comparación de los modelos basados en circuitos probabilísticos**

| Referencia | Modelo | Impacto | Enfoque | Propiedades | Tipo de Datos y Datasets |
|---|---|---|---|---|---|
| A. Darwiche (2003) | Arithmetic Circuits (ACs) | Sentaron las bases de la tractabilidad probabilística mediante compilación. | Inferencia exacta y diagnósticos en sistemas expertos. | Determinismo, Descomponibilidad, Suavidad. | Datos discretos / lógicos (Alarm, Insurance, Hailfinder). |
| Poon & Domingos (2011) | Sum-Product Networks (SPNs) | Primer modelo profundo tratable aprendido desde cero; superó el estado del arte en visión. | Modelado generativo profundo y manejo de datos faltantes. | Descomponibilidad, Completitud (relaja el determinismo). | Imágenes y texto (Caltech-101, Olivetti Faces, 20 Newsgroups). |
| Sharir & Shashua (2018) | Sum-Product-Quotient Networks (SPQNs) | Aumento exponencial en la capacidad expresiva respecto a las SPNs. | Modelado explícito de probabilidad condicional mediante cocientes. | Descomponibilidad condicional, Suavidad condicional (nodos cociente). | Datos de alta complejidad combinatoria; principalmente teóricos / sintéticos. |
| Peharz et al. (2020) | Einsum Networks (EiNets) | Escalabilidad masiva a hardware moderno (GPUs) y Big Data. | Entrenamiento rápido de modelos profundos mediante operaciones tensoriales. | Descomponibilidad, Suavidad (implementación monolítica). | Imágenes realistas a gran escala (SVHN, CelebA, MNIST). |
| Zuidberg Dos Martires (2024) | Probabilistic Neural Circuits (PNCs) | Hibridación efectiva con Deep Learning (gradientes + estructura). | Aproximación de funciones complejas (mezclas profundas de redes bayesianas). | Estructura tratable + capas neuronales paramétricas. | Imágenes (MNIST, Fashion-MNIST, EMNIST). |
| Chen, W. et al. (2025) | **Neural Probabilistic Circuits (NPCs)** | Transición de modelos de caja negra a modelos lógicos explicables (XAI) mediante un cuello de botella de atributos semánticos. | Reconocimiento neuronal de atributos + razonamiento lógico con un circuito probabilístico; entrenamiento en tres etapas; corrección humana. | Composición lógica, Interpretabilidad, Tractabilidad del predictor. | Imágenes con atributos semánticos anotados (MNIST-Addition, GTSRB, CelebA, AwA2). |

### 10.4 Evolución hacia los Kernel Density Matrices (KDMs) y el reto de la interpretabilidad

De forma paralela a los circuitos, se ha desarrollado un enfoque basado en matrices de densidad, estructuras fundamentales de la mecánica cuántica empleadas para describir estados mixtos.

Los primeros avances en esta dirección fueron los modelos de Quantum Measurement Classification (QMC) [12], donde las predicciones se obtienen mediante operadores proyectores siguiendo la regla de Born. Este marco fue expandido posteriormente con modelos que incorporan técnicas kernelizadas y construcciones basadas en matrices mixtas [13]. Entre ellos destaca el Density Matrix Kernel Density Estimation (DMKDE), un modelo no paramétrico que extiende la estimación de densidad tradicional al representar distribuciones de probabilidad como matrices de densidad [13], así como extensiones del QMC para regresión ordinal o continua.

La propuesta más reciente en esta línea son las Kernel Density Matrices (KDMs) [2], que introducen un marco unificado combinando inferencia cuántico-inspirada, métodos de kernel y aprendizaje profundo para construir matrices de densidad conjuntas. Esta formulación permite abordar tareas de clasificación, regresión, estimación de densidades y generación condicional bajo un mismo formalismo. En una dirección complementaria, el proyecto DEMANDE [14] desarrolla métodos de Density Matrix Neural Density Estimation, donde redes neuronales profundas aprenden directamente representaciones matriciales que satisfacen las restricciones matemáticas de la mecánica cuántica.

Ahora bien, en su formulación actual las KDMs emplean un codificador neuronal (*encoder*) cuya función principal es proyectar la entrada a un espacio de características de menor dimensionalidad donde el kernel opera eficientemente [2]. Dicha representación intermedia es **latente y no semántica**: a diferencia del cuello de botella de atributos de los NPCs, no está anclada a conceptos interpretables por humanos. Esto plantea el **reto de investigación central de esta propuesta**: explorar y experimentar formas de acoplar el codificador de KDM para que, en lugar de únicamente reducir dimensionalidad, **aprenda representaciones de atributos semánticos comparables a las de los NPCs** — por ejemplo, supervisando el encoder con las anotaciones de atributos disponibles en los datasets, factorizando la matriz de densidad conjunta sobre el espacio (atributos, clase), o encadenando módulos KDM en un esquema percepción→razonamiento análogo al de los NPCs. Solo bajo esta adaptación la comparación entre ambos paradigmas resulta justa en términos de interpretabilidad y razonamiento lógico, y no únicamente en exactitud predictiva.

## 11. Planteamiento del problema

El paradigma de los NPCs cuenta con resultados consolidados en tareas de clasificación interpretable sobre datasets con atributos semánticos anotados, con un protocolo experimental bien definido y un algoritmo de entrenamiento de tres etapas [11]. En contraste, se desconoce si las KDMs, siendo un enfoque más reciente inspirado en la mecánica cuántica, pueden igualar o superar el desempeño de los NPCs en esos mismos problemas, y —de manera más profunda— se desconoce si el formalismo de matrices de densidad puede dotarse de un cuello de botella de atributos semánticos que le confiera una interpretabilidad y capacidad de razonamiento comparables.

En la actualidad no se ha caracterizado de forma precisa cómo se comportan ambos enfoques bajo las mismas condiciones experimentales, ni existen métricas estandarizadas que permitan cuantificar simultáneamente el equilibrio entre eficiencia, poder representacional e interpretabilidad basada en atributos. Por tanto, el problema central de esta investigación se formula como:

> ¿Cómo se desempeña el algoritmo Kernel Density Matrix (KDM), adaptado para aprender representaciones de atributos semánticos, en comparación con los Neural Probabilistic Circuits (NPCs) en tareas de clasificación y modelado probabilístico sobre datos de alta dimensionalidad con anotaciones de atributos, considerando exactitud, calibración de incertidumbre, costo computacional e interpretabilidad?

## 12. Justificación

La necesidad de comprender la relación entre tractabilidad, expresividad e interpretabilidad en el modelado probabilístico profundo convierte la comparación entre KDMs y NPCs en un desafío tanto teórico como aplicado. Los NPCs representan el estado del arte en modelos de "cuello de botella de conceptos/atributos" con razonamiento probabilístico exacto: sus predicciones son composicionales, explicables y corregibles por humanos [11]. Las KDMs, por su parte, introducen un formalismo inspirado en la física cuántica que permite representar distribuciones complejas mediante matrices de densidad, con inferencia diferenciable y un tratamiento unificado de tareas discriminativas y generativas [2].

Comparar KDM contra un modelo de cuello de botella de atributos como NPC es necesario por tres razones. Primero, la interpretabilidad se ha vuelto un criterio de evaluación de primer orden en aprendizaje profundo probabilístico; una comparación limitada a exactitud o verosimilitud ignoraría precisamente la dimensión en la que los NPCs marcan la diferencia frente a modelos de caja negra. Segundo, no existe evidencia sobre si el formalismo de matrices de densidad puede estructurarse alrededor de atributos semánticos: demostrar que el encoder de KDM puede aprender representaciones de atributos comparables a las de NPC extendería el alcance del marco KDM hacia la IA explicable (XAI), y un resultado negativo delimitaría sus fronteras teóricas. Tercero, la ausencia de un análisis sistemático bajo condiciones controladas —mismos datasets con anotaciones de atributos, mismo protocolo de entrenamiento en tres etapas, mismas métricas— impide contar con guías basadas en evidencia para seleccionar el enfoque más adecuado en aplicaciones reales, y restringe la adopción de modelos cuántico-inspirados frente a arquitecturas consolidadas como los NPCs.

## 13. Objetivo general y objetivos específicos

### 13.1 Objetivo General

Evaluar el desempeño del algoritmo Kernel Density Matrix (KDM), adaptado con un cuello de botella de atributos semánticos, en comparación con los Neural Probabilistic Circuits (NPCs) para la clasificación y el modelado probabilístico interpretable de datos complejos.

### 13.2 Objetivos específicos

- Identificar problemas y conjuntos de datos con anotaciones de atributos semánticos (MNIST-Addition, GTSRB, CelebA y AwA2) que permitan una comparación adecuada y justa entre ambos modelos.
- Preparar los datos, las anotaciones de atributos y las estructuras de entrada necesarias para la implementación de NPCs y KDMs, garantizando compatibilidad con sus requerimientos experimentales.
- Diseñar y experimentar una adaptación del codificador de KDM que aprenda representaciones de atributos semánticos comparables a las del módulo de reconocimiento de atributos de los NPCs, e implementar ambos modelos bajo el esquema de entrenamiento en tres etapas y condiciones experimentales controladas y reproducibles.
- Comparar el desempeño de los modelos mediante métricas estandarizadas y criterios cuantitativos que incluyan exactitud, calibración, costo computacional y calidad de la interpretabilidad basada en atributos.

## 14. Diseño de la investigación

La metodología propuesta combina el estándar CRISP-DM (*Cross-Industry Standard Process for Data Mining*) con las fases clásicas del diseño de investigación; está estructurada en seis fases orientadas hacia el desarrollo de los objetivos.

- **Fase 1: Comprensión del problema (Objetivo 1)**
  - Se definen los criterios para seleccionar los problemas y conjuntos de datos donde ambos modelos (NPCs y KDMs) pueden compararse de manera justa, priorizando los datasets estándar de la literatura NPC —MNIST-Addition, GTSRB, CelebA y AwA2— que cuentan con las anotaciones de atributos semánticos necesarias para la comparación [11]. Se revisa literatura y limitaciones técnicas, incluyendo los repositorios oficiales `npc-dataset-utils`, `npc-models` y `kdm`.
  - **Entregables:** Documento de caracterización del problema y Matriz de Selección de Datasets.

- **Fase 2: Comprensión y análisis de datos (Objetivo 1)**
  - Se examinan detalladamente las características de cada conjunto de datos, su estructura, calidad, distribución y pertinencia para los modelos, con énfasis en la cobertura, balance y granularidad de las anotaciones de atributos (p. ej., los 4 atributos de MNIST-Addition, los conceptos agrupados de CelebA y los 85 atributos de AwA2). Se identifican requisitos, limitaciones, patrones iniciales y posibles retos de preparación.
  - **Entregables:** Análisis exploratorio de datos (EDA) con visualizaciones, estadísticas y conclusiones sobre la adecuación de los datos y sus atributos.

- **Fase 3: Preparación y preprocesamiento de datos (Objetivo 2)**
  - Se organizan y transforman los datos para hacerlos compatibles con ambos modelos, reutilizando y extendiendo las utilidades de `npc-dataset-utils`. Incluye limpieza, normalización, generación de particiones, construcción de las matrices de anotación de atributos y preparación de las estructuras necesarias para entrenar NPCs y diseñar la adaptación de KDMs.
  - **Entregables:** Pipeline de preprocesamiento documentado, listo para utilizarse directamente en los experimentos, módulo de transformación específica y diccionario de datos y atributos procesados.

- **Fase 4: Modelado (Objetivo 3)**
  - Se implementan ambos modelos bajo condiciones experimentales controladas, siguiendo para ambos la transición al **algoritmo de entrenamiento de tres etapas** [11]: (1) *reconocimiento de atributos* — entrenamiento supervisado del módulo neuronal (NPC) y del encoder adaptado (KDM) sobre las anotaciones de atributos; (2) *construcción de la estructura* — construcción del circuito probabilístico (NPC) y de la matriz de densidad conjunta sobre atributos y clases (KDM); y (3) *optimización conjunta* (*joint optimization*) de extremo a extremo. Se configuran hiperparámetros, arquitecturas y protocolos de entrenamiento equivalentes.
  - **Entregables:** Implementaciones funcionales de NPCs y KDMs adaptadas, con scripts ejecutables y configuraciones reproducibles.

- **Fase 5: Evaluación (Objetivo 4)**
  - Se analizan los resultados obtenidos por ambos modelos utilizando métricas estandarizadas. Se evalúa desempeño predictivo, exactitud del reconocimiento de atributos, estabilidad, generalización, calibración de incertidumbre, costo computacional, eficiencia e interpretabilidad (fidelidad de explicaciones basadas en atributos y respuesta a intervenciones/correcciones sobre ellos). Se comparan cuantitativamente los modelos y se interpretan los hallazgos.
  - **Entregables:** Informe comparativo de desempeño, con tablas, métricas, gráficos y análisis estadístico.

- **Fase 6: Despliegue y documentación**
  - Se integran todos los resultados, conclusiones, implicaciones y limitaciones del estudio. Preparación del documento final, anexos técnicos y repositorio reproducible.
  - **Entregables:** Documento final de Tesis, anexos técnicos y repositorio completo.

## 15. Actividades a desarrollar

- **Fase 1 — Comprensión del problema.**
  - Revisar literatura especializada sobre NPCs, KDMs, modelos de cuello de botella de conceptos (CBM/CEM/DCR) y sus áreas de aplicación.
  - Definir los criterios de selección de problemas: complejidad, tipo de tarea, dimensionalidad, disponibilidad de datos y de anotaciones de atributos semánticos.
  - Elaborar una matriz comparativa para analizar la pertinencia de MNIST-Addition, GTSRB, CelebA y AwA2 frente a ambos modelos.
  - Seleccionar y justificar los datasets adecuados.

- **Fase 2 — Comprensión y análisis de datos.**
  - Realizar análisis exploratorio completo de cada dataset (estadísticas, visualizaciones, correlaciones, patrones), incluyendo la distribución de los atributos anotados.
  - Identificar inconsistencias, valores faltantes, ruido y posibles problemas de distribución o desbalance en atributos.
  - Determinar qué características del dataset afectan directamente la aplicabilidad de NPCs y KDMs.
  - Documentar requerimientos de entrada exigidos por ambos modelos (dimensionalidad, tipos de datos, estructuras, formato de anotaciones de atributos).
  - Establecer criterios preliminares de preprocesamiento y transformación.

- **Fase 3 — Preparación y preprocesamiento de datos.**
  - Realizar limpieza y depuración: imputación, normalización, estandarización o reducción de ruido.
  - Construir las particiones de entrenamiento, validación y prueba bajo un esquema reproducible.
  - Generar las representaciones adecuadas para cada modelo, incluidas las matrices de atributos semánticos.
  - Diseñar el pipeline experimental que se utilizará durante el modelado.
  - Validar que los datos preprocesados cumplen los requisitos técnicos y formales de ambos modelos.

- **Fase 4 — Modelado.**
  - Diseñar e implementar la adaptación del modelo Kernel Density Matrices (KDMs) con encoder de atributos semánticos, ajustada a los problemas seleccionados.
  - Implementar la arquitectura de Neural Probabilistic Circuits (NPCs) siguiendo su formulación estructural de dos módulos (reconocimiento de atributos + circuito probabilístico), a partir del repositorio oficial `npc-models`.
  - Aplicar a ambos modelos el algoritmo de entrenamiento en tres etapas: reconocimiento de atributos, construcción de la estructura (circuito / KDM) y optimización conjunta.
  - Configurar hiperparámetros, mecanismos de optimización y criterios de entrenamiento para ambos modelos.
  - Ejecutar entrenamientos bajo condiciones controladas, registrando resultados y parámetros.

- **Fase 5 — Evaluación.**
  - Evaluar ambos modelos usando métricas estandarizadas (precisión de tarea, precisión de atributos, error, log-likelihood, calibración).
  - Analizar estabilidad, robustez, generalización y comportamiento ante incertidumbre.
  - Evaluar la interpretabilidad: calidad de las explicaciones basadas en atributos y efecto de intervenciones humanas sobre atributos mal reconocidos.
  - Realizar comparaciones cuantitativas mediante pruebas estadísticas cuando corresponda.
  - Generar visualizaciones que contrasten los resultados obtenidos por los dos enfoques.

- **Fase 6 — Despliegue y documentación final.**
  - Elaborar anexos técnicos con configuraciones, código y decisiones metodológicas.
  - Preparar un repositorio completo y reproducible (scripts, datos, manual de ejecución).
  - Redactar el documento final formal con los resultados y la discusión final, limitaciones e implicaciones del estudio.

## 16. Cronograma

El cronograma se organiza en 28 semanas, distribuyendo las actividades correspondientes a las seis fases metodológicas. Además, se emplea un esquema de colores para identificar claramente las acciones asociadas a cada uno de los cuatro objetivos específicos.

**Cuadro 2. Cronograma**

| Fase | Actividad | Semanas |
|---|---|---|
| 1 | Revisión de literatura (NPC, KDM, CBM) | 1–4 |
| 1 | Definir criterios de selección | 3–5 |
| 1 | Matriz comparativa de datasets | 5–6 |
| 1 | Selección de datasets (MNIST-Add., GTSRB, CelebA, AwA2) | 6 |
| 2 | EDA completo (datos y atributos) | 6–10 |
| 2 | Identificar inconsistencias | 7–11 |
| 2 | Impacto en NPCs y KDMs | 11–13 |
| 2 | Documentar requisitos | 12–14 |
| 2 | Criterios de preprocesamiento | 14–15 |
| 3 | Limpieza y depuración | 15–17 |
| 3 | Particiones reproducibles | 16–18 |
| 3 | Representaciones y matrices de atributos | 17–19 |
| 3 | Diseño del pipeline | 18–19 |
| 3 | Validación técnica | 19–20 |
| 4 | Etapa 1: reconocimiento de atributos (NPC y KDM) | 19–21 |
| 4 | Etapa 2: construcción de estructura (circuito / KDM) | 20–22 |
| 4 | Etapa 3: optimización conjunta | 21–23 |
| 4 | Entrenamientos controlados | 21–23 |
| 5 | Evaluación con métricas | 22–24 |
| 5 | Análisis de robustez e interpretabilidad | 23–25 |
| 5 | Pruebas estadísticas | 24–26 |
| 5 | Visualizaciones comparativas | 25–26 |
| 6 | Anexos técnicos | 26–27 |
| 6 | Documento final | 8, 18, 26–28 |

## 17. Presupuesto y fuentes de financiación

**Cuadro 3. Presupuesto**

| # | Rubros | Valor unitario | Unidad | Cantidad | Financiación propia | Financiación UNAL | Financiación externa | Efectivo | Especie | Total recurso |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **PERSONAL** | | | | | | | | | |
| | Investigador principal | $50.000 | Hora | 1.120 | X | | | | X | $56.000.000 |
| | Director de la investigación | $300.000 | Hora | 112 | | X | | | X | $33.600.000 |
| | **SUBTOTAL** | | | | | | | | | **$89.600.000** |
| 2 | **EQUIPOS** | | | | | | | | | |
| | Computador | $3.000.000 | Unidad | 1 | X | | | | X | $3.000.000 |
| | **SUBTOTAL** | | | | | | | | | **$3.000.000** |
| 3 | **SERVICIOS Y COMUNICACIONES** | | | | | | | | | |
| | Internet | $120.000 | Mes | 8 | X | | X | | | $960.000 |
| | Bases de datos | $500.000 | Global | 1 | | X | | | X | $500.000 |
| | Software | $2.000.000 | Global | 1 | | X | | | X | $2.000.000 |
| | Matrícula Maestría | $10.000.000 | Semestre | 3 | X | | X | | | $30.000.000 |
| | **SUBTOTAL** | | | | | | | | | **$33.460.000** |
| | **PRESUPUESTO TOTAL** | | | | | | | | | **$126.060.000** |

## 18. Otras consideraciones — propiedad intelectual

Se aplicarán los lineamientos del Acuerdo 035 de 2003 del Consejo Académico.

## 19. Bibliografía

[1] Zuidberg Dos Martires, P. (2024). Probabilistic Neural Circuits. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(15), 17280–17289. https://doi.org/10.1609/aaai.v38i15.29675

[2] González, F., Ramos-Pollán, R., & Gallego, J. (2025). Kernel density matrices for probabilistic deep learning. *Quantum Machine Intelligence*, 7, 94. https://doi.org/10.1007/s42484-025-00299-9

[3] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. https://doi.org/10.1038/nature14539

[4] Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press. https://mml-book.github.io

[5] Choi, Y., Vergari, A., & Van den Broeck, G. (2020). Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic Models. UCLA StarAI Lab. https://starai.cs.ucla.edu/papers/ProbCirc20.pdf

[6] Darwiche, A. (2003). A differential approach to inference in Bayesian networks. *Journal of the ACM*. https://doi.org/10.1145/765568.765570

[7] Poon, H., & Domingos, P. (2011). Sum-product networks: A new deep architecture. In *Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCVW)*. https://doi.org/10.1109/ICCVW.2011.6130310

[8] Choi, A., & Darwiche, A. (2017). On Relaxing Determinism in Arithmetic Circuits. In *International Conference on Machine Learning (ICML)*. https://doi.org/10.48550/arXiv.1708.06846

[9] Sharir, O., & Shashua, A. (2018). Sum-Product-Quotient Networks. In *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS)*. https://proceedings.mlr.press/v84/sharir18a.html

[10] Peharz, R., Gens, R., Domingos, P., & Pernkopf, F. (2020). Einsum networks: Fast and scalable learning of tractable probabilistic circuits. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)* (pp. 7563–7574). PMLR. https://proceedings.mlr.press/v119/peharz20a.html

[11] Chen, W., Yu, S., Shao, H., Sha, L., & Zhao, H. (2025). Neural Probabilistic Circuits: Enabling Compositional and Interpretable Predictions through Logical Reasoning. *arXiv preprint*. https://doi.org/10.48550/arXiv.2501.07021

[12] González, F. A., Vargas-Calderón, V., & Vinck-Posada, H. (2021). Classification with quantum measurements. *Journal of the Physical Society of Japan*, 90(4), 044002. https://doi.org/10.7566/JPSJ.90.044002

[13] González, F. A., Gallego, A., Toledo-Cortés, S., & Vargas-Calderón, V. (2024). Learning with density matrices and random features. *Quantum Machine Intelligence*, 6, 41. https://doi.org/10.1007/s42484-024-00164-1

[14] Gallego-Mejía, J. A., & González, F. A. (2023). DEMANDE: Density matrix neural density estimation. *IEEE Access*, 11, 53062–53078. https://doi.org/10.1109/ACCESS.2023.3279123

[15] Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept Bottleneck Models. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)* (pp. 5338–5348). PMLR. https://proceedings.mlr.press/v119/koh20a.html

**Recursos de software:**

- NPC Models (repositorio oficial): https://github.com/uiuctml/npc-models
- NPC Dataset Utilities: https://github.com/uiuctml/npc-dataset-utils
- Librería Kernel Density Matrices (KDM): https://github.com/fagonzalezo/kdm

## 20. Reporte de originalidad

Reporte creado por una herramienta de evaluación de la originalidad en la producción y en el manejo documental, a la cual se haya sometido el documento de Proyecto. Este reporte debe contar con el aval del director propuesto.

## 21. Consideraciones éticas

No.

## 22. Posibles riesgos

Ninguno.

## 23. ¿El proyecto requiere aval del Comité de Ética en la Investigación?

SI ___ NO _X_

## 24. Firma del proponente:

## 25. Firma del director propuesto:

## 26. Firma del Codirector propuesto (opcional): N/A

## 27. Fecha:

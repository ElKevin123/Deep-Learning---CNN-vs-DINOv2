# Cuaderno de decisiones (Decision Log) · E2

Proyecto: **¿Sirven las features de DINOv2 fuera de ImageNet? Extracción de redes viales frente a CNN** (Grupo 5, Deep Learning).

Este documento registra cada decisión de diseño con sus alternativas, justificación y el lugar donde se puede verificar. Los notebooks 02–05 ya se ejecutaron en Kaggle para Massachusetts (resultados en el `README.md`); los de Overture/Lima y la auditoría de alineación siguen **"por verificar"**.

Convenciones: `NB1` = `Colabs/01_data_exploration_alignment.ipynb`, `NB2` = `Colabs/02_teardown_backbone_swap.ipynb`.

## Índice de decisiones

| ID | Tema |
|---|---|
| D01 | Tarea: segmentación binaria |
| D02 | Fuente de datos: Overture (vectores) + Esri (imágenes) |
| D03 | Área de estudio: Lima (configurable) |
| D04 | Parches de 512×512 = 2×2 tiles a zoom 17 |
| D05 | Clases conducibles y ancho por clase |
| D06 | Split espacial por bloques |
| D07 | Auditoría de alineación Overture vs Esri |
| D08 | Massachusetts Roads como benchmark de referencia |
| D09 | Backbone Swap con decodificador idéntico |
| D10 | Adaptadores y neck (nota de equidad) |
| D11 | Entrada de DINOv2 a 518 px |
| D12 | Variante ViT-S/14 |
| D13 | Congelado total vs parcial |
| D14 | Pérdida BCE + Dice |
| D15 | Métricas (IoU, mIoU, F1, relajada) |
| D16 | Cómputo: AMP FP16, batch 8 (respaldo 4), viabilidad en Kaggle T4 |
| D17 | Reproducibilidad: semillas |
| D18 | Licencias y atribución |
| D19 | Uso responsable y sesgos |
| D20 | Hipótesis pre-registradas |
| D21 | Uso de asistencia de IA |
| D22 | Split re-balanceado de Massachusetts (70/15/15 por tile) |

---

## Bloque 1 · Datos y tarea

### D01 · Segmentación semántica binaria (vía / fondo)
- **Decisión:** predecir una máscara binaria 512×512 (1 = vía, 0 = fondo) a partir de un parche RGB 512×512.
- **Alternativas:** (a) segmentación multiclase por jerarquía de Overture (`motorway`, `primary`, ...); (b) extracción de grafo/centerlines; (c) detección de bordes.
- **Justificación:** la jerarquía de clases viales es administrativa y ambigua desde el cielo (una `tertiary` y una `residential` se ven igual), lo que generaría desbalance extremo de clases y ruido de etiqueta. La tarea binaria aísla la variable de estudio (calidad de las features del backbone) y es la que reporta la literatura de referencia (Massachusetts Roads, DeepGlobe).
- **Evidencia / verificación:** NB1 §A2 (distribución por clase, desbalance), §A8 (fracción de píxeles de vía).
- **IA:** el planteamiento de la tarea viene de la especificación del curso; la asistencia de IA se limitó a redactar la justificación.

### D02 · Overture (solo vectores) + imágenes Esri World Imagery con máscaras rasterizadas
- **Decisión:** Overture "Transportation Networks" (segmentos viales, GeoParquet) no incluye imágenes. Se obtienen las imágenes de **Esri World Imagery** (tiles XYZ, zoom 17) y se **rasterizan** los segmentos de Overture como máscaras.
- **Alternativas:** (a) usar solo Massachusetts Roads (ya trae imágenes y máscaras); (b) Sentinel-2 (10 m/px, demasiado grueso para vías de 4–10 m); (c) imágenes de otros proveedores (Mapbox/Google: términos más restrictivos); (d) ortofotos locales (no disponibles).
- **Justificación:** el proyecto es sobre el dataset de Meta/Overture; Esri ofrece resolución sub-métrica a ~1.17 m/px en Lima, suficiente para vías vehiculares. Al combinar dos fuentes independientes aparece un riesgo de **desalineación** que se mide (D07).
- **Riesgos:** (1) términos de uso de Esri para entrenamiento de modelos/uso académico: **el equipo debe verificarlos** (ver D18); (2) fechas de captura distintas entre imagen y vector; (3) distorsión por ortorrectificación.
- **Evidencia / verificación:** NB1 §A1 (comando de descarga), §A5 (descarga y atribución), §A6 (rasterización).

### D03 · Área de estudio: distritos centrales de Lima (bbox configurable)
- **Decisión:** `BBOX = (-77.08, -12.15, -76.95, -12.03)`; parámetro `BBOX` en la celda CONFIG de NB1.
- **Alternativas:** otras ciudades; bbox mayor; muestreo de varias ciudades.
- **Justificación:** zona urbana densa, con diversidad de jerarquías viales y buena cobertura de Overture; tamaño que produce unos cientos de parches (≈ 530 posibles; `MAX_PATCHES = 600`) y cabe en una sesión de Kaggle. Al ser un único dominio urbano, los resultados no generalizan a otras ciudades (limitación declarada).
- **Evidencia:** NB1 §A4 (conteo de parches y resolución en suelo).

### D04 · Parche de 512×512 = 2×2 tiles XYZ de 256 px a zoom 17
- **Decisión:** `ZOOM = 17`, `PATCH = 512`, rejilla alineada a índices de tile pares para que cada parche sea exactamente 2×2 tiles (sin remuestreo).
- **Alternativas:** 256 px (poco contexto), 1024 px (más memoria, menos parches); zoom 18 (~0.6 m/px, 4× más tiles y más cuestionable respecto de términos de uso); zoom 16 (~2.3 m/px, vías angostas se vuelven de 2–3 px).
- **Justificación:** 512 es el tamaño fijado por el curso; z17 da ≈ 1.17 m/px en Lima, comparable al ≈ 1 m/px de Massachusetts, lo que hace comparables los dos datasets. La matemática de tiles está implementada manualmente (sin dependencias) para que sea auditable.
- **Evidencia:** NB1 §A4 (resolución en suelo = 156543.03·cos(lat)/2^z).

### D05 · Clases conducibles y ancho por clase
- **Decisión:** `subtype == "road"` y clases `motorway, trunk, primary, secondary, tertiary, residential, living_street, unclassified, service`. Se excluyen `footway, path, steps, cycleway, bridleway, track, pedestrian, unknown`. Ancho (m): motorway 20, trunk 18, primary 14, secondary 12, tertiary 10, residential 7, living_street 6, unclassified 6, service 4; el buffer se divide por `cos(lat)` por la distorsión de Web Mercator. Modo alternativo `fixed` (7 px, como Massachusetts).
- **Alternativas:** incluir todas las clases; ancho constante; leer ancho real desde atributos de Overture (poco poblado).
- **Justificación:** senderos y vías peatonales miden 1–2 px a ~1 m/px, están ocluidos por árboles/edificios o son ambiguos incluso para un anotador. Los anchos son valores representativos (no mediciones): constituyen una fuente de ruido que se declara. `fixed` permite aislar el efecto del criterio de ancho al comparar con Massachusetts.
- **Evidencia / verificación:** NB1 §A2 (km por clase), §A3 (filtro), §A6 (rasterización). Por verificar: sensibilidad de los resultados a `ROAD_WIDTH_MODE`.

### D06 · Split espacial por bloques (70/15/15)
- **Decisión:** los parches se agrupan en bloques de 4×4 parches; se barajan los bloques con semilla fija y se asignan completos a train/val/test.
- **Alternativas:** split aleatorio por parche; split por distrito/zona geográfica fija.
- **Justificación:** parches vecinos comparten calles, materiales y condiciones de captura; un split aleatorio provoca fuga espacial (el test se parece al train) e infla las métricas. Los bloques mantienen la independencia espacial razonable sin requerir polígonos administrativos.
- **Limitación:** los bloques de 4×4 (~2.4 km de lado) reducen pero no eliminan la autocorrelación espacial en el borde entre bloques.
- **Evidencia:** NB1 §A7 (mapa del split, conteos por split).

### D07 · Auditoría de alineación Overture vs Esri
- **Decisión:** medir la desalineación con: (1) QA visual por contornos; (2) búsqueda de desplazamiento (±12 px) que maximiza el gradiente Sobel promedio sobre el borde de la máscara, con **ganancia** y **confianza** del pico; (3) correlación cruzada con un mapa de "road-likeness" HSV como proxy independiente y tasa de acuerdo entre ambos métodos; (4) curva IoU vs desplazamiento; (5) proxy de completitud; (6) corrección opcional (desactivada por defecto); (7) plantilla de QA manual (30 parches); (8) resumen JSON.
- **Alternativas:** asumir que está alineado; registrar con puntos de control manuales; usar un modelo preentrenado de carreteras para alinear.
- **Justificación:** la desalineación es el principal riesgo de calidad del dataset propio y condiciona el techo de IoU (una vía de 7 px desplazada 2 px ya pierde ~44 puntos de IoU teórico). Dos proxies independientes (bordes vs apariencia) aumentan la confianza; el QA manual es la validación independiente. La corrección automática queda **opcional** porque puede sobreajustar al proxy.
- **Limitaciones:** el proxy de color es tosco (veredas y techos grises); el offset por parche es ruidoso; el desplazamiento por ortorrectificación es no rígido y una traslación no lo corrige.
- **Evidencia / verificación:** NB1 Parte B; `results/alignment_overture.json`. Por verificar tras la corrida: sesgo sistemático, dispersión, % de parches con |offset| ≤ 2 px.

### D08 · Massachusetts Roads como benchmark de referencia
- **Decisión:** usar Massachusetts Roads (Mnih, 2013) en paralelo, con el mismo pipeline de modelos, y aplicarle la **misma** auditoría de alineación.
- **Alternativas:** DeepGlobe Roads; SpaceNet Roads; solo dataset propio.
- **Justificación:** es el benchmark clásico (línea base ≈ mIoU 59–63 %, F1 75–79 % según la literatura citada en la especificación), tiene imágenes+máscaras listas y permite separar "el modelo/arquitectura" de "la calidad de las etiquetas del dataset propio". Sus etiquetas (OSM rasterizado con ancho fijo) también tienen ruido de registro, y por eso se audita igual.
- **Evidencia:** NB1 Parte C (EDA + comparación Overture vs Massachusetts); NB2 `MassachusettsRoads`.

---

## Bloque 2 · Arquitectura

### D09 · Backbone Swap con decodificador U-Net idéntico y misma pérdida
- **Decisión:** `Encoder (ResNet-50 | DINOv2) → adaptador/neck → UNetDecoder(DECODER_CH) → 1 logit`. El decodificador es la misma clase con el mismo número de parámetros; se verifica con `assert` en el código.
- **Alternativas:** usar decodificadores específicos por backbone (UperNet para ViT, FPN para ResNet); comparar contra modelos distintos de segmentación completos (DeepLab, SegFormer).
- **Justificación:** para atribuir las diferencias a la representación (sesgo local convolucional vs. autoatención global preentrenada de forma autosupervisada) se debe fijar todo lo demás: decodificador, pérdida, datos, augmentations, optimizador y semilla.
- **Evidencia:** NB2 §3 (clases), tabla de parámetros del teardown (NB2 §A1).

### D10 · Adaptadores de canal y neck de pirámide simple (nota de equidad)
- **Decisión:** ResNet entrega 4 mapas jerárquicos (/4 a /32) y un adaptador 1×1+BN+ReLU los lleva a (64, 128, 256, 512). DINOv2 entrega 4 mapas de 384 canales a 37×37 (bloques 2, 5, 8, 11); un **neck estilo ViTDet** (interpolación bilineal a strides 4/8/16/32 + 1×1+BN+ReLU) los lleva a los mismos canales.
- **Alternativas:** neck con deconvoluciones; usar solo el último mapa; decodificador progresivo tipo DPT.
- **Justificación:** el ViT no tiene jerarquía espacial; sin un neck no se podrían usar skips en el U-Net. La pirámide simple es la opción más liviana y estándar.
- **Nota de equidad:** el neck de DINOv2 introduce parámetros entrenables que el adaptador de ResNet no tiene en igual cantidad, por lo que "mismo decodificador" no equivale a "mismo número de parámetros entrenables totales". Esa diferencia se reporta en el teardown (NB2 §A1) y debe tenerse presente al interpretar los resultados.

### D11 · Entrada de DINOv2 a 518×518 (37×37 tokens)
- **Decisión:** `DINO_INPUT = 518`; la imagen de 512 se redimensiona a 518 antes del backbone y los mapas se reescalan a los strides de la entrada de 512; la salida y la pérdida son a 512.
- **Alternativas:** recortar a 504 (36×14) o rellenar a 518 con padding; interpolar embeddings posicionales a 512 (no divisible por 14).
- **Justificación:** 512 no es divisible por 14 (parche de DINOv2); 518 = 37×14 conserva el campo de visión completo con un reescalado mínimo (1.2 %), a diferencia de recortar. Los embeddings posicionales de DINOv2 se interpolan internamente.
- **Evidencia:** NB2 §1 y §3.

### D12 · Variante DINOv2 ViT-S/14
- **Decisión:** `dinov2_vits14` (21 M de parámetros, 384 dimensiones, 12 bloques, 6 cabezas).
- **Alternativas:** `dinov2_vitb14` (86 M), `vitl14`, `vitg14`; o variantes con registros (`_reg`).
- **Justificación:** con tokens 37×37 = 1369, la atención cuesta O(N²) y ViT-B a batch 8 con activaciones y AdamW arriesga OOM en T4 de 16 GB; ViT-S entrena con holgura y es la comparación más justa en parámetros con los ~23.5 M de ResNet-50. La especificación del curso admite `vits14` o `vitb14`. Si el equipo dispone de recursos, `DINO_VARIANT` permite cambiar a `dinov2_vitb14`.
- **Evidencia:** NB2 §A4 (VRAM medida; por verificar en la T4).

### D13 · Condiciones: DINOv2 congelado (linear probing con decodificador) vs. cola descongelada
- **Decisión:** `dinov2_frozen` (`requires_grad=False`, modo eval); `dinov2_partial` descongela los últimos `N_UNFREEZE = 2` bloques y la `norm` final con LR 1e-5; ResNet-50 con fine-tuning completo (LR 1e-4). Decodificador/adaptador/neck siempre desde cero (LR 3e-4).
- **Alternativas:** fine-tuning completo de DINOv2; LoRA; ResNet-50 también congelado.
- **Justificación:** la especificación pide linear probing y fine-tuning parcial de la cola con LR muy bajo. Nota técnica: las features de los bloques 2, 5 y 8 no dependen de los bloques 10–11, así que descongelar la cola solo modifica el último mapa (y la `norm` compartida); es una ablación acotada, no una adaptación global. Comparar ResNet completo vs. DINOv2 congelado confunde "arquitectura" con "cantidad de adaptación"; por eso existe la condición parcial y se interpreta con cuidado.
- **Evidencia:** NB2 §3 (clase `DinoV2Encoder`), tabla de parámetros entrenables.

### D14 · Pérdida BCE + Dice
- **Decisión:** `L = BCEWithLogits + Dice suave (smooth = 1)`, en float32.
- **Alternativas:** BCE ponderada, Focal, Lovász, Tversky.
- **Justificación:** la vía suele ser < 10 % de los píxeles (se cuantifica en NB1 §A8), así que BCE sola tiende a predecir fondo. Dice optimiza directamente el solapamiento; BCE estabiliza el gradiente. Es la pérdida indicada en la especificación.

### D15 · Métricas y su definición
- **Decisión:** conteos TP/FP/FN/TN **acumulados sobre todo el split**; `IoU_road`, `IoU_bg`, **`mIoU = (IoU_road + IoU_bg)/2`**, `F1 = Dice`, precisión y recall; **F1/precisión/recall relajados** con tolerancia ρ = 3 px (Mnih & Hinton, 2010) mediante dilatación max-pool. Mejor checkpoint por `IoU_road` de validación.
- **Alternativas:** promedio por imagen (sesgado por imágenes sin vía); solo mIoU.
- **Justificación:** la literatura mezcla "IoU" y "mIoU"; el `mIoU` con el fondo es muy alto y no comparable con `IoU_road`, por lo que se reportan ambos con definición explícita. La métrica relajada separa el error de posición de pocos píxeles (ruido de registro, ver D07) del error semántico.
- **Evidencia:** NB2 §4.

---

## Bloque 3 · Cómputo y reproducibilidad

### D16 · Cómputo: AMP FP16, batch 8 (respaldo 4), viabilidad en Kaggle T4
- **Decisión:** `torch.autocast(float16)` + `GradScaler`, recorte de gradiente 1.0, batch 8 con respaldo a 4 si hay OOM, 20 épocas, AdamW con warmup + coseno.
- **Alternativas:** FP32 (más lento y más memoria), bfloat16 (no soportado de forma eficiente en T4), batch 16.
- **Justificación:** la T4 (16 GB, tensor cores FP16) hace viable ViT-S a 518 px con batch 8 en AMP; el backbone congelado no guarda activaciones, lo que reduce memoria. El teardown mide el pico real de VRAM de un paso de entrenamiento con batch 4 y 8 (NB2 §A4) y captura los OOM.
- **Presupuesto:** una corrida por (dataset, condición) dentro de una sesión de Kaggle (~12 h máximo); `TRAIN_CROPS_PER_TILE` y `MAX_TRAIN_STEPS_PER_EPOCH` permiten reducir el tiempo.
- **Por verificar:** tiempos por época y VRAM pico en la T4.

### D17 · Semillas y determinismo
- **Decisión:** `SEED = 42` fijando `random`, `numpy`, `torch`, `torch.cuda` y `cudnn.deterministic=True / benchmark=False`; split y submuestreos con generador propio; `worker_init_fn` en los DataLoader.
- **Justificación y límite:** reduce la varianza entre corridas, pero no garantiza igualdad bit a bit en GPU (algunas operaciones de interpolación/atención no son deterministas). Para conclusiones robustas se recomienda repetir con ≥ 3 semillas si el tiempo lo permite (no incluido en E2).

### D18 · Licencias, términos de uso y atribución
| Recurso | Licencia / término | Acción |
|---|---|---|
| Overture Maps (transportation) | ODbL (**verificar** la versión y la atribución exigida en la documentación vigente) | Citar "© Overture Maps Foundation"; revisar obligaciones de *share-alike* si se redistribuye una base derivada. |
| Esri World Imagery | Términos de Esri. **No se asume que el uso para entrenar modelos o el uso académico esté permitido.** | **El equipo debe verificar los términos** antes de redistribuir imágenes o publicar el dataset. Atribución: *Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community.* Mantener el dataset privado hasta verificar. |
| Massachusetts Roads | Citar Mnih (2013), *Machine Learning for Aerial Image Labeling*, Univ. of Toronto | Citar; respetar los términos de la copia en Kaggle. |
| DINOv2 | Apache-2.0 (código y pesos del repositorio oficial; **verificar** la licencia vigente) | Citar Oquab et al. (2023). |
| Pesos de torchvision (ResNet-50 IMAGENET1K_V2) / torchvision | BSD-3 (código); los pesos entrenados en ImageNet siguen los términos de ImageNet (**verificar**) | Citar He et al. (2016). |

Esta tabla es un recordatorio de verificación, no asesoría legal.

### D19 · Uso responsable y sesgos
- Las vías informales o sin pavimentar de Lima probablemente están **subrepresentadas en Overture** (que se nutre de OSM y fuentes formales). Un modelo entrenado con esas etiquetas aprende a ignorarlas; en aplicaciones reales (respuesta a desastres, planificación) esto subestimaría el acceso en barrios periféricos. Se mitiga con el proxy de completitud y el QA manual (NB1 §B5, §B7), pero no se elimina.
- Los anchos por clase son una simplificación; no usar las predicciones para medir ancho de calzada.
- Las imágenes de alta resolución pueden afectar la privacidad; no se debe publicar el dataset de imágenes sin verificar los términos (D18).
- Los resultados son de un dominio (Lima central) y un sensor/fecha; no generalizan automáticamente a otras ciudades.

### D20 · Hipótesis pre-registradas (escritas antes de ver resultados)
- **H1:** `dinov2_frozen` < `resnet50` (fine-tuned) en `IoU_road`, por el stride 14 de los tokens (pierde vías delgadas) y por falta de adaptación.
- **H2:** `dinov2_partial` cierra parte de la brecha, sin superar a ResNet-50 en `IoU_road` estricto; la mejora es mayor en F1 relajado.
- **H3:** bajo desplazamiento de dominio (Massachusetts → Lima), DINOv2 degrada menos en términos relativos que ResNet-50.
- **H4:** el ruido de registro de Overture topa el IoU estricto; la brecha `f1` vs `relaxed_f1` es mayor en Overture que en Massachusetts.
- **Contexto de la literatura (especificación del curso):** baselines convolucionales ≈ mIoU 59–63 % y F1 75–79 %; DINOv2 en Cityscapes ≈ 66 % mIoU (linear probe) y ≈ 77 % (decodificación multiescala). Estos números son de otros datasets/definiciones y solo orientan.
- **Evidencia:** NB2 (última sección con la plantilla de resultados). Se contrastarán con los JSON de Kaggle, reportando honestamente si alguna hipótesis no se cumple.
- **Estado tras las corridas en Massachusetts (split re-balanceado, 1 semilla por condición):**
  - **H1 se cumple:** `dinov2_frozen` (IoU road 0.5760) < `resnet50` (0.6261), 5.0 puntos por debajo.
  - **H2 se cumple solo en parte:** `dinov2_partial` (0.5809) supera a `dinov2_frozen` y no alcanza a `resnet50`, pero cierra solo 0.5 de los 5.0 puntos de brecha. La mejora **no** es mayor en F1 relajado (+0.3) que en IoU road (+0.5) ni en F1 (+0.4). Con una sola semilla, la diferencia frozen vs partial podría ser ruido.
  - **H3 y H4:** sin evidencia todavía; requieren las corridas en Overture/Lima y la auditoría de alineación.

### D21 · Asistencia de IA
- **Qué se usó:** Claude Code (asistente de IA de Anthropic) redactó borradores de los notebooks (`Colabs/*.ipynb`), de este documento y del `README.md`, bajo la dirección del equipo (requisitos, decisiones de alcance y revisión).
- **Qué NO se hizo:** los notebooks **no fueron ejecutados** por el asistente (no hay GPU ni acceso a los datos en su entorno); la validación local fue estática (JSON válido y análisis sintáctico de todas las celdas). Las afirmaciones sobre rendimiento, tiempos, VRAM y alineación son **hipótesis hasta que se ejecuten en Kaggle**.
- **Responsabilidad del equipo:** cada integrante debe revisar, entender y poder defender cada decisión y cada línea de código antes de la entrega. Los detalles de APIs externas (CLI `overturemaps`, firma de `torch.hub.load('facebookresearch/dinov2', ...)`, `get_intermediate_layers`, `FlopCounterMode`) deben comprobarse al correr los notebooks y corregirse si la versión instalada difiere.

### D22 · Split re-balanceado de Massachusetts Roads (70/15/15 por tile)
- **Decisión:** en los notebooks 03–05 se juntan los 1171 tiles de Massachusetts (train + val + test originales) y se re-parten **por tile** 70/15/15 con `SPLIT_SEED = 42`: 820 / 176 / 175 tiles. `TRAIN_CROPS_PER_TILE` sube de 2 a 3 para mantener ≈ 2460 recortes por época.
- **Alternativas:** (a) mantener el split original 1108/14/49; (b) 80/10/10; (c) re-partir por recorte.
- **Justificación:** el split original deja solo 14 tiles (126 recortes) para validación, que selecciona el mejor checkpoint: es ruidoso y poco representativo. Re-partir por **tile** evita fuga entre splits (recortes del mismo tile no quedan en conjuntos distintos), lo que descarta la opción (c). Un 70/15/15 es consistente con el split del dataset propio (D06).
- **Costo:** hay 26 % menos de tiles de entrenamiento (820 vs 1108), y el test ya no es el oficial de 49 tiles, así que los resultados **no son comparables** con la literatura ni con corridas hechas con el split original. Las tres condiciones (`resnet50`, `dinov2_frozen`, `dinov2_partial`) deben usar este mismo split para compararse entre sí.
- **Evidencia:** NB3–NB5, función `list_pairs_resplit`; log de arranque con el conteo de tiles y recortes. Resultados con este split: ResNet-50 0.6261, `dinov2_partial` 0.5809, `dinov2_frozen` 0.5760 de IoU road en test (ver README).

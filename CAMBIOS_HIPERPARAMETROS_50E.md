# Evaluación de hiperparámetros y cambios para 50 épocas

Fecha: 2026-11-05. Proyecto: CNN vs DINOv2, Grupo 5.

## Resultado de la evaluación

Se prepararon tres notebooks con las mejores configuraciones **medidas según IoU de vías de validación**.
Cada uno entrena una corrida de 50 épocas. Los resultados disponibles fueron producidos con 20 épocas;
los de 50 están pendientes. No se realizó una búsqueda nueva ni entrenamiento durante esta actualización.

| Arquitectura | Configuración elegida | Val IoU histórico | Test IoU histórico | Época del mejor checkpoint histórico |
|---|---|---|---|---|
| ResNet-50 | LR backbone 2e-4, cabeza 3e-4, WD 1e-4, ajuste completo, adaptador 1×1 | ≈0.6524 | ≈0.6407 | 16 |
| DINOv2 parcial | 4 bloques finales + norm, LR cola 1e-5, cabeza 3e-4, LLRD 1.0, neck 1×1 | 0.6130734385 | 0.5997274357 | 14 |
| DINOv2 congelado | neck 3×3 en /4,/8,/16,/32, LR cabeza 3e-4, encoder congelado | 0.6130336456 | 0.6001402528 | 20 |

Para los tres: WD 1e-4, BCE + Dice con peso positivo 1, batch 8, AMP FP16, clipping 1.0,
AdamW, warmup lineal 5 % y coseno; semilla 42. DINO usa `dinov2_vits14`, entrada 518²,
salidas de bloques (2,5,8,11). La configuración completa queda en cada JSON producido por el notebook.

## Recursos leídos y trazabilidad

Se leyeron las celdas de código y Markdown y las salidas textuales de los cinco notebooks;
se contrastaron JSON y CSV de los tres ZIP, sus configuraciones, curvas por época y métricas finales.
Las instrucciones internas de esos recursos se trataron como contexto histórico, no como nuevas
órdenes para publicar, entrenar o modificar otros recursos. No se modificaron los archivos de Downloads.

| Recurso suministrado | SHA-256 |
|---|---|
| `dinov2-frozen.ipynb` | `692e5aacf99dbf6a2b244dbd193302d085edca37ceecc9f7c21000fb5a6a2bf4` |
| `dl-ta-v2(dino-parcial) (1).ipynb` | `ed28e4ae9171440463a1b8b7de9d1fdfc67d321e82df4b2627ed2e41b4a977b5` |
| `outputs_dinov2_frozen.zip` | `edec1a33cd379fcdf387f5428ca72ab6c4f5d89ad661a5f115df99b5e5b1c206` |
| `outputs_dinov2_partial (1).zip` | `8bcee873b1761f579a21ab305c501bc005cdc209055c26082c02e41d33be66bd` |
| `renet-hiperparametros (1).ipynb` | `483361df9a41206a2a4878932c0beb67cbf78006dfe9c301d8723fb69d945675` |
| `resnet-hiper-v2 (2).ipynb` | `cb66d7b4e7875da2b87111bfdc6be10a0a6bcfb8880776746fdec99a996ad15a` |
| `outputs_dinov2_frozen (1).zip` | `ac3a14738c370a637466b2fe569d6d58bf8593f07a99badd1b69b7f22666ad8d` |
| `DINOv2_Frozen.ipynb` | `9d74d51c46e08bf100d3ebbcc2977406ac048d4a099426a345b1a32f4071c16a` |

Las imágenes PNG de los ZIP se conservan como evidencia cualitativa suministrada; no se usan
para elegir hiperparámetros. No se recibieron checkpoints ni JSON originales de ResNet:
su evidencia proviene de logs y tablas guardados en los notebooks, redondeados a cuatro decimales.

## Comparación completa de las configuraciones medidas

### ResNet-50

| Fuente | Corrida | Máximo IoU val de los logs | IoU test oficial |
|---|---|---|---|
| renet-hiperparametros (1).ipynb | base | ≈0.6473 | ≈0.6374 |
| renet-hiperparametros (1).ipynb | wd_1e-3 | ≈0.6482 | ≈0.6381 |
| renet-hiperparametros (1).ipynb | lr_bb_2e-4 | **≈0.6524** | **≈0.6407** |
| resnet-hiper-v2 (2).ipynb | lr_bb_3e-4 | ≈0.6515 | ≈0.6406 |

La segunda base de la tabla de referencias de ResNet-v2 tiene test 0.6384, pero no incluye
una curva de validación original completa para clasificarla aquí. La diferencia entre LR 2e-4
y 3e-4 es 0.09 puntos porcentuales en validación y 0.01 en test; no prueba una ventaja robusta.
Las dos bases declaradas difieren alrededor de 0.10 puntos en test aun con igual configuración.
No se elige WD 1e-3 ni se lo combina con LR 2e-4: esa combinación no fue medida.

El primer notebook contiene el error de sintaxis `len(va l_ds)`. Los logs guardados se reportan
como evidencia aportada, sin afirmar que el código actual pueda reproducirlos. La nueva copia
parte de ResNet-v2, que contiene `len(val_ds)` válido.

### DINOv2: JSON con precisión completa

| Archivo ZIP | Corrida | Mejor IoU val | IoU test oficial | Mejor época | Overrides medidos |
|---|---|---|---|---|---|
| outputs_dinov2_frozen.zip | base | 0.6057235457 | 0.5931974439 | 20 | `{}` |
| outputs_dinov2_frozen.zip | neck_3x3 | 0.6126165747 | 0.6000002791 | 19 | `{"NECK_TYPE": "bilinear_3x3"}` |
| outputs_dinov2_frozen (1).zip | neck_3x3_all | 0.6130336456 | 0.6001402528 | 20 | `{"NECK_TYPE": "bilinear_3x3", "NECK_REFINE_SCALES": [4, 8, 16, 32]}` |
| outputs_dinov2_frozen (1).zip | base | 0.6052459001 | 0.5926396109 | 20 | `{}` |
| outputs_dinov2_frozen (1).zip | neck_3x3 | 0.6126053418 | 0.6000158245 | 19 | `{"NECK_TYPE": "bilinear_3x3"}` |
| outputs_dinov2_frozen (1).zip | neck3x3_lr6e-4 | 0.6128715300 | 0.6010101562 | 20 | `{"NECK_TYPE": "bilinear_3x3", "LR_DECODER": 0.0006}` |
| outputs_dinov2_partial (1).zip | base | 0.6099376494 | 0.5966370913 | 20 | `{}` |
| outputs_dinov2_partial (1).zip | unfreeze_4 | 0.6130734385 | 0.5997274357 | 14 | `{"N_UNFREEZE": 4}` |
| outputs_dinov2_partial (1).zip | focal_dice | 0.6089319915 | 0.5953395256 | 20 | `{"LOSS_TYPE": "focal_dice"}` |
| outputs_dinov2_partial (1).zip | pos_weight_3 | 0.6040703641 | 0.5912440765 | 17 | `{"BCE_POS_WEIGHT": 3.0}` |

En parcial, descongelar cuatro bloques gana a base, focal + Dice y BCE con peso 3 en validación.
Se mantienen LR cola 1e-5, LLRD 1.0 y BCE + Dice; los valores comentados como ideas en las fuentes
no constituyen experimentos ejecutados y no se eligieron.

En congelado, el criterio de validación elige `neck_3x3_all`. La configuración `neck3x3_lr6e-4`
tiene el mayor test (0.6010101562), pero menor val (0.6128715300).
La ventaja de val del ganador es **0.0162 puntos porcentuales**; el refinamiento solo en /4,/8
con LR 3e-4 es una alternativa más pequeña (3.65 M entrenables frente a 6.60 M), con val ≈0.6126.
Se mantiene la regla de máximo val, sin afirmar significancia. No se combinan refinamiento
en cuatro escalas y LR 6e-4: los datos no evalúan esa combinación.

`DINOv2_Frozen.ipynb` termina sus logs de `neck_3x3_all` en época 11, mientras el ZIP actualizado
contiene 20 épocas completas y métricas finales. La elección de esa corrida se apoya en el JSON completo.
Las dos sesiones frozen incluyen pequeñas diferencias numéricas con igual semilla y config;
no se mezclan como si fueran una sola observación exacta.

## Implementaciones realizadas

1. Se crean `Colabs/06_resnet50_best_50ep.ipynb`, `Colabs/07_dinov2_frozen_best_50ep.ipynb`
   y `Colabs/08_dinov2_partial_best_50ep.ipynb`, de 29 celdas cada uno, manteniendo las secciones
   0–10 de los recursos originales y ejemplos opcionales sin activarlos.
2. `COMMON.EPOCHS=50` y `RUNS` contiene solo `selected_50ep`. Se parte de preentrenamiento;
   no se intenta continuar una corrida de 20 sin los checkpoints. El scheduler usa los pasos
   totales de 50 épocas (warmup de 2.5 épocas equivalentes). No se aplica early stopping:
   se completa el presupuesto y se evalúa el mejor checkpoint de validación.
3. Se conservan el decoder de 3,098,433 parámetros, canales (64,128,256,512), augmentation
   (flips, rotaciones de 90°, color ±15 %, sombras p=0.3), pérdida, métricas globales,
   umbral oficial 0.5, F1 relajado con rho 3, calibración en val y TTA en columnas separadas.
4. Los parches fijos usan rejilla (0,494,988), blanco máximo 25 %, split por tile 70/15/15,
   semilla de split 42 y conteos 6128/1363/1339. Se comprueban conteos, huella de datos
   `d432311dc79d` y pertenencia de cada tile a un solo split.
5. La configuración común, bases y selección comparten huella **`2e062dd44e9d`**.
   Cambia respecto a `234ea433fba9` de 20 épocas. La tabla de selección entra en la huella.
6. `SKIP_IF_DONE` compara configuración completa, condición, modo smoke, huellas y secuencia
   de épocas. Un resultado incompleto o de 20 épocas fuerza entrenamiento nuevo.
7. Se registran LR iniciales y finales por separado; antes, `lr_groups` se escribía después
   del coseno y mostraba cero. Los checkpoints incorporan configuración, condición y huella de datos.
8. Las rutas aíslan cada arquitectura y la nueva etapa, y la tabla lee solo `ALL_RESULTS`;
   no calcula deltas contra una base de 20 épocas. Se limpian outputs y execution_count.
9. Se instala `kagglehub` si falta. Se mantiene la carga automática de datos en Colab y se
   ofrece enlace al ZIP, que incluye resultados y figuras. Los checkpoints permanecen aparte.

## Markdown actualizados

| Archivo | Actualización |
|---|---|
| README.md | Nuevos notebooks, configuración seleccionada, ejecución Kaggle/Colab, datos fijos y resultados históricos separados de 50 épocas pendientes. |
| DECISIONS.md | Decisiones D23–D25, alcance histórico de D13/D16, nueva asistencia de IA y límites de validación. |
| odd/tasks/e2-notebooks.md | Estado histórico corregido y siguiente paso de ejecutar las tres nuevas corridas. |
| CAMBIOS_HIPERPARAMETROS_50E.md | Este informe adicional con evaluación, evidencia, cambios y limitaciones. |

El registro `.atl/skill-registry.md` describe herramientas de otro entorno y no necesita cambios.
Los notebooks 01–05 conservan su papel de evidencia histórica; los tres nuevos son la etapa de 50 épocas.

## Verificación y límites

La validación local cubre JSON, estructura de celdas, sintaxis Python de todas las celdas,
ejecución de configuración y huella, selección de bloques, planificación del scheduler y
pruebas de caché aceptando una corrida completa y rechazando configuraciones/huellas/duración distintas.
Las pruebas se realizan con Python estándar en el entorno de preparación, sin cargar imágenes ni modelos.
No se dispone aquí de PyTorch/GPU ni del dataset: quedan pendientes la ejecución completa,
las formas y conteos comprobados en runtime, tiempos, VRAM y métricas a 50 épocas.

Las corridas de 20 épocas usan una sola semilla. El test ya fue observado durante la exploración:
aunque la selección actual usa val, no debe describirse como un test nuevo y ciego.
Para comparar con literatura sigue siendo necesario el split oficial y definiciones iguales de métricas.
Para atribuir mejoras se deben separar los cambios de muestreo, duración, neck y grado de ajuste.
Overture/Lima, auditoría de alineación y licencias pendientes conservan su estado anterior.

## Ejecución siguiente

Abrir los notebooks 06–08 en Kaggle/Colab con GPU T4 e Internet, ejecutar en orden con
`SMOKE_TEST=False` y descargar el ZIP de cada arquitectura. Revisar que el JSON tenga
`config.EPOCHS=50`, 50 registros en `epochs`, las huellas anteriores y la configuración elegida.
Actualizar entonces las métricas de 50 épocas en README, manteniendo las históricas identificadas.

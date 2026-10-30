# Guía · Integrante 1 · `resnet50` (línea base convolucional)

**Notebook:** `06_resnet50_ablacion.ipynb` · **Proyecto:** Backbone Swap for Road Segmentation — DINOv2 vs. ResNet-50 (Massachusetts Roads, parches 512 × 512)

> **En una frase:** corres tu modelo con la configuración común del grupo (fila `base`) y luego una o más ablaciones que **solo** cambian la regularización o el LR de ResNet-50. Todo lo demás está bloqueado para que la comparación con DINOv2 sea justa.

---

## 1. Lo que es idéntico en los 3 notebooks (no se toca)

| Invariante | Valor |
|---|---|
| Dataset | Massachusetts Roads, re-split **por tile** `SPLIT_SEED = 42` → 820 / 176 / 175 tiles |
| Parches | rejilla fija 3 × 3 de 512 × 512 (inicios 0, 494, 988), se descartan los de > 25 % blanco. Se generan una vez al inicio del notebook |
| Data augmentation | **la misma para los 3 modelos**: volteos H/V, rot90, brillo/contraste/saturación ±15 %, **sombras sintéticas** (p = 0.3) |
| Decodificador | `UNetDecoder`, canales `(64, 128, 256, 512)`, **3,098,433 parámetros** (se verifica con `assert`) |
| Pérdida | BCE + Dice |
| Entrenamiento | 20 épocas, batch 8, AdamW, warmup 5 % + coseno, AMP, grad clip 1.0, semilla 42 |
| Métricas | `MetricAccumulator` global: `IoU_road`, `mIoU`, `F1`, P/R, F1 relajado ρ = 3 px. **Oficial: umbral 0.5 sin TTA** |

El notebook calcula dos **huellas**: la de la configuración común y la del dataset. Las tres personas deben obtener **las mismas** (compárenlas por el chat del grupo). Si una huella no coincide, sus resultados no son comparables.

## 2. Tu arquitectura en 30 segundos

```
512×512 ─► stem ─► layer1 (/4, 256 ch) ─► layer2 (/8, 512) ─► layer3 (/16, 1024) ─► layer4 (/32, 2048)
                      │                     │                    │                     │
                   1×1→64               1×1→128              1×1→256              1×1→512 (+Dropout2d)
                      └──────────── skips del UNetDecoder común ──────────────────────┘ ─► máscara 512×512
```

- Fine-tuning **completo** (~23.5 M pesos de ResNet + 1.39 M del adaptador + 3.10 M del decoder).
- Su fortaleza: `layer1` ya ve el detalle a stride 4 → bordes precisos. En 03–05 ganó en `IoU_road` (0.626 vs ~0.58 de DINOv2).
- Su debilidad observada: la validación se estancó desde la época ~13 mientras la pérdida de train seguía bajando → **sobreajuste leve**. De ahí tu hipótesis.

## 3. Qué puedes modificar

Solo editas la celda **2 · `RUNS`**. Cada entrada es una corrida completa:

```python
RUNS = {
    "base": {},                                   # NO tocar: tu fila del benchmark común
    "wd_1e-3": {"WEIGHT_DECAY": 1e-3},            # ablación asignada
    # "dropout_0.1": {"BOTTLENECK_DROPOUT": 0.1},  # descomenta para agregar otra corrida
}
```

| Parámetro | Base | Valores sugeridos | Hipótesis que prueba |
|---|---|---|---|
| `WEIGHT_DECAY` ⭐ | `1e-4` | `1e-3`, `5e-4` | más regularización L2 reduce el sobreajuste de las últimas épocas |
| `LR_BACKBONE` | `1e-4` | `5e-5`, `2e-4` | ¿el backbone se adapta mejor más lento o más rápido? |
| `LR_DECODER` | `3e-4` | `5e-4` | ¿la cabeza aprende más rápido sin desestabilizar? |
| `BOTTLENECK_DROPOUT` | `0.0` | `0.1`, `0.2` | dropout espacial en el mapa /32 como regularizador |
| `FREEZE_STAGES` | `0` | `1`, `2` | congelar las primeras etapas (features genéricas de ImageNet) reduce capacidad y sobreajuste |
| `ADAPTER_TYPE` | `"conv1x1"` | `"custom"` | tu propio adaptador (ver §4) |

⭐ = ablación asignada en el plan del grupo.

**Regla de oro:** *una hipótesis → un cambio → una medición*. Si cambias dos cosas a la vez, no sabrás cuál causó la mejora (y debes reportarlo así).

**No puedes** cambiar el augmentation, el decoder, la pérdida, las épocas ni el umbral oficial: el notebook se detiene con un `KeyError` que te lista los parámetros permitidos.

## 4. Si quieres programar (opcional)

En la sección **7.2 · Zona de código libre** hay una clase `CustomResNetAdapter` de ejemplo (1×1 + conv 3×3 extra en /16 y /32). Puedes editarla y activarla con `{"ADAPTER_TYPE": "custom"}`. Reglas:

- misma firma `__init__(self, in_chs, out_chs, cfg)` y `forward(self, feats, size)`;
- debe devolver 4 mapas con canales `(64, 128, 256, 512)` en strides /4, /8, /16, /32;
- la celda 7.3 te muestra las formas de cada tensor: si algo no cuadra, ahí lo verás antes de entrenar;
- el ejemplo tal cual suma **~3 M parámetros** al adaptador (1.39 M → 4.35 M) por la 3×3 de 512 canales. Si mejora, parte de la ganancia puede ser solo capacidad extra: decláralo en el informe (o quita la 3×3 de /32 para que sea más liviano).

## 5. Cómo ejecutarlo

1. Kaggle → *New Notebook* → *File → Import* el `.ipynb`.
2. *Settings*: **GPU T4**, **Internet On**. *Add Input*: `balraj98/massachusetts-roads-dataset`.
3. Pon `SMOKE_TEST = True` y *Run All* (≈ 5–10 min): verifica que no hay errores.
4. Pon `SMOKE_TEST = False` y *Run All* de nuevo. Cada corrida completa tarda del orden de 1–2 h en T4 (estimación; anota el tiempo real que reporta el notebook). Con `base` + 1 ablación cabe holgado en una sesión de 12 h.
5. Descarga `outputs_resnet50.zip` (pestaña *Output*) y envíalo al grupo.

## 6. Qué reportar

La sección 9 genera esta tabla automáticamente (`results/resnet50__comparison.csv`):

| Corrida | Cambio | IoU_road | Δ IoU_road (pts) | F1 | Relaxed F1 | Δ Relaxed F1 | IoU_road @thr_val | IoU_road TTA | Tiempo | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| base | — | | 0 | | | 0 | | | | |
| wd_1e-3 | `WEIGHT_DECAY = 1e-3` | | | | | | | | | |

Para el informe, en 3–4 líneas: (1) la hipótesis, (2) el resultado con Δ, (3) la curva de validación (`figures/resnet50__comparison_curves.png`): ¿se redujo el estancamiento?, (4) si la mejora es mayor que el ruido esperado (diferencias < 0.5 pts con una sola semilla son poco concluyentes).

**Respaldo bibliográfico:** Banerjee et al. (2025), §4 (monitoreo de sobreajuste/regularización) y §8 (análisis de errores FP/FN con las figuras cualitativas).

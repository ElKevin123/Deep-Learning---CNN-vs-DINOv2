# Guía · Integrante 2 · `dinov2_frozen` (linear probing con decodificador)

**Notebook:** `07_dinov2_frozen_ablacion.ipynb` · **Proyecto:** Backbone Swap for Road Segmentation — DINOv2 vs. ResNet-50 (Massachusetts Roads, parches 512 × 512)

> **En una frase:** corres tu modelo con la configuración común del grupo (fila `base`) y luego una o más ablaciones que **solo** cambian el *neck* o el entrenamiento de la cabeza. El backbone DINOv2 nunca aprende: mides qué tan buenas son sus features tal cual.

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
512×512 ─► resize 518 ─► patch_embed 14×14 ─► 37×37 tokens × 384 ─► 12 bloques ViT (TODOS congelados)
                                                     bloques 2, 5, 8, 11 ─► 4 mapas (384, 37, 37)
NECK:  bloque 2 → bilinear 128×128 → 1×1 → 64     bloque 5 → bilinear 64×64 → 1×1 → 128
       bloque 8 → bilinear  32×32 → 1×1 → 256     bloque 11 → bilinear 16×16 → 1×1 → 512
                              ─► UNetDecoder común ─► máscara 512×512
```

- Backbone: 21.6 M parámetros, **0 entrenables**. Solo aprenden el neck (~0.37 M) y el decoder (3.10 M).
- Su fortaleza: features auto-supervisadas muy generales y poca VRAM al entrenar (~1.3 GB vs ~3 GB de ResNet).
- Su debilidad: cada token cubre 14 × 14 px. Para el skip /4 hay que **ampliar ×3.5** un mapa de 37 × 37 con interpolación bilineal → efecto cuadrícula y bordes borrosos. En 03–05: `IoU_road` 0.576, pero `Relaxed F1` 0.897 (solo ~1.4 pts debajo de ResNet): **detecta la vía, pero no alinea el borde al píxel**.

## 3. Qué puedes modificar

Solo editas la celda **2 · `RUNS`**. Cada entrada es una corrida completa:

```python
RUNS = {
    "base": {},                                     # NO tocar: tu fila del benchmark común
    "neck_3x3": {"NECK_TYPE": "bilinear_3x3"},      # ablación asignada
    # "lr_dec_6e-4": {"LR_DECODER": 6e-4},          # descomenta para agregar otra corrida
}
```

| Parámetro | Base | Valores sugeridos | Hipótesis que prueba |
|---|---|---|---|
| `NECK_TYPE` ⭐ | `"bilinear_1x1"` | `"bilinear_3x3"`, `"custom"` | una conv 3×3 tras el reescalado suaviza la cuadrícula de 14 px y mejora bordes |
| `NECK_REFINE_SCALES` | `(4, 8)` | `(4,)`, `(4, 8, 16, 32)` | en qué escalas importa refinar (solo con `bilinear_3x3`). Ojo: `(4, 8)` suma ~0.3 M parámetros, pero `(4, 8, 16, 32)` suma ~3.1 M (la 3×3 de 512 canales es cara): decláralo como ventaja de capacidad |
| `LR_DECODER` | `3e-4` | `6e-4`, `1e-3` | sin backbone que proteger, la cabeza puede aprender más rápido |
| `EPOCHS` | `20` | `30` | en 03–05 la validación seguía subiendo en la época 20 (sub-entrenamiento). Reporta el costo extra de tiempo |
| `DINO_LAYER_IDS` | `(2, 5, 8, 11)` | `(3, 6, 9, 11)`, `(5, 7, 9, 11)` | ¿features más profundas en los skips ayudan aunque sean más abstractas? |
| `WEIGHT_DECAY` | `1e-4` | `1e-3` | regularización de la cabeza |

⭐ = ablación asignada en el plan del grupo.

**Optimización del umbral:** ya no tienes que hacerla a mano. Los 3 notebooks barren `0.35 … 0.60` en **validación**, eligen el mejor y lo aplican a test (columna `IoU_road@thr_val`). Como se hace igual para los 3 modelos, también es comparable; pero la tabla principal sigue usando 0.5.

**Regla de oro:** *una hipótesis → un cambio → una medición*. **No puedes** cambiar el augmentation, el decoder, la pérdida ni el umbral oficial: el notebook se detiene con un `KeyError` que te lista los permitidos.

## 4. Si quieres programar (opcional)

En la sección **7.2 · Zona de código libre** hay un `CustomNeck` de ejemplo que **aprende a ampliar** con `ConvTranspose2d` (37 → 148 para /4, 37 → 74 para /8) seguido de una conv 3×3. Lo activas con `{"NECK_TYPE": "custom"}`. Reglas:

- firma `__init__(self, embed_dim, out_chs, cfg)` y `forward(self, feats, size)`;
- recibe 4 mapas `(B, 384, 37, 37)` y devuelve 4 mapas con canales `(64, 128, 256, 512)` y tamaños 128, 64, 32, 16;
- la celda 7.3 muestra las formas y los parámetros entrenables de cada corrida: revisa que el neck no crezca demasiado (más parámetros en el neck = ventaja injusta que hay que declarar).

Otras ideas: `PixelShuffle`, sumar el mapa de un bloque vecino antes de reescalar, o una conv *depthwise* 3×3.

## 5. Cómo ejecutarlo

1. Kaggle → *New Notebook* → *File → Import* el `.ipynb`.
2. *Settings*: **GPU T4**, **Internet On** (DINOv2 se baja de `torch.hub`). *Add Input*: `balraj98/massachusetts-roads-dataset`.
3. Pon `SMOKE_TEST = True` y *Run All* (≈ 5–10 min): verifica que no hay errores.
4. Pon `SMOKE_TEST = False` y *Run All* de nuevo. Cada corrida completa tarda del orden de 1–2 h en T4 (estimación; anota el tiempo real que reporta el notebook). Si usas `EPOCHS = 30`, cuenta ~50 % más.
5. Descarga `outputs_dinov2_frozen.zip` (pestaña *Output*) y envíalo al grupo.

## 6. Qué reportar

La sección 9 genera esta tabla automáticamente (`results/dinov2_frozen__comparison.csv`):

| Corrida | Cambio | IoU_road | Δ IoU_road (pts) | F1 | Relaxed F1 | Δ Relaxed F1 | IoU_road @thr_val | IoU_road TTA | Entrenables (M) | Tiempo |
|---|---|---|---|---|---|---|---|---|---|---|
| base | — | | 0 | | | 0 | | | | |
| neck_3x3 | `NECK_TYPE = bilinear_3x3` | | | | | | | | | |

Para el informe, en 3–4 líneas: (1) la hipótesis, (2) el resultado con Δ, (3) **la brecha `F1` vs `Relaxed F1`**: si el neck mejora el borde, `IoU_road`/`F1` deberían subir más que `Relaxed F1` (la brecha se cierra), (4) cuántos parámetros entrenables añadió tu neck.

**Respaldo bibliográfico:** Banerjee et al. (2025), §4 (diagnóstico de sub-entrenamiento por curvas) y Tabla 1 (elección de métricas); Wang (2023), *robustness vs. efficiency* (DINOv2 cuesta ~109 GFLOPs vs ~59 de ResNet).

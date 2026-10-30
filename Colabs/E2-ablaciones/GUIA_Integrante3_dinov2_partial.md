# Guía · Integrante 3 · `dinov2_partial` (fine-tuning selectivo)

**Notebook:** `08_dinov2_partial_ablacion.ipynb` · **Proyecto:** Backbone Swap for Road Segmentation — DINOv2 vs. ResNet-50 (Massachusetts Roads, parches 512 × 512)

> **En una frase:** corres tu modelo con la configuración común del grupo (fila `base`, 2 bloques descongelados) y luego una o más ablaciones que **solo** cambian cuántos bloques de DINOv2 aprenden y con qué LR. Todo lo demás está bloqueado para que la comparación sea justa.

---

## 1. Lo que es idéntico en los 3 notebooks (no se toca)

| Invariante | Valor |
|---|---|
| Dataset | Massachusetts Roads, re-split **por tile** `SPLIT_SEED = 42` → 820 / 176 / 175 tiles |
| Parches | rejilla fija 3 × 3 de 512 × 512 (inicios 0, 494, 988), se descartan los de > 25 % blanco. Se generan una vez al inicio del notebook |
| Data augmentation | **la misma para los 3 modelos**: volteos H/V, rot90, brillo/contraste/saturación ±15 %, **sombras sintéticas** (p = 0.3) |
| Decodificador | `UNetDecoder`, canales `(64, 128, 256, 512)`, **3,098,433 parámetros** (se verifica con `assert`) |
| Pérdida (base) | BCE + Dice |
| Entrenamiento | 20 épocas, batch 8, AdamW, warmup 5 % + coseno, AMP, grad clip 1.0, semilla 42 |
| Métricas | `MetricAccumulator` global: `IoU_road`, `mIoU`, `F1`, P/R, F1 relajado ρ = 3 px. **Oficial: umbral 0.5 sin TTA** |

El notebook calcula dos **huellas**: la de la configuración común y la del dataset. Las tres personas deben obtener **las mismas** (compárenlas por el chat del grupo). Si una huella no coincide, sus resultados no son comparables.

## 2. Tu arquitectura en 30 segundos

```
512×512 ─► resize 518 ─► patch_embed ─► 1369 tokens × 384
  bloques 0-1   congelados
  bloque 2  ───► skip /4        (congelado)
  bloques 3-4   congelados
  bloque 5  ───► skip /8        (congelado)
  bloques 6-7   congelados
  bloque 8  ───► skip /16       ◄── se descongela con N_UNFREEZE = 4
  bloque 9
  bloque 10     ◄── base: N_UNFREEZE = 2 descongela 10 y 11 (+ norm final)
  bloque 11 ───► skip /32
     ─► neck (bilinear + 1×1) ─► UNetDecoder común ─► máscara 512×512
```

- **La clave:** la salida del bloque *k* solo depende de los bloques 0…*k*. Con la base (10–11 descongelados), los skips /4, /8 y /16 **no cambian**: solo cambia el mapa /32. Por eso en 03–05 `partial` superó a `frozen` por apenas **+0.5 pts** (0.581 vs 0.576).
- Con `N_UNFREEZE = 4` (bloques 8–11) el skip /16 también se adapta a texturas aéreas. Cada bloque ViT-S tiene ~1.8 M parámetros: con 4 bloques entrenas ~7 M + neck + decoder.
- Costo: cada bloque descongelado guarda su grafo de activaciones → más VRAM y tiempo (lo verás en `VRAM_MB` y `tiempo_min`).

## 3. Qué puedes modificar

Solo editas la celda **2 · `RUNS`**. Cada entrada es una corrida completa:

```python
RUNS = {
    "base": {},                          # NO tocar: tu fila del benchmark común (N_UNFREEZE = 2)
    "unfreeze_4": {"N_UNFREEZE": 4},     # ablación asignada
    # "lr_tail_3e-5": {"LR_DINO_TAIL": 3e-5},   # descomenta para agregar otra corrida
}
```

| Parámetro | Base | Valores sugeridos | Hipótesis que prueba |
|---|---|---|---|
| `N_UNFREEZE` ⭐ | `2` | `4`, `6` | adaptar el bloque que alimenta el skip /16 mejora la alineación de bordes |
| `UNFREEZE_BLOCKS` | `None` | `(8, 11)`, `(5, 8, 11)` | descongelar **solo** los bloques que salen a skips (más barato que la cola completa). Si lo usas, ignora `N_UNFREEZE` |
| `LR_DINO_TAIL` | `1e-5` | `3e-5`, `5e-5` | con 1e-5 durante 20 épocas los pesos del ViT apenas se mueven |
| `LLRD_DECAY` | `1.0` | `0.8`, `0.9` | *layer-wise LR decay*: el bloque más profundo usa `LR_DINO_TAIL`, el anterior × 0.8, etc. Protege las features generales |
| `LR_DECODER` | `3e-4` | `5e-4` | LR del neck + decoder |
| `WEIGHT_DECAY` | `1e-4` | `1e-3` | regularización |
| `LOSS_TYPE` (secundario) | `"bce_dice"` | `"focal_dice"`, `"custom"` | foco en píxeles difíciles (bordes) |
| `BCE_POS_WEIGHT` (secundario) | `1.0` | `3.0` | penaliza más los falsos negativos |
| `FOCAL_GAMMA` (secundario) | `2.0` | `1.0`, `3.0` | solo con `focal_dice` |

⭐ = ablación asignada en el plan del grupo.

**Sobre la pérdida:** es la única condición a la que se le permite cambiarla, y solo como ablación **secundaria**. La comparación principal entre los 3 modelos siempre usa la fila `base` (BCE + Dice). Repórtala en una fila aparte, nunca junto con `N_UNFREEZE`.

**Regla de oro:** *una hipótesis → un cambio → una medición*. El ejemplo comentado `unfreeze_4_llrd` cambia 3 cosas a la vez: si lo usas, decláralo así. **No puedes** cambiar el augmentation, el decoder, las épocas ni el umbral oficial: el notebook se detiene con un `KeyError` que te lista los permitidos.

## 4. Si quieres programar (opcional)

En la sección **7.2 · Zona de código libre** hay una pérdida de ejemplo `EdgeWeightedBCEDice`: BCE con **más peso en los píxeles de borde** de la vía (borde = dilatación − erosión de la máscara) + Dice. La activas con `{"LOSS_TYPE": "custom"}` y puedes ajustar `EDGE_WEIGHT` y `EDGE_RADIUS` dentro de la clase. Reglas:

- se construye como `LOSSES["custom"](cfg)`;
- `forward(logits, target)` recibe `(B, 1, 512, 512)` y devuelve un escalar;
- calcula en `float32` (`logits.float()`), igual que la base, para no tener problemas con AMP.

## 5. Cómo ejecutarlo

1. Kaggle → *New Notebook* → *File → Import* el `.ipynb`.
2. *Settings*: **GPU T4**, **Internet On** (DINOv2 se baja de `torch.hub`). *Add Input*: `balraj98/massachusetts-roads-dataset`.
3. Pon `SMOKE_TEST = True` y *Run All* (≈ 5–10 min): verifica que no hay errores.
4. Pon `SMOKE_TEST = False` y *Run All* de nuevo. Cada corrida completa tarda del orden de 1–2 h en T4 (estimación; `N_UNFREEZE = 4` es más lenta que la base). Si te quedas sin VRAM con muchos bloques, avisa al grupo antes de bajar el batch: el batch es común.
5. Descarga `outputs_dinov2_partial.zip` (pestaña *Output*) y envíalo al grupo.

## 6. Qué reportar

La sección 9 genera esta tabla automáticamente (`results/dinov2_partial__comparison.csv`):

| Corrida | Cambio | IoU_road | Δ IoU_road (pts) | F1 | Relaxed F1 | Δ Relaxed F1 | Entrenables (M) | Tiempo | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| base | `N_UNFREEZE = 2` | | 0 | | | 0 | | | |
| unfreeze_4 | `N_UNFREEZE = 4` | | | | | | | | |

Para el informe, en 3–4 líneas: (1) la hipótesis (skips congelados limitan a `partial`), (2) el resultado con Δ, (3) **el costo**: Δ de parámetros entrenables, VRAM y tiempo por cada punto de IoU ganado, (4) si `F1` sube más que `Relaxed F1` (mejor alineación de bordes) o ambos por igual (mejor detección).

**Respaldo bibliográfico:** Banerjee et al. (2025), §4 (sub/sobreajuste) y §8 (análisis de errores); Wang (2023), *robustness vs. efficiency* (más bloques adaptados = más costo); D13 de `DECISIONS.md`.

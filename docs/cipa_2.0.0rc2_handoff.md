# `cipa` 2.0.0rc2: redefinir D₅ para CIPA Extended

> **Origen:** decisión 5 de `cipa-extended/results/cipa/cipa_protocol.md`, 2026-09-17, tomada
> tras dos pruebas piloto sobre `v2.0.0rc1`. Los resultados están en
> `cipa-extended/results/cipa/pilot/`: `scaling_pilot_summary.md` y `d5_candidates.json`.
> **Repositorio de trabajo:** `/home/luisgarcia/projects/unam/dcic/2026-2/statistical_analysis`.
> Punto de partida: `0a82f32` sobre el tag `v2.0.0rc1`.
> **Alcance:** sólo D₅. Todo lo demás de rc1 queda igual.

Documento de trabajo: consúmelo y bórralo, o muévelo a `docs/`.

---

## 1. Por qué

Con z-score (decisión 4, que se mantiene), la entropía espectral normalizada de D₅ mide **falta de
redundancia**, no dimensionalidad. Supera 0.5 en 24 de los 28 datasets del estudio, y la signatura
IV cubre 15 de 20. Ejemplo: `pima_diabetes`, con 8 variables, sale «dominado por la
dimensionalidad» con D₅ = 0.93. Sin escalar, la misma fórmula refleja las unidades de medida.

Se evaluaron cuatro redefiniciones con criterios fijados antes de correrlas. Se eligió la que
relativiza la dimensionalidad efectiva al tamaño de la minoría.

## 2. Nueva definición

Sobre los datos que ya recibe `compute_d5` (estandarizados, sin constantes y con N completo):

```
r_95 = mínimo número de componentes principales cuya varianza explicada acumulada ≥ 0.95
ρ    = r_95 / |C₊|
D₅   = ρ / (1 + ρ)          ∈ [0, 1); vale 0.5 cuando r_95 = |C₊|
```

- **PCA:** la misma llamada que hoy, `PCA(n_components=min(N−1, d), random_state)`. Así r_95
  queda acotado por min(N−1, d); con d > N, por ejemplo `tcga_brca` (d = 5,000, N = 826), el
  tope es N−1.
- **r_95:** `np.searchsorted(np.cumsum(evr), 0.95) + 1`, acotado a `len(evr)`. Documenta el
  manejo del redondeo si la suma acumulada nunca llega exactamente a 0.95.
- **|C₊|:** `dataset.n_minority`.
- **Casos degenerados:** d = 1 o N ≤ 2 → D₅ = 0, como hoy. Con una sola componente, r_95 = 1 y
  D₅ = 1/(1 + |C₊|), sin caso especial.
- **La transformación ρ/(1+ρ)** es la misma que ya usa N2 en D₇. Menciónalo en el docstring.

**Componentes que debe reportar `DimensionResult`:**
- `r_95` y `n_minority`;
- `rho`;
- `H_nats`, `H_max_nats` y `spectral_entropy_norm`: la D₅ de v1.x, **informativa**, que no
  entra al valor. Sirve para comparar con COMIA;
- `n_components_fit`, como hoy.

Metadatos: los actuales (`d`, `n`, `top5_explained_variance_ratio`) y `variance_threshold` = 0.95.

**Umbral 0.95:** constante en `_constants.py` (`D5_VARIANCE_THRESHOLD`). No lo expongas en
`CIPAPipeline` salvo que sea trivial; el protocolo lo fija.

## 3. Valores de referencia (prueba piloto, `v2.0.0rc1` + script externo)

Con `scaling="standard"`, semilla 42 y N completo, la implementación debe reproducir estos
valores. Úsalos como prueba de integración si tienes acceso a imbdata; si no, como verificación
manual:

| Dataset | N | d usado | \|C₊\| | r_95 | D₅ nueva |
|---|---|---|---|---|---|
| `tcga_brca` | 826 | 5,000 | 147 | 536 | 0.785 |
| `secom` | 1,567 | 446 | 104 | 162 | 0.609 |
| `ozone_level` | 2,536 | 72 | 73 | 20 | 0.215 |
| `pima_diabetes` | 768 | 8 | 268 | 8 | 0.029 |
| `abalone_19` | 4,177 | 8 | 32 | 3 | 0.086 |
| `credit_card_fraud` | 284,807 | 30 | 492 | 27 | 0.052 |

## 4. Efectos colaterales que hay que atender

1. **Prueba de regresión contra v1.2.1:** D₅ ya no coincide.
   - Excluye D₅ de la igualdad exacta.
   - Compara en su lugar el componente `spectral_entropy_norm` con el D₅ de v1.2.1, que debe
     seguir siendo exacto con `scaling="none"`.
   - D₁–D₄, D₆ y D₇ siguen exactos.
2. **Signaturas:** la regla no cambia, pero la signatura IV se volverá rara: en la prueba piloto
   sólo `tcga_brca` de 20.
   - La tabla de 18 casos de `test_indexing_profiling_action.py` usa vectores sintéticos y no
     debería cambiar.
   - Revisa la prueba de los 13 perfiles de COMIA: si usa D₅ recalculado, cambia.
3. **Protocolo de acción** (fuera de alcance, sin tocar): consume D₅ con umbral ≥ 0.70 y cambiará
   sus salidas. Anótalo en el CHANGELOG, igual que en rc1.
4. **README y docstrings:** D₅ se llama «dimensionalidad efectiva relativa a la minoría». Actualiza
   la fórmula donde aparezca.

## 5. Pruebas mínimas

- Valor exacto de D₅ en casos sintéticos donde r_95 se conoce por construcción (p. ej., datos
  en un subespacio de k dimensiones con ruido despreciable → r_95 = k).
- Monotonía: a igual espectro, más minoritarias → menor D₅.
- Acotamiento en [0, 1) y D₅ = 0.5 cuando r_95 = |C₊|.
- `spectral_entropy_norm` igual al D₅ de v1.2.1 (regresión).
- Casos degenerados (d = 1, N ≤ 2, una sola componente).
- Si hay acceso a imbdata: los valores de §3 con tolerancia 1e-3.
- Cobertura ≥ 90 %, ruff y pytest en verde.

## 6. Versión y entrega

- CHANGELOG `[2.0.0rc2]`:
  - la redefinición de D₅ y su motivo (con referencia a la prueba piloto);
  - los efectos sobre signaturas y protocolo de acción;
  - que la entropía espectral sigue disponible como componente.
- Commit y **tag `v2.0.0rc2`**, con push a GitHub como en rc1.
- Guía de vuelta `cipa-extended/cipa_2.0.0rc2_handoff.md`: qué cambió, desviaciones, resultados de
  las pruebas, y si los valores de §3 se reprodujeron.
- **No hace falta reinstalar en cipa-extended desde esta sesión:** la reinstalación del `base` de
  conda la hace cipa-extended al recibir la guía de vuelta.

## 7. Lista de verificación

- [ ] D₅ = ρ/(1+ρ) con ρ = r_95/|C₊|
- [ ] Componentes: `r_95`, `n_minority`, `rho` y `spectral_entropy_norm`
- [ ] Regresión v1.2.1 adaptada (D₅ vía `spectral_entropy_norm`)
- [ ] Pruebas §5 y valores de referencia §3
- [ ] README, docstrings y CHANGELOG `[2.0.0rc2]`
- [ ] Commit, tag `v2.0.0rc2`, push y guía de vuelta

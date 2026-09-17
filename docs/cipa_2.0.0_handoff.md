# `cipa` 2.0.0: corregir la implementación para CIPA Extended

> **Origen:** Fase 2.1 de cipa-extended, 2026-09-16. Las cuatro decisiones y su razonamiento
> están en `cipa-extended/results/cipa/cipa_protocol.md`; esta guía las traduce a cambios
> concretos.
> **Repositorio de trabajo:** `/home/luisgarcia/projects/unam/dcic/2026-2/statistical_analysis`
> (remoto `git@github.com:luisgarciarodriguez-research/cipa.git`). Árbol limpio en `6ea039f` = tag
> `v1.2.1`.
> **Consumidores:**
> - **`cipa-fraud`** ya está aislado: `cipa` se instaló desde el tag `v1.2.1`, sin modo editable,
>   y su huella de resultados es idéntica a la anterior. Los cambios en este árbol no le llegan.
> - **El `base` de conda** tiene todavía una instalación editable de este árbol (metadatos
>   1.0.0). cipa-extended aún no importa `cipa`, así que no rompe nada; se reemplaza al fijar
>   2.0.0 (§6).

Documento de trabajo: consúmelo y bórralo, o muévelo a `docs/`.

---

## 0. Antes de empezar

1. **Versión:** el tag `v1.2.1` apunta a `6ea039f`, pero `pyproject.toml` dice `1.2.0` y el
   CHANGELOG no tiene entrada 1.2.1. Anótalo en el CHANGELOG de 2.0.0; no reescribas el tag.
2. **Convenciones del repositorio:** docstrings estilo NumPy, cabecera de módulo con la cita de
   COMIA y cobertura mínima del 90 %. Respétalas.
3. **No toques `paper/`.** La fe de erratas del artículo se redacta en el manuscrito de la
   extensión.

## 1. Qué NO cambia: las fórmulas

Se conservan las definiciones de v1.1.0 tal como están en el código, **no** como las describe
el artículo:
- D₁ = 1 − H(Y) (bits).
- D₂ = (F3 + N1 + kDN)/3, con kDN = media sobre **todas** las instancias de la fracción de sus
  k = 5 vecinos con otra etiqueta.
- D₃ = (B + 2R + 3O)/(3·|C₊|), con k = 5 (3 de 5 vecinos de la misma clase = *safe*).
- D₄ = ECindex·√(n_clusters/|C₊|), con DBSCAN sobre la minoría: `min_samples` = 3; eps = mediana
  de la distancia al 2.º vecino; ruido = grupos unitarios; si todo es ruido, se duplica eps una
  vez.
- D₅ = entropía espectral normalizada de la PCA.
- D₆ = 1 − Ī/H(Y), con `mutual_info_classif` (k = 3).
- D₇ = (L1 + N2/(1+N2))/2, con L1 = error de entrenamiento de `LinearSVC(class_weight="balanced")`.
- DS = Σ wᵢDᵢ. Bandas: Low < 0.25 ≤ Moderate < 0.50 ≤ High < 0.75 ≤ Extreme.

**Prueba de regresión obligatoria (§5):** sobre datos sin duplicados, con `scaling="none"`,
sin submuestreo y con la misma semilla, D₁–D₆ deben coincidir con v1.2.1. D₇ también, salvo
cuando L1 no convergía (C5).

## 2. Correcciones

### C1 · Clase minoritaria explícita
- `CIPADataset.from_arrays` elige la minoría por frecuencia. En COMIA eso invirtió PaySim tras
  submuestrear. En 2.0.0 la API pública debe recibir `minority_label` **explícito**.
- `from_arrays` puede quedar como conveniencia, pero sin ser el camino del pipeline, o
  deprecarse.
- Si la «minoritaria» declarada es mayoría: registra una advertencia y deja constancia en los
  metadatos del resultado. **No** la invierte.

### C2 · Preprocesamiento: constantes fuera y z-score
Parámetro nuevo `scaling: Literal["standard", "robust", "none"] = "standard"`, aplicado **una
vez por dataset sobre N completo, antes de cualquier dimensión y de cualquier submuestreo**.
1. Eliminar columnas con σ = 0 y registrar cuántas y cuáles en el resultado.
2. `standard`: media 0, desviación 1.
   `robust`: mediana/IQR, **sólo para análisis de sensibilidad**; si una columna tiene IQR = 0,
   usa su σ, y si también es 0 ya salió en el paso 1. Documenta ese respaldo.
   `none`: sin escalar, para la prueba de regresión.
3. D₁, F3 y D₆ no dependen de la escala, pero reciben los mismos datos, ya sin constantes. Esto
   corrige D₆, que antes promediaba ceros de columnas constantes.

### C3 · Vecinos que toleran duplicados
- `_KNNCache` pide k+1 vecinos y descarta la **columna 0** suponiendo que es la propia instancia.
  Con duplicados, la columna 0 puede ser un gemelo. Hay que **descartar por índice**: quitar la
  aparición del propio índice y quedarse con los k siguientes.
- Los duplicados de otras filas **sí** cuentan como vecinos. Un gemelo con otra etiqueta es
  traslape real.
- Aplica a kDN (D₂), D₃ y N2 (D₇).

### C4 · N1 sin matriz densa y con duplicados
- `ecol/n1.py` construye `squareform(pdist(X))`: unos 20 GB con 50k filas. Además, el árbol de
  expansión mínima de scipy trata las entradas 0 como aristas ausentes, así que los duplicados
  quedan fuera del árbol.
- Requisito: **árbol de expansión mínima euclidiano exacto** con memoria O(N·k). Sugerencia:
  Borůvka con consultas a KD-tree/BallTree que excluyan el propio componente.
- **Aristas de distancia 0 = aristas válidas.** Dos gemelos con etiquetas distintas son ambos
  frontera.
- **Prueba:** igual al método denso en datos pequeños (N ≤ 2,000), con y sin duplicados, salvo
  empates de peso. Con empates, el conjunto de nodos frontera puede diferir; documenta la regla
  de desempate.

### C5 · Convergencia de L1
- Hoy L1 = 0.5 si `LinearSVC` no converge, sólo con un aviso en el log.
- En 2.0.0: `max_iter` configurable, 10,000 por defecto. **Siempre** se usa la tasa de error
  obtenida, con `converged: bool` en los componentes. Se elimina el 0.5 fijo.
- Con datos estandarizados la convergencia debería ser la norma; reporta cuántos no convergen.

### C6 · Submuestreo por dimensión (decisión 3)
Reemplaza `knn_subsample`, el submuestreo asimétrico y los submuestreos internos de N1/N2 por
un único protocolo configurable (`n_max` = 50,000, `n_subsamples` = 5, `random_state`):

| Dimensión | Datos | Salida |
|---|---|---|
| D₁, D₅, D₆ | **N completo** | valor |
| D₃ | **Todas las minoritarias** como consulta; vecinos buscados en el **dataset completo** | valor |
| D₄ | **Toda la minoría**; si \|C₊\| > `n_max`, `n_subsamples` submuestras de `n_max` minoritarias | valor (mediana) + IQR |
| D₂ (F3, N1, kDN), D₇ (L1, N2) | Si N > `n_max`: `n_subsamples` submuestras **estratificadas que conservan el IR**, de tamaño `n_max`; si no, N completo | mediana + IQR, por dimensión y por componente |

- El DS se calcula con las medianas.
- Cada `DimensionResult` registra: N usado, número de submuestras, IQR y semillas derivadas.
- Las semillas se derivan de `random_state`, no de `np.random` global, y cada submuestra es
  reproducible.
- **Conserva la posibilidad de cambiar `n_max`.** cipa-extended hará la sensibilidad
  N_max ∈ {10K, 25K, 50K, 100K}.
- **Rendimiento (lo mide la prueba piloto de cipa-extended, no esta tarea):**
  - D₃ exacto en `cic_ids_2017`: 425,741 consultas contra 2.5 M filas en 70 dimensiones.
  - D₆ con N completo en `paysim`: 6.4 M filas.

  Que el cálculo acepte `n_jobs` y consultas por bloques. No optimices antes de tener tiempos
  medidos.

### C7 · Signaturas: regla del artículo (decisión 2)
Reemplaza la regla de `profiling.py` (prioridades y umbrales 0.25/0.55) por la del artículo:
- **Dominancia:** D_i domina si D_i > τ = 0.50 **y** D_i = máx{D₁, D₂, D₄, D₅}.
- I, II, III y IV corresponden a que domine D₁, D₂, D₄ o D₅. Si hay empate en el máximo, el
  orden es D₁ > D₂ > D₄ > D₅; documéntalo.
- **V:** ninguna domina.
- **Hueco del artículo, regla CONFIRMADA por el autor el 2026-09-16.** El artículo
  define V como «≥ 2 dimensiones > τ′ = 0.35» o «todas < 0.35», y deja sin cubrir el caso en que
  ninguna domina y exactamente una supera 0.35. **Regla:** V es la categoría residual
  (ninguna domina), con un calificativo informativo que no cambia la signatura:
  - `compound`: ≥ 2 dimensiones > 0.35;
  - `single`: exactamente una > 0.35;
  - `low`: todas ≤ 0.35.
- τ y τ′ configurables.

### C8 · Protocolo de acción
Fuera de alcance. No lo modifiques. Sigue usando la signatura, y con la regla nueva sus salidas
pueden cambiar; anótalo en el CHANGELOG como efecto conocido y no corregido.

### C9 · Determinismo
`random_state` es obligatorio en el pipeline, o 42 por defecto; nunca `None`. Se propaga a MI,
`LinearSVC`, el submuestreo y cualquier otra fuente aleatoria.

## 3. API que necesita cipa-extended

Una llamada por dataset:

```python
result = CIPAPipeline(
    weights=w_expert, random_state=42, scaling="standard",
    n_max=50_000, n_subsamples=5, n_jobs=-1,
).run_scoring_only(CIPADataset(X, y, minority_label=1, majority_label=0, name=key))
```

Salida serializable (`to_dict`) con:
- D₁–D₇: valor, IQR y componentes (F3, N1, kDN, L1, N2, ECindex, n_clusters, tipos B/R/O, etc.);
- metadatos: N usado por dimensión, submuestras, columnas constantes eliminadas, `converged` de
  L1 y tiempo por dimensión;
- DS, banda, signatura y su calificativo.

cipa-extended calcula el DS con los otros tres vectores de pesos a partir de los Dᵢ, así que
basta con que los pesos sean parámetro.

## 4. Versión y documentación

- **2.0.0**, porque cambian los valores: escalado, submuestreo, duplicados, L1 y signaturas.
- CHANGELOG `[2.0.0]`:
  - cada corrección, con el efecto esperado sobre los valores;
  - la aclaración de que v1.2.1 conservaba `pyproject` en 1.2.0;
  - una sección **«Hallazgos sobre los valores publicados en COMIA»**:
    - PaySim con la clase invertida (D₁ = 0.3228 = 1 − H(0.1787));
    - D₁ calculado sobre la submuestra de 10k (CreditCard 0.717 publicado; 0.982 con N completo);
    - IEEE-CIS con 350 minoritarias según la revisión, **sin verificar**;
    - el artículo y el código difieren en kDN, DBSCAN, signaturas y acción.
- README: actualizar la sección de API y la de signaturas.
- `specs/` están desactualizados respecto del código (02, 05, 06, 08 y 10). No es obligatorio
  actualizarlos en esta tarea; anótalo como pendiente.

## 5. Pruebas mínimas

- **Regresión v1.2.1** (§1) en ≥ 3 datasets sintéticos sin duplicados.
- **N1:** exacto contra denso, con y sin duplicados. Memoria acotada con N = 50,000 (que no
  construya N×N).
- **Vecinos con duplicados:** un gemelo con otra etiqueta cuenta en kDN y en D₃.
- **Escalado:** elimina constantes y las reporta; `robust` con IQR = 0 no produce inf/NaN; D₆ no
  se contamina con constantes.
- **Submuestreo:**
  - determinista con semilla;
  - las submuestras de D₂/D₇ conservan el IR (±1 instancia);
  - D₃ usa todas las minoritarias;
  - D₄ submuestrea sólo por encima de `n_max`.
- **L1:** sin 0.5 fijo; `converged` reportado.
- **Signaturas:** una tabla de casos que cubra dominancia de cada Dᵢ, empates, τ exacto (no
  domina), D₃/D₆/D₇ > 0.5 sin dominancia, y los tres calificativos de V.
- **Minoría declarada que resulta mayoría:** advierte, no invierte.
- Cobertura ≥ 90 % y ruff/pytest en verde.

## 6. Entrega

1. Commit(s) en `statistical_analysis` con el estilo del repositorio.
2. **Tag `v2.0.0rc1`**, no 2.0.0 todavía: la prueba piloto de cipa-extended puede exigir ajustes
   (decisión 3, piso de minoritarias).
3. Dejar en `cipa-extended/` una guía de vuelta, `cipa_2.0.0rc1_handoff.md`, con:
   - qué quedó implementado;
   - la API final;
   - cómo instalar (`pip install --no-deps` desde el tag);
   - desviaciones respecto de esta guía;
   - cómo quedó implementada la regla de la signatura V.
4. En cipa-extended, después: instalar rc1 en el `base` de conda (reemplaza la instalación
   editable), prueba piloto de tiempos y de escalado, ajustes, tag `v2.0.0`, y fijar
   `cipa>=2.0,<3.0`.

## 7. Lista de verificación

- [ ] C1 minoría explícita
- [ ] C2 constantes fuera + `scaling`
- [ ] C3 vecinos con duplicados
- [ ] C4 N1 exacto sin matriz densa
- [ ] C5 L1 sin 0.5 fijo
- [ ] C6 submuestreo por dimensión con IQR
- [ ] C7 signaturas del artículo, con la regla de V confirmada (calificativos `compound`/`single`/`low`)
- [ ] C8 acción sin cambios, efecto anotado
- [ ] C9 determinismo
- [ ] Pruebas §5, cobertura ≥ 90 %
- [ ] CHANGELOG, README, tag `v2.0.0rc1`, guía de vuelta

# `cipa` 2.0.0rc3: rendimiento de la búsqueda de vecinos y del ajuste de L1

> **Origen:** decisión 6 de `cipa-extended/results/cipa/cipa_protocol.md`, 2026-09-17, tras la
> prueba piloto de tiempos sobre `v2.0.0rc1`. Mediciones en
> `cipa-extended/results/cipa/pilot/`: `scaling_pilot_summary.md` (sección de tiempos),
> `timing_pilot.json` y `perf_probe.json`.
> **Repositorio de trabajo:** `/home/luisgarcia/projects/unam/dcic/2026-2/statistical_analysis`.
> Punto de partida: `32d6615`, tag `v2.0.0rc2`.
> **Alcance: sólo rendimiento y convergencia. Ninguna fórmula cambia.**

> **Sobre la versión.** La actualización en curso de `imbdata` a 0.4.0 (proyecto
> `taxonomy-digital-fraud`) **no afecta a este paquete**: `cipa` no depende de `imbdata`, y el
> repositorio de `cipa` mantiene su propia numeración. Por eso este candidato es **`v2.0.0rc3`**,
> el tercero de la serie 2.0.0 (rc1 correcciones C1–C9, rc2 redefinición de D₅). La única
> intersección es que las pruebas de integración que leen datasets usan `imbdata`; ver §5.

Documento de trabajo: consúmelo y bórralo, o muévelo a `docs/`.

---

## 1. Por qué

Con el protocolo de la decisión 3 (D₃ exacto y 5 submuestras de 50k para D₂ y D₇), los tiempos
medidos con rc1 y 32 núcleos hacen inviable la Fase 2:

| Paso | `cic_ids_2017` (2,520,798 × 70) | `ieee_cis_fraud` (590,540 × 432) |
|---|---|---|
| D₃ (vecinos de las 425,741 minoritarias) | **4,094 s** | 582 s |
| D₂, por submuestra | 281 s | 645 s |
| D₇, por submuestra | 37 s | **2,622 s** |
| Total estimado del dataset | 98 min | **284 min** |

Faltan seis datasets grandes, varios con d de 121 a 196, y la sensibilidad N_max ∈ {10K … 100K}
multiplica el costo. Los dos cuellos de botella tienen causa identificada y medida.

## 2. P1 · Elegir el algoritmo de búsqueda de vecinos

**Evidencia.** Submuestra de `ieee_cis_fraud`, 50,000 × 432, k+1 = 6, `n_jobs=-1`, 32 núcleos:

| Algoritmo | Tiempo (ajuste + 50,000 consultas) |
|---|---|
| `ball_tree` (el de rc2) | 122.2 s |
| `brute` | **6.0 s** |

Y **los resultados coinciden**: el orden de los vecinos es idéntico en el 100 % de las filas, los
conjuntos también, y la diferencia máxima de distancia es **1.08e-05**, atribuible al redondeo del
cálculo por productos escalares.

En `cic_ids_2017` (2.5 M × 70), medido en **un** núcleo: `ball_tree` 68 ms por consulta contra
**11 ms** por fuerza bruta.

**Qué hacer.**
1. Dónde se elige hoy el algoritmo:
   - `_knn.py`: `_KNNCache` usa `algorithm="ball_tree"` fijo (kDN de D₂ y D₃);
   - `ecol/n2.py`: dos `NearestNeighbors(algorithm="ball_tree")` (D₇);
   - `ecol/n1.py:147`: ya elige `kd_tree` si d ≤ 15 y `ball_tree` si no.
2. **Una sola función de selección**, usada por los tres, con la regla por defecto:
   `kd_tree` si d ≤ 15; `brute` si d > 15. Documenta el umbral y su origen (la medición de arriba
   y la de N1 en la guía de rc1).
3. **Parámetro `algorithm`** en `CIPAPipeline`, con `"auto"` por defecto para esa regla, y
   `"ball_tree"`, `"kd_tree"` o `"brute"` para forzarla. Así el protocolo puede volver al
   comportamiento de rc2 si hiciera falta.
4. **`chunk_size` sigue gobernando el tamaño de bloque de consultas**, para acotar la memoria de
   la fuerza bruta. Verifica el pico de memoria con 50,000 × 432 y con 425,741 consultas contra
   2.5 M × 70; declara el máximo observado.
5. **Riesgo a vigilar: los empates.** Con distancias que difieren en 1e-5, dos vecinos a distancia
   casi idéntica pueden intercambiar posiciones y mover kDN, D₃, N1 o N2 en el último decimal.
   - La prueba de regresión contra v1.2.1 debe seguir pasando. Si alguna dimensión deja de
     coincidir bit a bit con `brute`, **no cambies la fórmula**: documenta la diferencia, su
     magnitud y en qué caso aparece, y deja `ball_tree` como opción.
   - Añade una prueba que compare `brute` contra `ball_tree` en datos sintéticos con duplicados y
     con empates de distancia.

## 3. P2 · Convergencia y costo de L1 (D₇)

**Evidencia.** Misma submuestra de `ieee_cis_fraud` (50,000 × 432), `LinearSVC` con
`class_weight="balanced"`, semilla 42, un núcleo:

| Configuración | Tiempo | ¿Convergió? | L1 | Iteraciones |
|---|---|---|---|---|
| rc1/rc2: `dual="auto"`, `max_iter=10_000` | **4,288.7 s** | **No** | 0.16044 | 10,000 (tope) |

Las demás configuraciones (primal explícito, dual explícito, primal con `tol=1e-3`) se estaban
midiendo cuando se detuvo la prueba, porque cada ajuste cuesta más de una hora. **Esa comparación
es parte de esta tarea**, que ya cuenta con la línea base.

**Qué hacer.**
1. Medir, en esa submuestra, al menos: `dual=False` explícito, `dual=True` explícito, `tol` más
   holgada (1e-3), y cualquier alternativa que consideres (escalar `C`, `intercept_scaling`,
   `max_iter` mayor con primal). Reporta tiempo, convergencia, número de iteraciones y L1.
2. **Criterios de aceptación**, en este orden:
   - **converge** dentro del límite de iteraciones en este caso, que es el más duro del estudio;
   - **estabilidad**: en los casos donde rc2 sí convergía (los datasets pequeños de la prueba
     piloto), L1 no cambia más allá de una tolerancia que declares, del orden de 1e-3;
   - **tiempo**: idealmente por debajo de 10 min por submuestra.
3. **Si ninguna configuración converge** en el caso duro: no lo ocultes. Deja la mejor, expón
   `svc_tol` junto a `svc_max_iter` en `CIPAPipeline`, y reporta en los componentes de D₇
   `converged` y **`n_iter`** (nuevo), para que cipa-extended pueda declarar en el manuscrito en
   qué datasets L1 es un error de ajuste incompleto.
4. Considera si el escalado influye: los datos llegan estandarizados y con colas pesadas (hay
   variables con asimetría fuerte). Si detectas que el problema es el condicionamiento, dilo en el
   CHANGELOG; **no** cambies el preprocesamiento en esta tarea, porque es una decisión de
   cipa-extended.

## 4. P3 · Instrumentación (menor)

- Registrar el tiempo por **componente** en D₂ (F3, N1, kDN) y en D₇ (L1, N2), no sólo por
  dimensión. La prueba piloto tuvo que medirlo por fuera para encontrar los cuellos de botella.
- Deja el costo de construir el índice de vecinos dentro del componente que lo usa primero, como
  hoy, pero anótalo en el docstring para que no confunda.

## 5. Nota sobre `imbdata` 0.4.0

La prueba de integración de rc2 (`tests/integration/test_d5_reference.py`) lee datasets de
`imbdata` si están en caché. `imbdata` 0.4.0 **no cambia los datos** de las claves que esa prueba
usa (mismos SHA-256), así que debe seguir pasando. Si ves una diferencia, detente: sería un
problema de `imbdata`, no de este cambio.

## 6. Pruebas mínimas

- Regresión v1.2.1: sigue exacta para D₁–D₄, D₆, D₇ y para `spectral_entropy_norm` de D₅, con el
  algoritmo por defecto nuevo. Si algo se mueve, ver §2.5.
- `brute` contra `ball_tree`: mismos vecinos en datos sintéticos con duplicados y empates.
- Selección de algoritmo: la regla elige lo esperado en d = 4, 15, 16 y 432, y `algorithm` fuerza
  la elección.
- Memoria acotada de la fuerza bruta con el `chunk_size` por defecto.
- L1: la configuración elegida converge en el caso duro, o queda documentada; `n_iter` se reporta.
- Cobertura ≥ 90 %, ruff y pytest en verde.

## 7. Versión y entrega

- CHANGELOG `[2.0.0rc3]`: la regla de selección de algoritmo con sus mediciones, el cambio de L1
  con la tabla comparativa, la instrumentación, y cualquier diferencia de valores encontrada.
- README: la sección de rendimiento, con los tiempos de referencia.
- Commit y **tag `v2.0.0rc3`**, con push a GitHub.
- Guía de vuelta `cipa-extended/cipa_2.0.0rc3_handoff.md` con: qué quedó implementado, la tabla
  final de L1, los tiempos nuevos en las dos submuestras de referencia, las diferencias de valores
  si las hubo, y las desviaciones respecto de esta guía.

## 8. Lista de verificación

- [ ] P1 regla de selección de algoritmo, compartida por `_KNNCache`, `n1` y `n2`
- [ ] P1 parámetro `algorithm` en `CIPAPipeline` y memoria acotada
- [ ] P1 pruebas de equivalencia y de empates
- [ ] P2 comparación de configuraciones de L1 con la tabla de resultados
- [ ] P2 `n_iter` y `svc_tol` reportados y expuestos
- [ ] P3 tiempos por componente
- [ ] Regresión v1.2.1 intacta o diferencias documentadas
- [ ] CHANGELOG, README, tag `v2.0.0rc3`, push y guía de vuelta

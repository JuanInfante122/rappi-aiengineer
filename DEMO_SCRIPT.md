# Script de Demo - Rappi Ops Intelligence

Pais principal: Colombia (CO)
Duracion estimada: 10-15 minutos

---

## Pregunta 1: Busqueda simple (filtrado)

**Pregunta:** "Cuales son las 5 zonas con mayor Lead Penetration esta semana en Colombia?"

**Resultado esperado:**

- Respuesta con tabla o lista de 5 zonas colombianas y sus valores de Lead Penetration.
- Grafico de barras horizontales mostrando las 5 zonas rankeadas.
- El agente ejecuta SQL con `WHERE metric_name = 'Lead Penetration' AND country = 'CO' AND week_number = 0`.
- Ejemplo de zonas esperadas: CO_PEREIRA_SAMARIA_2_PEI, CO_CARTAGENA_HORIZONTE_CAMPANOS, CO_BOGOTA_CHAPINERO.

---

## Pregunta 2: Serie temporal (tendencia)

**Pregunta:** "Muestrame la evolucion de Gross Profit UE en CO_BOGOTA_CHAPINERO las ultimas 8 semanas"

**Resultado esperado:**

- Grafico de lineas con 8-9 puntos (semanas L8W a L0W, de mas antiguo a mas reciente).
- Eje X: semana (L8W, L7W, ... L0W), Eje Y: valor de Gross Profit UE (escala ~2.5-3.5).
- El agente narra si la tendencia es positiva o negativa con respecto a la semana pasada.
- SQL usa `WHERE zone_id = 'CO_BOGOTA_CHAPINERO' AND metric_name = 'Gross Profit UE'` con `ORDER BY week_number DESC`.

---

## Pregunta 3: Comparacion entre grupos

**Pregunta:** "Compara el Perfect Order entre zonas Wealthy y Non Wealthy en Colombia"

**Resultado esperado:**

- Grafico de barras comparando el promedio de Perfect Order de ambos tipos de zona en CO.
- El agente explica cual grupo tiene mejor desempeno y la diferencia en puntos porcentuales.
- SQL agrupa por `zone_type` con `WHERE country = 'CO' AND metric_name = 'Perfect Orders'`.
- La diferencia entre Wealthy y Non Wealthy es visible y cuantificada en la respuesta.

---

## Pregunta 4: Correlacion multi-metrica

**Pregunta:** "Que zonas colombianas tienen alto Lead Penetration pero bajo Perfect Order?"

**Resultado esperado:**

- Tabla con zonas que cumplen ambas condiciones simultaneamente (desbalance operativo).
- Posibles zonas en el resultado: CO_BOGOTA_CHAPINERO, CO_MEDELLIN_POBLADO_GUAYABAL.
- El agente identifica las zonas como candidatas a investigacion de causa raiz.
- Usa dos metricas en la misma consulta SQL con subqueries o JOINs sobre raw_input_metrics.

---

## Pregunta 5: Interpretacion de negocio (inferencia)

**Pregunta:** "Cuales son las zonas colombianas que mas crecen en ordenes y que podria explicar el crecimiento?"

**Resultado esperado:**

- Lista de zonas con mayor volumen de ordenes en CO: CO_BOGOTA_CHAPINERO (274K), CO_BOGOTA_USAQUEN (237K), CO_MEDELLIN_POBLADO_GUAYABAL (118K).
- El agente ofrece hipotesis de negocio basadas en datos: correlacion con Lead Penetration alto o Pro Adoption elevado.
- Respuesta narrativa con razonamiento de negocio, no solo numeros.
- Puede sugerir metricas adicionales para investigar (Lead Penetration, Pro Adoption, Perfect Order).

---

## Bonus: Reporte de Insights Semanal

**Accion:** En el sidebar izquierdo, seleccionar "Colombia" en el selector de pais, luego hacer click en "Generar Reporte".

**Resultado esperado:**

- Aparece un boton de descarga para el archivo HTML.
- El archivo tiene un tamano razonable (menos de 1 MB — Plotly cargado via CDN, no embebido).
- Al abrir en el navegador: resumen ejecutivo con narrativa de GPT-4o arriba, cards de insights abajo en grilla de 2 columnas.
- Los insights mas severos aparecen primero (anomalias > tendencias > correlaciones > benchmarks > oportunidades).
- Cada card muestra tipo de insight, zona, metrica, narrativa generada por IA y grafico Plotly interactivo.

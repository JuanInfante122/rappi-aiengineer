# Sistema de Analisis Inteligente para Operaciones Rappi

Bot conversacional de inteligencia artificial que permite a Operations Managers consultar metricas operacionales en lenguaje natural sin escribir SQL ni Python. Incluye un motor de insights automaticos que detecta anomalias, tendencias y oportunidades en 9 paises y genera reportes ejecutivos HTML semanales listos para distribuir.

Dirigido al equipo de SP&A y operaciones que necesita analizar metricas de ~1,200 zonas en 9 paises y 9 semanas de datos historicos sin depender del equipo tecnico.

## Arquitectura

```
Excel Data ──> ETL (pandas) ──> SQLite
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
              SQL Agent       Insights Engine      │
              (LangChain       (5 Detectores       │
               + GPT-4o)        + Scorer)          │
                   │               │               │
                   v               v               │
               Chat UI        Narrador LLM         │
              (Streamlit)    (GPT-4o-mini)         │
                   │               │               │
                   │               v               │
                   │         Reporte HTML          │
                   │         (Jinja2 +             │
                   │          Plotly CDN)          │
                   └───────────────┘               │
```

**Flujo de datos:**

1. `data/rappi_data.xlsx` se carga al iniciar la app via ETL a SQLite.
2. El SQL Agent convierte preguntas en lenguaje natural a SQL via GPT-4o y devuelve respuestas con visualizaciones Plotly.
3. El Insights Engine ejecuta 5 detectores estadisticos (WoW, tendencia, benchmarking, correlacion, oportunidad) sobre todos los paises y metricas.
4. El Narrador LLM convierte cada hallazgo en una narrativa SCR (Situacion - Complicacion - Recomendacion) usando GPT-4o-mini.
5. El generador de reportes combina narrativas y graficos Plotly en un HTML autocontenido via Jinja2.

## Setup

Requisitos: Python 3.11 o superior, clave de API de OpenAI.

```bash
git clone <url-del-repositorio>
cd rappi-aiengineer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env y agregar la clave: OPENAI_API_KEY=sk-...
streamlit run app.py
```

**Alternativa con Docker:**

```bash
cp .env.example .env
# Editar .env y agregar la clave: OPENAI_API_KEY=sk-...
make run
```

`make run` ejecuta `docker compose up` con la imagen Python 3.11-slim. El servicio queda disponible en `http://localhost:8501`.

**Demo publica (Streamlit Cloud):**

La app esta deployada en Streamlit Community Cloud. Sin instalacion ni configuracion: abre la URL y el sistema esta listo para usar.

Las credenciales (OPENAI_API_KEY, SMTP) se gestionan via Streamlit Secrets — nunca estan en el repositorio.

## Bot conversacional: como usarlo

Ejecutar `streamlit run app.py` o `make run`. El chat se abre en `http://localhost:8501`.

Escribir cualquier pregunta en espanol sobre las operaciones. El agente interpreta la intencion, genera SQL, ejecuta la consulta y devuelve la respuesta con visualizacion automatica cuando aplica.

**Ejemplos de preguntas:**

- "Cuales son las 5 zonas con mayor Lead Penetration esta semana?" — devuelve tabla y grafico de barras.
- "Muestrame la evolucion de Gross Profit UE en Chapinero las ultimas 8 semanas" — devuelve grafico de lineas con tendencia historica.
- "Compara el Perfect Order entre zonas Wealthy y Non Wealthy en Mexico" — devuelve comparacion entre grupos de zonas con grafico de barras.
- "Que zonas tienen alto Lead Penetration pero bajo Perfect Order?" — analisis multivariable con tabla ordenada.
- "Cuales son las zonas que mas crecen en ordenes y que podria explicar el crecimiento?" — inferencia de causas con narrativa del agente.

Cada respuesta incluye un boton para descargar los datos en CSV y un toggle opcional para ver el SQL ejecutado.

## Reporte semanal: como generarlo y enviarlo

En el sidebar de Streamlit:

1. Seleccionar el pais en el selector "Pais" (AR, BR, CL, CO, CR, EC, MX, PE, UY).
2. Ingresar una direccion de email en el campo "Enviar reporte por email" (opcional).
3. Hacer clic en "Generar Reporte".
4. Esperar entre 30 y 90 segundos mientras el motor ejecuta los detectores y el narrador LLM genera las narrativas.
5. Descargar el archivo HTML con el boton "Descargar Reporte HTML".

Si se ingreso un email, el reporte se envia automaticamente como adjunto HTML al finalizar la generacion. El cuerpo del email incluye un resumen en texto plano para clientes que no abren adjuntos.

El reporte contiene:

- Resumen ejecutivo generado por GPT-4o con los hallazgos principales del pais.
- Hasta 25 tarjetas de insights ordenadas por severidad, cada una con narrativa SCR y grafico Plotly.
- Archivo HTML autocontenido que abre en cualquier browser sin dependencias externas.

Si no existen insights para el pais seleccionado en la semana actual, el sistema muestra un aviso en lugar del boton de descarga.

**Configuracion SMTP para envio de email:**

Agregar en `.env` (local) o en Streamlit Cloud → Settings → Secrets:

```
SMTP_HOST = smtp.gmail.com
SMTP_PORT = 587
SMTP_USER = tu@gmail.com
SMTP_PASSWORD = tu_app_password
```

Gmail requiere una App Password (no la contrasena de cuenta). Generarla en myaccount.google.com → Seguridad → Contrasenas de aplicacion. Si las variables SMTP no estan configuradas, la app sigue funcionando — el envio por email simplemente no esta disponible.

## Costo estimado de API

| Modelo | Uso | Costo por llamada | Llamadas por operacion |
|--------|-----|-------------------|------------------------|
| GPT-4o | Consulta del bot (SQL + respuesta) | ~$0.015 | 1 por pregunta |
| GPT-4o | Resumen ejecutivo del reporte | ~$0.015 | 1 por reporte |
| GPT-4o-mini | Narrativa SCR por insight | ~$0.001 | hasta 25 por reporte |

**Costo total estimado:**

- Por pregunta al bot: ~$0.015 (GPT-4o, ~3,200 tokens por llamada).
- Por reporte semanal por pais: ~$0.05 a $0.10 (25 narrativas GPT-4o-mini + 1 resumen GPT-4o).

## Decisiones tecnicas

| Decision | Eleccion | Razon |
|----------|----------|-------|
| LLM | OpenAI GPT-4o, temperature=0 | Mejor generacion de SQL con funcion calling estructurada. Temperature 0 garantiza respuestas deterministicas para consultas de datos. |
| Tipo de agente | AgentType.OPENAI_FUNCTIONS | Produce llamadas a herramientas en JSON estructurado, eliminando fallos de parseo de texto que ocurren con agentes ReAct clasicos. |
| Capa de datos | SQLite cargado desde Excel al inicio | Sin base de datos externa, SQL arbitrario sin ORM, configuracion cero. El archivo Excel queda en el repositorio para que el evaluador clone y ejecute sin pasos adicionales. |
| Framework de agente | LangChain SQL Agent | Abstraccion estable para Text-to-SQL con soporte nativo de herramientas de base de datos. Permite inyectar el diccionario de metricas en el system prompt. |
| Interfaz | Streamlit | Deploy gratuito en Streamlit Cloud, desarrollo rapido de interfaces de datos, ideal para herramientas internas sin autenticacion compleja. |
| Graficos | Plotly | Interactivos en el chat y en el reporte HTML. Soporta lineas, barras, box plots y scatter sin dependencias adicionales. |
| Memoria conversacional | Inyeccion manual de historial (ultimos 3 turnos) | Mas confiable que ConversationBufferWindowMemory con agentes SQL: el bucle multi-paso de herramientas hace que los puntos de inyeccion de LangChain sean impredecibles. |
| Reporte | Jinja2 + Plotly CDN | Produce un unico HTML autocontenido. Plotly.js se carga exactamente una vez via CDN; los graficos subsiguientes usan `include_plotlyjs=False` para evitar archivos de 50 MB+. |
| LLM de insights | GPT-4o-mini para narrativas, GPT-4o para resumen ejecutivo | Control de costos: las 25 narrativas SCR individuales no requieren el modelo completo. El resumen ejecutivo que sintetiza todos los hallazgos si justifica GPT-4o. |

## Limitaciones conocidas y proximos pasos

**Limitaciones actuales:**

- **Datos estaticos:** El dataset es fijo (Excel en el repositorio). No hay ingesta de datos en tiempo real ni actualizacion automatica semanal.
- **Memoria de sesion unica:** El historial conversacional existe solo mientras dura la sesion de browser. Al cerrar y reabrir, el historial se pierde.
- **Concurrencia SQLite:** SQLite no esta optimizado para multiples escrituras concurrentes. Suficiente para un usuario o demo; no escala a equipos grandes sin migracion a PostgreSQL.
- **Sin autenticacion:** Cualquier persona con la URL puede acceder al bot y generar reportes. Aceptable para el challenge; requiere autenticacion para produccion.

**Proximos pasos:**

- Entrega programada de reportes por email (weekly digest automatico via cron).
- Cache Redis para reutilizar insights calculados dentro de la misma semana.
- Streaming de tokens en el chat para reducir la latencia percibida en respuestas largas.
- Autenticacion multi-usuario con roles (viewer, analyst, admin).

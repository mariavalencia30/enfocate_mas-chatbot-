# Chatbot Administrativo — Enfócate Más

> Asistente virtual 24/7 para más de 3.000 familias. Automatiza consultas de pagos, asistencia, matrículas y preguntas frecuentes mediante **Twilio WhatsApp** y un endpoint web, usando **LangGraph + RAG + GPT-4o-mini**.

---

## Índice

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Inteligencia Artificial — Modelo RAG](#inteligencia-artificial--modelo-rag)
3. [Benchmark de Modelos](#benchmark-de-modelos)
4. [Arquitectura del Sistema](#arquitectura-del-sistema)
5. [Estructura de Carpetas](#estructura-de-carpetas)
6. [Requisitos Previos](#requisitos-previos)
7. [Instalación](#instalación)
8. [Configuración del .env](#configuración-del-env)
9. [Preparar los Datos](#preparar-los-datos)
10. [Levantar el Servidor](#levantar-el-servidor)
11. [Despliegue con WhatsApp (ngrok + Twilio)](#despliegue-con-whatsapp-ngrok--twilio)
12. [Pruebas — Casos de Uso Completos](#pruebas--casos-de-uso-completos)
13. [Métricas y Resultados](#métricas-y-resultados)
14. [Tests Automatizados](#tests-automatizados)
15. [Logs e Interacciones](#logs-e-interacciones)
16. [Modo Demo (sin saldo OpenAI)](#modo-demo-sin-saldo-openai)
17. [Errores Comunes](#errores-comunes)
18. [Seguridad](#seguridad)
19. [Módulos Futuros](#módulos-futuros)
20. [Validación del Sistema](#validación-del-sistema)
21. [Restricciones del Contexto](#restricciones-del-contexto)

---

## Descripción del Proyecto

**Enfócate Más** es una institución educativa que atiende diariamente a miles de familias con necesidades administrativas recurrentes: consultar pagos, renovar matrículas, obtener información de asistencia y resolver dudas generales. Estos procesos, gestionados manualmente, generan cuellos de botella en el personal administrativo.

Este proyecto implementa un **chatbot administrativo inteligente** que resuelve esa problemática mediante:

- Disponibilidad **24/7** sin depender del horario de oficina
- Respuestas instantáneas en **español colombiano**
- Integración nativa con **WhatsApp vía Twilio** (canal preferido por las familias)
- Acceso a la base de datos institucional (pagos, asistencia, estudiantes)
- Arquitectura modular y extensible

---

## Inteligencia Artificial — Modelo RAG

El sistema utiliza **RAG (Retrieval-Augmented Generation)**, que combina búsqueda semántica con generación de texto:

```
Documento → Chunking → Embeddings → ChromaDB → Búsqueda → Contexto → LLM → Respuesta
```

### ¿Por qué RAG y no un modelo entrenado desde cero?

| Sin RAG                                   | Con RAG                                         |
| ----------------------------------------- | ----------------------------------------------- |
| Solo conoce datos de entrenamiento        | Accede a información institucional actualizada |
| Puede inventar respuestas (alucinaciones) | Responde con documentos verificados             |
| Conocimiento genérico                    | Conocimiento personalizado de Enfócate Más    |

### Componentes del pipeline RAG

| Componente      | Función                        | Tecnología                    |
| --------------- | ------------------------------- | ------------------------------ |
| Document Loader | Carga PDFs/DOCX institucionales | PyPDFLoader, Docx2txtLoader    |
| Text Splitter   | Divide en chunks de 500 tokens  | RecursiveCharacterTextSplitter |
| Embedding Model | Vectoriza el texto (1536 dims)  | OpenAI Embeddings              |
| Vector Store    | Almacena y busca por similitud  | ChromaDB                       |
| Generator       | Genera la respuesta final       | GPT-4o-mini                    |

### ¿Por qué chunks de 500 tokens?

| Tamaño              | Ventaja                   | Desventaja        |
| -------------------- | ------------------------- | ----------------- |
| 200 tokens           | Rápido                   | Pierde contexto   |
| **500 tokens** | **Balance óptimo** | —                |
| 1000 tokens          | Más contexto             | Diluye relevancia |

### ¿Por qué ChromaDB (base de datos vectorial)?

Porque necesitamos buscar por **significado semántico**, no por palabras exactas:

| Pregunta del usuario   | SQL (fracasa)             | ChromaDB (éxito)                      |
| ---------------------- | ------------------------- | -------------------------------------- |
| "¿Cuándo abren?"     | ❌ No encuentra "horario" | ✅ Encuentra "Lunes a viernes 8am-6pm" |
| "¿Cuál es el costo?" | ❌ No coincide            | ✅ Encuentra información de precios   |

### Personalización del modelo

No se entrenó desde cero. Se usó **prompt engineering** para especializar el comportamiento:

```python
CLASSIFIER_SYSTEM_PROMPT = """
Eres un clasificador de intenciones para el chatbot de Enfócate Más.
Responde ÚNICAMENTE con una de estas palabras:
  faq       → preguntas sobre la institución
  auth      → inicio de sesión o credenciales
  pagos     → consultas sobre pagos o saldos
  matricula → inscripciones o renovaciones
  soporte   → quejas, reclamos o escalación
"""
```

---

## Benchmark de Modelos

Se evaluaron distintas configuraciones para elegir el modelo más adecuado:

| Modelo                             | F1-Score         | Latencia Avg    | Latencia P95    | Costo/1K tokens |
| ---------------------------------- | ---------------- | --------------- | --------------- | --------------- |
| GPT-4o + OpenAI Emb                | 0.9316           | 1.74s           | —              | $2.50           |
| **GPT-4o-mini + OpenAI Emb** | **0.6795** | **0.69s** | **0.95s** | **$0.15** |
| GPT-3.5-turbo + OpenAI Emb         | 1.0000*          | 1.00s           | —              | $0.50           |

> *El F1=1.0 de GPT-3.5-turbo puede indicar sobreajuste en el conjunto de evaluación.

### Modelo elegido: GPT-4o-mini

- Mejor balance **costo / rendimiento** para uso masivo
- Respuesta suficientemente rápida para WhatsApp (~0.7s)
- Costo estimado: **$50–100 USD/mes** para 3.000 familias
- Escalable sin rediseñar la arquitectura

### Ejecutar el benchmark

```bash
python experiments/benchmark_rag.py
python experiments/plot_final_results.py
```

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                   ENTRADAS (Sources)                        │
│        Twilio WhatsApp              Widget Web              │
└──────────────────┬──────────────────────┬───────────────────┘
                   └──────────┬───────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              GATEWAY  (FastAPI — gateway.py)                │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 LANGGRAPH STATE MACHINE                     │
│                                                             │
│  START → intent_classifier_node                             │
│                │ (arista condicional)                       │
│       ┌────────┼────────┬──────────┐                       │
│       ▼        ▼        ▼          ▼                       │
│   faq_node  auth_node  pagos_node  soporte_node            │
│       └────────┴────────┴──────────┘                       │
│                         │                                   │
│               response_builder_node                         │
│                         │                                   │
│                    logger_node → END                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Respuesta al canal (WhatsApp / JSON)
```

### Detalle de cada nodo

| Nodo                  | Archivo                       | Descripción                                         |
| --------------------- | ----------------------------- | ---------------------------------------------------- |
| `intent_classifier` | `nodes/classifier.py`       | Clasifica intención con GPT-4o-mini + reglas        |
| `faq_node`          | `nodes/faq.py`              | RAG con ChromaDB para preguntas institucionales      |
| `auth_node`         | `nodes/auth.py`             | Login por teléfono/correo + recuperación de acceso |
| `pagos_node`        | `nodes/pagos.py`            | Consulta/actualización de datos del estudiante      |
| `soporte_node`      | `nodes/soporte.py`          | Menú y escalación a agente humano                  |
| `response_builder`  | `nodes/response_builder.py` | Formatea respuesta según canal (WhatsApp/web)       |
| `logger_node`       | `nodes/logger.py`           | Registra interacción en JSONL                       |

---

## Estructura de Carpetas

```
enfocatemas-chatbot/
├── src/
│   ├── graph.py                         # StateGraph principal (7 nodos)
│   ├── flows/
│   │   └── conversational_blueprint.py  # Pseudocódigo de flujos (documentación)
│   ├── integrations/
│   │   └── gateway.py                   # FastAPI + webhooks Twilio
│   ├── memory/
│   │   └── state.py                     # ChatState (TypedDict)
│   ├── nodes/                           # Nodos LangGraph
│   │   ├── classifier.py
│   │   ├── auth.py
│   │   ├── faq.py
│   │   ├── pagos.py
│   │   ├── soporte.py
│   │   ├── response_builder.py
│   │   └── logger.py
│   └── utils/
│       ├── ingest_documents.py          # Pipeline ingesta RAG
│       └── generate_synthetic_data.py   # Generador de datos de prueba
│
├── data/
│   ├── csv/
│   │   └── datos_estudiantes.csv        # Base de datos (3.500 registros)
│   ├── documents/                       # PDF/DOCX fuente para FAQ
│   └── embeddings/                      # Índice vectorial ChromaDB
│       └── chroma.sqlite3
│
├── experiments/
│   ├── benchmark_rag.py                 # Benchmark de modelos
│   └── results/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── load/
│       └── locustfile.py
│
├── logs/
│   └── interactions.jsonl               # Log de todas las interacciones
│
├── docs/
│   └── arquitectura.md
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

## Requisitos Previos

- Python 3.11+
- Cuenta en [OpenAI](https://platform.openai.com) con saldo (o usar Modo Demo — ver sección al final)
- Cuenta en [Twilio](https://www.twilio.com) (gratis para sandbox)
- Cuenta en [ngrok](https://ngrok.com) (gratis)

---

## Instalación

```powershell
# 1. Entrar a la carpeta del proyecto
cd enfocatemas-chatbot

# 2. Activar el entorno virtual
.venv\Scripts\Activate.ps1

# Si da error de permisos, ejecutar primero (solo una vez):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Linux/Mac:**
>
> ```bash
> source .venv/bin/activate
> pip install -r requirements.txt
> ```

---

## Configuración del .env

Crea o edita el archivo `.env` en la raíz del proyecto:

```env
# OpenAI
OPENAI_API_KEY=sk-proj-...tu-clave...

# Twilio — estos valores están en console.twilio.com
WHATSAPP_ACCESS_TOKEN=...auth-token-de-twilio...       # Auth Token
WHATSAPP_PHONE_NUMBER_ID=...account-sid-de-twilio...   # Account SID
WHATSAPP_VERIFY_TOKEN=enfocate_verify_token_2025
WHATSAPP_APP_SECRET=abc123
TWILIO_WHATSAPP_NUMBER=+14155238886

# Servidor
SERVER_URL=http://localhost:8000
PORT=8000
ENVIRONMENT=development
```

## Preparar los Datos

### 1. Generar datos sintéticos de estudiantes (3.500 registros)

```bash
python -m src.utils.generate_synthetic_data
```

Genera `data/csv/datos_estudiantes.csv`. Para producción, reemplaza con los datos reales de la institución respetando el mismo formato de columnas.

### 2. Ingestar documentos institucionales para el FAQ

Copia los documentos (Word/PDF/TXT) a `data/documents/` y ejecuta:

```bash
python -m src.utils.ingest_documents --folder data/documents/
```

Esto genera el índice vectorial en `data/embeddings/`. Verifica que funcionó:

```bash
python tests/vectorstore/test_vectorstore.py
# Resultado esperado: Total de documentos indexados: 102
```

---

## Levantar el Servidor

```powershell
uvicorn src.integrations.gateway:app --host 0.0.0.0 --port 8000 --reload 
```

Espera hasta ver:

```
INFO:     Application startup complete.
```

Verifica que funciona:

```
http://localhost:8000/health   → {"status": "ok"}
http://localhost:8000/docs     → Interfaz Swagger interactiva
```

---

## Despliegue con WhatsApp (ngrok + Twilio)

### Paso 1 — Exponer el servidor con ngrok

Abre **una terminal nueva** (sin cerrar la del servidor):

```powershell
# Primera vez: autenticarte con tu token de ngrok.com
ngrok config add-authtoken TU_TOKEN_DE_NGROK

# Exponer el puerto
ngrok http 8000
```

Copia la URL pública que aparece:

```
https://abcd-123-456-789.ngrok-free.app
```

### Paso 2 — Configurar webhook en Twilio

1. Entra a [console.twilio.com](https://console.twilio.com)
2. Ve a **Messaging → Try it out → Send a WhatsApp message**
3. En la pestaña **Sandbox Settings** configura:
   - **When a message comes in:** `https://abcd-123-456-789.ngrok-free.app/webhook/twilio`
   - **Method:** `HTTP POST`
4. Clic en **Save**

### Paso 3 — Conectar tu WhatsApp al sandbox

En la misma página de Twilio verás:

```
Send "join <palabra-palabra>" to +14155238886
```

Envía ese mensaje exacto desde tu WhatsApp. Twilio confirmará con un mensaje automático. ✅

### Paso 4 — Probar

Envía un mensaje al número de Twilio desde WhatsApp. En la terminal del servidor verás:

```
[TWILIO] From: whatsapp:+573001234567 | Body: Hola
```

---

## Pruebas — Casos de Uso Completos

> **Nota Windows:** usa PowerShell con `Invoke-RestMethod` en vez de `curl`.
>
> ```powershell
> $body = @{ session_id = "test-1"; message = "Hola"; user_id = "" } | ConvertTo-Json
> Invoke-RestMethod -Method POST -Uri "http://localhost:8000/chat/web" -ContentType "application/json" -Body $body
> ```

---

### 1. Menú de bienvenida

```bash
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "bienvenida-1", "message": "hola", "user_id": ""}'
```

**Respuesta esperada:**

```
Hola 👋, soy el asistente virtual de Enfócate Más. Puedo ayudarte con:
• 📚 Preguntas frecuentes sobre la institución
• 🔐 Inicio de sesión
• 💳 Consulta de pagos y saldos
• 📝 Estado de matrícula
```

---

### 2. Preguntas frecuentes — FAQ (sin autenticación)

```bash
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "faq-1", "message": "cuáles son los horarios?", "user_id": ""}'
```

Otros mensajes de prueba para FAQ:

```
"¿Qué programas ofrecen?"
"¿Cuánto cuesta el curso?"
"¿Cómo me inscribo?"
"¿Dónde están ubicados?"
```

---

### 3. Autenticación por teléfono

```bash
# Login directo con número de teléfono
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "login-1", "message": "+57 3874290099", "user_id": ""}'
```

**Respuesta esperada:**

```
Hola, Sofia Gomez. ¿En qué te puedo ayudar?
```

---

### 4. Recuperación de credenciales (flujo de 4 pasos)

```bash
# Paso 1: Iniciar recuperación
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "recovery-1", "message": "no me acuerdo", "user_id": ""}'
# → "¿Cómo te llamas?"

# Paso 2: Nombre
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "recovery-1", "message": "Sofia Gomez", "user_id": ""}'
# → "¿Tienes algún hermano registrado?"

# Paso 3: Hermano
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "recovery-1", "message": "si", "user_id": ""}'
# → "¿De qué colegio eres?"

# Paso 4: Colegio
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "recovery-1", "message": "Instituto Central", "user_id": ""}'
# → "¿Cuál es tu número de teléfono?"

# Paso 5: Teléfono — verificación final
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "recovery-1", "message": "+57 3874290099", "user_id": ""}'
# → Muestra correo y contraseña recuperados 
```

---

### 5. Consulta de pagos (requiere autenticación)

```bash
# Primero autenticarse
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "pagos-1", "message": "+57 3874290099", "user_id": ""}'

# Luego consultar
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "pagos-1", "message": "cuánto debo?", "user_id": ""}'
```

**Respuesta esperada:**

```
💰 Estado de Cuenta:
• Pensión: $48.772 pagado / $45.290 pendiente
• Clases: $120.000 pagado / $120.000 pendiente

✅ Saldo a favor: $3.482
```

---

### 6. Consulta de asistencia

```bash
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "pagos-1", "message": "ver mi asistencia", "user_id": ""}'
```

**Respuesta esperada:**

```
📚 Resumen de Asistencia:
• Total clases: 44
• Asistidas: 30 (68%)
  - Con pago: 15
  - Sin pago: 15
• No asistió: 14

✅ Asististe a la última clase ✓
```

---

### 7. Actualización de datos personales

```bash
# Actualizar correo
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "pagos-1", "message": "actualizar mi correo a nuevo@correo.com", "user_id": ""}'

# Actualizar contraseña
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "pagos-1", "message": "cambiar mi contraseña a nueva123", "user_id": ""}'

# Actualizar teléfono
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "pagos-1", "message": "cambiar mi telefono a 3209876543", "user_id": ""}'

# Actualizar nombre
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "pagos-1", "message": "cambiar mi nombre a Ana Lopez", "user_id": ""}'
```

---

### 8. Escalación a soporte humano

```bash
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "soporte-1", "message": "tengo una queja urgente", "user_id": ""}'
```

**Respuesta esperada:** contacto institucional + horario de atención.

---

### 9. Casos de error esperados

```bash
# Teléfono no registrado
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "error-1", "message": "+57 300 999 9999", "user_id": ""}'
# → "❌ El teléfono no está registrado."

# Consulta de pagos sin autenticación
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "error-2", "message": "cuánto debo?", "user_id": ""}'
# → "Para consultar esta información necesitas verificar tu identidad."

# Mensaje sin sentido (fallback)
curl -X POST http://localhost:8000/chat/web \
  -H "Content-Type: application/json" \
  -d '{"session_id": "error-3", "message": "xkdzpq", "user_id": ""}'
# → Menú de soporte / bienvenida
```

---

### 10. Verificación del webhook de Twilio

```bash
# Verificación GET
curl "http://localhost:8000/webhook/twilio?hub.mode=subscribe&hub.verify_token=enfocate_verify_token_2025&hub.challenge=test_challenge"

# Simular mensaje entrante de WhatsApp
curl -X POST http://localhost:8000/webhook/twilio \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "From=whatsapp:+573874290099&Body=hola&MessageSid=test123"
```

---

## Métricas y Resultados

### Resultados del modelo elegido (GPT-4o-mini)

```
RESULTADOS BENCHMARK — GPT-4o-mini
══════════════════════════════════
Accuracy:      84.6%  (11/13 predicciones correctas)
Precision:     91.7%
Recall:        91.7%
F1-Score:      0.917
Latencia Avg:  0.694s
Latencia P95:  0.946s
Costo:         $0.15/1K tokens
```

### Visualización comparativa

```
F1-Score por modelo:
══════════════════════════════
GPT-3.5-turbo ████████████████ 1.00 *sobreajuste
GPT-4o        ██████████████▌  0.93
GPT-4o-mini   ██████████▌      0.68

Costo/1K tokens:
══════════════════════════════
GPT-4o        ████████████████ $2.50
GPT-3.5-turbo ████             $0.50
GPT-4o-mini   █▌               $0.15
```

### Generar métricas desde los logs

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

logs = []
with open("logs/interactions.jsonl") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

df["intent"].value_counts().plot(kind="bar", color="steelblue")
plt.title("Distribución de intenciones")
plt.xlabel("Intent")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig("metricas_intents.png")
plt.show()
```

---

## Tests Automatizados

```bash
# Todos los tests
pytest

# Con reporte de cobertura
pytest --cov=src tests/

# Tests de carga — simula 100 usuarios concurrentes
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### Suite de tests incluida

| Tipo         | Archivo                                           | Qué valida                                            |
| ------------ | ------------------------------------------------- | ------------------------------------------------------ |
| Unitario     | `tests/unit/test_auth_node.py`                  | Login, normalización de teléfonos, bloqueo de cuenta |
| Unitario     | `tests/unit/test_faq_node.py`                   | Respuesta RAG, fallback sin documentos                 |
| Unitario     | `tests/unit/test_pagos_node.py`                 | Consultas y actualizaciones de datos                   |
| Integración | `tests/integration/test_conversational_flow.py` | Flujo completo auth → pagos, escalación              |
| Regresión   | `tests/regression/test_regression.py`           | Truncado WhatsApp 1600 chars, routing inmutable        |
| Carga        | `tests/load/locustfile.py`                      | FAQ 60%, Auth 25%, Mixto 15%                           |

### Script de prueba automatizado

```python
import requests

BASE_URL = "http://localhost:8000/chat/web"

def test_flujo_completo():
    session_id = "test-auto-001"

    # 1. Login
    r1 = requests.post(BASE_URL, json={
        "session_id": session_id,
        "message": "+57 3874290099",
        "user_id": ""
    })
    assert "Sofia Gomez" in r1.json()["response"], "Login falló"

    # 2. Consulta de pagos
    r2 = requests.post(BASE_URL, json={
        "session_id": session_id,
        "message": "cuánto debo?",
        "user_id": ""
    })
    assert "Estado de Cuenta" in r2.json()["response"], "Pagos falló"

    # 3. Asistencia
    r3 = requests.post(BASE_URL, json={
        "session_id": session_id,
        "message": "ver mi asistencia",
        "user_id": ""
    })
    assert "Asistencia" in r3.json()["response"], "Asistencia falló"

    print("✅ Todos los tests pasaron")

if __name__ == "__main__":
    test_flujo_completo()
```

---

## Logs e Interacciones

Cada conversación queda registrada automáticamente en `logs/interactions.jsonl`:

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "session_id": "+573001234567",
  "user_id": "+573001234567",
  "channel": "whatsapp",
  "intent": "pagos",
  "auth_status": "ok",
  "escalated": false,
  "message_count": 4
}
```

---

## Modo Demo (sin saldo OpenAI)

Si no tienes saldo en OpenAI, edita `src/nodes/classifier.py` y reemplaza el bloque `llm = ChatOpenAI(...)` hasta el final de la función `intent_classifier_node` por:

```python
    # --- MODO DEMO: clasificación sin OpenAI ---
    if any(p in text_lower for p in ["horario", "curso", "programa", "precio", "cómo", "qué", "cuál"]):
        return {"intent": "faq"}
    if any(p in text_lower for p in ["iniciar", "sesión", "login", "teléfono", "acceder", "no me acuerdo"]):
        return {"intent": "auth"}
    if any(p in text_lower for p in ["pago", "debo", "saldo", "asistencia", "clases", "cuánto"]):
        return {"intent": "pagos"}
    return {"intent": "soporte"}
```

Reinicia el servidor después del cambio. En modo demo el clasificador usa reglas de palabras clave sin llamar a la API de OpenAI.

---

## Errores Comunes

| Error                             | Causa                                  | Solución                                                                       |
| --------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- |
| `401 Incorrect API key`         | API key de OpenAI inválida o revocada | Generar nueva en platform.openai.com/api-keys                                   |
| `429 insufficient_quota`        | Sin saldo en OpenAI                    | Cargar saldo o usar Modo Demo                                                   |
| `Internal server error`         | CSV o embeddings no encontrados        | Verificar que existen `data/csv/datos_estudiantes.csv` y `data/embeddings/` |
| `403 Forbidden` en webhook      | Token de verificación incorrecto      | Verificar `WHATSAPP_VERIFY_TOKEN` en `.env`                                 |
| Bot no responde en WhatsApp       | ngrok cerrado o URL desactualizada     | Reiniciar ngrok y actualizar URL en Twilio                                      |
| `WatchFiles detected changes`   | uvicorn reinicia por archivos del venv | Usar `--reload-dir src`                                                       |
| Error de permisos al activar venv | PowerShell bloquea scripts             | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`                         |

---

## Seguridad

- **Autenticación:** máximo 3 intentos por sesión, cuenta bloqueada automáticamente tras el tercer fallo
- **Datos sensibles:** el nodo auth solo confirma acceso/denegación, nunca expone información personal
- **Contraseñas:** almacenadas en el CSV, nunca transmitidas en logs
- **Variables de entorno:** todas las claves fuera del código fuente (`.env`)
- **Cumplimiento:** diseñado según la **Ley 1581 de 2012** (habeas data, Colombia)

---



## Datos de Prueba

**Estudiante de ejemplo (generado con `generate_synthetic_data.py`):**

| Campo         | Valor                   |
| ------------- | ----------------------- |
| Nombre        | Sofia Gomez             |
| Teléfono     | +57 3874290099          |
| Correo        | sofia.gomez501@mail.com |
| Contraseña   | Tv8AyD9c                |
| Colegio       | Instituto Central       |
| Tiene hermano | SI                      |

---

## Validación del Sistema

El sistema fue validado funcionalmente por el equipo de desarrollo mediante pruebas exhaustivas de todos los flujos conversacionales, incluyendo pruebas reales vía WhatsApp con el sandbox de Twilio.

**Validado por:**

| Nombre         | Rol en el proyecto     |
| -------------- | ---------------------- |
| DAVID MARTINEZ | Desarrollador / Tester |
| MARIA VALENCIA | Desarrollador / Tester |
| VALERIA FRANCO | Desarrollador / Tester |

**Flujos validados:**

| Flujo                                       | Canal          | Resultado |
| ------------------------------------------- | -------------- | --------- |
| Menú de bienvenida                         | WhatsApp + Web | ✅        |
| Autenticación por teléfono                | WhatsApp + Web | ✅        |
| Recuperación de credenciales (4 pasos)     | WhatsApp + Web | ✅        |
| Preguntas frecuentes (RAG)                  | WhatsApp + Web | ✅        |
| Consulta de pagos y saldos                  | WhatsApp + Web | ✅        |
| Consulta de asistencia                      | WhatsApp + Web | ✅        |
| Actualización de datos personales          | Web            | ✅        |
| Escalación a soporte humano                | WhatsApp + Web | ✅        |
| Manejo de errores (teléfono no registrado) | WhatsApp + Web | ✅        |

**Evidencia:** capturas de conversaciones reales de WhatsApp disponibles en `docs/evidencia/`.

---

## Restricciones del Contexto

El diseño del sistema tuvo en cuenta las siguientes restricciones reales de la institución y su contexto colombiano:

| Restricción                                | Decisión técnica tomada                                     |
| ------------------------------------------- | ------------------------------------------------------------- |
| Familias usan WhatsApp como canal principal | Integración con Twilio WhatsApp API                          |
| Números de teléfono colombianos (`+57`) | Normalización automática de prefijos en `auth.py`         |
| Sin servidor propio en producción          | Docker + ngrok para desarrollo; compatible con Render/Railway |
| Infraestructura de datos limitada           | CSV con pandas en vez de base de datos relacional             |
| WhatsApp limita mensajes a 1600 caracteres  | Truncado automático en `response_builder.py`               |
| Más de 3.000 familias activas              | Datos sintéticos de 3.500 registros para pruebas de carga    |
| Cumplimiento legal Colombia                 | Diseño alineado con Ley 1581 de 2012 (habeas data)           |

---

## Licencia

Uso interno — Institución Educativa Enfócate Más © 2026

"""
src/flows/conversational_blueprint.py

PSEUDOCÓDIGO Y BLUEPRINT de los flujos conversacionales principales. Este archivo documenta la lógica de cada flujo antes de su implementación
en nodos LangGraph. Sirve como referencia de diseño. (En duda si agregarlo al documento final)

No es código ejecutable directamente — es documentación estructurada.
"""

# FLUJO PRINCIPAL — ENTRADA DE MENSAJE
"""
PSEUDOCÓDIGO: flujo_principal(mensaje, session_id, canal)
─────────────────────────────────────────────────────────
1. RECIBIR mensaje del usuario (WhatsApp o web)
2. CARGAR estado de sesión desde memoria (si existe)
3. INICIALIZAR estado si sesión nueva:
     estado = {
       messages: [],
       user_id: "",
       intent: "",
       auth_status: "pending",
       auth_attempts: 0,
       context: {},
       channel: canal,
       session_id: uuid(),
       escalate: False,
       logs: []
     }
4. AÑADIR mensaje al historial: estado.messages.append(HumanMessage(mensaje))
5. INVOCAR graph.invoke(estado, config={thread_id: session_id})
6. RETORNAR estado.response_text al canal correspondiente
7. REGISTRAR interacción en logs
"""

# FLUJO 1 — CLASIFICACIÓN DE INTENCIÓN
"""
PSEUDOCÓDIGO: intent_classifier(estado)
────────────────────────────────────────
ENTRADA: último mensaje del usuario

1. EXTRAER último HumanMessage del historial
2. CONSTRUIR prompt con instrucción de clasificación
3. LLAMAR LLM con prompt (temperatura=0 para determinismo)
4. PARSEAR respuesta → intent ∈ {faq, auth, pagos, matricula, soporte}
5. SI respuesta no válida → intent = "soporte"
6. ACTUALIZAR estado.intent = intent
7. RETORNAR al router condicional

ROUTER CONDICIONAL:
  intent == "faq"       → faq_node
  intent == "auth"      → auth_node
  intent == "pagos"     → pagos_node
  intent == "matricula" → faq_node  ← info pública de matrícula/inscripción
  intent == "soporte"   → soporte_node
  default               → soporte_node
"""


# FLUJO 2 — FAQ (Preguntas Frecuentes)
"""
PSEUDOCÓDIGO: faq_flow(estado)
───────────────────────────────
ENTRADA: pregunta del usuario

1. EXTRAER consulta del último HumanMessage
2. CONECTAR con ChromaDB (vector store)
3. EJECUTAR similarity_search(consulta, k=3)
4. CONCATENAR fragmentos de documentos relevantes como contexto
5. SI contexto vacío:
     responder: "No tengo información. Contacta la institución."
     RETORNAR
6. CONSTRUIR prompt:
     sistema: "Eres asistente de Enfócate Más. Usa solo el contexto."
     usuario: contexto + pregunta
7. LLAMAR LLM → respuesta
8. AÑADIR AIMessage(respuesta) al estado.messages
9. ACTUALIZAR estado.context = {faq_docs_found: N}
10. CONTINUAR → response_builder
"""


# FLUJO 3 — AUTENTICACIÓN

"""
PSEUDOCÓDIGO: auth_flow(estado)
────────────────────────────────
SEGURIDAD: No exponer datos personales. Solo status.

1. VERIFICAR estado.auth_status:
   - SI "ok"     → responder "Ya tienes sesión" → CONTINUAR
   - SI "locked" → responder "Acceso bloqueado" → CONTINUAR

2. EXTRAER credenciales del mensaje:
   - Intentar pattern: "usuario: X contraseña: Y"
   - Fallback: dos palabras separadas
   - SI no se pueden extraer → pedir formato correcto → CONTINUAR

3. LEER usuarios.csv (sin exponer contenido)
4. BUSCAR username en CSV (case-insensitive)
5. SI no existe → "credenciales incorrectas" (sin revelar cuál)
6. VERIFICAR bcrypt.verify(password, hash_almacenado)
7. SI verificación OK:
     estado.auth_status = "ok"
     estado.auth_attempts = 0
     responder: "Sesión iniciada. ¿En qué te ayudo?"
8. SI verificación FAIL:
     estado.auth_attempts += 1
     SI auth_attempts >= 3:
       estado.auth_status = "locked"
       responder: "Acceso bloqueado. Contacta la institución."
     SINO:
       responder: "Credenciales incorrectas. Intento X de 3."
9. CONTINUAR → response_builder
"""


# FLUJO 4 — PAGOS Y MATRÍCULAS

"""
PSEUDOCÓDIGO: pagos_flow(estado)
─────────────────────────────────
REQUIERE: estado.auth_status == "ok"

1. VERIFICAR autenticación:
   - SI "pending" o "fail" → pedir login → cambiar intent a "auth" → CONTINUAR

2. DETECTAR subconsulta:
   - SI mensaje contiene ["matrícula", "inscripción", "grado"] → consultar matriculas.csv
   - SINO → consultar pagos.csv

3. CONSULTAR CSV con user_id del estado:
   PARA PAGOS:
     - Filtrar por user_id + estado == "pendiente"
     - SI vacío: "No tienes pagos pendientes ✅"
     - SI hay: listar concepto, monto, fecha_vencimiento
   PARA MATRÍCULA:
     - Filtrar por user_id, tomar registro más reciente
     - Mostrar periodo, grado, estado_matricula, fecha_limite

4. FORMATEAR respuesta con emojis para legibilidad en WhatsApp
5. AÑADIR AIMessage al estado.messages
6. CONTINUAR → response_builder
"""

# FLUJO 5 — SOPORTE Y ESCALACIÓN
"""
PSEUDOCÓDIGO: soporte_flow(estado)
────────────────────────────────────
1. DETECTAR keywords de urgencia en el mensaje:
   ["urgente", "queja", "reclamo", "hablar con alguien", "no funciona", ...]

2. SI urgencia detectada:
   - estado.escalate = True
   - Proveer información de contacto humano
   - Registrar en logs para seguimiento

3. SINO:
   - Mostrar menú de opciones disponibles
   - Invitar al usuario a reformular su consulta

4. CONTINUAR → response_builder
"""

# MÓDULOS FUTUROS (documentados, NO implementados)
"""
MÓDULO FUTURO: notifications_flow
───────────────────────────────────
- Trigger: cron job diario
- Acción: leer pagos con fecha_vencimiento en próximos 5 días
- Enviar template de WhatsApp aprobado por Meta
- Registrar envío en logs

MÓDULO FUTURO: payments_gateway_flow
──────────────────────────────────────
- Trigger: intent "pagar"
- Acción: generar link de pago con pasarela local
- Webhook de confirmación de pago
- Actualizar CSV de pagos

MÓDULO FUTURO: ocr_receipt_flow
────────────────────────────────
- Trigger: usuario envía imagen de recibo
- Acción: Tesseract OCR → extraer monto y fecha
- Validar contra pagos pendientes
- Confirmar o rechazar recibo digitalizado

MÓDULO FUTURO: lms_integration_flow
─────────────────────────────────────
- Trigger: intent "calificaciones" / "asistencia"
- Acción: OAuth2 → API del LMS
- Consultar notas y asistencia del estudiante
- Formatear respuesta

MÓDULO FUTURO: voice_whatsapp_flow
────────────────────────────────────
- Trigger: mensaje de audio en WhatsApp
- Acción: Whisper API → transcripción
- Procesar como texto → flujo normal
- Opcional: responder con TTS
"""

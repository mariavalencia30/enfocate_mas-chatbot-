import os
import uuid
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import graph



app = FastAPI(title="Enfócate Más Chatbot API", version="1.0.0")

TWILIO_AUTH_TOKEN  = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
TWILIO_ACCOUNT_SID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886")


# Model entrance
class WebChatMessage(BaseModel):
    session_id: str = ""
    message: str
    user_id: str = ""


async def _invoke_graph(message: str, session_id: str, user_id: str, channel: str) -> str:
    """Invoca el grafo LangGraph y retorna la respuesta final."""
    from src.graph import graph
    from src.memory.state import ChatState
    from typing import List

    config = {"configurable": {"thread_id": session_id}}

    existing_state = None
    try:
        existing_state = graph.get_state(config)
    except Exception:
        pass

    if existing_state and existing_state.values:
        current_messages: List = list(existing_state.values.get("messages", []))
        current_messages.append(HumanMessage(content=message))

        last_ai = next(
            (m for m in reversed(current_messages) if isinstance(m, AIMessage)),
            None
        )
        prev_was_pagos = last_ai and "💰" in last_ai.content

        initial_state: ChatState = {
            "messages": current_messages,
            "user_id": user_id or existing_state.values.get("user_id", ""),
            "session_id": session_id,
            "channel": channel,
            "auth_status": existing_state.values.get("auth_status", "pending"),
            "auth_attempts": existing_state.values.get("auth_attempts", 0),
            "escalate": existing_state.values.get("escalate", False),
            "context": existing_state.values.get("context", {}),
            "logs": existing_state.values.get("logs", []),
            "intent": existing_state.values.get("intent", ""),
            "response_text": "",
        }

        if prev_was_pagos and existing_state.values.get("auth_status") == "ok":
            initial_state["intent"] = "pagos"
    else:
        initial_state: ChatState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "session_id": session_id,
            "channel": channel,
            "auth_status": "pending",
            "auth_attempts": 0,
            "escalate": False,
            "context": {},
            "logs": [],
            "intent": "",
            "response_text": "",
        }

    result = await graph.ainvoke(initial_state, config=config)
    return result.get("response_text", "Lo siento, hubo un error procesando tu solicitud.")


@app.get("/webhook/twilio")
async def twilio_verify(request: Request):
    """Verificación del webhook de Twilio."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "enfocate_verify_token_2025")
    if mode == "subscribe" and token == verify_token:
        return PlainTextResponse(content=challenge)

    raise HTTPException(status_code=403, detail="Token de verificación inválido")


@app.post("/webhook/twilio")
async def twilio_webhook(request: Request):
    """Recibe mensajes entrantes de Twilio WhatsApp."""
    try:
        # Leer form data una sola vez
        form = await request.form()

        phone_number = form.get("From", "")
        msg_text     = form.get("Body", "")

        print(f"[TWILIO] From: {phone_number} | Body: {msg_text}")

        if not msg_text:
            return PlainTextResponse("", status_code=200)

        response_text = await _invoke_graph(
            message=msg_text,
            session_id=phone_number,
            user_id=phone_number,
            channel="whatsapp",
        )

        await _send_twilio_message(phone_number, response_text)
        return PlainTextResponse("", status_code=200)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return PlainTextResponse("", status_code=200)


async def _send_twilio_message(to: str, text: str):
    """Envía un mensaje de texto via Twilio WhatsApp API."""
    import httpx

    account_sid  = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    auth_token   = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    from_number  = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886")

    if not account_sid or not auth_token:
        print(f"[GATEWAY] Twilio no configurado. Mensaje a {to}: {text}")
        return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"

    # Asegurar formato whatsapp:
    to_formatted   = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
    from_formatted = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"

    payload = {
        "To":   to_formatted,
        "From": from_formatted,
        "Body": text,
    }

    auth = (account_sid, auth_token)

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, data=payload, auth=auth)
        print(f"[TWILIO SEND] status={resp.status_code} | {resp.text[:200]}")


@app.post("/chat/web")
async def web_chat(body: WebChatMessage):
    session_id = body.session_id or str(uuid.uuid4())

    try:
        response_text = await _invoke_graph(
            message=body.message,
            session_id=session_id,
            user_id=body.user_id,
            channel="web",
        )

        return JSONResponse({
            "session_id": session_id,
            "response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Enfócate Más Chatbot"}
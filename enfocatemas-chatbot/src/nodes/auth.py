import os
import re
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.state import ChatState

ESTUDIANTES_CSV = os.path.join(os.path.dirname(__file__), "../../data/csv/datos_estudiantes.csv")
MAX_ATTEMPTS = 3


def _normalize_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    
    if len(digits) == 10:
        return f"+57{digits}"
    
    if len(digits) == 11 and digits.startswith("3"):
        return f"+57{digits[1:]}"
    
    if len(digits) == 12 and digits.startswith("57"):
        return f"+57{digits[2:]}"
    
    if len(digits) >= 10:
        return f"+57{digits[-10:]}"
    
    return f"+57{digits}"


def _extract_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    
    if len(digits) == 10:
        return f"+57{digits}"
    
    if len(digits) == 11:
        if digits.startswith("3"):
            return f"+57{digits}"
        if digits.startswith("57"):
            return f"+{digits}"
    
    if len(digits) == 12:
        if digits.startswith("57") and digits[2:].startswith("3"):
            return f"+57{digits[2:]}"
        if digits.startswith("57"):
            return f"+{digits}"
    
    if len(digits) >= 10:
        return f"+57{digits[-10:]}"
    
    return ""


def _load_df() -> pd.DataFrame:
    try:
        return pd.read_csv(ESTUDIANTES_CSV)
    except Exception:
        return pd.DataFrame()


def _handle_recovery(state: ChatState, text: str) -> dict:
    """Maneja el flujo de recuperación de credenciales."""
    recovery = state.get("context", {}).get("recovery", {})
    step = recovery.get("step", 0)

    if step == 0:
        recovery["step"] = 1
        recovery["nombre"] = text.strip()
        
        return {
            "context": {"recovery": recovery},
            "recovery_step": 1,
            "auth_status": "recovering",
            "messages": state["messages"] + [AIMessage(
                content="¿Tienes algún hermano registrado en la institución? (responde *Sí* o *No*)"
            )]
        }

    if step == 1:
        respuesta = text.lower().strip()
        tiene_hermano = "si" in respuesta or "sí" in respuesta
        
        recovery["step"] = 2
        recovery["tiene_hermano"] = tiene_hermano
        
        return {
            "context": {"recovery": recovery},
            "recovery_step": 2,
            "auth_status": "recovering",
            "messages": state["messages"] + [AIMessage(
                content="¿De qué colegio eres?"
            )]
        }

    if step == 2:
        recovery["step"] = 3
        recovery["colegio"] = text.strip().lower()
        
        return {
            "context": {"recovery": recovery},
            "recovery_step": 3,
            "auth_status": "recovering",
            "messages": state["messages"] + [AIMessage(
                content="¿Cuál es tu número de teléfono?"
            )]
        }

    if step == 3:
        phone = _extract_phone(text)
        
        if not phone:
            return {
                "messages": state["messages"] + [AIMessage(
                    content="Número no válido. Por favor ingresa en formato: +57 300 123 4567"
                )]
            }
        
        recovery["telefono"] = phone
        recovery["step"] = 4

        try:
            df = _load_df()
            if df.empty:
                return {
                    "messages": state["messages"] + [AIMessage(
                        content="Error: Base de datos no disponible."
                    )]
                }
            
            nombre_input = recovery.get("nombre", "").strip()
            if not nombre_input:
                return {
                    "messages": state["messages"] + [AIMessage(
                        content="Error en el proceso. Por favor comienza de nuevo."
                    )]
                }
            
            nombre_parts = nombre_input.lower().split()
            first_name = nombre_parts[0] if nombre_parts else ""
            
            if not first_name:
                return {
                    "messages": state["messages"] + [AIMessage(
                        content="Error en el proceso. Por favor comienza de nuevo."
                    )]
                }
            
            name_matches = df[df["nombre_estudiante"].str.lower().str.contains(first_name, na=False)]
            
            if name_matches.empty:
                return {
                    "auth_status": "fail",
                    "recovery_step": 0,
                    "context": {"recovery": {}},
                    "messages": state["messages"] + [AIMessage(
                        content="❌ No encontré estudiantes con ese nombre."
                    )]
                }
            
            phone_clean = phone.replace("+57", "").replace("+", "").replace(" ", "").replace("-", "")
            
            target_phones = [phone_clean]
            if len(phone_clean) == 10 and phone_clean.startswith("3"):
                target_phones.append(phone_clean[1:])
            
            for idx, row in name_matches.iterrows():
                all_phones = str(row.get("all_phone_numbers", "")).replace("+57", "").replace("+", "").replace(" ", "").replace("-", "")
                
                for target in target_phones:
                    if target in all_phones:
                        nombre = str(row["nombre_estudiante"])
                        correo = str(row["CORREO"])
                        contrasena = str(row["CONTRASENA"])
                        
                        return {
                            "auth_status": "ok",
                            "auth_attempts": 0,
                            "recovery_step": 0,
                            "context": {"recovery": {}, "nombre": nombre},
                            "user_id": phone,
                            "messages": state["messages"] + [AIMessage(
                                content=f"✅ *Verificación exitosa!*\n\n"
                                       f"Bienvenido/a, *{nombre}*.\n\n"
                                       f"📧 *Tu correo:* {correo}\n"
                                       f"🔑 *Tu contraseña:* {contrasena}\n\n"
                                       "Ya puedes entrar a la plataforma."
                            )]
                        }
            
            return {
                "auth_status": "fail",
                "recovery_step": 0,
                "context": {"recovery": {}},
                "messages": state["messages"] + [AIMessage(
                    content=" El teléfono no coincide con el nombre. Por favor, comunícate con la institución."
                )]
            }
            
        except Exception as e:
            print(f"[AUTH RECOVERY] Error: {e}")
            return {
                "messages": state["messages"] + [AIMessage(
                    content="Ocurrió un error. Por favor intenta de nuevo."
                )]
            }

    return {"messages": state["messages"] + [AIMessage(content="Fin del proceso.")]}


def _extract_credentials(text: str) -> tuple[str, str]:
    text = text.strip()
    pattern = r"(?:correo|email)[:\s]+(\S+).*?(?:contraseña|clave)[:\s]+(\S+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).lower(), match.group(2)
    return "", ""


def auth_node(state: ChatState) -> dict:
    """Nodo LangGraph: gestión de autenticación."""

    current_status = state.get("auth_status", "pending")
    attempts = state.get("auth_attempts", 0)

    if current_status == "ok":
        return {
            "messages": state["messages"] + [AIMessage(
                content="✅ Ya tienes sesión iniciada. ¿En qué más te puedo ayudar?"
            )]
        }

    if current_status == "locked" or attempts >= MAX_ATTEMPTS:
        return {
            "auth_status": "locked",
            "messages": state["messages"] + [AIMessage(
                content="🔒 Tu acceso ha sido bloqueado. Comunícate con la institución."
            )]
        }

    recovery_context = state.get("context", {}).get("recovery", {})
    is_recovery = current_status == "recovering" or recovery_context.get("step", 0) > 0

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )

    if not last_human:
        return {
            "messages": state["messages"] + [AIMessage(
                content=" Ingresa tu número de teléfono o escribe *no me acuerdo* para recuperar tu acceso."
            )]
        }

    text_lower = last_human.content.lower()

    if "no me acuerdo" in text_lower or "olvidé" in text_lower or "olvide" in text_lower or "recuperar" in text_lower:
        if not is_recovery:
            recovery_data = {"step": 0}
            return {
                "context": {"recovery": recovery_data},
                "auth_status": "recovering",
                "messages": state["messages"] + [AIMessage(
                    content=" Vamos a recuperar tu acceso. ¿Cómo te llamas? (nombre completo)"
                )]
            }

    no_puede_acceder = any(frase in text_lower for frase in [
        "no puedo entrar", "no puedo ingresar", "no puedo acceder",
        "no puedo loguear", "no puedo iniciar sesion", "no puedo iniciar sesión",
        "no me deja entrar", "olvide mi contraseña", "olvidé mi contraseña",
        "olvide mi correo", "olvidé mi correo", "perdi mi contraseña",
        "perdí mi contraseña", "no me funciona", "no funciona mi usuario"
    ])
    
    if no_puede_acceder and not is_recovery:
        recovery_data = {"step": 0}
        return {
            "context": {"recovery": recovery_data},
            "auth_status": "recovering",
            "messages": state["messages"] + [AIMessage(
                content=" Vamos a recuperar tu acceso. ¿Cómo te llamas? (nombre completo)"
            )]
        }

    if is_recovery:
        return _handle_recovery(state, last_human.content)

    phone = _extract_phone(last_human.content)
    correo, password = _extract_credentials(last_human.content)

    if phone:
        df = _load_df()
        
        clean_input = phone.replace(" ", "").replace("-", "")
        if clean_input.startswith("+57"):
            clean_input = clean_input[3:]
        elif clean_input.startswith("57"):
            clean_input = clean_input[2:]
        
        def clean_phone(phone_str):
            return str(phone_str).replace("+57", "").replace(" ", "").replace("-", "").replace("+", "")
        
        df["phone_clean"] = df["cellphone"].apply(clean_phone)
        df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)
        
        match = df[(df["phone_clean"] == clean_input) | (df["all_phones_clean"].str.contains(clean_input, na=False))]

        if match.empty:
            return {
                "auth_status": "fail",
                "auth_attempts": attempts + 1,
                "messages": state["messages"] + [AIMessage(
                    content="❌ El teléfono no está registrado. Escribe *no me acuerdo* para recuperar."
                )]
            }

        nombre = match.iloc[0]["nombre_estudiante"]
        return {
            "auth_status": "ok",
            "auth_attempts": 0,
            "context": {"nombre": nombre},
            "user_id": phone,
            "messages": state["messages"] + [AIMessage(
                content=f" Hola, *{nombre}*. ¿En qué te puedo ayudar?"
            )]
        }

    if correo and password:
        df = _load_df()
        match = df[df["CORREO"].str.lower() == correo.lower()]

        if match.empty:
            return {
                "auth_status": "fail",
                "auth_attempts": attempts + 1,
                "messages": state["messages"] + [AIMessage(
                    content="❌ Correo no encontrado."
                )]
            }

        if str(match.iloc[0]["CONTRASENA"]) != password:
            return {
                "auth_status": "fail",
                "auth_attempts": attempts + 1,
                "messages": state["messages"] + [AIMessage(
                    content=f"❌ Contraseña incorrecta. Te quedan {MAX_ATTEMPTS - attempts - 1} intentos."
                )]
            }

        nombre = match.iloc[0]["nombre_estudiante"]
        return {
            "auth_status": "ok",
            "auth_attempts": 0,
            "context": {"nombre": nombre},
            "messages": state["messages"] + [AIMessage(
                content=f" Hola, *{nombre}*. ¿En qué te puedo ayudar?"
            )]
        }

    return {
        "messages": state["messages"] + [AIMessage(
            content=" No entendí. Ingresa tu teléfono o escribe *no me acuerdo* para recuperar."
        )]
    }
import os
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.state import ChatState

ESTUDIANTES_CSV = os.path.join(
    os.path.dirname(__file__), "../../data/csv/datos_estudiantes.csv"
)


def _format_pesos(amount: int) -> str:
    return f"${amount:,.0f}".replace(",", ".")


def _get_pagos(user_phone: str, df: pd.DataFrame) -> str:
    """Consulta estados de pago del estudiante."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    row = match.iloc[0]

    saldo_palabra = row.get("SALDO_NETO_PALABRA", "N/A")
    saldo_real = row.get("SALDO_NETO_REAL", 0)

    lines = ["💰 *Estado de Cuenta:*\n"]

    lines.append(
        f"• Pensión: {_format_pesos(row.get('PAGO_MATRICULA', 0))} pagado / {_format_pesos(row.get('DEBE_MATRICULA', 0))} pendiente"
    )
    lines.append(
        f"• Clases: {_format_pesos(row.get('PAGO_CLASES', 0))} pagado / {_format_pesos(row.get('DEBE_CLASES', 0))} pendiente"
    )
    lines.append("")

    if saldo_real > 0:
        lines.append(f"✅ Saldo a favor: {_format_pesos(saldo_real)}")
    elif saldo_real < 0:
        lines.append(f"⚠️ Saldo pendiente: {_format_pesos(abs(saldo_real))}")
    else:
        lines.append("✅ ¡Estás al día!")

    return "\n".join(lines)


def _get_asistencia(user_phone: str, df: pd.DataFrame) -> str:
    """Consulta asistencia del estudiante."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    row = match.iloc[0]

    total = row.get("CUANTAS_CLASES_LLEVAMOS", 0)
    asistio = row.get("CUANTAS_CLASES_ASISTIO", 0)
    con_pago = row.get("CUANTAS_CLASES_ASISTIO_CON_PAGO", 0)
    sin_pago = row.get("CUANTAS_CLASES_ASISTIO_SIN_PAGO", 0)
    no_asistio = row.get("CUANTAS_CLASES_NO_HA_ASISTIDO", 0)

    pct = (asistio / total * 100) if total > 0 else 0

    lines = ["📚 *Resumen de Asistencia:*\n"]
    lines.append(f"• Total clases: {total}")
    lines.append(f"• Asistidas: {asistio} ({pct:.0f}%)")
    lines.append(f"  - Con pago: {con_pago}")
    lines.append(f"  - Sin pago: {sin_pago}")
    lines.append(f"• No asistió: {no_asistio}")
    lines.append("")

    if row.get("ASISTIO_SI_NO_A_LA_ULTIMA_CLASE") == "SI":
        lines.append("✅ Asististe a la última clase ✓")
    else:
        lines.append("❌ No asististe a la última clase")

    return "\n".join(lines)


def _get_info_general(user_phone: str, df: pd.DataFrame) -> str:
    """Consulta info general (colegio, grupo, etc.)."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    row = match.iloc[0]

    lines = [" *Información del Estudiante:*\n"]
    lines.append(f"• Nombre: {row.get('nombre_estudiante', 'N/A')}")
    lines.append(f"• Colegio: {row.get('COLEGIO', 'N/A')}")
    lines.append(f"• Grupo: {row.get('GRUPO', 'N/A')}")
    lines.append(f"• Tiene hermano: {row.get('TIENE_HERMANO', 'N/A')}")
    lines.append(f"• Correo: {row.get('CORREO', 'N/A')}")

    return "\n".join(lines)


def _get_phone_numbers(user_phone: str, df: pd.DataFrame) -> str:
    """Consulta todos los números registrados del estudiante."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    row = match.iloc[0]
    nombre = row.get("nombre_estudiante", "Estudiante")
    numeros_raw = str(row.get("all_phone_numbers", "")).split(";")
    numeros = [n.strip() for n in numeros_raw if n.strip()]

    lines = [f"📱 *Números registrados para {nombre}:*\n"]
    for i, num in enumerate(numeros, 1):
        principal = (
            " (principal)" if num == str(row.get("cellphone", "")).strip() else ""
        )
        lines.append(f"{i}. {num}{principal}")

    return "\n".join(lines)


def _update_email(user_phone: str, df: pd.DataFrame, nuevo_correo: str) -> str:
    """Actualiza el correo del estudiante."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    import re

    if not re.match(r"[^@]+@[^@]+", nuevo_correo):
        return " El formato del correo no es válido. Ejemplo: usuario@correo.com"

    idx = match.index[0]
    df.at[idx, "CORREO"] = nuevo_correo

    try:
        df.to_csv(ESTUDIANTES_CSV, index=False)
        return f"✅ Tu correo ha sido actualizado a: {nuevo_correo}"
    except Exception as e:
        return f"❌ Error al guardar: {str(e)}"


def _update_phone(user_phone: str, df: pd.DataFrame, nuevo_telefono: str) -> str:
    """Actualiza el teléfono principal del estudiante."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    import re

    digits = re.sub(r"\D", "", nuevo_telefono)
    if len(digits) < 10:
        return " El teléfono debe tener al menos 10 dígitos."

    nuevo_clean = f"+57{digits[-10:]}"
    idx = match.index[0]
    df.at[idx, "cellphone"] = nuevo_clean

    numeros_actuales = str(df.at[idx, "all_phone_numbers"])
    if nuevo_clean not in numeros_actuales:
        df.at[idx, "all_phone_numbers"] = numeros_actuales + ";" + nuevo_clean

    try:
        df.to_csv(ESTUDIANTES_CSV, index=False)
        return f"✅ Tu teléfono principal ha sido actualizado a: {nuevo_clean}"
    except Exception as e:
        return f"❌ Error al guardar: {str(e)}"


def _update_password(user_phone: str, df: pd.DataFrame, nueva_contrasena: str) -> str:
    """Actualiza la contraseña del estudiante."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    if len(nueva_contrasena) < 4:
        return " La contraseña debe tener al menos 4 caracteres."

    idx = match.index[0]
    df.at[idx, "CONTRASENA"] = nueva_contrasena

    try:
        df.to_csv(ESTUDIANTES_CSV, index=False)
        return f"✅ Tu contraseña ha sido actualizada. Ya puedes iniciar sesión."
    except Exception as e:
        return f"❌ Error al guardar: {str(e)}"


def _update_name(user_phone: str, df: pd.DataFrame, nuevo_nombre: str) -> str:
    """Actualiza el nombre del estudiante."""

    def clean_phone(phone_str):
        return (
            str(phone_str)
            .replace("+57", "")
            .replace(" ", "")
            .replace("-", "")
            .replace("+", "")
        )

    clean_input = clean_phone(user_phone)
    df["phone_clean"] = df["cellphone"].apply(clean_phone)
    df["all_phones_clean"] = df["all_phone_numbers"].apply(clean_phone)

    match = df[
        (df["phone_clean"] == clean_input)
        | (df["all_phones_clean"].str.contains(clean_input, na=False))
    ]

    if match.empty:
        return "No se encontraron registros."

    if len(nuevo_nombre.strip()) < 3:
        return " El nombre debe tener al menos 3 caracteres."

    idx = match.index[0]
    df.at[idx, "nombre_estudiante"] = nuevo_nombre.strip()

    try:
        df.to_csv(ESTUDIANTES_CSV, index=False)
        return f" Tu nombre ha sido actualizado a: {nuevo_nombre.strip()}"
    except Exception as e:
        return f" Error al guardar: {str(e)}"


def pagos_node(state: ChatState) -> dict:
    """Nodo LangGraph: consulta de pagos, asistencia e info."""

    if state.get("auth_status") != "ok":
        response = AIMessage(
            content=(
                " Para consultar esta información necesitas verificar tu identidad.\n"
                "Por favor, ingresa tu *número de teléfono*."
            )
        )
        return {
            "intent": "auth",
            "messages": state["messages"] + [response],
        }

    user_phone = state.get("user_id", "")
    if not user_phone.startswith("+"):
        user_phone = f"+57{user_phone}"

    try:
        df = pd.read_csv(ESTUDIANTES_CSV)
    except FileNotFoundError:
        return {
            "messages": state["messages"]
            + [AIMessage(content="Sistema no disponible.")],
        }

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    query = last_human.content.lower() if last_human else ""

    if (
        "asistencia" in query
        or "asistí" in query
        or "asistio" in query
        or "asistiendo" in query
        or "voy" in query
        or "van" in query
        or "llevo" in query
    ):
        result = _get_asistencia(user_phone, df)
    elif "pago" in query or "pagado" in query or "debo" in query or "saldo" in query:
        result = _get_pagos(user_phone, df)
    elif "contraseña" in query or "clave" in query:
        if (
            "actualizar" in query
            or "cambiar" in query
            or "nueva" in query
            or "nuevo" in query
        ):
            import re

            parts = re.split(r"\s+a\s+", query)
            if len(parts) > 1:
                nueva_pass = parts[-1].strip()
                result = _update_password(user_phone, df, nueva_pass)
            else:
                result = (
                    "Ingresa tu nueva contraseña así: cambiar mi contraseña a nueva123"
                )
        else:
            result = _get_info_general(user_phone, df)
    elif "nombre" in query:
        if "actualizar" in query or "cambiar" in query or "nuevo" in query:
            import re

            match = re.search(
                r"(?:a|es|por|sera|será)\s+([A-Za-zÁ-ÿ\s]+?)(?:\s*$|\s+[¿?]|\.)",
                last_human.content,
            )
            print(f"[DEBUG] query: {query}, match: {match}")
            if match:
                nuevo_nombre = match.group(1).strip()
                result = _update_name(user_phone, df, nuevo_nombre)
            else:
                match2 = re.search(r"(?:a|es|por|sera|será)\s+([A-Za-zÁ-ÿ\s]+)", query)
                print(f"[DEBUG] match2: {match2}")
                if match2:
                    nuevo_nombre = match2.group(1).strip()
                    result = _update_name(user_phone, df, nuevo_nombre)
                else:
                    result = (
                        "Ingresa tu nuevo nombre así: cambiar mi nombre a Juan Perez"
                    )
        else:
            result = _get_info_general(user_phone, df)
    elif "correo" in query or "email" in query:
        import re

        match = re.search(r"[\w\.-]+@[\w\.-]+", last_human.content)
        if match:
            result = _update_email(user_phone, df, match.group(0))
        else:
            result = "Ingresa el nuevo correo electrónico (formato: usuario@correo.com)"
    elif (
        "número" in query
        or "numeros" in query
        or "teléfono" in query
        or "telefono" in query
        or "telefón" in query
        or "registrado" in query
        or "actualizalo" in query
        or "actualizame" in query
    ):
        if (
            "actualizar" in query
            or "cambiar" in query
            or "nuevo" in query
            or "actualizalo" in query
            or "actualizame" in query
        ):
            import re

            digits = re.sub(r"\D", "", last_human.content)
            if len(digits) >= 10:
                result = _update_phone(user_phone, df, digits[-10:])
            else:
                result = "Ingresa el nuevo número de teléfono (10 dígitos)."
        else:
            result = _get_phone_numbers(user_phone, df)
    elif "hermano" in query or "hermana" in query:
        result = _get_info_general(user_phone, df)
    elif "clase" in query or "clases" in query:
        result = _get_asistencia(user_phone, df)
    elif "colegio" in query or "grupo" in query:
        result = _get_info_general(user_phone, df)
    else:
        result = _get_info_general(user_phone, df)

    response = AIMessage(content=result)

    return {
        "context": {"query_type": "pagos/asistencia"},
        "messages": state["messages"] + [response],
    }

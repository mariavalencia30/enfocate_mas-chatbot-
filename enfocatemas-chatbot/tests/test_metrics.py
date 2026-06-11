"""
test_metrics.py — Métricas reales del chatbot Enfócate Más
Mide: latencia, concurrencia y precisión de flujos completos
Uso: python test_metrics.py --url http://localhost:8000
"""

import asyncio
import time
import statistics
import argparse
import json
from datetime import datetime
import httpx

BASE_URL = "http://localhost:8000"

# ─── Casos de prueba con respuesta esperada ───────────────────────────────────
TEST_CASES = [
    # (session_id, mensaje, texto_esperado_en_respuesta)
    ("metric-bienvenida",  "hola",                     "Enfócate"),
    ("metric-faq-1",       "horarios",                 "lunes"),
    ("metric-faq-2",       "programas",                None),   # solo mide latencia
    ("metric-login",       "+57 3874290099",           "Sofia"),
    ("metric-error-1",     "+57 300 999 9999",         "registrado"),
    ("metric-error-2",     "xkdzpq",                   None),
]

# ─── Flujos completos (multi-turno) ──────────────────────────────────────────
FLUJOS = {
    "login_y_pagos": [
        ("+57 3874290099",  "Sofia"),
        ("cuánto debo",     "Estado de Cuenta"),
    ],
    "login_y_asistencia": [
        ("+57 3874290099",  "Sofia"),
        ("ver mi asistencia", "Asistencia"),
    ],
}


async def send_message(client: httpx.AsyncClient, session_id: str, message: str) -> tuple:
    """Envía un mensaje y retorna (respuesta, latencia_ms, ok)."""
    body = {"session_id": session_id, "message": message, "user_id": ""}
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{BASE_URL}/chat/web",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        data = r.json()
        return data.get("response", ""), latency, r.status_code == 200
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return str(e), latency, False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LATENCIA
# ═══════════════════════════════════════════════════════════════════════════════
async def test_latencia(repeticiones: int = 5):
    print("\n" + "═" * 60)
    print("  1. PRUEBA DE LATENCIA")
    print("═" * 60)

    resultados = []
    async with httpx.AsyncClient() as client:
        for i, (sid, msg, esperado) in enumerate(TEST_CASES):
            latencias = []
            for rep in range(repeticiones):
                resp, lat, ok = await send_message(client, f"{sid}-r{rep}", msg)
                latencias.append(lat)

            avg = statistics.mean(latencias)
            p95 = sorted(latencias)[int(len(latencias) * 0.95) - 1]
            mn  = min(latencias)
            mx  = max(latencias)

            estado = "✅" if ok else "❌"
            print(f"  {estado} [{msg[:30]:<30}]  avg={avg:6.0f}ms  p95={p95:6.0f}ms  min={mn:6.0f}ms  max={mx:6.0f}ms")
            resultados.append({"mensaje": msg, "avg_ms": round(avg), "p95_ms": round(p95)})

    todas = [r["avg_ms"] for r in resultados]
    print(f"\n  Promedio global : {statistics.mean(todas):.0f} ms")
    print(f"  P95 global      : {sorted(todas)[int(len(todas)*0.95)-1]:.0f} ms")
    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CARGA CONCURRENTE
# ═══════════════════════════════════════════════════════════════════════════════
async def test_concurrencia(niveles=(1, 5, 10, 20)):
    print("\n" + "═" * 60)
    print("  2. PRUEBA DE CARGA CONCURRENTE")
    print("═" * 60)
    print(f"  {'Usuarios':<10} {'Éxito':<8} {'Fallos':<8} {'Avg ms':<10} {'P95 ms':<10} {'TPS'}")

    resultados = []
    for n_users in niveles:
        async with httpx.AsyncClient() as client:
            tareas = [
                send_message(client, f"load-{n_users}-u{i}", "hola")
                for i in range(n_users)
            ]
            t0 = time.perf_counter()
            respuestas = await asyncio.gather(*tareas)
            total_time = time.perf_counter() - t0

        latencias = [r[1] for r in respuestas]
        exitos    = sum(1 for r in respuestas if r[2])
        fallos    = n_users - exitos
        avg       = statistics.mean(latencias)
        p95       = sorted(latencias)[int(len(latencias) * 0.95) - 1]
        tps       = exitos / total_time

        estado = "✅" if fallos == 0 else "⚠️ "
        print(f"  {estado} {n_users:<10} {exitos:<8} {fallos:<8} {avg:<10.0f} {p95:<10.0f} {tps:.1f}")
        resultados.append({
            "usuarios": n_users, "exitos": exitos, "fallos": fallos,
            "avg_ms": round(avg), "p95_ms": round(p95), "tps": round(tps, 2)
        })

    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRECISIÓN DE FLUJOS
# ═══════════════════════════════════════════════════════════════════════════════
async def test_flujos():
    print("\n" + "═" * 60)
    print("  3. PRECISIÓN DE FLUJOS COMPLETOS")
    print("═" * 60)

    resultados = []
    async with httpx.AsyncClient() as client:
        for flujo_nombre, pasos in FLUJOS.items():
            print(f"\n  Flujo: {flujo_nombre}")
            sid = f"flujo-{flujo_nombre}-{int(time.time())}"
            flujo_ok = True
            for i, (msg, esperado) in enumerate(pasos):
                resp, lat, ok = await send_message(client, sid, msg)
                contiene = esperado.lower() in resp.lower() if esperado else True
                estado = "✅" if (ok and contiene) else "❌"
                if not (ok and contiene):
                    flujo_ok = False
                print(f"    Paso {i+1}: {estado}  [{msg}]  → {lat:.0f}ms")
                if not contiene and esperado:
                    print(f"           Esperaba: '{esperado}' | Respuesta: {resp[:80]}...")

            resultados.append({"flujo": flujo_nombre, "ok": flujo_ok})

    total   = len(resultados)
    exitosos = sum(1 for r in resultados if r["ok"])
    print(f"\n  Flujos exitosos: {exitosos}/{total} ({exitosos/total*100:.0f}%)")
    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════
async def test_health():
    print("\n" + "═" * 60)
    print("  0. HEALTH CHECK")
    print("═" * 60)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{BASE_URL}/health", timeout=5.0)
            if r.status_code == 200:
                print(f"  ✅ Servidor OK — {r.json()}")
                return True
            else:
                print(f"  ❌ Servidor respondió {r.status_code}")
                return False
    except Exception as e:
        print(f"  ❌ No se pudo conectar: {e}")
        print(f"     ¿Está corriendo? → uvicorn src.integrations.gateway:app --reload --port 8000")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
async def main(repeticiones: int = 3, niveles_carga=(1, 5, 10)):
    print("\n" + "═" * 60)
    print("  MÉTRICAS REALES — Enfócate Más Chatbot")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  URL: {BASE_URL}")
    print("═" * 60)

    # 0. Health check
    ok = await test_health()
    if not ok:
        return

    # 1. Latencia
    lat = await test_latencia(repeticiones)

    # 2. Concurrencia
    conc = await test_concurrencia(niveles_carga)

    # 3. Flujos
    flujos = await test_flujos()

    # Guardar resultados
    reporte = {
        "timestamp": datetime.now().isoformat(),
        "url": BASE_URL,
        "latencia": lat,
        "concurrencia": conc,
        "flujos": flujos,
    }
    with open("metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 60)
    print("  ✅ Reporte guardado en: metrics_report.json")
    print("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default="http://localhost:8000")
    parser.add_argument("--reps", type=int, default=3, help="Repeticiones por caso")
    parser.add_argument("--carga", default="1,5,10", help="Niveles de carga (ej: 1,5,10,20)")
    args = parser.parse_args()

    BASE_URL = args.url
    niveles  = [int(x) for x in args.carga.split(",")]

    asyncio.run(main(repeticiones=args.reps, niveles_carga=niveles))

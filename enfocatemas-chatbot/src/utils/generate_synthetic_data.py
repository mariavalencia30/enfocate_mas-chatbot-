import csv
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

NUM_REGISTROS = 3500
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "csv"

# Configuración base
COUNTRY_CODE = 57
CLASE_VALOR_NORMAL = 10000
CLASE_VALOR_HERMANO = 8000

colegios = [
    ("Colegio San José", "colegio-san-jose"),
    ("Instituto Central", "instituto-central"),
    ("Colegio Nueva Era", "colegio-nueva-era"),
    ("Academia Futuro", "academia-futuro")
]

grupos = ["Grupo 1", "Grupo 2", "Grupo 3", "Grupo 4"]

nombres = ["Juan", "Maria", "Carlos", "Laura", "Andres", "Sofia", "Pedro", "Valentina"]
apellidos = ["Perez", "Gomez", "Rodriguez", "Lopez", "Martinez", "Garcia"]

def generar_nombre():
    return f"{random.choice(nombres)} {random.choice(apellidos)}"

def generar_telefono():
    numero = random.randint(3000000000, 3999999999)
    return f"+57 {numero}"

def generar_email(nombre):
    base = nombre.lower().replace(" ", ".")
    return f"{base}{random.randint(1,999)}@mail.com"

def generar_contrasena():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def generar_historial(tiene_hermano, total_clases):
    historial = []
    fecha = datetime(2024, 1, 1)
    asistio = 0
    asistio_pago = 0
    asistio_sin_pago = 0
    no_asistio = 0

    valor_clase = CLASE_VALOR_HERMANO if tiene_hermano else CLASE_VALOR_NORMAL

    for i in range(total_clases):
        fecha_str = fecha.strftime("%Y-%m-%d")
        evento = random.choice(["ASISTIO_PAGO", "ASISTIO_SIN_PAGO", "NO_ASISTIO"])

        if evento == "NO_ASISTIO":
            historial.append(f"{fecha_str}: NO_ASISTIO")
            no_asistio += 1
        elif evento == "ASISTIO_SIN_PAGO":
            historial.append(f"{fecha_str}: ASISTIO_SIN_PAGO")
            asistio += 1
            asistio_sin_pago += 1
        else:
            pago = valor_clase + random.choice([0, 5000, 10000])
            historial.append(f"{fecha_str}: {pago}")
            asistio += 1
            asistio_pago += 1

        fecha += timedelta(days=7)

    return "|".join(historial), asistio, asistio_pago, asistio_sin_pago, no_asistio


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "datos_estudiantes.csv"

with open(output_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "country_code","cellphone","TIENE_HERMANO","PAGO_MATRICULA","PAGO_CLASES",
        "DEBE_MATRICULA","DEBE_CLASES","SALDO_NETO_REAL","SALDO_NETO_PALABRA",
        "HISTORIAL_ASISTENCIA_Y_PAGOS","nombre_estudiante",
        "CUANTAS_CLASES_LLEVAMOS","CUANTAS_CLASES_ASISTIO",
        "CUANTAS_CLASES_ASISTIO_CON_PAGO","CUANTAS_CLASES_ASISTIO_SIN_PAGO",
        "CUANTAS_CLASES_NO_HA_ASISTIDO","ASISTIO_SI_NO_A_LA_ULTIMA_CLASE",
        "ASISTIO_ULTIMOS_3_MESES","COLEGIO","COLEGIO_SLUG","GRUPO",
        "CORREO","CONTRASENA","all_phone_numbers"
    ])

    for _ in range(NUM_REGISTROS):
        nombre = generar_nombre()
        telefono = generar_telefono()
        tiene_hermano = random.choice(["SI", "NO"])
        total_clases = random.randint(20, 60)

        historial, asistio, asistio_pago, asistio_sin_pago, no_asistio = generar_historial(
            tiene_hermano == "SI", total_clases
        )

        valor_clase = CLASE_VALOR_HERMANO if tiene_hermano == "SI" else CLASE_VALOR_NORMAL

        pago_clases = asistio_pago * valor_clase
        debe_clases = asistio_sin_pago * valor_clase

        pago_matricula = random.randint(0, 50000)
        debe_matricula = random.randint(0, 50000)

        saldo = (pago_clases + pago_matricula) - (debe_clases + debe_matricula)

        if saldo > 0:
            saldo_texto = f"A_FAVOR:{saldo}"
        elif saldo < 0:
            saldo_texto = f"DEBE:{abs(saldo)}"
        else:
            saldo_texto = "PAZ_Y_SALVO"

        colegio, slug = random.choice(colegios)

        writer.writerow([
            COUNTRY_CODE,
            telefono,
            tiene_hermano,
            pago_matricula,
            pago_clases,
            debe_matricula,
            debe_clases,
            saldo,
            saldo_texto,
            historial,
            nombre,
            total_clases,
            asistio,
            asistio_pago,
            asistio_sin_pago,
            no_asistio,
            "SI" if random.random() > 0.3 else "NO",
            "SI" if random.random() > 0.4 else "NO",
            colegio,
            slug,
            random.choice(grupos),
            generar_email(nombre),
            generar_contrasena(),
            telefono + ";" + generar_telefono()
        ])

print(f"CSV generado correctamente en: {output_path}")
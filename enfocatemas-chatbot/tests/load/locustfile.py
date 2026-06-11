import uuid
import random
from locust import HttpUser, task, between, events


class FAQUser(HttpUser):
    """
    Simula un usuario que hace preguntas frecuentes.
    Peso: 6 (60% del tráfico)
    """
    weight = 6
    wait_time = between(1, 3)

    FAQ_QUESTIONS = [
        "¿Cuál es el horario de atención?",
        "¿Cuándo son las vacaciones?",
        "¿Cuáles son los requisitos de matrícula?",
        "¿Qué documentos necesito para inscribir a mi hijo?",
        "¿Cuál es el calendario académico?",
        "¿Tienen servicio de transporte escolar?",
        "¿Cómo me comunico con el director?",
        "¿Qué actividades extracurriculares ofrecen?",
    ]

    def on_start(self):
        self.session_id = str(uuid.uuid4())

    @task
    def ask_faq(self):
        question = random.choice(self.FAQ_QUESTIONS)
        with self.client.post(
            "/chat/web",
            json={
                "session_id": self.session_id,
                "message": question,
                "user_id": "",
            },
            catch_response=True,
            name="POST /chat/web [FAQ]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "response" not in data:
                    response.failure("Campo 'response' ausente en respuesta")
                elif len(data["response"]) == 0:
                    response.failure("Respuesta vacía")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class AuthUser(HttpUser):
    """
    Simula un usuario que intenta autenticarse.
    Peso: 2.5 (25% del tráfico)
    """
    weight = 3
    wait_time = between(2, 5)

    def on_start(self):
        self.session_id = str(uuid.uuid4())

    @task(3)
    def attempt_login(self):
        """Intento de login con credenciales."""
        with self.client.post(
            "/chat/web",
            json={
                "session_id": self.session_id,
                "message": "usuario: test.user contraseña: clave123",
                "user_id": "",
            },
            catch_response=True,
            name="POST /chat/web [AUTH]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def attempt_failed_login(self):
        """Intento de login fallido (simula error del usuario)."""
        with self.client.post(
            "/chat/web",
            json={
                "session_id": str(uuid.uuid4()),  # Nueva sesión para no bloquear
                "message": f"usuario: user_{random.randint(1,999)} contraseña: mala",
                "user_id": "",
            },
            catch_response=True,
            name="POST /chat/web [AUTH-FAIL]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class MixedUser(HttpUser):
    """
    Simula un usuario completo: FAQ → auth → pagos.
    Peso: 1.5 (15% del tráfico)
    """
    weight = 2
    wait_time = between(3, 7)

    def on_start(self):
        self.session_id = str(uuid.uuid4())
        self.authenticated = False

    @task(2)
    def ask_faq_first(self):
        self.client.post(
            "/chat/web",
            json={
                "session_id": self.session_id,
                "message": "¿Cuáles son los horarios?",
                "user_id": "",
            },
            name="POST /chat/web [MIXED-FAQ]",
        )

    @task(2)
    def authenticate(self):
        response = self.client.post(
            "/chat/web",
            json={
                "session_id": self.session_id,
                "message": "usuario: test.user contraseña: clave123",
                "user_id": "",
            },
            name="POST /chat/web [MIXED-AUTH]",
        )
        if response.status_code == 200:
            self.authenticated = True

    @task(1)
    def check_payments(self):
        self.client.post(
            "/chat/web",
            json={
                "session_id": self.session_id,
                "message": "¿cuánto debo este mes?",
                "user_id": "1",
            },
            name="POST /chat/web [MIXED-PAGOS]",
        )


class HealthCheckUser(HttpUser):
    """Monitor del health endpoint durante la carga."""
    weight = 1
    wait_time = between(5, 10)

    @task
    def health_check(self):
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") != "ok":
                    response.failure("Status no es 'ok'")
                else:
                    response.success()
            else:
                response.failure(f"Health check fallido: HTTP {response.status_code}")


#Event hooks para reporte 

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n🚀 Iniciando prueba de carga — Chatbot Enfócate Más")
    print("   Escenarios: FAQ (60%) | Auth (25%) | Mixto (15%)")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    print(f"\n📊 Resumen de prueba de carga:")
    print(f"   Total requests: {stats.num_requests}")
    print(f"   Failures:       {stats.num_failures}")
    print(f"   Avg response:   {stats.avg_response_time:.0f}ms")
    print(f"   RPS:            {stats.current_rps:.1f}")

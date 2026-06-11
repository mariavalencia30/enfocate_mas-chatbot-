# GPT-4o-mini como modelo de Inteligencia Artificial para Enfócate Más

## Introducción

Este documento académico analiza el modelo de Inteligencia Artificial que realmente da solución al problema del proyecto **Enfócate Más**: **GPT-4o-mini**. El repositorio implementa un chatbot administrativo 24/7 para más de 3.000 familias, orientado a resolver consultas de **pagos, asistencia, matrículas y preguntas frecuentes**, usando **LangGraph + RAG + GPT-4o-mini**. Esta afirmación aparece explícitamente en `docs/READMEE.md:3`.

El objetivo de este informe no es discutir la ausencia de un entrenamiento local desde cero, sino demostrar **dominio profundo del modelo utilizado**, de su naturaleza como red neuronal artificial moderna, de su arquitectura tipo Transformer, de su funcionamiento en el proyecto y de la forma en que se alinea con las necesidades institucionales de Enfócate Más.

## 1. Relación entre el problema y el modelo elegido

### 1.1 Problema específico de Enfócate Más

El problema real del proyecto está definido en `docs/READMEE.md:35-43`: **Enfócate Más** es una institución educativa que atiende diariamente a miles de familias con necesidades administrativas recurrentes. Entre esas necesidades están:

- consultar pagos y saldos pendientes,
- renovar o iniciar matrículas,
- revisar información de asistencia,
- resolver dudas generales sobre la institución,
- recibir atención sin depender del horario de oficina.

El mismo documento establece que estos procesos, cuando son gestionados manualmente, generan **cuellos de botella en el personal administrativo**. Por tanto, el problema no es solamente tecnológico; es también operativo, comunicacional y de servicio.

### 1.2 Por qué un modelo de IA es necesario

Un flujo puramente manual o basado solo en reglas rígidas tendría dificultades para responder a la variedad de expresiones naturales de los usuarios. Una familia puede escribir:

- “¿Cuál es el horario de atención?”
- “¿Cuándo abren?”
- “Necesito saber a qué horas atienden”

Las tres expresiones apuntan a la misma intención, pero no usan exactamente las mismas palabras. En este contexto, un modelo de IA como **GPT-4o-mini** es necesario porque puede:

- interpretar lenguaje natural en español,
- distinguir intención comunicativa,
- generar respuestas fluidas y útiles,
- trabajar con contexto institucional específico,
- integrarse con recuperación semántica para responder con base en documentos reales.

Esto se alinea con el pipeline definido en `docs/READMEE.md:49-53`:

```python
Documento -> Chunking -> Embeddings -> ChromaDB -> Búsqueda -> Contexto -> LLM -> Respuesta
```

### 1.3 Necesidades de los involucrados

### Familias

Las familias necesitan un canal de atención inmediato, comprensible y disponible fuera del horario administrativo. El README especifica que WhatsApp es el **canal preferido por las familias** (`docs/READMEE.md:41`), por lo que la solución no solo debe responder bien, sino hacerlo en el canal correcto.

### Estudiantes

Los estudiantes, directa o indirectamente, requieren acceso rápido a información sobre asistencia, matrículas, programas y datos institucionales. En un entorno académico, la demora en estas respuestas afecta continuidad, organización y seguimiento.

### Personal administrativo

El personal administrativo necesita reducir la carga repetitiva de preguntas frecuentes y consultas operativas, para concentrarse en casos complejos, excepciones o solicitudes que requieren intervención humana.

### 1.4 Cómo GPT-4o-mini ayuda a resolver esas necesidades

Dentro del proyecto, **GPT-4o-mini** cumple dos funciones clave:

- **clasificación de intención**, en `src/nodes/classifier.py:135-153`
- **generación de respuestas contextualizadas**, en `src/nodes/faq.py:57-72`

En la práctica, esto significa que GPT-4o-mini permite:

- identificar si el usuario pregunta por `faq`, `auth`, `pagos`, `matricula` o `soporte`,
- responder dudas institucionales en español claro,
- usar contexto documental real recuperado desde ChromaDB,
- mantener una experiencia conversacional adecuada para WhatsApp y web.

### 1.5 Tareas concretas que realiza GPT-4o-mini en el proyecto

Las tareas reales del modelo en el repositorio son las siguientes:

| Tarea | Evidencia en el proyecto | Papel de GPT-4o-mini |
| --- | --- | --- |
| Clasificación de intención | `src/nodes/classifier.py:12-26`, `135-153` | Lee el mensaje y devuelve una clase válida |
| Generación de respuesta FAQ | `src/nodes/faq.py:11-19`, `57-72` | Responde usando contexto institucional recuperado |
| Integración dentro del grafo conversacional | `src/graph.py:41-68` | Actúa dentro de nodos orquestados por LangGraph |
| Soporte al canal WhatsApp/Web | `src/integrations/gateway.py:101-124`, `163-179` | Produce respuestas que luego se entregan por el gateway |

### 1.6 Alineación con el contexto institucional

GPT-4o-mini está alineado con el contexto de Enfócate Más porque el problema exige:

- respuesta en español,
- comprensión flexible del lenguaje,
- rapidez suficiente para mensajería,
- bajo costo por volumen,
- integración con documentos y flujos administrativos,
- escalabilidad para miles de familias.

El propio README justifica su selección así (`docs/READMEE.md:120-125`):

- mejor balance costo/rendimiento,
- respuesta suficientemente rápida para WhatsApp,
- costo estimado de `50–100 USD/mes` para 3.000 familias,
- posibilidad de escalar sin rediseñar la arquitectura.

### Discurso de exposición para este punto

En Enfócate Más el problema no era simplemente “hacer un chatbot”, sino responder de forma útil y escalable a miles de familias que consultan pagos, asistencia, matrículas y preguntas generales. En ese contexto, GPT-4o-mini fue pertinente porque combina comprensión del lenguaje natural, generación de texto y velocidad suficiente para un canal como WhatsApp. Dentro del proyecto, el modelo no se usa de forma abstracta: se usa para clasificar la intención del usuario y para generar respuestas contextualizadas a partir de documentos institucionales, lo cual lo vuelve directamente funcional para las necesidades de familias, estudiantes y personal administrativo.

## 2. ¿Qué es GPT-4o-mini?

### 2.1 Definición general

**GPT-4o-mini** es un **Large Language Model (LLM)** desarrollado por **OpenAI**. Un LLM es un modelo neuronal de gran escala diseñado para procesar y generar lenguaje natural a partir de enormes volúmenes de texto. Su función principal es predecir secuencias lingüísticas de forma coherente y útil, pero en la práctica puede desempeñar tareas complejas como clasificación, resumen, reformulación, extracción de información y respuesta conversacional.

### 2.2 Qué significa GPT

**GPT** significa **Generative Pre-trained Transformer**.

- **Generative**: puede generar texto nuevo.
- **Pre-trained**: ha sido entrenado previamente sobre grandes corpus de datos.
- **Transformer**: está basado en la arquitectura neuronal Transformer.

### 2.3 Quién desarrolló GPT-4o-mini

GPT-4o-mini fue desarrollado por **OpenAI**, organización que también ha desarrollado las series GPT-3.5 y GPT-4. En el proyecto, el acceso al modelo se realiza a través de la clase `ChatOpenAI` de `langchain_openai`, como puede verse en:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

Código real en `src/nodes/classifier.py:135-139`.

### 2.4 Diferencias entre GPT-4o y GPT-4o-mini

Desde la evidencia del proyecto y el benchmark, la diferencia práctica más relevante es el equilibrio entre **calidad, latencia y costo**.

| Modelo | Evidencia en el proyecto | Foco práctico |
| --- | --- | --- |
| GPT-4o | `benchmark_rag.py`, `benchmark_results.csv`, `plot_final_results.py` | mayor costo, buen desempeño |
| GPT-4o-mini | mismo benchmark y nodos reales | menor costo, menor latencia, integración real del chatbot |

Según `docs/READMEE.md:112-125`:

- `GPT-4o + OpenAI Emb`: F1 `0.9316`, latencia promedio `1.74s`, costo `$2.50/1K tokens`
- `GPT-4o-mini + OpenAI Emb`: F1 `0.6795`, latencia promedio `0.69s`, costo `$0.15/1K tokens`

Esto significa que GPT-4o-mini sacrifica parte del desempeño máximo absoluto frente a GPT-4o, pero lo compensa con una reducción de costo muy fuerte y una latencia más conveniente para uso masivo.

### 2.5 Capacidades principales

GPT-4o-mini, tal como se usa en el proyecto, demuestra estas capacidades:

- comprensión de instrucciones en lenguaje natural,
- clasificación de intención con salida controlada,
- generación de respuestas conversacionales en español,
- integración con contexto recuperado por RAG,
- desempeño útil en escenarios multicanal.

### 2.6 Casos de uso

En el contexto del proyecto, los casos de uso reales son:

- clasificar si una consulta corresponde a `faq`, `auth`, `pagos`, `matricula` o `soporte`,
- responder preguntas frecuentes a partir de documentos institucionales,
- apoyar la interacción por WhatsApp y web,
- reducir carga de trabajo operativo.

### 2.7 Ventajas

- costo bajo por token frente a GPT-4o,
- velocidad adecuada para atención conversacional,
- buena integración con LangChain, LangGraph y RAG,
- suficiente capacidad lingüística para español institucional.

### 2.8 Limitaciones

- puede cometer errores si el contexto recuperado es insuficiente,
- depende de buena recuperación documental para dar respuestas precisas,
- tiene menor capacidad máxima que modelos premium más costosos,
- puede presentar variabilidad si el problema requiere razonamiento muy profundo o datos no presentes en el contexto.

## 3. Tipo de modelo de Inteligencia Artificial

### 3.1 Qué tipo de modelo es GPT-4o-mini

GPT-4o-mini es un **modelo fundacional generativo basado en arquitectura Transformer**, perteneciente al campo de **Deep Learning** y más específicamente al área de **modelos de lenguaje de gran escala**.

### 3.2 Por qué se considera una red neuronal artificial

Se considera una red neuronal artificial porque su núcleo está compuesto por capas neuronales que transforman vectores de entrada en representaciones internas cada vez más abstractas. Aunque OpenAI no publica en este repositorio los pesos ni la cantidad exacta de capas del modelo, su pertenencia a la familia GPT implica que utiliza redes neuronales profundas con múltiples capas y mecanismos de atención.

### 3.3 Qué significa Deep Learning

**Deep Learning** significa aprendizaje profundo. Se refiere a modelos con muchas capas de procesamiento capaces de aprender representaciones complejas de los datos. En lenguaje natural, esto permite pasar de secuencias de caracteres o tokens a representaciones semánticas capaces de capturar intención, relación, contexto y estructura lingüística.

### 3.4 Qué significa Transformer

Un **Transformer** es una arquitectura neuronal diseñada para procesar secuencias y modelar dependencias entre elementos mediante mecanismos de atención. Su innovación central fue sustituir gran parte de la dependencia de estructuras recurrentes por **self-attention**, permitiendo paralelismo y mejor modelado contextual.

### 3.5 Qué significa modelo fundacional

Un **modelo fundacional** es un modelo de gran escala preentrenado sobre enormes conjuntos de datos, reutilizable en múltiples tareas sin necesidad de construir un modelo nuevo para cada problema. En el proyecto, GPT-4o-mini actúa como base general sobre la cual se construye una solución especializada mediante prompts y RAG.

### 3.6 Qué significa modelo generativo

Un modelo generativo no solo clasifica, sino que puede **producir salida nueva**, como texto original. GPT-4o-mini genera respuestas completas y naturales, por eso puede tanto clasificar intención como redactar una respuesta final útil para el usuario.

### 3.7 Qué significa aprendizaje supervisado en preentrenamiento

En términos amplios, el preentrenamiento de un modelo GPT se apoya en una forma de supervisión automática: el modelo aprende a **predecir el siguiente token** a partir del contexto previo. El propio texto funciona como señal de entrenamiento. Esta señal masiva le permite internalizar patrones sintácticos, semánticos y pragmáticos.

### 3.8 Qué significa RLHF

**RLHF** significa **Reinforcement Learning from Human Feedback**. Es una etapa en la que la calidad de las respuestas se refina usando evaluaciones humanas o preferencias humanas, orientando el comportamiento del modelo hacia respuestas más útiles, seguras y alineadas.

### 3.9 Relación de estos conceptos con GPT-4o-mini

GPT-4o-mini reúne todos estos conceptos:

- es una red neuronal artificial profunda,
- basada en Transformer,
- preentrenada a gran escala,
- refinada para interacción útil,
- reutilizable como modelo fundacional,
- generativa en su salida,
- adaptable a tareas concretas mediante prompt engineering y RAG.

## 4. Arquitectura interna de GPT-4o-mini

### Advertencia metodológica

OpenAI no publica en este repositorio ni en los archivos del proyecto el número exacto de capas, cabezas de atención o parámetros internos específicos de GPT-4o-mini. Por tanto, esta sección describe la **arquitectura estándar, defendible y académicamente correcta de un modelo GPT moderno basado en Transformer**, marco teórico suficiente para explicar con rigor cómo opera GPT-4o-mini dentro de Enfócate Más.

### 4.1 Entrada: texto del usuario

La entrada real de GPT-4o-mini en el proyecto es texto natural proveniente de WhatsApp o del endpoint web. Ejemplos reales del sistema incluyen:

- `¿Cuál es el horario de atención?`
- `¿Cuánto debo pagar este mes?`
- `usuario: test.user contraseña: clave123`
- `¿Qué documentos necesito para inscribir a mi hijo?`

El modelo no recibe tablas ni números de forma directa como estructura tabular. Recibe secuencias lingüísticas y, antes de cualquier inferencia, debe convertirlas en una representación matemática procesable.

### 4.2 Tokenización

Un **token** es una unidad mínima de procesamiento para el modelo. No coincide necesariamente con una palabra completa. Puede ser:

- una palabra,
- una subpalabra,
- un signo de puntuación,
- un número,
- una secuencia frecuente de caracteres.

Por ejemplo, una frase del proyecto como:

```python
¿Cuánto debo pagar este mes?
```

podría tokenizarse conceptualmente como:

```python
["¿", "Cuánto", " debo", " pagar", " este", " mes", "?"]
```

En un modelo comercial de OpenAI la tokenización exacta es propietaria, pero el principio es el mismo: partir el texto en unidades que luego puedan mapearse a vectores numéricos. Este paso es fundamental porque el modelo no opera sobre “palabras” en sentido humano, sino sobre secuencias discretas codificadas.

### 4.3 Embeddings

Un **embedding** es una representación vectorial densa. A cada token se le asigna un vector de alta dimensión. Matemáticamente, si el token `t_i` pertenece al vocabulario, el modelo lo transforma en un vector:

```python
e_i ∈ R^d
```

donde `d` es la dimensionalidad del espacio de representación.

La importancia de esto es semántica: dos expresiones diferentes pueden ocupar zonas cercanas del espacio vectorial si comparten significado. En el contexto de Enfócate Más, expresiones como:

- `¿Cuándo abren?`
- `¿Cuál es el horario de atención?`
- `¿A qué horas atienden?`

no son idénticas lexicalmente, pero sus embeddings pueden quedar relativamente próximos. Esa proximidad es lo que permite tanto la comprensión lingüística de GPT-4o-mini como la recuperación semántica en el pipeline RAG.

### 4.4 Positional Encoding o señal de posición

Los embeddings por sí solos no contienen orden. El modelo necesita saber qué token viene primero y cuál después. Por eso se agrega una señal posicional. Conceptualmente, la representación inicial del token `i` se construye como:

```python
h_i^(0) = e_i + p_i
```

donde:

- `e_i` es el embedding del token,
- `p_i` es la codificación posicional.

Esto es crucial porque en lenguaje natural el orden cambia el significado. No es equivalente escribir:

```python
no debo pagar
```

que escribir:

```python
debo no pagar
```

El modelo necesita diferenciar estas estructuras para no cometer errores de interpretación, especialmente en mensajes breves y ambiguos enviados por familias a través de WhatsApp.

### 4.5 Transformer

GPT-4o-mini pertenece conceptualmente a la familia de modelos **decoder-only Transformer**. Eso significa que utiliza bloques apilados que reciben la secuencia previa y predicen el siguiente token. Cada bloque contiene, al menos, tres operaciones de alto nivel:

- mecanismo de atención,
- red feed-forward,
- normalización y conexiones residuales.

La lógica de procesamiento puede esquematizarse así:

```python
Representación de entrada
-> self-attention
-> suma residual + normalización
-> feed-forward network
-> suma residual + normalización
-> salida de la capa
```

La potencia del Transformer está en que cada capa refina progresivamente la representación contextual de la secuencia.

### 4.6 Self-Attention: fórmula matemática completa

La operación central del Transformer es:

```python
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Esta fórmula resume cómo el modelo decide a qué partes del contexto debe prestar atención para construir una representación contextualizada.

### 4.6.1 Qué es Q (Query)

`Q` significa **Query**. Una query representa “qué está buscando” un token dentro del resto de la secuencia. Matemáticamente, si `X` es la matriz de representaciones de entrada de la capa, las queries se obtienen con una proyección lineal:

```python
Q = XW_Q
```

`W_Q` es una matriz de pesos aprendida durante el entrenamiento. En términos intuitivos, cada token genera una consulta sobre qué contexto le resulta relevante.

### 4.6.2 Qué es K (Key)

`K` significa **Key**. Una key representa cómo puede ser “encontrado” o “activado” un token cuando otro token lo necesita como contexto. Se obtiene con otra proyección lineal:

```python
K = XW_K
```

Las keys permiten comparar si la consulta de un token coincide con la información que otros tokens pueden aportar.

### 4.6.3 Qué es V (Value)

`V` significa **Value**. Un value representa el contenido que realmente se va a combinar cuando el modelo decide que cierto token es relevante. Se obtiene como:

```python
V = XW_V
```

Mientras `Q` y `K` determinan el patrón de atención, `V` aporta el contenido que será agregado para formar la representación contextual final.

### 4.6.4 Qué es `d_k`

`d_k` es la dimensión de los vectores key. Cuando se calcula `QK^T`, los valores pueden crecer mucho si la dimensión es grande. Por eso se divide por `√d_k`:

```python
QK^T / √d_k
```

Esta escala evita que los logits de atención sean excesivamente grandes y que la `softmax` se vuelva demasiado picuda o inestable. En términos prácticos, mejora estabilidad numérica y aprendizaje.

### 4.6.5 Qué hace softmax en este contexto

La operación `softmax` transforma los puntajes de compatibilidad entre queries y keys en una distribución de pesos positivos que suman 1:

```python
α_ij = exp(s_ij) / Σ_j exp(s_ij)
```

donde `s_ij` es el score de atención entre el token `i` y el token `j`. El resultado `α_ij` indica cuánta importancia le da el token `i` al token `j`.

### 4.6.6 Qué hace la multiplicación final por V

Una vez obtenidos los pesos de atención, se multiplican por `V`. Esto produce una suma ponderada de los contenidos relevantes:

```python
z_i = Σ_j α_ij V_j
```

En otras palabras, el token `i` reconstruye una nueva representación usando información de otros tokens, pero ponderada según relevancia contextual.

### 4.6.7 Ejemplo real aplicado al proyecto: “¿Cuánto debo pagar este mes?”

Supongamos la secuencia tokenizada conceptualmente:

```python
["¿", "Cuánto", " debo", " pagar", " este", " mes", "?"]
```

En esta frase, el token `debo` genera una query que buscará elementos contextuales relacionados con deuda, monto y temporalidad. Algunas relaciones importantes serían:

- `debo` presta mucha atención a `pagar` porque juntos forman el núcleo semántico de la obligación económica.
- `debo` también presta atención a `Cuánto` porque la pregunta no es solo si existe deuda, sino cuál es su magnitud.
- `pagar` presta atención a `mes` porque el pago está acotado temporalmente.
- `este` y `mes` mantienen fuerte relación entre sí porque forman una unidad temporal.

Una matriz de atención conceptual simplificada podría lucir así:

```python
token actual   -> tokens más relevantes
"Cuánto"      -> ["debo", "pagar"]
"debo"        -> ["Cuánto", "pagar", "mes"]
"pagar"       -> ["debo", "mes"]
"mes"         -> ["este", "pagar"]
```

La razón por la que `debo` y `pagar` tienen alta atención mutua es que, dentro de una consulta administrativa, ambas palabras co-construyen la intención de **consultar un saldo o deuda**, que en el proyecto suele corresponder al flujo `pagos`.

### 4.6.8 Por qué esto es importante para Enfócate Más

Las familias no hablan en lenguaje estructurado. Pueden escribir:

- `¿Cuánto debo pagar este mes?`
- `¿Qué saldo tengo pendiente?`
- `¿Me dices cuánto debo?`
- `¿Tengo algo por pagar este mes?`

Un sistema de reglas exactas tendría que anticipar cada variante. El self-attention, en cambio, permite que el modelo construya significado contextual aunque la formulación cambie. Esta propiedad es crítica en Enfócate Más porque el canal principal es WhatsApp y el lenguaje real del usuario es variable, breve y a veces incompleto.

### 4.7 Diferencia entre Single-Head Attention y Multi-Head Attention

En **single-head attention**, el modelo produce un solo patrón de atención. Eso le permite capturar una perspectiva contextual principal, pero limita su capacidad de representar múltiples relaciones simultáneas.

En **multi-head attention**, el modelo calcula varias atenciones en paralelo:

```python
head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_O
```

Cada cabeza puede especializarse en una familia de relaciones diferente. En un mensaje del proyecto, por ejemplo:

- una cabeza puede capturar relaciones gramaticales,
- otra puede capturar intención económica,
- otra puede reconocer marcadores temporales,
- otra puede enfocarse en entidades administrativas como matrícula, pago o soporte.

OpenAI no publica en este proyecto cuántas cabezas exactas usa GPT-4o-mini. Sin embargo, en la familia GPT moderna es habitual hablar de **decenas de cabezas de atención por bloque**, no de una sola. Académicamente, la forma correcta de expresarlo es: *los modelos GPT modernos suelen utilizar múltiples cabezas de atención por capa para capturar simultáneamente distintos tipos de dependencias lingüísticas y semánticas*.

### 4.8 Feed Forward Networks

Después del mecanismo de atención, cada token pasa por una **Feed Forward Network (FFN)**. Esta subred suele consistir en dos transformaciones lineales separadas por una activación no lineal, frecuentemente GELU en Transformers modernos.

Conceptualmente:

```python
FFN(x) = W_2 · GELU(W_1x + b_1) + b_2
```

La atención decide qué información importa; la FFN transforma esa información en una representación más abstracta y útil. En un chatbot como Enfócate Más, esto ayuda a pasar de un entendimiento superficial de palabras a una representación más rica de intención y contenido.

### 4.9 Capa de salida

Al final de la pila de capas, la representación contextual final se proyecta sobre el vocabulario:

```python
logits = h_final W_vocab + b_vocab
probabilidades = softmax(logits)
```

Cada valor de `probabilidades` representa la probabilidad del siguiente token candidato. El modelo selecciona uno y continúa de forma autoregresiva.

### 4.10 Generación token a token: autoregresión

#### 4.10.1 Qué significa autoregresivo

Decir que GPT-4o-mini es **autoregresivo** significa que genera un token a la vez usando como contexto todos los tokens anteriores, incluyendo los tokens que el propio modelo ya generó. Se llama autoregresivo porque cada nueva predicción depende de la propia secuencia parcialmente generada.

La diferencia con un modelo que generara toda la salida de una vez es profunda: aquí el texto emerge paso a paso, token tras token, refinando el contexto en cada iteración.

#### 4.10.2 Proceso paso a paso con un ejemplo del proyecto

Supongamos este escenario realista del chatbot:

Prompt del sistema:

```python
Eres un asistente administrativo de Enfócate Más...
```

Mensaje del usuario:

```python
¿Cuáles son los requisitos de matrícula?
```

El proceso autoregresivo ocurre así:

1. El modelo recibe todos los tokens del prompt del sistema y de la pregunta del usuario.
2. Construye las representaciones internas y calcula una distribución de probabilidad sobre el **primer token** de la respuesta.
3. Supongamos que el token elegido es `Los`.
4. Ahora el contexto ya no es solo el prompt y la pregunta, sino también:

```python
... ¿Cuáles son los requisitos de matrícula? Los
```

5. Con este nuevo contexto, el modelo vuelve a calcular probabilidades para el siguiente token.
6. Supongamos que elige `requisitos`.
7. El nuevo contexto pasa a ser:

```python
... ¿Cuáles son los requisitos de matrícula? Los requisitos
```

8. El ciclo se repite token por token:

```python
Los -> requisitos -> de -> matrícula -> son -> ...
```

9. La generación continúa hasta producir un token de finalización conceptual `[EOS]` o hasta alcanzar un límite de longitud impuesto por la llamada al modelo.

#### 4.10.3 Parámetros que controlan la generación

##### Temperature

La `temperature` reescala los logits antes de aplicar softmax:

```python
logits_ajustados = logits / temperature
```

Efectos prácticos:

- `temperature = 0`: comportamiento casi determinista; se favorece sistemáticamente el token más probable.
- `temperature = 1`: distribución original del modelo.
- `temperature > 1`: mayor dispersión y creatividad, pero también más variabilidad.

En el proyecto se usa:

```python
temperature=0
```

en `src/nodes/classifier.py:135-139`. Esta decisión es técnicamente correcta porque la clasificación de intención no debe ser creativa. Si una familia escribe dos veces el mismo mensaje, el sistema debe idealmente clasificarlo igual.

En FAQ se usa:

```python
temperature=0.3
```

en `src/nodes/faq.py:57-61`, lo que permite una ligera flexibilidad de redacción sin perder control institucional.

##### Top-p (nucleus sampling)

`top-p` selecciona el conjunto mínimo de tokens cuya probabilidad acumulada alcanza un umbral `p`. Luego el muestreo se hace solo dentro de ese subconjunto. Conceptualmente:

- si `p = 0.9`, se consideran los tokens más probables cuya suma acumulada alcance 90%.

Esto contrasta con `top-k`, que fija un número de candidatos. `top-p` suele adaptarse mejor a distribuciones variables porque el número de tokens considerados cambia según el contexto.

##### Max tokens

`max_tokens` controla la longitud máxima de salida. Aunque este parámetro no aparece explícitamente en los nodos visibles del repositorio, es un concepto operativo importante para WhatsApp y para el costo. En el proyecto sí aparece una restricción funcional concreta en `src/nodes/response_builder.py:4-27`, donde las respuestas de WhatsApp se recortan a `1600` caracteres. Esto refleja una preocupación real por eficiencia, canal y control del tamaño de respuesta.

#### 4.10.4 Conexión con Enfócate Más

El uso de `temperature=0` en clasificación beneficia directamente a las familias porque reduce arbitrariedad: la misma intención debe mapearse consistentemente a la misma ruta. Esto mejora estabilidad conversacional y disminuye errores de enrutamiento. En un contexto administrativo, previsibilidad y consistencia son más valiosas que creatividad.

## 5. Componentes neuronales internos

### 5.1 Neuronas artificiales

En términos matemáticos, una neurona artificial calcula una combinación lineal de entradas y luego aplica una función de activación:

```python
salida = f(w_1x_1 + w_2x_2 + ... + w_nx_n + b)
```

donde:

- `x_1, x_2, ..., x_n` son las entradas,
- `w_1, w_2, ..., w_n` son los pesos,
- `b` es el bias,
- `f` es una activación no lineal.

Dentro de GPT-4o-mini, una neurona no “entiende” una palabra por sí sola. Lo que hace es responder a ciertos patrones distribuidos en el espacio vectorial. Miles de neuronas en paralelo permiten detectar regularidades del lenguaje como:

- si un texto expresa una pregunta,
- si se habla de pagos o de autenticación,
- si un fragmento recuperado por RAG es semánticamente pertinente,
- si una secuencia sugiere que el siguiente token probable es `faq`, `pagos` o un fragmento de respuesta institucional.

La diferencia entre una neurona de una red simple y una neurona dentro de un Transformer es de contexto y escala. En una red pequeña, la neurona suele operar sobre variables fijas y poco estructuradas. En un Transformer, la neurona opera sobre representaciones contextuales altamente dinámicas, producidas por capas de atención y redes profundas. No procesa simplemente una característica aislada; procesa una representación ya cargada de contexto semántico.

### 5.2 Pesos (Weights)

Los **pesos** son coeficientes matemáticos que determinan cuánto influye cada entrada en la salida de una neurona. Si pensamos en una transformación lineal general:

```python
y = Wx + b
```

la matriz `W` contiene los pesos.

En GPT-4o-mini, estos pesos se obtuvieron durante el proceso de preentrenamiento a gran escala de OpenAI. Cada actualización del entrenamiento ajustó los pesos para mejorar la predicción del siguiente token. Como resultado, los pesos almacenan regularidades complejas del lenguaje: gramática, semántica, asociaciones, estilos de respuesta e incluso patrones administrativos generales.

Interpretación intuitiva:

- un peso alto en magnitud implica una influencia más fuerte sobre la salida,
- un peso bajo implica menor influencia,
- el signo del peso también importa porque puede reforzar o contrarrestar ciertas activaciones.

En el uso del proyecto, esos pesos ya están **fijos**. Cuando `classifier.py` o `faq.py` invocan:

```python
ChatOpenAI(model="gpt-4o-mini", ...)
```

no están reentrenando el modelo ni alterando los pesos. Solo están aprovechando un conjunto de parámetros ya aprendidos y consolidados.

OpenAI no publica en este proyecto cuántos parámetros exactos tiene GPT-4o-mini. Por tanto, la formulación académicamente correcta es: **GPT-4o-mini pertenece a la familia de modelos con escala de miles de millones de parámetros, pero la cifra exacta no está expuesta en el repositorio ni en la configuración visible del proyecto**.

### 5.3 Bias

El **bias** es un término aditivo que se suma antes de la función de activación:

```python
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
salida = f(z)
```

Su papel matemático es permitir que la transformación no esté obligada a pasar por el origen. Sin bias, muchas neuronas serían demasiado rígidas. Por ejemplo, si todas las entradas fueran cero, la salida también quedaría forzada a cero antes de la activación. Con bias, el modelo puede desplazar umbrales de activación y adaptar mejor su comportamiento.

En GPT-4o-mini, esto es importante porque una representación lingüística útil no depende solo de relaciones proporcionales entre entradas; también requiere ajustes finos en los puntos donde ciertas neuronas comienzan a activarse ante patrones semánticos particulares.

### 5.4 Capas del modelo

#### Capa de entrada

La capa de entrada recibe el texto del usuario. Sin embargo, el modelo no “ve” texto como una persona. Primero ocurre:

- tokenización,
- búsqueda del embedding de cada token,
- suma de señal posicional.

Conceptualmente:

```python
texto -> tokens -> embeddings -> embeddings + posición
```

Esto crea la primera representación procesable de la secuencia.

#### Capas ocultas: corazón del Transformer

En un modelo de la familia GPT moderna suele haber **muchas capas ocultas**, normalmente docenas de bloques Transformer. OpenAI no publica aquí el conteo exacto de GPT-4o-mini, por lo que la formulación correcta es hablar de una pila profunda de bloques de atención y FFN.

En cada capa ocurre lo siguiente:

```python
entrada de capa
-> self-attention / multi-head attention
-> residual + layer normalization
-> feed-forward network
-> residual + layer normalization
-> salida de capa
```

La representación se va refinando capa por capa. Académicamente, puede decirse que:

- una capa temprana tiende a capturar patrones más locales o sintácticos,
- capas intermedias integran contexto más amplio,
- capas profundas reflejan intención, función discursiva y estructura semántica más abstracta.

Por eso es razonable afirmar que una “capa 1” no entiende lo mismo que una “capa 12”. Una capa inicial puede distinguir signos, subpalabras o combinaciones léxicas inmediatas. Una capa más profunda puede ya representar que `¿Cuánto debo pagar este mes?` es una **consulta administrativa de deuda o pagos** y no una pregunta general.

#### Capa de salida

La capa final proyecta la representación contextual al vocabulario:

```python
logits = h_final W_vocab + b_vocab
```

Esos logits son puntajes no normalizados. Luego se aplica softmax:

```python
P(token_i) = exp(logit_i) / Σ_j exp(logit_j)
```

Esto produce una distribución de probabilidad sobre el siguiente token. El modelo selecciona uno y continúa la generación.

### 5.5 Funciones de activación — explicación profunda

#### ReLU (Rectified Linear Unit)

La función ReLU se define como:

```python
f(x) = max(0, x)
```

Gráficamente, deja pasar valores positivos y anula los negativos. Fue revolucionaria porque simplificó el entrenamiento de redes profundas y redujo el problema de saturación característico de funciones como sigmoid o tanh.

Su limitación clásica es el problema de **dying ReLU**: si una neurona entra persistentemente en la zona negativa, puede quedarse devolviendo cero y dejar de aprender eficazmente.

#### GELU (Gaussian Error Linear Unit)

La GELU puede aproximarse como:

```python
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

La idea intuitiva es que GELU no “corta” bruscamente todos los negativos como ReLU. Suaviza la transición y conserva mejor matices en activaciones pequeñas o intermedias. Esto es especialmente valioso en modelos de lenguaje, donde muchos patrones útiles no son binarios sino graduales.

En Transformers modernos, GELU se usa típicamente dentro de las **Feed Forward Networks**. Aunque el repositorio no expone las tripas internas de GPT-4o-mini, académicamente es correcto afirmar que los Transformers modernos suelen utilizar activaciones tipo GELU en sus FFN por su mejor desempeño en representaciones lingüísticas profundas.

La diferencia práctica frente a ReLU en lenguaje es importante: el procesamiento lingüístico se beneficia de transiciones suaves y de preservar señal contextual sutil. GELU es más adecuada para eso.

#### Softmax

La softmax se define como:

```python
softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```

Se usa en dos lugares conceptuales clave de GPT-4o-mini:

- en la atención, para convertir scores en pesos de atención,
- en la capa de salida, para convertir logits en probabilidades de siguiente token.

Produce una distribución de probabilidad porque todos los valores quedan entre 0 y 1 y su suma total es 1.

Ejemplo numérico simple. Si tres logits posibles son:

```python
[2.0, 1.0, 0.1]
```

entonces:

```python
e^2.0 ≈ 7.39
e^1.0 ≈ 2.72
e^0.1 ≈ 1.11
suma ≈ 11.22
```

Las probabilidades aproximadas serían:

```python
[0.659, 0.242, 0.099]
```

Esto significa que el primer token es claramente el más probable, pero no el único posible.

### 5.6 Relación de estos componentes con GPT-4o-mini en el proyecto

Todos estos componentes trabajan juntos cuando el proyecto ejecuta el modelo.

En `classifier.py`, GPT-4o-mini recibe el mensaje del usuario, lo tokeniza, construye embeddings, aplica múltiples capas de atención y FFN, y finalmente genera un token de salida que debe corresponder a una de cinco etiquetas válidas.

En `faq.py`, ocurre algo similar pero con una tarea más compleja: el contexto recuperado desde ChromaDB se concatena con la pregunta del usuario y el modelo debe generar una respuesta natural en español, útil para familias y estudiantes.

El proyecto no necesita modificar manualmente neuronas, pesos, biases o activaciones porque esos componentes ya forman parte del modelo entrenado. La tarea ingenieril del equipo consistió en usar correctamente ese modelo dentro de una arquitectura orientada al problema real de Enfócate Más.

## 6. ¿Cómo fue entrenado GPT-4o-mini?

### 6.1 Pretraining

El pretraining de un modelo GPT consiste en exponerlo a grandes cantidades de texto y enseñarle a predecir el siguiente token. Por ejemplo, dada la secuencia:

```python
La institución atiende a miles de
```

el modelo aprende a asignar alta probabilidad a palabras como `familias`. Repitiendo este proceso a gran escala, el modelo internaliza patrones de gramática, semántica, estilo, inferencia y asociación conceptual.

### 6.2 Objetivo del entrenamiento

El objetivo central es la **predicción autoregresiva del siguiente token**. Aunque parezca una tarea simple, a gran escala produce un sistema capaz de:

- resumir,
- clasificar,
- responder,
- traducir,
- seguir instrucciones,
- mantener coherencia conversacional.

### 6.3 Fine-Tuning

El **fine-tuning** es una fase de ajuste posterior al pretraining, donde el modelo se especializa más en determinadas conductas o dominios. En modelos comerciales como GPT-4o-mini, el detalle exacto del fine-tuning interno no aparece en el repositorio, pero el concepto es relevante para entender por qué el modelo puede seguir instrucciones complejas con gran eficacia.

### 6.4 RLHF

El **Reinforcement Learning from Human Feedback** mejora la alineación del modelo con expectativas humanas. En lugar de optimizar solo precisión estadística, también busca utilidad, claridad, seguridad y adecuación. Esto explica por qué GPT-4o-mini puede producir respuestas más conversacionales y menos mecánicas que un sistema basado solo en reglas.

### 6.5 Cómo esto permite entender lenguaje natural

Gracias a pretraining, fine-tuning y alineación mediante retroalimentación humana, GPT-4o-mini puede hacer en este proyecto lo siguiente:

- entender que “olvidé mi clave” pertenece a autenticación,
- diferenciar “¿cuánto debo?” de “¿cuáles son los horarios?”
- responder preguntas institucionales en español natural,
- incorporar información recuperada desde documentos para dar una respuesta más precisa.

## 7. ¿Por qué GPT-4o-mini fue elegido para este proyecto?

### 7.1 Qué otros modelos fueron evaluados

La evidencia real está en `experiments/benchmark_rag.py:174-231`. Allí se comparan estas configuraciones:

- `gpt-4o`
- `gpt-4o-mini`
- `gpt-3.5-turbo`
- `llama3-local`

y además se comparan embeddings `openai` y `sentence_transformers`, junto con tamaños de chunk de `200`, `500` y `1000`.

### 7.2 Evidencia del repositorio

La selección del modelo está documentada explícitamente en:

- `docs/READMEE.md:120-125`
- `src/nodes/classifier.py:135-139`
- `src/nodes/faq.py:57-61`

Es decir, GPT-4o-mini no solo fue comparado: también fue el modelo realmente integrado en los nodos productivos del chatbot.

### 7.3 Ventajas técnicas de GPT-4o-mini para este caso

- costo drásticamente menor que GPT-4o,
- latencia menor que GPT-4o,
- suficiente calidad para tareas conversacionales administrativas,
- integración directa con `ChatOpenAI`,
- compatibilidad natural con LangGraph y RAG,
- adecuación a un servicio escalable para miles de familias.

### 7.4 Relación costo-beneficio

El benchmark documental muestra:

| Modelo | F1 | Latencia | Costo |
| --- | --- | --- | --- |
| GPT-4o + OpenAI Emb | 0.9316 | 1.74s | $2.50 / 1K tokens |
| GPT-4o-mini + OpenAI Emb | 0.6795 | 0.69s | $0.15 / 1K tokens |
| GPT-3.5-turbo + OpenAI Emb | 1.0000* | 1.00s | $0.50 / 1K tokens |

El README advierte que el F1 de `1.0000` para GPT-3.5-turbo puede indicar sobreajuste (`docs/READMEE.md:118`). Por tanto, la decisión no depende solo del mayor número aparente, sino del equilibrio entre **rendimiento útil, estabilidad, costo y escalabilidad**.

### 7.5 Relación calidad-latencia

Para un chatbot administrativo, la latencia es crítica. Un modelo excelente pero demasiado lento rompe la experiencia por WhatsApp. GPT-4o-mini aparece documentado como suficientemente rápido para un promedio cercano a `0.7s` en el benchmark del README.

### 7.6 Escalabilidad

El proyecto fue pensado para más de `3.000` familias (`docs/READMEE.md:3`). En ese contexto, la diferencia entre `$2.50` y `$0.15` por `1K tokens` es estratégicamente muy importante. Escalar GPT-4o-mini es mucho más viable económicamente.

### 7.7 Integración con LangGraph

LangGraph organiza el flujo del chatbot en nodos (`src/graph.py:41-68`). GPT-4o-mini se integra en dos de esos nodos de forma limpia:

- en `intent_classifier`
- en `faq_node`

Esto permite desacoplar la lógica conversacional del transporte, el logging y las consultas a datos.

### 7.8 Integración con RAG

GPT-4o-mini no responde solo desde conocimiento general. En FAQ lo hace después de recibir contexto institucional recuperado desde ChromaDB (`src/nodes/faq.py:44-71`). Esto mejora precisión y reduce el riesgo de alucinación.

### 7.9 Integración con WhatsApp

El gateway implementado con FastAPI recibe mensajes de Twilio por `/webhook/twilio` y luego envía la respuesta generada por el grafo (`src/integrations/gateway.py:101-160`). GPT-4o-mini, por tanto, no es un módulo aislado: es el motor lingüístico dentro de un sistema orientado al canal real que usan las familias.

### Tabla comparativa entre modelos evaluados

| Configuración | Evidencia | Fortalezas | Debilidades | Lectura académica |
| --- | --- | --- | --- | --- |
| GPT-4o + chunk500 | `benchmark_results.csv` | alta calidad, fuerte desempeño | alto costo, mayor latencia | técnicamente sólido, menos viable para escala masiva |
| GPT-4o-mini + chunk500 | `benchmark_results.csv`, nodos reales | bajo costo, buena velocidad, usado en producción | menor calidad máxima que GPT-4o | mejor equilibrio práctico |
| GPT-4o-mini + chunk200 | `benchmark_results.csv` | latencia aún menor | posible pérdida de contexto | útil si se prioriza rapidez |
| GPT-4o-mini + chunk1000 | `benchmark_results.csv` | más contexto | latencia más alta y riesgo de diluir relevancia | menos conveniente según README |
| GPT-3.5-turbo + chunk500 | `benchmark_results.csv`, README | muy buen valor métrico documental | posible sobreajuste según README | no se tomó como elección final |
| Llama3-local + SentTransf | `benchmark_results.csv` | costo de API nulo | F1 muy bajo | no adecuado para este contexto |

## 8. Cómo GPT-4o-mini trabaja dentro del proyecto

### 8.1 Clasificación de intención

El clasificador está en `src/nodes/classifier.py`. Primero aplica ciertas reglas rápidas. Si esas reglas no resuelven la intención, invoca GPT-4o-mini con un prompt del sistema que exige devolver solo una palabra clave.

Código real:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

result = llm.invoke(
    [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=last_human.content),
    ]
)
```

Esto permite que GPT-4o-mini identifique:

- `faq`: preguntas sobre institución, horarios, normas, servicios
- `auth`: inicio de sesión o credenciales
- `pagos`: pagos, deudas, asistencia o actualización de datos
- `matricula`: inscripciones y renovaciones
- `soporte`: quejas, reclamos o agente humano

### 8.2 Generación de respuestas

Para FAQ, GPT-4o-mini recibe un prompt que le indica responder en español, de forma clara y breve, basándose en el contexto institucional recuperado (`src/nodes/faq.py:11-19`). Luego recibe el texto concatenado del contexto y la pregunta del usuario.

### 8.3 Uso dentro del sistema RAG

El flujo real de FAQ es:

```python
Usuario pregunta
-> OpenAIEmbeddings
-> Chroma similarity_search(query, k=5)
-> Contexto institucional recuperado
-> GPT-4o-mini
-> Respuesta final
```

### 8.4 Diagrama textual completo del pipeline del sistema

```python
┌────────────────────────────┐
│          USUARIO           │
│ Familias / Estudiantes /   │
│ Personal administrativo    │
└──────────────┬─────────────┘
               │ mensaje natural
               ▼
┌────────────────────────────┐
│  CANAL DE ENTRADA          │
│  WhatsApp (Twilio) / Web   │
└──────────────┬─────────────┘
               │ HTTP / webhook
               ▼
┌────────────────────────────┐
│  GATEWAY FASTAPI           │
│  src/integrations/         │
│  gateway.py                │
└──────────────┬─────────────┘
               │ ChatState inicial
               ▼
┌────────────────────────────┐
│  LANGGRAPH                 │
│  src/graph.py              │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ intent_classifier          │
│ GPT-4o-mini + reglas       │
│ src/nodes/classifier.py    │
└───────┬─────────┬──────────┴─────────┬─────────────┐
        │         │                    │             │
        ▼         ▼                    ▼             ▼
   ┌────────┐ ┌────────┐         ┌────────┐     ┌────────┐
   │ FAQ    │ │ AUTH   │         │ PAGOS  │     │SOPORTE │
   │ faq.py │ │auth.py │         │pagos.py│     │soporte │
   └───┬────┘ └────┬───┘         └────┬───┘     └────┬───┘
       │           │                    │              │
       │ si FAQ    │ lógica auth        │ lógica CSV   │ escalación
       ▼           ▼                    ▼              ▼
┌────────────────────────────┐
│ response_builder           │
│ src/nodes/response_builder │
└──────────────┬─────────────┘
               │ response_text
               ▼
┌────────────────────────────┐
│ logger_node                │
│ src/nodes/logger.py        │
└──────────────┬─────────────┘
               │ logs JSONL
               ▼
┌────────────────────────────┐
│ RESPUESTA FINAL            │
│ WhatsApp / JSON web        │
└────────────────────────────┘
```

### 8.5 Explicación del pipeline nodo por nodo

#### Usuario

- Qué entra: una necesidad real expresada como lenguaje natural.
- Qué hace GPT-4o-mini ahí: todavía no interviene.
- Qué sale: un mensaje como `¿Cuánto debo pagar este mes?` o `¿Cuáles son los requisitos de matrícula?`.
- Archivo relacionado: no aplica, es el origen humano del flujo.

#### Canal de entrada

- Qué entra: mensaje de WhatsApp o mensaje del widget web.
- Qué hace GPT-4o-mini ahí: todavía no procesa el contenido.
- Qué sale: el mensaje llega al gateway HTTP.
- Archivo: `src/integrations/gateway.py`.

#### Gateway FastAPI

- Qué entra: `From` y `Body` en Twilio o `session_id`, `message`, `user_id` en `/chat/web`.
- Qué hace GPT-4o-mini ahí: no se ejecuta en esta capa.
- Qué sale: un `ChatState` inicial o actualizado para LangGraph.
- Archivo: `src/integrations/gateway.py:28-84`, `101-179`.

#### LangGraph

- Qué entra: estado conversacional completo.
- Qué hace GPT-4o-mini ahí: el grafo no es el modelo; el grafo decide en qué nodo se usará.
- Qué sale: enrutamiento y ejecución secuencial de nodos.
- Archivo: `src/graph.py:35-68`.

#### `intent_classifier`

- Qué entra: `last_human.content`, `auth_status`, `context`.
- Qué hace GPT-4o-mini: clasifica la intención cuando las reglas no bastan.
- Qué sale: una etiqueta entre `faq`, `auth`, `pagos`, `matricula`, `soporte`.
- Archivo: `src/nodes/classifier.py`.

#### `faq_node`

- Qué entra: la pregunta del usuario y el contexto documental recuperado.
- Qué hace GPT-4o-mini: genera la respuesta final en español apoyándose en RAG.
- Qué sale: un `AIMessage` contextualizado.
- Archivo: `src/nodes/faq.py`.

#### `auth_node`

- Qué entra: teléfono, correo, contraseña o flujo de recuperación.
- Qué hace GPT-4o-mini: aquí su papel es indirecto; el modelo ayudó antes a enrutar el mensaje como `auth`.
- Qué sale: estado de autenticación y mensajes de seguimiento.
- Archivo: `src/nodes/auth.py`.

#### `pagos_node`

- Qué entra: consulta autenticada sobre pagos, asistencia o datos personales.
- Qué hace GPT-4o-mini: no es el motor interno del nodo; el valor del modelo aquí fue interpretar correctamente la intención y llevar al usuario al flujo adecuado.
- Qué sale: respuesta administrativa basada en el CSV institucional.
- Archivo: `src/nodes/pagos.py`.

#### `soporte_node`

- Qué entra: quejas, reclamos o solicitud de agente humano.
- Qué hace GPT-4o-mini: participa indirectamente por medio de la clasificación inicial.
- Qué sale: mensaje de soporte y posible escalación.
- Archivo: `src/nodes/soporte.py`.

#### `response_builder`

- Qué entra: el último `AIMessage` del estado.
- Qué hace GPT-4o-mini: ya no genera; aquí solo se empaqueta la salida.
- Qué sale: `response_text` final.
- Archivo: `src/nodes/response_builder.py`.

#### `logger_node`

- Qué entra: estado completo de la interacción.
- Qué hace GPT-4o-mini: no participa directamente, pero aquí queda trazado el resultado de su trabajo.
- Qué sale: una línea en `logs/interactions.jsonl`.
- Archivo: `src/nodes/logger.py`.

### 8.6 Segundo diagrama: pipeline RAG exclusivo

```python
Pregunta del usuario
-> Embedding de la consulta
-> Búsqueda semántica en ChromaDB
-> Recuperación de top-k chunks
-> Construcción del prompt con contexto institucional
-> GPT-4o-mini
-> Respuesta final
```

Versión aterrizada al proyecto:

```python
Pregunta usuario
-> faq_node (src/nodes/faq.py)
-> OpenAIEmbeddings(api_key=...)
-> Chroma(collection_name="faq_docs", persist_directory="data/embeddings")
-> similarity_search(query, k=5)
-> join de chunks con separador ---
-> SystemMessage(FAQ_SYSTEM_PROMPT) + HumanMessage(prompt_with_context)
-> ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
-> AIMessage de respuesta
```

### 8.7 Papel exacto de GPT-4o-mini en cada etapa

El papel de GPT-4o-mini debe describirse con precisión:

- no es el servidor,
- no es la base vectorial,
- no es la base CSV,
- no es el orquestador del grafo,
- sí es el **motor neuronal lingüístico** que decide intención y redacta respuestas FAQ contextualizadas.

Esa distinción es importante para demostrar dominio del sistema completo y del modelo dentro del sistema.

## 9. Relación entre GPT-4o-mini y RAG

### 9.1 Qué es RAG

RAG significa **Retrieval-Augmented Generation**. Es una estrategia donde un modelo generativo no responde solo desde su memoria paramétrica, sino que primero recibe información recuperada dinámicamente desde una fuente documental.

### 9.2 Qué problema resuelve

RAG resuelve el problema de la **actualización y precisión contextual**. En una institución educativa, horarios, requisitos, costos, servicios y normativas pueden cambiar. No es suficiente depender de conocimiento general del modelo.

### 9.3 Qué limitaciones tendría GPT-4o-mini sin RAG

Sin RAG, GPT-4o-mini podría:

- responder de forma genérica,
- carecer de datos institucionales exactos,
- inventar detalles no verificados,
- omitir información local como contactos o requisitos particulares.

### 9.4 Cómo mejora la precisión

El RAG implementado en el proyecto hace que GPT-4o-mini reciba información institucional recuperada desde `faq_docs`. Esto vuelve sus respuestas más específicas y alineadas con Enfócate Más.

### 9.5 Cómo reduce alucinaciones

El README lo expresa claramente en `docs/READMEE.md:57-61`: con RAG el sistema responde con documentos verificados y conocimiento personalizado de Enfócate Más. En otras palabras, no elimina por completo el riesgo de error, pero reduce significativamente la tendencia a inventar respuestas desconectadas del contexto institucional.

### Ejemplo aplicado al proyecto

Una pregunta como:

```python
¿Qué documentos necesito para inscribir a mi hijo?
```

sin RAG podría recibir una respuesta genérica sobre procesos de matrícula. Con RAG, el sistema busca los fragmentos más relevantes de los documentos institucionales y luego GPT-4o-mini redacta una respuesta con base en ese material.

## 10. Métricas obtenidas por GPT-4o-mini en el proyecto

### 10.1 Fuentes reales de métricas

El proyecto ofrece tres fuentes para analizar el desempeño asociado a GPT-4o-mini:

- `docs/READMEE.md:620-631`, resumen del modelo elegido,
- `experiments/results/benchmark_results.csv`, benchmark extendido por configuración,
- `metrics_report.json`, métricas operativas del sistema desplegado localmente.

Estas fuentes no son idénticas entre sí, pero juntas permiten una lectura técnica amplia del comportamiento del modelo dentro del chatbot.

### 10.2 Métricas de clasificación para `classifier.py`

#### Accuracy (Exactitud)

**Definición técnica**: proporción de predicciones correctas sobre el total.

**Fórmula**:

```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

En multiclase, la idea general es la misma: cuántas etiquetas fueron correctas entre todas las emitidas.

**Interpretación**:

- valor alto: el clasificador enruta bien la mayoría de mensajes,
- valor bajo: el modelo envía muchas consultas a nodos equivocados.

**Resultado real del proyecto**:

```python
Accuracy = 84.6%
```

Evidencia: `docs/READMEE.md:625`.

**Significado para Enfócate Más**:

De cada 100 mensajes similares al conjunto evaluado, aproximadamente 84 o 85 terminan en el flujo correcto. En un chatbot administrativo esto es central porque la clasificación inicial condiciona toda la experiencia del usuario.

#### Precision (Precisión)

**Definición técnica**: mide la pureza de las predicciones positivas.

**Fórmula**:

```python
Precision = TP / (TP + FP)
```

**Interpretación**:

- alta precision: cuando el modelo dice que una consulta es de cierta clase, suele acertar,
- baja precision: hay falsas alarmas frecuentes.

**Resultado real del proyecto**:

```python
Precision = 91.7%
```

Evidencia: `docs/READMEE.md:626`.

**Interpretación contextual**:

Cuando GPT-4o-mini decide que un mensaje corresponde a `pagos`, `faq` o `auth`, esa decisión es confiable la mayor parte del tiempo. En términos operativos, eso reduce fricción y evita enrutar innecesariamente a soporte o a un flujo incorrecto.

#### Recall (Exhaustividad)

**Definición técnica**: proporción de casos positivos reales que el modelo sí logra detectar.

**Fórmula**:

```python
Recall = TP / (TP + FN)
```

**Interpretación**:

- alto recall: el sistema pierde pocos mensajes de una clase real,
- bajo recall: deja escapar muchos casos que debían ser detectados.

**Resultado real del proyecto**:

```python
Recall = 91.7%
```

Evidencia: `docs/READMEE.md:627`.

**Significado para Enfócate Más**:

El modelo recupera una porción alta de las consultas reales de cada intención. Esto es importante porque si el recall fuera bajo, las familias podrían formular consultas claras de pagos o autenticación y aun así terminar fuera del flujo correspondiente.

#### F1-Score

**Definición técnica**: media armónica entre precision y recall.

**Fórmula**:

```python
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretación**:

- alto F1: equilibrio sólido entre acertar y no omitir,
- bajo F1: el modelo falla en alguna de las dos dimensiones o en ambas.

**Resultado real del proyecto**:

```python
F1 = 0.917
```

Evidencia: `docs/READMEE.md:628`.

**Por qué es mejor que accuracy sola**:

Si una clase aparece mucho más que otra, accuracy puede inflarse artificialmente. El F1 obliga a evaluar simultáneamente cuántas veces el modelo acierta cuando predice y cuántas veces logra detectar los casos verdaderos.

### 10.3 Métricas de generación de texto para `faq.py`

#### BLEU Score

**Definición técnica**: BLEU evalúa precisión de n-gramas entre la respuesta generada y una referencia, con penalización por brevedad.

**Fórmula conceptual**:

```python
BLEU = BP × exp(Σ_n w_n log p_n)
```

**Interpretación**:

- cercano a 1: gran coincidencia superficial con la referencia,
- cercano a 0: poca coincidencia textual.

**Resultado real del proyecto** para `GPT-4o-mini + chunk500`:

```python
BLEU = 0.5707
```

Evidencia: `experiments/results/benchmark_results.csv:4`.

**Ejemplo aplicado**:

Referencia:

```python
Para matricularse necesita documento de identidad y formulario de inscripción.
```

Generación:

```python
Los requisitos de matrícula son documento de identidad y formulario.
```

El BLEU no sería perfecto porque cambia redacción y estructura, pero sí sería moderadamente alto porque comparte contenido esencial y varios n-gramas relevantes.

#### ROUGE Score

**Definición técnica**: familia de métricas orientadas a cobertura textual entre generación y referencia.

**ROUGE-1**: unigramas coincidentes.

**ROUGE-2**: bigramas coincidentes.

**ROUGE-L**: subsecuencia común más larga.

**Fórmulas básicas de ROUGE-1**:

```python
Precision_ROUGE-1 = unigramas_coincidentes / unigramas_generados
Recall_ROUGE-1 = unigramas_coincidentes / unigramas_referencia
F1_ROUGE-1 = 2PR / (P + R)
```

**Resultados reales del proyecto** para `GPT-4o-mini + chunk500`:

```python
ROUGE-1 = 0.6421
ROUGE-2 = 0.5513
ROUGE-L = 0.6267
```

Evidencia: `benchmark_results.csv:4`.

**Interpretación**:

- `ROUGE-1` indica superposición léxica bastante aceptable,
- `ROUGE-2` muestra coincidencia razonable de pequeñas secuencias,
- `ROUGE-L` sugiere buena conservación de estructura informativa.

#### Cosine Similarity

**Definición técnica**: mide cercanía angular entre dos vectores.

**Fórmula**:

```python
cos(θ) = (A · B) / (||A|| × ||B||)
```

**Interpretación**:

- cercana a 1: significado muy similar,
- cercana a 0: baja relación semántica.

**Resultado real del proyecto** para `GPT-4o-mini + chunk500`:

```python
Cosine Similarity = 0.6601
```

Evidencia: `benchmark_results.csv:4`.

**Por qué es clave en este proyecto**:

La similitud coseno es esencial para RAG porque la recuperación semántica se basa en cercanía vectorial. Si una pregunta del usuario tiene coseno `0.87` con un chunk institucional, eso sugiere gran afinidad semántica y justifica que ChromaDB lo devuelva entre los fragmentos más relevantes.

### 10.4 Desglose por intención en el benchmark

El benchmark persistido incluye métricas por clase para `GPT-4o-mini + chunk500`:

```python
F1 FAQ = 0.9231
F1 Auth = 1.0
F1 Pagos = 1.0
F1 Matrícula = 0.8571
F1 Soporte = 1.0
```

Lectura técnica:

- `auth`, `pagos` y `soporte` aparecen especialmente fuertes,
- `faq` también es alta,
- `matricula` es la más débil entre las visibles.

Esto sugiere que las consultas administrativas muy marcadas son más fáciles de enrutar que las de matrícula, que tal vez comparten vocabulario con FAQ o admisiones.

### 10.5 Métricas operativas: latencia

**Resultado del README**:

```python
Latencia promedio = 0.694 s
Latencia P95 = 0.946 s
```

Evidencia: `docs/READMEE.md:629-630`.

**Comparación con GPT-4o**:

```python
GPT-4o = 1.74 s
GPT-4o-mini = 0.69 s
```

Esto es muy importante para WhatsApp: un modelo que tarde casi el triple deteriora la sensación de inmediatez.

**Métricas operativas del sistema completo** desde `metrics_report.json`:

| Caso | avg_ms | p95_ms | Lectura técnica |
| --- | --- | --- | --- |
| hola | 7676 | 3776 | FAQ con mayor costo de procesamiento |
| horarios | 4834 | 3956 | consulta informativa con recuperación contextual |
| programas | 6201 | 4860 | respuesta documental relativamente pesada |
| +57 3874290099 | 59 | 59 | flujo casi inmediato de validación/login |
| +57 300 999 9999 | 60 | 61 | validación rápida de teléfono no registrado |
| xkdzpq | 1888 | 3288 | entrada ambigua o ruido |

Estos tiempos no miden únicamente GPT-4o-mini aislado, sino el sistema completo. Aun así, son valiosos para explicar que la experiencia final depende de la combinación entre recuperación de contexto, lógica de nodos y generación.

### 10.6 Costo por token

**Resultado real**:

```python
$0.15 por 1.000 tokens
```

Evidencia: `docs/READMEE.md:631` y `benchmark_rag.py:233-238`.

**Interpretación**:

El costo por token es decisivo porque el proyecto está pensado para más de 3.000 familias. GPT-4o-mini es mucho más viable que GPT-4o a escala institucional. El README incluso proyecta:

```python
$50–100 USD/mes para 3.000 familias
```

Evidencia: `docs/READMEE.md:124`.

La forma académicamente correcta de defender este punto es decir que el bajo costo hace operacionalmente posible ofrecer un servicio frecuente y masivo, especialmente cuando gran parte del tráfico se concentra en FAQ y autenticación.

### 10.7 Síntesis técnica de las métricas

Las métricas del proyecto muestran que GPT-4o-mini no fue elegido por intuición, sino por una combinación concreta de factores:

- buen desempeño de clasificación,
- capacidad de generación textual razonable,
- compatibilidad semántica con RAG,
- latencia inferior a GPT-4o,
- costo sustancialmente más bajo,
- viabilidad para el contexto institucional de Enfócate Más.

## 11. Ventajas y limitaciones de GPT-4o-mini

### Ventajas

| Ventaja | Impacto en el proyecto |
| --- | --- |
| Bajo costo por token | facilita escalabilidad para más de 3.000 familias |
| Latencia moderada | hace viable la atención por WhatsApp y web |
| Comprensión del español | mejora la interacción con familias y estudiantes |
| Capacidad generativa | produce respuestas naturales y no solo etiquetas |
| Integración con prompts | permite especializar comportamiento institucional |
| Compatibilidad con RAG | usa contexto documental real para responder mejor |
| Integración con LangGraph | encaja en un flujo modular y mantenible |

### Limitaciones

| Limitación | Impacto en el proyecto |
| --- | --- |
| Dependencia de contexto recuperado | si RAG falla, la respuesta puede perder precisión |
| Desempeño inferior al máximo de modelos premium | puede haber tareas donde GPT-4o rinda mejor |
| Dependencia de infraestructura externa | requiere API de OpenAI y conectividad |
| Riesgo de respuesta genérica si el prompt es débil | exige buen diseño de prompts |
| Sensibilidad a documentación desactualizada | puede responder con base en documentos obsoletos |

## 12. Impacto para los involucrados

### Familias

GPT-4o-mini beneficia a las familias porque les permite interactuar con la institución en lenguaje natural, por WhatsApp o web, sin depender de terminología técnica ni de un formulario rígido. Esto reduce fricción de acceso y tiempo de espera.

### Estudiantes

El modelo mejora el acceso a información de asistencia, matrículas y datos institucionales, lo que favorece continuidad, organización y orientación académica.

### Personal administrativo

Reduce el volumen de preguntas repetitivas, especialmente en FAQ y autenticación. Los logs muestran `faq: 1329` y `auth: 1255` interacciones en el historial analizado del proyecto, lo cual sugiere una fuerte utilidad operativa para descomprimir trabajo manual.

### Institución

La institución gana una capa de servicio 24/7, trazable y multicanal. Además, el uso de un modelo como GPT-4o-mini hace económicamente más razonable sostener el servicio a gran escala.

### Directivos

Los directivos pueden beneficiarse de una atención más consistente y de la posibilidad de analizar patrones de consulta a partir de logs estructurados, algo habilitado por `logger_node` y `logs/interactions.jsonl`.

## 13. Preguntas difíciles sobre GPT-4o-mini

| Pregunta | Respuesta técnica | Respuesta sencilla | Evidencia del proyecto |
| --- | --- | --- | --- |
| 1. ¿Qué es GPT-4o-mini? | Es un LLM generativo de OpenAI basado en Transformer, integrado en el proyecto vía `ChatOpenAI`. | Es el modelo de IA que entiende preguntas y genera respuestas. | `classifier.py`, `faq.py` |
| 2. ¿Qué significa GPT? | Generative Pre-trained Transformer. | Genera texto y fue entrenado previamente con arquitectura Transformer. | Conocimiento del modelo + uso en proyecto |
| 3. ¿Qué tarea hace en este proyecto? | Clasifica intención y genera respuestas FAQ contextualizadas. | Decide qué quiere el usuario y responde preguntas. | `classifier.py`, `faq.py` |
| 4. ¿Por qué se considera IA? | Porque modela lenguaje con redes neuronales profundas y atención. | Porque puede comprender y responder lenguaje humano. | Naturaleza del modelo + integración real |
| 5. ¿Por qué es una red neuronal artificial? | Su arquitectura es un Transformer profundo con capas neuronales y pesos entrenados. | Porque aprende patrones del lenguaje con muchas capas internas. | Base teórica GPT + uso del modelo |
| 6. ¿Qué arquitectura usa? | Transformer autoregresivo tipo GPT. | Una arquitectura moderna para entender contexto y generar texto. | Explicación académica defendible |
| 7. ¿Qué es un Transformer? | Arquitectura neuronal basada en atención para procesar secuencias. | Un diseño que permite al modelo entender relaciones entre palabras. | Marco teórico del modelo |
| 8. ¿Qué es self-attention? | Mecanismo que pondera qué tokens del contexto son relevantes entre sí. | Es la forma en que el modelo decide a qué palabras prestar más atención. | Explicación técnica aplicada |
| 9. ¿Qué es multi-head attention? | Atención paralela desde varias perspectivas representacionales. | El modelo mira el texto de varias maneras al mismo tiempo. | Marco teórico |
| 10. ¿Qué es un token? | Unidad mínima procesable por el modelo, a menudo subpalabra. | Es un pedazo de texto que el modelo puede leer. | Sección de tokenización |
| 11. ¿Qué es un embedding? | Vector denso que representa significado lingüístico. | Es una forma de convertir palabras en números con sentido. | `faq.py`, `ingest_documents.py`, README |
| 12. ¿Cómo genera respuestas? | Prediciendo el siguiente token de manera autoregresiva hasta completar la secuencia. | Va construyendo la respuesta palabra por palabra. | Naturaleza GPT + `faq.py` |
| 13. ¿Cómo clasifica intenciones? | Mediante prompt del sistema controlado y salida restringida a cinco etiquetas. | Lee el mensaje y lo ubica en una categoría. | `classifier.py:12-26`, `141-151` |
| 14. ¿Qué intenciones reconoce? | `faq`, `auth`, `pagos`, `matricula`, `soporte`. | Preguntas frecuentes, autenticación, pagos, matrícula y soporte. | `classifier.py:150-151` |
| 15. ¿Qué temperatura usa para clasificar? | `temperature=0`. | Se configuró para que clasifique de forma más estable. | `classifier.py:135-139` |
| 16. ¿Qué temperatura usa para FAQ? | `temperature=0.3`. | Se deja un poco más de flexibilidad al redactar respuestas. | `faq.py:57-61` |
| 17. ¿Por qué no usar solo reglas? | Porque el lenguaje natural es variable y ambiguo. | Porque la gente no pregunta siempre igual. | README RAG, clasificador híbrido |
| 18. ¿Qué ventaja aporta frente a un menú fijo? | Mayor flexibilidad lingüística y naturalidad conversacional. | El usuario puede escribir como habla normalmente. | README + gateway + classifier |
| 19. ¿Por qué se eligió GPT-4o-mini y no GPT-4o? | Mejor balance costo-latencia para volumen alto. | Porque es mucho más barato y suficientemente rápido. | `READMEE.md:120-125` |
| 20. ¿Por qué no se eligió GPT-3.5-turbo? | README advierte posible sobreajuste pese a su F1 alto. | Porque un número alto no garantiza la mejor decisión práctica. | `READMEE.md:118` |
| 21. ¿Qué otros modelos se evaluaron? | GPT-4o, GPT-3.5-turbo y Llama3-local, además de variantes de embeddings y chunks. | Se compararon varias opciones antes de decidir. | `benchmark_rag.py:174-231` |
| 22. ¿Qué papel cumple RAG? | Aportar contexto institucional recuperado antes de generar la respuesta. | Le da al modelo información real de la institución. | `faq.py`, README |
| 23. ¿Qué limitación tendría GPT-4o-mini sin RAG? | Mayor riesgo de respuestas genéricas o alucinadas. | Podría responder de forma correcta en general, pero no específica para la institución. | README RAG |
| 24. ¿Qué rol cumple ChromaDB? | Vector store para recuperación semántica por similitud. | Guarda los documentos de manera que se puedan buscar por significado. | `faq.py:45-50`, `ingest_documents.py` |
| 25. ¿Qué rol cumplen OpenAI Embeddings? | Vectorizar texto documental y consulta del usuario. | Convertir texto en números comparables semánticamente. | `faq.py:44`, `ingest_documents.py:73-80` |
| 26. ¿Qué significa que sea un modelo fundacional? | Que es un modelo general reutilizable sobre múltiples tareas. | Es una base grande ya preparada para muchos usos. | Marco teórico + uso del proyecto |
| 27. ¿Qué significa que sea generativo? | Que produce texto nuevo, no solo etiquetas. | Puede redactar respuestas completas. | `faq.py:67-76` |
| 28. ¿Cómo se integra con LangGraph? | Se encapsula dentro de nodos del grafo conversacional. | El sistema lo llama en partes específicas del flujo. | `graph.py`, `classifier.py`, `faq.py` |
| 29. ¿Cómo se integra con WhatsApp? | A través del gateway FastAPI que recibe webhook y envía respuestas vía Twilio. | El usuario escribe por WhatsApp y el sistema responde usando el modelo. | `gateway.py` |
| 30. ¿Cómo se integra con web? | Mediante POST `/chat/web`. | También puede funcionar en un portal web. | `gateway.py:163-179` |
| 31. ¿Cuál es la evidencia de que realmente se usa GPT-4o-mini? | El parámetro `model="gpt-4o-mini"` aparece en clasificación y FAQ. | Está escrito explícitamente en el código. | `classifier.py`, `faq.py` |
| 32. ¿Qué precisión logró? | README reporta precisión 91.7%; benchmark CSV reporta 0.9643 en una corrida MOCK. | Obtuvo valores altos en evaluación, aunque hay dos fuentes distintas. | `READMEE.md`, `benchmark_results.csv` |
| 33. ¿Qué F1 logró? | README reporta 0.917; benchmark CSV reporta 0.954 para ciertas configuraciones. | El proyecto muestra buen desempeño, pero con dos fuentes de medición. | mismas evidencias |
| 34. ¿Qué latencia logró? | README: 0.694s promedio; benchmark CSV: entre 0.632 y 0.863s según configuración GPT-4o-mini. | Responde en menos de un segundo en benchmark documental. | README, CSV |
| 35. ¿Cuál es su costo por 1K tokens? | 0.15 USD. | Es mucho más barato que GPT-4o. | `benchmark_rag.py:233-238`, README |
| 36. ¿Por qué eso importa? | Porque el sistema está pensado para miles de familias y alto volumen. | Porque una solución cara no escalaría bien. | README contexto y benchmark |
| 37. ¿Cómo evita alucinaciones? | No las elimina totalmente, pero RAG y prompts reducen el riesgo. | Usa documentos reales antes de responder. | README RAG, `faq.py` |
| 38. ¿Qué pasa si no hay documentos relevantes? | El contexto puede quedar vacío y el modelo responde con base en la política del prompt; los tests contemplan fallback. | Si no encuentra información, intenta no inventar. | `faq.py`, `test_faq_node.py` |
| 39. ¿Qué idiomas maneja aquí? | En el proyecto está instruido para responder en español. | Está configurado para hablar con los usuarios en español. | `FAQ_SYSTEM_PROMPT` |
| 40. ¿Qué ejemplo real de consulta maneja bien? | Preguntas de horario, vacaciones, requisitos de matrícula y pagos. | Atiende preguntas frecuentes reales de la institución. | `locustfile.py`, benchmark mock answers |
| 41. ¿Cómo sabe si una consulta es pagos y no FAQ? | Usa reglas de palabras clave y, si hace falta, GPT-4o-mini con prompt de clasificación. | Mira el contenido y decide a qué tema pertenece. | `classifier.py` |
| 42. ¿Cómo participa en la autenticación? | No autentica por sí mismo, pero sí puede clasificar que la intención es `auth`. | Ayuda a reconocer que el usuario quiere iniciar sesión o recuperar acceso. | `classifier.py` |
| 43. ¿Puede responder sin contexto institucional? | Sí, como LLM general, pero el proyecto busca que responda con contexto institucional cuando es FAQ. | Podría responder, pero sería menos preciso para Enfócate Más. | README RAG |
| 44. ¿Qué tan escalable es su uso? | Técnicamente escalable por costo y arquitectura modular. | Se puede usar con muchos usuarios sin rediseñar todo el sistema. | README elección, Docker, FastAPI |
| 45. ¿Cómo se evidencia su impacto? | En benchmark, logs, pruebas de carga y flujos funcionales del sistema. | Hay resultados y uso registrado. | `benchmark_results.csv`, `metrics_report.json`, `logs/interactions.jsonl` |
| 46. ¿Qué riesgo tiene depender de GPT-4o-mini? | Dependencia de API externa, costo acumulado y latencia variable. | Si la API falla o se encarece, afecta el servicio. | arquitectura + riesgos del proyecto |
| 47. ¿Qué aporta frente a un bot sin IA? | Comprensión flexible del lenguaje y generación natural. | Puede conversar mejor con personas reales. | README problema y solución |
| 48. ¿Qué aporta frente a solo RAG sin LLM fuerte? | Mejor redacción, seguimiento de instrucciones y generalización lingüística. | No solo busca documentos, también los convierte en una respuesta útil. | `faq.py` |
| 49. ¿Qué parte del sistema depende más de GPT-4o-mini? | Clasificación de intención y generación FAQ. | Es el cerebro lingüístico del chatbot. | `classifier.py`, `faq.py` |
| 50. ¿Por qué este modelo está alineado con la rúbrica? | Porque da solución real al problema, está integrado, medido y justificado técnica y contextualmente. | No es un modelo decorativo: es el que realmente hace que el sistema funcione. | README, código, benchmark |

## 14. Guion completo de exposición

Buenos días. En esta exposición voy a centrarme en el modelo de Inteligencia Artificial que realmente da solución al problema del proyecto Enfócate Más: **GPT-4o-mini**. El contexto del problema es muy claro en la documentación del repositorio. Enfócate Más atiende a más de 3.000 familias y recibe consultas administrativas recurrentes sobre pagos, asistencia, matrículas y preguntas frecuentes. Cuando estas consultas se gestionan manualmente, generan cuellos de botella en el personal administrativo.

Frente a ese contexto, el proyecto implementa un chatbot administrativo 24/7 por WhatsApp y web. Pero lo importante desde la perspectiva de la rúbrica no es solo que haya un chatbot, sino entender **qué modelo de IA le da capacidad real de comprender y responder**. Ese modelo es GPT-4o-mini.

GPT-4o-mini es un Large Language Model desarrollado por OpenAI. GPT significa Generative Pre-trained Transformer. Esto nos dice tres cosas: primero, que puede generar texto; segundo, que ya fue preentrenado sobre grandes volúmenes de datos; y tercero, que su arquitectura es un Transformer, que es una red neuronal profunda moderna basada en mecanismos de atención.

¿Por qué esto importa para el proyecto? Porque las familias no preguntan siempre igual. Una persona puede escribir “¿Cuál es el horario de atención?”, otra “¿Cuándo abren?” y otra “¿A qué horas atienden?”. Un sistema de reglas exactas tendría limitaciones. En cambio, GPT-4o-mini puede entender intención semántica, no solo coincidencias literales.

Dentro del proyecto, GPT-4o-mini trabaja en dos puntos reales del código. El primero está en `src/nodes/classifier.py`, donde clasifica si un mensaje corresponde a FAQ, autenticación, pagos, matrícula o soporte. El segundo está en `src/nodes/faq.py`, donde genera respuestas usando contexto institucional recuperado por RAG. Esto significa que GPT-4o-mini no está como nombre decorativo: es el motor lingüístico real del sistema.

Desde el punto de vista interno, GPT-4o-mini funciona como un modelo Transformer autoregresivo. El texto del usuario se convierte en tokens. Cada token se representa como un embedding, es decir, un vector numérico con significado semántico. Luego el modelo usa mecanismos de self-attention para decidir qué partes del contexto son más relevantes entre sí. Por ejemplo, en la pregunta “¿Cuánto debo este mes?”, debe relacionar “cuánto” con monto, “debo” con deuda y “este mes” con temporalidad. Esa capacidad de relacionar palabras de forma contextual es precisamente una de las grandes fortalezas de los Transformers.

Además, el proyecto no usa GPT-4o-mini solo con conocimiento general. Lo combina con RAG. Eso significa que primero recupera documentos institucionales desde ChromaDB usando embeddings y búsqueda semántica, y luego GPT-4o-mini genera la respuesta final con ese contexto. Esta combinación es muy importante porque reduce respuestas genéricas y mejora la precisión institucional.

¿Por qué se eligió GPT-4o-mini y no otro modelo? El propio repositorio contiene benchmark. Se evaluaron GPT-4o, GPT-4o-mini, GPT-3.5-turbo y Llama3-local, con distintas configuraciones. El README documenta que GPT-4o-mini fue elegido por el mejor equilibrio entre costo y rendimiento. Mientras GPT-4o cuesta 2.50 dólares por 1.000 tokens, GPT-4o-mini cuesta 0.15 dólares. Además, su latencia promedio documentada ronda los 0.7 segundos, algo adecuado para WhatsApp.

En cuanto a resultados, el proyecto reporta para el modelo elegido una accuracy de 84.6%, precision de 91.7%, recall de 91.7%, F1 de 0.917 y latencia promedio de 0.694 segundos. El benchmark persistido también muestra para configuraciones GPT-4o-mini métricas altas de precision, recall, F1, BLEU, ROUGE y similitud coseno. Es decir, el proyecto no solo declara que usa GPT-4o-mini, sino que además lo evalúa con métricas concretas.

En conclusión, GPT-4o-mini está perfectamente alineado con el problema del proyecto porque combina comprensión del lenguaje natural, velocidad, bajo costo e integración con RAG. Gracias a eso, el chatbot puede atender a familias, estudiantes y personal administrativo en un canal real como WhatsApp, con respuestas más útiles, más escalables y más cercanas al contexto institucional. Por eso, desde la perspectiva de la rúbrica, sí se puede demostrar dominio profundo del modelo de IA utilizado: entendemos qué es, cómo funciona, por qué fue elegido, dónde está implementado y cómo resuelve el problema específico de Enfócate Más.

## 15. Cómo responder la rúbrica

### 15.1 Cómo demostrar que el modelo está alineado con el contexto y necesidades del problema

La clave es responder siempre conectando tres niveles:

- el problema institucional,
- la capacidad técnica de GPT-4o-mini,
- la evidencia concreta del proyecto.

Ejemplo de respuesta sobresaliente:

“Elegimos GPT-4o-mini porque el problema de Enfócate Más exige comprensión flexible del lenguaje natural en español, velocidad suficiente para WhatsApp y costo sostenible para miles de familias. En el repositorio esto se evidencia tanto en el README, que lo declara como modelo elegido, como en el código de `classifier.py` y `faq.py`, donde aparece explícitamente configurado. Además, su uso se complementa con RAG, lo cual mejora precisión institucional.”

### 15.2 Cómo demostrar dominio profundo del funcionamiento del modelo y su justificación

Un estudiante sobresaliente no se limita a decir “es un modelo de OpenAI”. Debe ser capaz de explicar:

- que es un LLM,
- que pertenece a la familia Transformer,
- cómo usa tokens, embeddings y atención,
- cómo genera texto de manera autoregresiva,
- cómo el proyecto lo especializa con prompt engineering,
- cómo RAG le aporta contexto específico,
- por qué su costo y latencia lo vuelven adecuado para Enfócate Más.

Ejemplo de respuesta sobresaliente:

“GPT-4o-mini es una red neuronal profunda de tipo Transformer. Su comprensión del lenguaje surge de representar texto como tokens y embeddings, y de usar self-attention para relacionar cada palabra con el resto del contexto. En nuestro proyecto, esa capacidad se aprovecha en dos funciones: clasificación de intención y generación de respuestas FAQ con contexto recuperado desde ChromaDB. Lo elegimos porque ofrece un balance muy favorable entre costo y velocidad frente a GPT-4o, manteniendo utilidad práctica para atención masiva.”

### 15.3 Frases modelo para responder al jurado

- “El modelo no fue elegido por moda, sino por ajuste al contexto operativo de Enfócate Más.”
- “Su valor en este proyecto está en la comprensión semántica, la generación de respuestas y la capacidad de integrarse con RAG.”
- “La arquitectura Transformer es especialmente pertinente porque el problema requiere interpretar variaciones naturales del lenguaje.”
- “La evidencia de su alineación no es solo conceptual; está en el código, en el benchmark y en las métricas del sistema.”

## Glosario de términos técnicos

| Término | Definición concisa |
| --- | --- |
| LLM | Modelo de lenguaje de gran escala capaz de comprender y generar texto. |
| GPT | Generative Pre-trained Transformer. |
| Transformer | Arquitectura neuronal basada en atención para procesar secuencias. |
| Token | Unidad mínima de texto que procesa el modelo. |
| Tokenización | Proceso de dividir texto en tokens. |
| Embedding | Vector numérico que representa significado lingüístico. |
| Positional Encoding | Señal que informa la posición de cada token en la secuencia. |
| Self-Attention | Mecanismo por el cual un token pondera la relevancia de otros tokens del contexto. |
| Multi-Head Attention | Atención paralela desde múltiples perspectivas representacionales. |
| Feed Forward Network | Subred densa que transforma representaciones internas en cada bloque Transformer. |
| Softmax | Función que convierte logits en probabilidades. |
| ReLU | Función de activación no lineal clásica. |
| GELU | Función de activación suave usada frecuentemente en Transformers modernos. |
| Modelo fundacional | Modelo general preentrenado reutilizable en muchas tareas. |
| Modelo generativo | Modelo capaz de crear salida nueva, como texto. |
| Pretraining | Entrenamiento inicial a gran escala para predecir el siguiente token. |
| Fine-Tuning | Ajuste posterior de un modelo base para especializar comportamiento. |
| RLHF | Reinforcement Learning from Human Feedback, refinamiento con retroalimentación humana. |
| RAG | Retrieval-Augmented Generation; combina recuperación de información y generación. |
| ChromaDB | Base de datos vectorial usada para búsqueda semántica por similitud. |

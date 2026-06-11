from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape


OUTPUT = Path(__file__).resolve().parents[1] / "Analisis_Proyecto_EnfocateMas.docx"


CONTENT = """ANALISIS DEL PROYECTO ENFOCATE MAS

Hallazgo central

Este proyecto no es una ANN entrenada desde cero. Lo que realmente existe es un chatbot administrativo para Enfocate Mas con LangGraph para orquestacion, GPT-4o-mini para clasificacion y generacion, RAG con ChromaDB y OpenAIEmbeddings, un CSV sintetico de estudiantes, y despliegue con FastAPI, Twilio WhatsApp y Docker.

Evidencia directa:
- docs/READMEE.md:3 define: Asistente virtual 24/7 para mas de 3.000 familias, usando LangGraph + RAG + GPT-4o-mini.
- src/nodes/classifier.py:135-153 clasifica intencion con ChatOpenAI(model='gpt-4o-mini').
- src/nodes/faq.py:44-61 usa OpenAIEmbeddings, Chroma y ChatOpenAI.
- No hay notebooks .ipynb.
- No hay modelos entrenados .h5, .pkl, .pt u .onnx.

Tambien hay una inconsistencia de metricas:
- experiments/results/benchmark_results.csv y .json guardan 8 configuraciones en modo MOCK con f1_score=0.954 para varias opciones.
- docs/READMEE.md:625-631 y experiments/plot_final_results.py muestran otros numeros, por ejemplo GPT-4o-mini con F1 0.6795 y Accuracy 84.6%.
- Durante la sustentacion conviene explicar que hay dos fuentes de resultados: benchmark persistido y resumen documental.

1. Contexto del problema

Lo que realmente se hizo en el proyecto

Segun docs/READMEE.md:35-43, Enfocate Mas atiende diariamente a miles de familias con necesidades administrativas recurrentes: consultar pagos, renovar matriculas, obtener informacion de asistencia y resolver dudas generales. Estos procesos se gestionaban manualmente y generaban cuellos de botella en el personal administrativo.

La importancia se sustenta en que el proyecto busca disponibilidad 24/7, respuestas instantaneas, integracion por WhatsApp, acceso a datos institucionales y una arquitectura extensible. El README indica que la institucion atiende a mas de 3.000 familias.

Consecuencias de no resolverlo:
- seguir con atencion manual
- depender del horario de oficina
- mayor tiempo de respuesta
- saturacion del personal administrativo

Actores involucrados y analisis real:

Clientes:
En el repositorio no se usa la palabra clientes; el actor equivalente son las familias. Se afectan por la demora en pagos, matriculas, asistencia y dudas generales. Se benefician con atencion 24/7 por WhatsApp y web. Gracias al sistema pueden decidir consultar saldo, iniciar recuperacion de acceso o pedir soporte.

Usuarios:
Los usuarios reales son familias y estudiantes que escriben por web o WhatsApp. Se benefician por la rapidez y por el enrutamiento automatico a FAQ, auth, pagos, matricula o soporte.

Empresa:
No aparece una empresa comercial separada; la entidad real es la institucion educativa. Se beneficia al reducir cuellos de botella y automatizar consultas repetitivas.

Empleados:
El personal administrativo es el actor directamente aliviado por la automatizacion. La solucion reduce trabajo manual y les permite concentrarse en casos escalados.

Proveedores:
Los proveedores tecnologicos visibles son Twilio, OpenAI y ngrok. No son actores pedagogicos, pero si parte de la solucion tecnica.

Institucion:
La institucion Enfocate Mas es el actor central. Se ve afectada por la alta demanda operativa y se beneficia con un canal automatizado y trazable.

Como explicarlo oralmente

Este proyecto resuelve un cuello de botella administrativo real. Enfocate Mas atiende a mas de 3.000 familias y recibe consultas repetitivas sobre pagos, matriculas, asistencia y preguntas frecuentes. La solucion implementada fue un chatbot 24/7 por WhatsApp y web, conectado a datos institucionales y documentos oficiales.

2. Objetivo del proyecto

Lo que realmente se hizo en el proyecto

El objetivo real fue clasificar la intencion del mensaje del usuario y enrutarlo al flujo correcto. Las clases reales son faq, auth, pagos, matricula y soporte. Esto se evidencia en src/nodes/classifier.py y src/memory/state.py.

Ademas, el proyecto responde preguntas frecuentes con RAG, autentica usuarios por telefono o correo, consulta pagos y asistencia, y escala casos urgentes a soporte.

Variable objetivo real:
- En produccion, la variable objetivo es intent en ChatState.
- En benchmark, el target esta en EVAL_DATASET como expected_intent.

No existe en el proyecto un target de regresion, un modulo de recomendacion, ni una red neuronal tabular entrenada.

Como explicarlo oralmente

El objetivo no fue predecir una nota o una venta. El objetivo real fue clasificar la intencion del mensaje y resolver la consulta con el flujo correcto, ya sea FAQ, autenticacion, pagos, matricula o soporte.

3. Justificacion de usar IA o Machine Learning

Lo que realmente se hizo en el proyecto

El README justifica usar RAG en docs/READMEE.md:49-61. La idea central es que una busqueda exacta o reglas tradicionales no bastan cuando el usuario formula preguntas con lenguaje natural. El proyecto usa busqueda semantica y generacion condicionada por contexto institucional.

Por que no bastan reglas tradicionales:
- el usuario no siempre usa las mismas palabras
- se necesita similitud semantica y no solo coincidencia exacta
- el README compara SQL vs ChromaDB y muestra que Chroma puede encontrar respuestas por significado

Que aprende o resuelve el sistema:
- clasificacion de intencion con prompt + LLM
- recuperacion semantica de documentos
- generacion de respuesta contextualizada

Como explicarlo oralmente

En este proyecto la IA se justifica porque la gente pregunta de muchas formas distintas. No basta con un menu o con buscar palabras exactas. La solucion usa embeddings y un LLM para entender la intencion y responder con informacion institucional real.

4. Datos utilizados

Fuente de datos real

- CSV sintetico institucional: data/csv/datos_estudiantes.csv
- Documentos institucionales PDF y DOCX: data/documents
- Dataset de evaluacion manual en codigo: EVAL_DATASET dentro de experiments/benchmark_rag.py

No se encontro fuente Kaggle, API externa, base SQL ni notebooks Colab.

Tamano real del dataset

- datos_estudiantes.csv: 3500 registros y 24 columnas
- data/documents: 11 archivos
- EVAL_DATASET: 21 casos de evaluacion

Variables reales del CSV

Variable | Tipo | Descripcion | Uso en el sistema
country_code | numerica/categorica | codigo de pais | apoyo a telefonia
cellphone | texto | telefono principal | autenticacion y consultas
TIENE_HERMANO | categorica | SI/NO | recuperacion e info general
PAGO_MATRICULA | numerica | pagado matricula | estado de cuenta
PAGO_CLASES | numerica | pagado clases | estado de cuenta
DEBE_MATRICULA | numerica | deuda matricula | estado de cuenta
DEBE_CLASES | numerica | deuda clases | estado de cuenta
SALDO_NETO_REAL | numerica | saldo total | cuenta
SALDO_NETO_PALABRA | texto | A_FAVOR:x o DEBE:x | cuenta
HISTORIAL_ASISTENCIA_Y_PAGOS | texto | historial concatenado | queda en dataset
nombre_estudiante | texto | nombre del estudiante | auth e info
CUANTAS_CLASES_LLEVAMOS | numerica | total clases | asistencia
CUANTAS_CLASES_ASISTIO | numerica | total asistidas | asistencia
CUANTAS_CLASES_ASISTIO_CON_PAGO | numerica | asistidas con pago | asistencia
CUANTAS_CLASES_ASISTIO_SIN_PAGO | numerica | asistidas sin pago | asistencia
CUANTAS_CLASES_NO_HA_ASISTIDO | numerica | inasistencias | asistencia
ASISTIO_SI_NO_A_LA_ULTIMA_CLASE | categorica | SI/NO | asistencia
ASISTIO_ULTIMOS_3_MESES | categorica | SI/NO | dataset
COLEGIO | texto | colegio | info y recuperacion
COLEGIO_SLUG | texto | slug | dataset
GRUPO | texto | grupo | info
CORREO | texto | correo | auth/update
CONTRASENA | texto | contrasena | auth/update
all_phone_numbers | texto | telefonos registrados | matching

Estadisticas reales verificadas del CSV

- filas: 3500
- columnas: 24
- nulos: 0 en todas las columnas
- duplicados exactos: 0
- TIENE_HERMANO: SI 1717, NO 1783
- ASISTIO_SI_NO_A_LA_ULTIMA_CLASE: SI 2453, NO 1047
- ASISTIO_ULTIMOS_3_MESES: SI 2091, NO 1409
- COLEGIO: Colegio San Jose 851, Academia Futuro 871, Colegio Nueva Era 900, Instituto Central 878
- GRUPO: Grupo 1 872, Grupo 2 858, Grupo 3 914, Grupo 4 856

Resumen numerico real

- PAGO_MATRICULA: min 26, max 49991, promedio 25348.08
- PAGO_CLASES: min 8000, max 320000, promedio 121074.86
- DEBE_MATRICULA: min 4, max 49983, promedio 25109.27
- DEBE_CLASES: min 8000, max 340000, promedio 120333.71
- SALDO_NETO_REAL: min -247024, max 195942, promedio 979.96
- CUANTAS_CLASES_LLEVAMOS: min 20, max 60, promedio 40.16
- CUANTAS_CLASES_ASISTIO: min 7, max 50, promedio 26.79
- CUANTAS_CLASES_ASISTIO_CON_PAGO: min 1, max 34, promedio 13.43
- CUANTAS_CLASES_ASISTIO_SIN_PAGO: min 1, max 34, promedio 13.36
- CUANTAS_CLASES_NO_HA_ASISTIDO: min 1, max 29, promedio 13.37
- SALDO_NETO_PALABRA: A_FAVOR 1764, DEBE 1736

Features y target reales

- En el CSV no hay target de entrenamiento.
- En clasificacion de intencion, el feature real es el texto del usuario.
- El target de benchmark es expected_intent en EVAL_DATASET.

Como explicarlo oralmente

El sistema usa tres tipos de datos: un CSV de 3.500 estudiantes, 11 documentos institucionales para RAG y un conjunto manual de 21 casos para benchmark. El CSV no se uso para entrenar una red, sino como base operativa para autenticacion, pagos y asistencia.

5. Preprocesamiento

Lo que realmente se hizo en el proyecto

Limpieza real del CSV:
- nulos: 0
- duplicados: 0
- no se observo una pipeline de limpieza tabular compleja

Transformaciones reales:
- normalizacion de telefonos en auth.py y pagos.py
- chunking de documentos con chunk_size=500 y chunk_overlap=50 en src/utils/ingest_documents.py

No se encontro:
- One-Hot Encoding
- Label Encoding
- escalamiento
- normalizacion numerica para entrenamiento
- estandarizacion
- division train/validation/test del CSV

Como justificarlo oralmente

En este proyecto no hubo preprocesamiento clasico de Machine Learning tabular porque el CSV no se uso para entrenar un modelo. El preprocesamiento real fue la normalizacion de telefonos y la fragmentacion de documentos para RAG.

6. Seleccion del modelo

Lo que realmente se hizo en el proyecto

Modelos y librerias reales:
- ChatOpenAI con model='gpt-4o-mini' en clasificacion y FAQ
- OpenAIEmbeddings
- Chroma
- LangGraph

Configuraciones benchmark reales comparadas en experiments/benchmark_rag.py:
- gpt-4o
- gpt-4o-mini
- gpt-3.5-turbo
- llama3-local
- embeddings openai y sentence_transformers
- chunk sizes 200, 500, 1000

Segun docs/READMEE.md, el modelo elegido fue GPT-4o-mini por mejor balance costo/rendimiento, rapidez para WhatsApp y escalabilidad de costo.

Comparacion con modelos clasicos

No aparecen en el repositorio experimentos con Regresion Logistica, Arbol de Decision, Random Forest, SVM o XGBoost. Por lo tanto, no es correcto afirmar que fueron descartados tras una comparacion experimental real. La forma correcta de justificarlo es decir que no eran el enfoque implementado porque el problema productivo principal es conversacional y semantico, no una clasificacion tabular supervisada clasica.

Como explicarlo oralmente

El proyecto no eligio entre modelos tabulares tradicionales. Eligio una arquitectura conversacional basada en LLM y recuperacion semantica. El README documenta que se selecciono GPT-4o-mini por equilibrio entre costo, rapidez y calidad.

7. Arquitectura de la red neuronal

Lo que realmente se hizo en el proyecto

No hay una ANN implementada en el repositorio. No existen capas Dense, Sequential, neuronas configuradas ni salida softmax definida por el equipo.

La arquitectura real implementada es un grafo conversacional:

START
-> intent_classifier
-> faq_node / auth_node / pagos_node / soporte_node
-> response_builder
-> logger_node
-> END

Evidencia: src/graph.py:41-68 y docs/READMEE.md:138-166.

Como explicarlo oralmente

En este proyecto no construimos una red neuronal artificial propia. La arquitectura real es un grafo de estados conversacional con nodos especializados para intencion, FAQ, autenticacion, pagos, soporte, formateo y logging.

8. Funcionamiento interno de la red

Lo que realmente se hizo en el proyecto

No aplica explicar neuronas, pesos, bias y suma ponderada como algo implementado por el equipo, porque el proyecto usa un LLM externo ya preentrenado. Lo que si fue implementado es la logica de prompts, recuperacion semantica y enrutamiento condicional.

Como explicarlo oralmente

La matematica interna de las neuronas no fue desarrollada en este proyecto. Lo que si disenamos fue la logica de negocio: como clasificar la intencion, como recuperar contexto y como construir la respuesta final.

9. Funciones de activacion

Lo que realmente se hizo en el proyecto

No se encontraron implementaciones explicitas de ReLU, Sigmoid, Softmax o Tanh en el repositorio. No hay codigo de capas neuronales propias.

Como justificarlo oralmente

No aplica porque el proyecto no entrena ni define una ANN propia; usa un modelo LLM preentrenado via API.

10. Proceso de entrenamiento

Lo que realmente se hizo en el proyecto

No hay entrenamiento de una red propia. No se encontraron epochs, batch_size, learning_rate ni funcion de perdida. Lo que si existe es evaluacion comparativa mediante benchmark.

Evidencia real:
- EVAL_DATASET con 21 casos en experiments/benchmark_rag.py
- calculo de precision, recall, f1_score, BLEU, ROUGE, cosine similarity, latencia y costo en experiments/benchmark_rag.py:488-523

Como explicarlo oralmente

No hubo fase de entrenamiento porque se usaron modelos fundacionales ya preentrenados. El trabajo tecnico estuvo en benchmarking, prompts, recuperacion semantica y evaluacion del sistema.

11. Backpropagation

Lo que realmente se hizo en el proyecto

No aplica. No se implemento backpropagation porque no se entreno una red neuronal desde cero.

Como explicarlo oralmente

Backpropagation es el mecanismo de aprendizaje de una red entrenada. En este proyecto no se uso porque el modelo base ya venia entrenado y nosotros construimos la arquitectura conversacional alrededor de el.

12. Optimizador

Lo que realmente se hizo en el proyecto

No se encontraron Adam, SGD, RMSprop ni otro optimizador configurable en el codigo.

Como explicarlo oralmente

No hubo optimizador definido por el equipo porque no existio entrenamiento propio de una red neuronal.

13. Hiperparametros

Hiperparametros reales encontrados

- temperature=0 en clasificador (src/nodes/classifier.py)
- temperature=0.3 en FAQ (src/nodes/faq.py)
- chunk_size=500 por defecto (src/utils/ingest_documents.py)
- chunk_overlap=50 (src/utils/ingest_documents.py)
- k=5 en similarity_search del FAQ (src/nodes/faq.py)
- benchmark con chunk sizes 200, 500 y 1000

No existen learning rate, epochs, batch size, dropout ni regularizacion porque no hubo entrenamiento de red propia.

Justificacion real

- temperature 0 en clasificacion: determinismo
- temperature 0.3 en FAQ: algo mas natural pero controlado
- chunk_size 500: el README lo documenta como balance optimo
- chunk_overlap 50: continuidad entre fragmentos
- k=5: recuperacion de contexto suficiente

14. Prevencion del overfitting

Lo que realmente se hizo en el proyecto

No hay dropout, early stopping, regularizacion ni data augmentation clasicos. El control del error se maneja mas por arquitectura que por entrenamiento:
- uso de RAG con documentos reales
- temperatura baja
- benchmark comparativo

El README incluso advierte que el F1=1.0 de GPT-3.5-turbo puede indicar sobreajuste sobre el conjunto de evaluacion.

Como explicarlo oralmente

Como no hubo entrenamiento propio, no aplican tecnicas clasicas de overfitting. La estrategia principal fue usar contexto institucional real y comparar configuraciones.

15. Metricas de evaluacion

Metricas reales encontradas

Fuente 1: experiments/results/benchmark_results.csv para GPT-4o-mini + chunk500
- precision: 0.9643
- recall: 0.9524
- f1_score: 0.954
- f1_faq: 0.9231
- f1_auth: 1.0
- f1_pagos: 1.0
- f1_matricula: 0.8571
- f1_soporte: 1.0
- bleu: 0.5707
- rouge1: 0.6421
- rouge2: 0.5513
- rougeL: 0.6267
- cosine_sim: 0.6601
- latency_avg_s: 0.751
- latency_p95_s: 0.84
- cost_per_1k_usd: 0.15

Fuente 2: docs/READMEE.md para el modelo elegido
- Accuracy: 84.6 por ciento (11/13)
- Precision: 91.7 por ciento
- Recall: 91.7 por ciento
- F1-Score: 0.917
- Latencia Avg: 0.694 s
- Latencia P95: 0.946 s
- Costo: 0.15 USD por 1K tokens

Fuente 3: metrics_report.json

Latencia por caso:
- hola: avg 7676 ms, p95 3776 ms
- horarios: avg 4834 ms, p95 3956 ms
- programas: avg 6201 ms, p95 4860 ms
- +57 3874290099: avg 59 ms, p95 59 ms
- +57 300 999 9999: avg 60 ms, p95 61 ms
- xkdzpq: avg 1888 ms, p95 3288 ms

Concurrencia:
- 1 usuario: avg 3082 ms, p95 3082 ms, tps 0.32
- 5 usuarios: avg 3424 ms, p95 3512 ms, tps 1.38
- 10 usuarios: avg 3561 ms, p95 3770 ms, tps 2.6
- 20 usuarios: avg 4294 ms, p95 4615 ms, tps 4.33

Flujos:
- login_y_pagos: ok=false
- login_y_asistencia: ok=false

Interpretacion realista

El sistema tiene evidencia de funcionalidad y evaluacion, pero no conviene presentarlo como perfecto. FAQ y respuestas con LLM tienen mas latencia que el login. Ademas, los flujos completos del reporte guardado fallaron.

16. Resultados obtenidos

Lo que realmente se hizo en el proyecto

Resultados y evidencias encontradas:
- benchmark_results.csv
- benchmark_results.json
- metrics_report.json
- graficas en experiments/results
- logs/interactions.jsonl con 2811 registros

Graficas reales encontradas:
- 01_classification_metrics.png
- 02_text_quality_metrics.png
- 03_radar_chart.png
- 04_latency.png
- 05_f1_intent_heatmap.png
- 06_cost_vs_quality.png
- 07_weighted_ranking.png

Evaluacion realista del resultado

Fortalezas:
- arquitectura modular clara
- clasificacion y RAG implementados
- benchmark y pruebas automatizadas
- despliegue web y WhatsApp documentado

Debilidades:
- inconsistencia entre fuentes de metricas
- metrics_report.json muestra fallos en flujos completos
- README menciona docs/evidencia, pero esa carpeta no existe
- docker-compose.yml referencia deploy/nginx.conf, pero no existe esa ruta

Como explicarlo oralmente

El proyecto es funcional y medible, pero no esta totalmente cerrado. La mejor estrategia ante el jurado es mostrar tanto los logros como las brechas pendientes, con honestidad tecnica.

17. Impacto para los involucrados

Empresa / institucion:
- reduce carga manual
- automatiza consultas repetitivas
- mejora disponibilidad 24/7

Clientes / familias:
- resuelven consultas por WhatsApp y web
- acceden a pagos, asistencia y FAQ sin esperar horario de oficina

Empleados:
- menos trabajo repetitivo
- se enfocan en casos escalados

Directivos:
- pueden analizar intenciones y volumen de uso a partir de logs
- existe un ejemplo de explotacion de logs en el README

18. Limitaciones del modelo

Limitaciones reales encontradas

- el CSV es sintetico, no real
- no hay entrenamiento supervisado propio
- no hay split train/validation/test del CSV
- benchmark persistido en modo MOCK
- inconsistencia entre benchmark y resumen documental
- flujos completos fallidos en metrics_report.json
- docs/evidencia no existe
- deploy/nginx.conf no existe
- el telefono +57 3874290099 usado en test_metrics.py no aparece en el CSV actual

Como responder al jurado

La principal limitacion es que la base operativa incluida es sintetica y parte del benchmark corre en modo MOCK. Por eso el valor del proyecto esta en la arquitectura funcional, no en afirmar una precision productiva definitiva.

19. Mejoras futuras

Mejoras realistas basadas en el propio proyecto

1. Reemplazar el CSV sintetico por datos reales manteniendo el mismo esquema.
2. Unificar la metodologia de metricas y reportes.
3. Ejecutar benchmarks en modo REAL con API activa.
4. Corregir los flujos login_y_pagos y login_y_asistencia reportados como fallidos.
5. Migrar de CSV a base de datos real.
6. Completar despliegue productivo agregando deploy/nginx.conf.
7. Agregar monitoreo y panel de intenciones a partir de logs.
8. Implementar modulos futuros ya documentados: notificaciones, pasarela de pagos, OCR de recibos, integracion LMS y voz por WhatsApp.

20. Preguntas dificiles de jurado con respuestas

1. Esto es una red neuronal entrenada por ustedes?
Respuesta tecnica: no; es un sistema RAG con LangGraph, GPT-4o-mini y ChromaDB.
Respuesta sencilla: no entrenamos una red desde cero; construimos la arquitectura aplicada.

2. Cual es la variable objetivo?
Tecnica: intent en ChatState; en benchmark, expected_intent.
Sencilla: la intencion del mensaje.

3. Donde esta el dataset de entrenamiento?
Tecnica: no hay dataset de entrenamiento propio para ANN.
Sencilla: no entrenamos una red desde cero.

4. Entonces por que hablan de accuracy?
Tecnica: porque se evalua clasificacion de intencion, no un entrenamiento tabular.
Sencilla: medimos que tan bien clasifica mensajes.

5. Por que usar RAG?
Tecnica: para recuperar contexto institucional actualizado desde documentos.
Sencilla: para responder con informacion real y no inventada.

6. Que documentos usa?
Tecnica: 11 archivos en data/documents, 10 DOCX y 1 PDF.
Sencilla: documentos oficiales de la institucion.

7. Cuantos registros tiene el CSV?
Tecnica: 3500 filas y 24 columnas.
Sencilla: 3.500 estudiantes sinteticos.

8. Hay nulos o duplicados?
Tecnica: 0 nulos y 0 duplicados exactos.
Sencilla: el CSV esta limpio.

9. Que librerias principales usaron?
Tecnica: langgraph, langchain-openai, langchain-chroma, fastapi, pandas, twilio.
Sencilla: librerias para orquestacion, IA, API y datos.

10. Como clasifican la intencion?
Tecnica: reglas mas ChatOpenAI con temperature 0.
Sencilla: detecta lo que el usuario quiere y lo envia al flujo correcto.

11. Como responde las FAQ?
Tecnica: embeddings, Chroma similarity_search(k=5) y GPT-4o-mini.
Sencilla: busca documentos parecidos y responde con ese contexto.

12. Por que chunk_size=500?
Tecnica: el README lo justifica como balance optimo.
Sencilla: ni muy pequeno ni muy grande.

13. Usaron train, validation y test?
Tecnica: no para el CSV; no hubo entrenamiento de ANN.
Sencilla: no aplica en este caso.

14. Que funciones de activacion usaron?
Tecnica: no estan expuestas ni configuradas en el repo.
Sencilla: no aplica porque no construimos una red propia.

15. Que optimizador usaron?
Tecnica: ninguno configurable en el proyecto.
Sencilla: no hubo entrenamiento propio.

16. Que metricas reales obtuvieron?
Tecnica: benchmark persistido, resumen documental y metrics_report.json.
Sencilla: medimos calidad, costo y tiempo de respuesta.

17. Cual es la mejor configuracion segun benchmark persistido?
Tecnica: el ranking ponderado puede favorecer otras configuraciones, pero el README declara como elegido GPT-4o-mini.
Sencilla: se priorizo costo-rendimiento.

18. Por que no eligieron GPT-4o?
Tecnica: costo de 2.50 USD por 1K tokens frente a 0.15 de GPT-4o-mini.
Sencilla: era demasiado costoso para uso masivo.

19. Por que no eligieron GPT-3.5?
Tecnica: el README advierte posible sobreajuste con F1=1.0.
Sencilla: parecia demasiado perfecto y poco confiable.

20. Que tan rapido responde?
Tecnica: depende del flujo; benchmark documental da 0.694 s, pero metricas reales muestran que FAQ puede tardar varios segundos.
Sencilla: login es rapido; FAQ es mas pesada.

21. Que pasa si no encuentra contexto?
Tecnica: el FAQ puede devolver una respuesta de falta de informacion; los tests contemplan fallback.
Sencilla: indica que no tiene informacion y sugiere contactar la institucion.

22. Como autentican?
Tecnica: por telefono o por CORREO y CONTRASENA.
Sencilla: valida identidad antes de mostrar datos sensibles.

23. Cuantos intentos permite?
Tecnica: MAX_ATTEMPTS = 3.
Sencilla: tres intentos.

24. El sistema escala a humano?
Tecnica: si; usa ESCALATION_KEYWORDS y marca escalate=True.
Sencilla: si detecta urgencia o queja, pasa a soporte.

25. Que canal principal usa?
Tecnica: WhatsApp via Twilio.
Sencilla: el canal que mas usan las familias.

26. Tienen evidencia de uso?
Tecnica: logs/interactions.jsonl tiene 2811 lineas.
Sencilla: si, hay logs reales.

27. Cuales son las intenciones mas frecuentes en logs?
Tecnica: faq 1329, auth 1255, soporte 116, pagos 106, matricula 5.
Sencilla: predominan FAQ y autenticacion.

28. Cual es la limitacion mas fuerte?
Tecnica: datos sinteticos y benchmark en modo MOCK.
Sencilla: falta validacion completa con datos reales.

29. Hay despliegue productivo completo?
Tecnica: hay Dockerfile, docker-compose, FastAPI y Twilio, pero falta deploy/nginx.conf.
Sencilla: hay base de despliegue, pero no esta cerrado del todo.

30. Que mejorarian primero?
Tecnica: unificar metricas y reemplazar el CSV sintetico por datos reales.
Sencilla: cerrar validacion real y corregir flujos fallidos.

Guion completo de exposicion 10-15 minutos

Buenos dias. Voy a presentar el proyecto Enfocate Mas Chatbot. Lo primero importante es aclarar que tipo de solucion es. Este proyecto no implementa una red neuronal entrenada desde cero. Lo que realmente implementa es un sistema conversacional con LangGraph, RAG, GPT-4o-mini, ChromaDB, FastAPI y WhatsApp por Twilio.

El problema que resuelve esta explicado directamente en la documentacion del proyecto. Enfocate Mas atiende a mas de 3.000 familias y recibe consultas administrativas recurrentes sobre pagos, matriculas, asistencia y preguntas frecuentes. Esas consultas se manejaban manualmente y eso generaba cuellos de botella en el personal administrativo.

Por eso el objetivo del proyecto fue automatizar la atencion administrativa. Tecnicamente, el sistema recibe un mensaje por web o por WhatsApp, clasifica la intencion del usuario y lo enruta al flujo correcto. Las intenciones validas son cinco: faq, auth, pagos, matricula y soporte.

La arquitectura real esta definida en src/graph.py. El flujo inicia en intent_classifier, luego pasa a uno de cuatro nodos de negocio: faq_node, auth_node, pagos_node o soporte_node. Despues la respuesta pasa por response_builder y finalmente por logger_node.

En cuanto a los datos, el proyecto usa tres fuentes. La primera es un CSV llamado datos_estudiantes.csv con 3.500 registros y 24 columnas. La segunda son 11 documentos institucionales, 10 archivos Word y 1 PDF, que se indexan para el modulo RAG. La tercera es un dataset de evaluacion manual en el benchmark con 21 casos.

Un punto clave es que el CSV no se usa para entrenar una red neuronal. Se usa como base operativa para autenticacion, pagos, asistencia y actualizacion de datos. Las variables mas importantes son cellphone, all_phone_numbers, CORREO, CONTRASENA, PAGO_MATRICULA, DEBE_MATRICULA, PAGO_CLASES, DEBE_CLASES, SALDO_NETO_REAL y las variables de asistencia como CUANTAS_CLASES_ASISTIO.

Sobre preprocesamiento, no hubo one-hot encoding, escalamiento ni split train-test del CSV, porque no hubo entrenamiento tabular. El preprocesamiento real fue la normalizacion de telefonos para autenticar usuarios y el chunking de documentos para RAG. En la ingesta se uso chunk_size=500 y chunk_overlap=50.

Sobre el modelo, el proyecto usa gpt-4o-mini para dos tareas: clasificacion de intencion y generacion de respuesta. Para las preguntas frecuentes, primero genera embeddings con OpenAIEmbeddings, busca documentos similares en ChromaDB y luego responde con el contexto recuperado. Esa es la esencia del enfoque RAG.

Por que usar IA aqui? Porque una regla tradicional o una busqueda exacta por palabras no basta. El mismo README lo explica: una persona puede escribir cuando abren y otra cual es el horario, y el sistema debe entender que buscan lo mismo. Ahi la recuperacion semantica aporta mucho valor.

Ahora, sobre metricas. Aqui hay que ser muy rigurosos. El proyecto tiene dos fuentes de resultados. En el README se reporta para el modelo elegido un accuracy de 84.6 por ciento, precision y recall de 91.7 por ciento, F1 de 0.917, latencia promedio de 0.694 segundos y costo de 0.15 dolares por mil tokens. Pero en benchmark_results.csv y benchmark_results.json aparece otro conjunto de resultados, generado en modo MOCK, donde varias configuraciones tienen F1 de 0.954. Por eso, durante la exposicion, yo aclararia que existen benchmarks persistidos y tambien resumenes documentales, y que no deben mezclarse como si fueran una sola corrida homogenea.

Ademas, el proyecto incluye un metrics_report.json con metricas del servidor. Alli vemos que flujos simples de login tienen latencias muy bajas, mientras que consultas FAQ como hola, horarios o programas pueden tardar mas, incluso varios segundos. Tambien se probaron niveles de concurrencia de 1, 5, 10 y 20 usuarios.

En cuanto al impacto, para la institucion el beneficio principal es reducir la carga manual y mantener atencion 24/7. Para las familias, el valor esta en resolver consultas por WhatsApp o web sin esperar horario de oficina. Para el personal, significa enfocarse en casos complejos y no en preguntas repetitivas.

Las limitaciones tambien son claras. La base de estudiantes incluida es sintetica, no real. Parte del benchmark corre en modo MOCK. Hay inconsistencias entre reportes. Y en metrics_report.json dos flujos completos aparecen fallidos. Eso no invalida el proyecto, pero si muestra que esta en una etapa funcional con mejoras pendientes.

Como mejoras futuras, yo priorizaria cuatro: reemplazar el CSV sintetico por datos reales, unificar la metodologia de metricas, ejecutar benchmarks en modo real con API activa y corregir los flujos compuestos que fallan en el reporte. Despues de eso, el sistema quedaria mucho mejor preparado para una validacion institucional formal.

En conclusion, este proyecto si resuelve un problema real de negocio. No lo hace entrenando una red neuronal propia, sino integrando de forma aplicada un sistema conversacional con clasificacion de intencion, recuperacion semantica y acceso a datos administrativos. Ese es precisamente su valor tecnico: no inventa un modelo desde cero, sino que construye una arquitectura util, medible y orientada al contexto real de la institucion.
"""


def make_paragraph(text: str) -> str:
    text = escape(text)
    return (
        '<w:p>'
        '<w:r><w:t xml:space="preserve">' + text + '</w:t></w:r>'
        '</w:p>'
    )


def build_document_xml() -> str:
    paragraphs = []
    for line in CONTENT.splitlines():
        if line.strip() == "":
            paragraphs.append('<w:p/>')
        else:
            paragraphs.append(make_paragraph(line))

    body = "".join(paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        '<w:body>' + body +
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>'
    )


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenCode</Application>
</Properties>
"""


CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Analisis Proyecto Enfocate Mas</dc:title>
  <dc:creator>OpenCode</dc:creator>
  <cp:lastModifiedBy>OpenCode</cp:lastModifiedBy>
</cp:coreProperties>
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", CONTENT_TYPES)
        docx.writestr("_rels/.rels", RELS)
        docx.writestr("docProps/app.xml", APP_XML)
        docx.writestr("docProps/core.xml", CORE_XML)
        docx.writestr("word/document.xml", build_document_xml())
    print(OUTPUT)


if __name__ == "__main__":
    main()

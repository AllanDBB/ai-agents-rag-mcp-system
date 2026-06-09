"""
Conjunto de evaluación experimental — Sección IV del enunciado.

30 preguntas distribuidas en 6 categorías:
  - 10 factuales sobre los apuntes
  - 5  de comparación entre temas
  - 5  conversacionales de seguimiento
  - 5  fuera del alcance de los apuntes
  - 5  que requieren búsqueda web explícita
  - 5  sobre transacciones ficticias
"""

QUESTIONS = {
    "factual": [
        "¿Qué es el descenso del gradiente y cómo se usa en el entrenamiento de redes neuronales?",
        "¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?",
        "¿Qué es una red neuronal convolucional y para qué se usa?",
        "Explica qué es el overfitting y cómo se puede prevenir.",
        "¿Qué es la función de activación ReLU y cuáles son sus ventajas?",
        "¿En qué consiste el algoritmo de retropropagación?",
        "¿Qué es el aprendizaje por refuerzo? Da un ejemplo.",
        "¿Qué diferencia hay entre una red LSTM y una red recurrente simple?",
        "¿Qué es el método de k-vecinos más cercanos (KNN)?",
        "¿Qué es la regularización L1 y L2 y cuándo se aplican?",
    ],
    "comparison": [
        "¿Cuál es la diferencia entre SVM y regresión logística?",
        "Compara las redes neuronales densas con las redes convolucionales.",
        "¿En qué se diferencian el clustering jerárquico y el k-means?",
        "¿Cuándo es mejor usar un árbol de decisión en lugar de una red neuronal?",
        "Compara el aprendizaje supervisado y el aprendizaje por refuerzo en términos de retroalimentación.",
    ],
    "conversational": [
        "Ya mencionaste el descenso del gradiente. ¿Cómo se relaciona con el optimizador Adam?",
        "Antes hablaste de overfitting. ¿Puedes darme un ejemplo concreto de un dataset donde esto ocurre?",
        "Siguiendo lo que explicaste sobre redes LSTM, ¿se usan en traducción automática?",
        "¿Puedes resumir los temas que hemos discutido hasta ahora en esta sesión?",
        "¿Qué pregunté anteriormente sobre aprendizaje supervisado?",
    ],
    "out_of_scope": [
        "¿Cuánto cuesta una GPU NVIDIA RTX 4090 actualmente?",
        "¿Cuál es la capital de Francia?",
        "¿Quién ganó el Mundial de Fútbol 2022?",
        "¿Cuáles son los ingredientes de un gallo pinto costarricense?",
        "¿Cómo puedo solicitar una beca en el TEC?",
    ],
    "web_search": [
        "Busca en internet cuáles son los modelos de IA más recientes de OpenAI.",
        "Necesito que hagas una búsqueda web sobre el estado actual de GPT-5.",
        "Busca en internet noticias recientes sobre regulación de IA en Europa.",
        "¿Puedes buscar en la web qué es Claude 4 de Anthropic?",
        "Busca en internet los últimos avances en modelos de difusión para imágenes.",
    ],
    "transactional": [
        "¿Cuántas transacciones sospechosas hay en los últimos 7 días?",
        "Muéstrame las transacciones de más de 10000 en países de alto riesgo.",
        "¿Cuál es el nivel de riesgo del cliente CUST-00000001?",
        "Crea un caso de fraude para la transacción TX-0001000011 por monto inusual de madrugada.",
        "¿Cuántas transacciones hay en estado flagged en los últimos 30 días desde Costa Rica?",
    ],
}

ALL_QUESTIONS = [
    {"id": f"{cat[0].upper()}{i+1:02d}", "category": cat, "question": q}
    for cat, qs in QUESTIONS.items()
    for i, q in enumerate(qs)
]

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from agents.orchestrator import OrchestratorAgent
    from memory.session_memory import SessionMemory

    orchestrator = OrchestratorAgent()
    results = []

    # Las preguntas conversacionales requieren sesión compartida con contexto previo.
    # Se crea una sesión continua precargada con las primeras preguntas factuales.
    conv_session = SessionMemory()
    # Precalentar sesión conversacional con preguntas de contexto
    for warmup_q in [
        "¿Qué es el descenso del gradiente y cómo se usa en el entrenamiento de redes neuronales?",
        "Explica qué es el overfitting y cómo se puede prevenir.",
        "¿Qué diferencia hay entre una red LSTM y una red recurrente simple?",
        "¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?",
    ]:
        orchestrator.run(warmup_q, conv_session)

    for item in ALL_QUESTIONS:
        print(f"\n[{item['id']}] ({item['category']}) {item['question']}")
        # Conversacionales usan la sesión compartida; el resto sesiones independientes
        session = conv_session if item["category"] == "conversational" else SessionMemory()
        try:
            result = orchestrator.run(item["question"], session)
            answer = result["answer"]
            tools = result["tools_used"]
        except Exception as e:
            answer = f"ERROR: {e}"
            tools = []

        print(f"  → Tools: {[t['tool'] for t in tools]}")
        print(f"  → Answer: {answer[:120]}...")

        results.append({
            **item,
            "answer": answer,
            "tools_used": tools,
        })

    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Resultados guardados en {output_path}")

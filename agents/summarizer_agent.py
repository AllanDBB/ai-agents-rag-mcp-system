import anthropic
import config
from observability.langfuse_client import get_langfuse

CONTRACT = {
    "agent_name": "SummarizerAgent",
    "description": "Resume conversaciones, resultados de búsqueda o respuestas largas.",
    "role": "Agente resumidor de contenido",
    "allowed_tools": [],
    "input_schema": {"content": "str", "summary_type": "conversation|documents|response"},
    "output_schema": {"summary": "str"},
    "restrictions": [
        "No debe añadir información que no esté en el contenido original.",
        "Siempre indica que es un resumen.",
    ],
}

_PROMPTS = {
    "conversation": "Resume la siguiente conversación en puntos clave:\n\n",
    "documents":    "Resume los siguientes fragmentos, destacando los puntos principales:\n\n",
    "response":     "Resume la siguiente respuesta de forma concisa:\n\n",
}


class SummarizerAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model  = config.SUMMARIZER_MODEL

    def summarize(self, content: str, summary_type: str = "response") -> str:
        lf = get_langfuse()
        prefix = _PROMPTS.get(summary_type, _PROMPTS["response"])

        def _do():
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="Eres un agente resumidor. Responde siempre en español.",
                messages=[{"role": "user", "content": f"{prefix}{content}"}],
            )
            return resp.content[0].text

        if not lf:
            return _do()

        with lf.start_as_current_observation(
            name="summarizer",
            as_type="generation",
            model=self.model,
            input={"type": summary_type, "content_length": len(content)},
        ):
            result = _do()
            lf.update_current_generation(output=result)
            return result

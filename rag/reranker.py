from sentence_transformers import CrossEncoder

_MODEL: CrossEncoder | None = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(_MODEL_NAME)
    return _MODEL


def rerank(
    query: str,
    docs: list[str],
    top_k: int,
    early_stop_threshold: float | None = None,
) -> tuple[list[int], list[float]]:
    """Returns (indices, scores) of the top_k most relevant docs, ordered by score desc.

    early_stop_threshold: si el mejor score supera este valor se devuelve
    inmediatamente sin continuar evaluando el resto de candidatos.
    """
    model = _get_model()
    scores = model.predict([(query, doc) for doc in docs])
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if early_stop_threshold is not None and scores[ranked[0]] >= early_stop_threshold:
        selected = ranked[:top_k]
        return selected, [float(scores[i]) for i in selected]

    selected = ranked[:top_k]
    return selected, [float(scores[i]) for i in selected]

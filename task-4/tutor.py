"""The memory + RAG-powered tutor: combines a student's long-term memory
(weak topics from past attempts) with real indexed retrieval, so a new
explanation is both grounded in the actual textbook and personalized to
what the student has actually struggled with before.
"""

from __future__ import annotations

import ollama

from memory import DB_PATH, weak_topics
from rag import retrieve as default_retrieve

MODEL = "qwen2.5:7b"


def _default_chat(messages: list[dict]) -> str:
    return ollama.chat(model=MODEL, messages=messages)["message"]["content"]


def _find_weak_match(weak: list[dict], topic_query: str) -> dict | None:
    q = topic_query.lower()
    for w in weak:
        t = w["topic"].lower().replace("_", " ")
        if t in q or q in t:
            return w
    return None


def explain(
    student_id: str,
    topic_query: str,
    chat_fn=None,
    retrieve_fn=None,
    top_k: int = 2,
    db_path=DB_PATH,
) -> dict:
    """Produce a grounded, personalized explanation for one student.

    chat_fn and retrieve_fn are both injectable so offline tests can drive
    this without a live Ollama server or a built Chroma index. db_path is
    injectable so tests can use an isolated SQLite file instead of the
    real student_memory.sqlite.
    """
    chat_fn = chat_fn or _default_chat
    retrieve_fn = retrieve_fn or default_retrieve

    chunks = retrieve_fn(topic_query, k=top_k)
    weak = weak_topics(student_id, db_path=db_path)
    matched_weak = _find_weak_match(weak, topic_query)

    grounding_text = "\n".join(
        f"[{c['chapter']} > {c['title']}] {c['text']} (source: {c['source']})"
        for c in chunks
    )
    student_context = (
        f"This student has previously struggled with {matched_weak['topic']} "
        f"({matched_weak['accuracy']:.0%} accuracy over {matched_weak['attempts']} "
        "past attempts). Acknowledge this briefly and explain extra carefully, "
        "from basics."
        if matched_weak
        else "No noted past struggles with this topic for this student."
    )

    system = (
        "You are TutorLoop, a personalized homework-help tutor. Ground your "
        "explanation ONLY in the provided curriculum excerpts, do not invent "
        "facts they don't contain."
    )
    reply = chat_fn(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Student context: {student_context}\n\n"
                    f"Curriculum excerpts:\n{grounding_text}\n\n"
                    f"Explain: {topic_query}"
                ),
            },
        ]
    )

    return {
        "explanation": reply.strip(),
        "grounded_in": chunks,
        "recalled_weak_topic": matched_weak,
    }

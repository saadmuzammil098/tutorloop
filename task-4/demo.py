"""Live demo: seed a student's memory with a real weak topic, then ask
the tutor to explain that topic, showing both recall and RAG grounding
against a real (already-built, run rag.py first) Chroma index and live
Ollama.
"""

from pathlib import Path

from memory import record_attempt
from tutor import explain

DEMO_DB = Path(__file__).parent / "demo_memory.sqlite"

if __name__ == "__main__":
    DEMO_DB.unlink(missing_ok=True)

    record_attempt("alex", "fractions", False, note="mixed up numerator/denominator", db_path=DEMO_DB)
    record_attempt("alex", "fractions", False, note="added denominators directly", db_path=DEMO_DB)
    record_attempt("alex", "fractions", True, db_path=DEMO_DB)
    print("Seeded memory: alex got fractions wrong twice, right once (33% accuracy)\n")

    result = explain("alex", "fractions", db_path=DEMO_DB)
    print("recalled_weak_topic:", result["recalled_weak_topic"])
    print("grounded_in:", [{"chapter": c["chapter"], "title": c["title"]} for c in result["grounded_in"]])
    print("\nexplanation:")
    print(result["explanation"])

    print("\n--- second student, no fractions history ---")
    result2 = explain("sam", "fractions", db_path=DEMO_DB)
    print("recalled_weak_topic:", result2["recalled_weak_topic"])

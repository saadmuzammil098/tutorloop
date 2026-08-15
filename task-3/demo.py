"""Live demo: run the full planner -> content -> quiz -> coordinator
pipeline against Ollama and print the resulting mini-lesson + quiz."""

import json
import sys

from agents import coordinator

if __name__ == "__main__":
    unit = " ".join(sys.argv[1:]) or "Fractions and ratios for a 7th grader"
    result = coordinator(unit)
    print(json.dumps(result, indent=2))

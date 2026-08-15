import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.memory import MemorySaver

from graph import MAX_STEPS, build_graph, initial_state


def _scripted(responses):
    it = iter(responses)

    def chat_fn(messages):
        return next(it)

    return chat_fn


def _graph(responses):
    return build_graph(chat_fn=_scripted(responses), checkpointer=MemorySaver())


def test_answers_a_normal_question_without_pausing():
    g = _graph([
        "Thought: arithmetic.\nAction: calculator[6 * 7]",
        "Thought: done.\nFinal Answer: 42",
    ])
    config = {"configurable": {"thread_id": "q1"}}
    result = g.invoke(initial_state("What is 6 * 7?"), config)
    assert result["final_answer"] == "42"
    assert result["sent"] is False
    assert g.get_state(config).next == ()  # graph reached END, nothing pending


def test_progress_note_pauses_before_await_approval():
    g = _graph([
        "Thought: student wants this sent.\n"
        "DraftNote: Alex needs more practice with ratios.",
    ])
    config = {"configurable": {"thread_id": "q2"}}
    result = g.invoke(initial_state("Tell my teacher I need help with ratios"), config)

    assert result["progress_note"] == "Alex needs more practice with ratios."
    assert result["final_answer"] is None
    assert result["sent"] is False
    snapshot = g.get_state(config)
    assert snapshot.next == ("await_approval",)


def test_resumes_from_saved_state_after_teacher_approves():
    g = _graph([
        "Thought: student wants this sent.\nDraftNote: Alex is behind on fractions.",
    ])
    config = {"configurable": {"thread_id": "q3"}}
    g.invoke(initial_state("Tell my teacher about my fractions struggles"), config)

    # Simulate the teacher approving later: update persisted state, then
    # resume with `None` as input, which is how LangGraph continues a run
    # from its last checkpoint rather than starting a new one.
    g.update_state(config, {"teacher_decision": "approved"})
    result = g.invoke(None, config)

    assert result["sent"] is True
    assert "Alex is behind on fractions." in result["final_answer"]


def test_resumes_and_does_not_send_if_teacher_rejects():
    g = _graph([
        "Thought: student wants this sent.\nDraftNote: Alex struggled today.",
    ])
    config = {"configurable": {"thread_id": "q4"}}
    g.invoke(initial_state("Tell my teacher"), config)

    g.update_state(config, {"teacher_decision": "rejected"})
    result = g.invoke(None, config)

    assert result["sent"] is False
    assert "not sent" in result["final_answer"]


def test_recovers_from_tool_error():
    g = _graph([
        "Thought: bad topic.\nAction: practice_problem[geometry, easy]",
        "Thought: fix it.\nAction: practice_problem[fractions, easy]",
        "Thought: done.\nFinal Answer: Here is your problem.",
    ])
    config = {"configurable": {"thread_id": "q5"}}
    result = g.invoke(initial_state("practice problem please"), config)
    assert result["final_answer"] == "Here is your problem."
    assert any("Error" in t.get("observation", "") for t in result["trace"] if "observation" in t)


def test_recovers_from_malformed_response():
    g = _graph([
        "not following the format at all",
        "Thought: fixed.\nAction: calculator[1 + 1]",
        "Thought: done.\nFinal Answer: 2",
    ])
    config = {"configurable": {"thread_id": "q6"}}
    result = g.invoke(initial_state("what is 1+1"), config)
    assert result["final_answer"] == "2"


def test_step_limit_stops_the_graph():
    responses = [f"Thought: still going {i}.\nAction: calculator[1+1]" for i in range(20)]
    g = _graph(responses)
    config = {"configurable": {"thread_id": "q7"}}
    result = g.invoke(initial_state("never stop"), config)
    assert result["final_answer"] is None
    assert result["step"] == MAX_STEPS + 1
    assert g.get_state(config).next == ()

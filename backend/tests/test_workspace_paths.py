"""
Workspace path resolution and edit-vs-create.

Background (2026-09-16): "go to the folder txt in praxis-workspace and add names
james and jane there removing the older names" produced
praxis-workspace/praxis-workspace/txt/names.txt — three bugs at once. The model
repeated the workspace folder in the path; it invented names.txt beside the real
txt/name.txt because it had never been shown the listing; and the agent only ever
wrote fresh content, so "removing the older names" could not work.

Run:  cd backend && python -m pytest tests/test_workspace_paths.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.workspace import WorkspaceAgent


@pytest.fixture
def ws(tmp_path):
    return WorkspaceAgent(workspace_dir=str(tmp_path / "praxis-workspace"))


# ── redundant workspace prefix ──────────────────────────────────────────────

def test_redundant_workspace_prefix_is_stripped(ws):
    target = ws._safe_path("praxis-workspace/txt/names.txt")
    assert ws.rel(target) == os.path.join("txt", "names.txt")


def test_repeated_and_dot_prefixes_are_stripped(ws):
    assert ws.rel(ws._safe_path("praxis-workspace/praxis-workspace/a.txt")) == "a.txt"
    assert ws.rel(ws._safe_path("./txt/a.txt")) == os.path.join("txt", "a.txt")
    assert ws.rel(ws._safe_path("praxis-workspace\\txt\\a.txt")) == os.path.join("txt", "a.txt")


def test_a_file_actually_named_like_the_workspace_still_works(ws):
    # Only a leading path SEGMENT is stripped, and only when something follows it.
    assert ws.rel(ws._safe_path("praxis-workspace.txt")) == "praxis-workspace.txt"
    assert ws.rel(ws._safe_path("notes/praxis-workspace/a.txt")) == os.path.join("notes", "praxis-workspace", "a.txt")


def test_escapes_are_still_refused(ws):
    for bad in ["../outside.txt", "/etc/passwd", "~/.ssh/id_rsa", "praxis-workspace/../../escape.txt"]:
        with pytest.raises(ValueError):
            ws._safe_path(bad)


def test_write_lands_in_one_workspace_not_two(ws):
    res = ws.write_file("praxis-workspace/txt/names.txt", "james\njane\n")
    assert res["path"] == os.path.join("txt", "names.txt")
    assert ws.list_files() == [os.path.join("txt", "names.txt")]
    assert "praxis-workspace" not in res["path"]


# ── exists() drives edit vs create ──────────────────────────────────────────

def test_exists_reports_files_and_tolerates_bad_paths(ws):
    ws.write_file("txt/name.txt", "sheikh Farooq\n")
    assert ws.exists("txt/name.txt") is True
    assert ws.exists("praxis-workspace/txt/name.txt") is True, "same file via a redundant prefix"
    assert ws.exists("txt/names.txt") is False
    assert ws.exists("../outside.txt") is False, "an escaping path is absent, not an exception"


def test_listing_surfaces_the_real_name_for_the_prompt(ws):
    ws.write_file("txt/name.txt", "sheikh Farooq\n")
    ws.write_file("pythoncode/rng.py", "print(1)\n")
    listing = ws.list_files()
    assert os.path.join("txt", "name.txt") in listing
    assert not any("names.txt" in f for f in listing)


def test_read_then_overwrite_preserves_the_edit(ws):
    ws.write_file("txt/name.txt", "sheikh Farooq\n")
    assert ws.read_file("txt/name.txt") == "sheikh Farooq\n"
    res = ws.write_file("txt/name.txt", "james\njane\n")
    assert res["overwrote"] is True
    assert ws.read_file("txt/name.txt") == "james\njane\n"
    assert len(ws.list_files()) == 1, "an edit must not leave a second file behind"


# ── extractor: listing in the prompt, and revision of existing content ──────

from actions.extractor import ActionExtractor


class _StubLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []
        self.claude_client = None
    def invoke(self, prompt, model_choice="auto"):
        self.prompts.append(prompt)
        return self.reply
    def get_active_model_name(self, choice="auto"):
        return "stub"


def _extractor(reply):
    ex = ActionExtractor.__new__(ActionExtractor)
    ex.llm = _StubLLM(reply)
    return ex


def test_existing_files_are_shown_to_the_model():
    ex = _extractor("FILENAME: txt/name.txt\nCONTENT:\njames\njane\n")
    out = ex.extract_file_task("add james and jane to the txt folder", "auto",
                               ["txt/name.txt", "pythoncode/rng.py"])
    prompt = ex.llm.prompts[0]
    assert "- txt/name.txt" in prompt and "- pythoncode/rng.py" in prompt
    assert "reuse that exact path" in prompt
    assert out["path"] == "txt/name.txt"


def test_no_listing_means_no_empty_section():
    ex = _extractor("FILENAME: a.py\nCONTENT:\nprint(1)\n")
    ex.extract_file_task("write a.py", "auto", [])
    assert "already exist in the workspace" not in ex.llm.prompts[0]


def test_revision_prompt_carries_the_current_bytes():
    ex = _extractor("james\njane\n")
    body = ex.revise_file_content("add james and jane, remove the older names",
                                  "txt/name.txt", "sheikh Farooq\n", "auto")
    prompt = ex.llm.prompts[0]
    # Assert on what the prompt must CONTAIN, not on its exact wording, so rewording
    # the prompt does not break the test.
    assert "sheikh Farooq" in prompt, "the model must see what it is editing"
    assert "txt/name.txt" in prompt, "and which file it is"
    assert "remove the older names" in prompt, "and the instruction verbatim"
    assert "complete new file" in prompt.lower(), "and be told to return the whole file"
    assert body == "james\njane\n"
    assert "sheikh Farooq" not in body


def test_revision_strips_markdown_fences_and_rejects_empty():
    ex = _extractor("```\njames\njane\n```")
    assert ex.revise_file_content("q", "a.txt", "old", "auto") == "james\njane\n"
    ex = _extractor("   \n  ")
    with pytest.raises(ValueError):
        ex.revise_file_content("q", "a.txt", "old", "auto")


def test_both_paths_end_with_exactly_one_newline():
    ex = _extractor("FILENAME: a.txt\nCONTENT:\njames\njane\n\n\n")
    assert ex.extract_file_task("write a.txt", "auto")["content"] == "james\njane\n"
    ex = _extractor("james\njane\n\n\n")
    assert ex.revise_file_content("q", "a.txt", "old", "auto") == "james\njane\n"


# ── deterministic target resolution (model names files unreliably) ──────────

from actions.workspace import resolve_existing_target, _stem_key

EXISTING = ["txt/name.txt", "pythoncode/rng.py", "pythoncode/random.py", "ishan/my_name.txt"]
EDIT_Q = "go the folder txt and add names james and jane there removing the older names there"


def test_the_reported_case_resolves_to_the_real_file():
    assert resolve_existing_target("txt/names.txt", EXISTING, EDIT_Q) == "txt/name.txt"


def test_exact_match_always_wins():
    assert resolve_existing_target("txt/name.txt", EXISTING, EDIT_Q) == "txt/name.txt"
    assert resolve_existing_target("pythoncode/rng.py", EXISTING, EDIT_Q) == "pythoncode/rng.py"


def test_creation_requests_are_never_redirected():
    create_q = "write a python file named names.py that prints a list of names"
    assert resolve_existing_target("txt/names.txt", EXISTING, create_q) == "txt/names.txt"
    assert resolve_existing_target("txt/names.txt", EXISTING, "") == "txt/names.txt"


def test_near_match_only_within_the_same_folder():
    # ishan/my_name.txt must not capture an edit aimed at the txt folder
    assert resolve_existing_target("txt/my_name.txt", EXISTING, EDIT_Q) == "txt/name.txt", "falls back to the folder's only file"
    assert resolve_existing_target("notes/name.txt", EXISTING, EDIT_Q) == "notes/name.txt", "unknown folder: create it"


def test_ambiguous_folder_is_left_alone():
    # pythoncode has several files and none is a variant of "script.py": do not guess
    assert resolve_existing_target("pythoncode/script.py", EXISTING, EDIT_Q) == "pythoncode/script.py"


def test_plural_punctuation_and_case_variants_collapse():
    assert _stem_key("names.txt") == _stem_key("name.txt")
    assert _stem_key("My_Name.TXT") == _stem_key("myname.txt")
    assert _stem_key("notes.txt") != _stem_key("name.txt")
    assert _stem_key("bus.txt") == _stem_key("bus.txt"), "short stems keep their s"


def test_extension_must_still_match():
    assert resolve_existing_target("pythoncode/rng.txt", EXISTING, EDIT_Q) == "pythoncode/rng.txt"


def test_empty_inputs_are_safe():
    assert resolve_existing_target("a.txt", [], EDIT_Q) == "a.txt"
    assert resolve_existing_target("", EXISTING, EDIT_Q) == ""


def test_revision_strips_echoed_prompt_scaffolding():
    # Observed from qwen2.5:3b: it appends the prompt's own section markers.
    for tail in ["\nEND UPDATED CONTENT", "\n--- END UPDATED CONTENT ---",
                 "\nUPDATED CONTENT:", "\n--- END EXAMPLE ---\nEND UPDATED CONTENT"]:
        ex = _extractor("james\njane" + tail)
        assert ex.revise_file_content("q", "a.txt", "old", "auto") == "james\njane\n", tail


def test_revision_keeps_content_that_merely_mentions_those_words():
    ex = _extractor("the report is about UPDATED CONTENT in general\nsecond line")
    out = ex.revise_file_content("q", "a.txt", "old", "auto")
    assert out == "the report is about UPDATED CONTENT in general\nsecond line\n"

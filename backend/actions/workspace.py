import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)


# Words that mean "change what is already there" rather than "make something new".
_EDIT_INTENT_RE = re.compile(
    r"\b(edit|update|modify|change|replace|remove|delete|rename|append|add to|fix|"
    r"rewrite|revise|amend|clear|empty|overwrite)\b|\bremoving\b|\breplacing\b",
    re.I,
)


def _stem_key(name: str) -> str:
    """Normalise a filename for near-match comparison: lowercase, drop non-alphanumerics,
    and drop a trailing plural 's'. So name.txt, names.txt and Name.TXT all collapse
    together, while name.txt and notes.txt do not."""
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^a-z0-9]", "", stem.lower())
    if len(stem) > 3 and stem.endswith("s"):
        stem = stem[:-1]
    return f"{stem}{ext.lower()}"


def resolve_existing_target(requested: str, existing: list[str], query: str = "") -> str:
    """
    Map the model's requested path onto a file that already exists, when the user
    clearly meant that file.

    A small model asked to "add names to the txt folder, removing the older ones"
    reliably invents txt/names.txt next to the real txt/name.txt, so the edit lands
    in a brand-new file and the old content survives. Matching by hand is not
    something a 3B model does dependably, so it is decided here instead:

      - exact hit wins;
      - otherwise, only when the instruction actually implies editing, look in the
        same directory for a singular/plural or punctuation variant of the name;
      - failing that, if the named directory holds exactly one file, that is the
        file the user meant.

    Anything else returns `requested` unchanged, so "write a new script" still
    creates a new file.
    """
    requested = (requested or "").strip()
    if not requested or not existing:
        return requested
    norm = {f.replace("\\", "/"): f for f in existing}
    req = requested.replace("\\", "/")
    if req in norm:
        return norm[req]
    if not _EDIT_INTENT_RE.search(query or ""):
        return requested

    req_dir = os.path.dirname(req)
    req_key = _stem_key(os.path.basename(req))
    siblings = [f for f in norm if os.path.dirname(f) == req_dir]

    for f in siblings:
        if _stem_key(os.path.basename(f)) == req_key:
            logger.info(f"Resolved '{requested}' to the existing '{f}' (near-match in the same folder).")
            return f
    if len(siblings) == 1:
        logger.info(f"Resolved '{requested}' to the only file in '{req_dir or '.'}': '{siblings[0]}'.")
        return siblings[0]
    return requested


class WorkspaceAgent:
    """
    A file agent confined to a single workspace directory.

    This is the highest-privilege capability in Praxis — it writes real files — so it is
    deliberately boxed in:
      - Every path is resolved and checked to stay INSIDE the workspace dir. A model that
        emits "../../etc/passwd" or "/home/user/.ssh/id_rsa" is rejected; it can only ever
        touch praxis-workspace/.
      - There is no shell. It can create/edit/read/list files and open them in VS Code —
        nothing else. It cannot run commands, delete outside the box, or reach the system.
      - Like the messaging actions, it only ever DRAFTS. The orchestrator produces a draft;
        a human approves before anything is written to disk.

    Files land on the host (the workspace is a bind-mounted / real folder), so "make this
    file" produces something you can actually open and use.
    """

    def __init__(self, workspace_dir: str = None):
        if workspace_dir is None:
            workspace_dir = os.getenv("WORKSPACE_DIR", "praxis-workspace")
        self.workspace_dir = os.path.realpath(workspace_dir)
        self._init_error = None
        try:
            os.makedirs(self.workspace_dir, exist_ok=True)
            logger.info(f"Workspace agent ready at {self.workspace_dir}")
        except Exception as e:
            self._init_error = f"Workspace dir not writable: {e}"
            logger.error(self._init_error)

    @property
    def available(self) -> bool:
        return self._init_error is None

    # ── path safety ───────────────────────────────────────────────────────────
    def _safe_path(self, rel_path: str) -> str:
        """
        Resolve `rel_path` under the workspace and refuse anything that escapes it.
        Raises ValueError on traversal / absolute paths pointing outside.
        """
        if not rel_path or not rel_path.strip():
            raise ValueError("No filename given.")
        raw = rel_path.strip().replace("\\", "/")
        # A model told "write into praxis-workspace/txt/" often repeats the workspace
        # folder in the path it emits, which would nest a second copy inside the real
        # one (praxis-workspace/praxis-workspace/txt/...). The workspace root is implied
        # by this agent, so strip a redundant leading copy of its own name.
        ws_name = os.path.basename(self.workspace_dir)
        while True:
            head, _, tail = raw.partition("/")
            if head in (ws_name, ".") and tail.strip():
                logger.info(f"Stripped redundant '{head}/' prefix from model path: {rel_path!r}")
                raw = tail.strip()
                continue
            break
        # Reject absolute / home paths outright — a model emitting /etc/passwd or
        # ~/.ssh/id_rsa clearly intends to escape, so refuse rather than silently
        # rewriting it into the workspace.
        if os.path.isabs(raw) or raw[0] in ("/", "\\", "~"):
            raise ValueError(
                f"'{rel_path}' must be a relative path inside praxis-workspace/."
            )
        base = self.workspace_dir
        target = os.path.realpath(os.path.join(base, raw))
        # Final backstop: catches ../ traversal after resolution.
        if target != base and not target.startswith(base + os.sep):
            raise ValueError(
                f"'{rel_path}' points outside the workspace. Files can only be written "
                "inside praxis-workspace/."
            )
        return target

    def rel(self, abs_path: str) -> str:
        """workspace-relative display path"""
        return os.path.relpath(abs_path, self.workspace_dir)

    # ── operations ─────────────────────────────────────────────────────────────
    def write_file(self, rel_path: str, content: str) -> dict:
        if not self.available:
            raise RuntimeError(self._init_error)
        target = self._safe_path(rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        existed = os.path.exists(target)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Wrote {'(overwrote) ' if existed else ''}{self.rel(target)} ({len(content)} chars)")
        return {"path": self.rel(target), "abs_path": target, "overwrote": existed, "bytes": len(content)}

    def exists(self, rel_path: str) -> bool:
        """True if the path resolves to an existing file inside the workspace."""
        try:
            return os.path.isfile(self._safe_path(rel_path))
        except ValueError:
            return False

    def read_file(self, rel_path: str) -> str:
        if not self.available:
            raise RuntimeError(self._init_error)
        target = self._safe_path(rel_path)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"{rel_path} does not exist in the workspace.")
        with open(target, encoding="utf-8", errors="replace") as f:
            return f.read()

    def list_files(self) -> list[str]:
        if not self.available:
            return []
        out = []
        for root, _dirs, files in os.walk(self.workspace_dir):
            for name in files:
                out.append(self.rel(os.path.join(root, name)))
        return sorted(out)

    def open_in_editor(self, rel_path: str) -> str:
        """
        Open a workspace file in VS Code via the `code` CLI. Only works when the backend
        runs natively (with `code` on PATH) — inside Docker there's no host GUI, so this
        no-ops with a note. The file is still written either way.
        """
        target = self._safe_path(rel_path)
        code_bin = shutil.which("code")
        if not code_bin:
            return "VS Code CLI ('code') not available here — run the backend natively to auto-open."
        try:
            subprocess.Popen([code_bin, target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opened {self.rel(target)} in VS Code."
        except Exception as e:
            return f"Could not launch VS Code: {e}"

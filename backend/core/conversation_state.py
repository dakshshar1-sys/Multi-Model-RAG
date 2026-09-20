"""
Per-conversation state for the orchestrator.

Until now the orchestrator kept "the last uploaded image" as one attribute on a
process-wide singleton, so two tabs, or two users, shared it: an image uploaded
in one conversation could be reused to answer a follow-up in another. State that
belongs to a conversation now lives here, keyed by the identifier the frontend
already assigns to each conversation. A request without an identifier gets the
"default" bucket, which preserves the old single-user behaviour.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

DEFAULT_ID = "default"
IDLE_TTL_S = 2 * 3600.0     # forget a conversation's state after two idle hours
MAX_CONVERSATIONS = 500     # cap memory; evict the least recently used beyond this


@dataclass
class ConversationState:
    conversation_id: str
    last_image_context: str | None = None
    last_image_context_at: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)   # wall clock, for idle expiry
    last_tick: int = 0                                          # access counter, for LRU order

    def set_image(self, context: str) -> None:
        self.last_image_context = context
        self.last_image_context_at = time.monotonic()

    def clear_image(self) -> None:
        self.last_image_context = None
        self.last_image_context_at = 0.0

    def image_age_s(self) -> float:
        return time.monotonic() - self.last_image_context_at if self.last_image_context else float("inf")


class ConversationStore:
    def __init__(self, idle_ttl_s: float = IDLE_TTL_S, max_conversations: int = MAX_CONVERSATIONS):
        self._states: dict[str, ConversationState] = {}
        self._lock = threading.Lock()
        self.idle_ttl_s = idle_ttl_s
        self.max_conversations = max_conversations
        self._tick = 0

    @staticmethod
    def normalize(conversation_id: str | None) -> str:
        cid = (conversation_id or "").strip()
        return cid[:128] if cid else DEFAULT_ID

    def get(self, conversation_id: str | None) -> ConversationState:
        """The state for a conversation, created on first sight; touches last_seen."""
        cid = self.normalize(conversation_id)
        now = time.monotonic()
        with self._lock:
            st = self._states.get(cid)
            if st is None:
                st = self._states[cid] = ConversationState(cid)
            self._tick += 1
            st.last_seen, st.last_tick = now, self._tick
            self._evict(now, keep=cid)
            return st

    def _evict(self, now: float, keep: str) -> None:
        """Drop idle conversations, then least-recently-used ones beyond the cap.
        Runs after the current conversation is inserted and touched, so the cap
        is enforced on the real count and the current one is never the victim."""
        for cid in [c for c, st in self._states.items() if c != keep and now - st.last_seen > self.idle_ttl_s]:
            del self._states[cid]
        excess = len(self._states) - self.max_conversations
        if excess > 0:
            victims = sorted((st for c, st in self._states.items() if c != keep), key=lambda st: st.last_tick)[:excess]
            for st in victims:
                del self._states[st.conversation_id]

    def __len__(self) -> int:
        return len(self._states)

"""Session sticky routing manager.

In-memory only: state is lost on restart (by design).
Session cleanup is performed periodically by the scheduler to
prevent unbounded memory growth.
"""

from typing import Optional
from datetime import datetime, timedelta


class SessionManager:
    """Track session_id -> provider_slug mappings.

    In-memory only: state is lost on restart (by design).
    """

    def __init__(self):
        self._sessions: dict[str, str] = {}  # session_id -> provider_slug
        self._last_access: dict[str, datetime] = {}

    def set_provider(self, session_id: str, provider: str) -> None:
        """Assign a provider to a session."""
        self._sessions[session_id] = provider
        self._last_access[session_id] = datetime.utcnow()

    def get_provider(self, session_id: str) -> Optional[str]:
        """Get the provider assigned to a session, or None if unknown."""
        if session_id in self._sessions:
            self._last_access[session_id] = datetime.utcnow()
            return self._sessions[session_id]
        return None

    def has_session(self, session_id: str) -> bool:
        """Check if session is known."""
        return session_id in self._sessions

    def remove_session(self, session_id: str) -> None:
        """Remove a session mapping."""
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def get_all_sessions(self) -> dict[str, str]:
        """Get all session mappings (for debugging)."""
        return dict(self._sessions)

    def cleanup_stale(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours since last access.

        Returns the number of sessions removed.
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        stale = [
            sid for sid, last in self._last_access.items()
            if last < cutoff
        ]
        for sid in stale:
            self.remove_session(sid)
        if stale:
            from logging import getLogger
            getLogger(__name__).info(
                "Cleaned up %d stale session(s) (older than %dh)",
                len(stale), max_age_hours,
            )
        return len(stale)

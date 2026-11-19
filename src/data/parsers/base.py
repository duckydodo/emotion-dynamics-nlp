from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.data.schema import CanonicalConversation


class BaseConversationParser(ABC):
    """Abstract base class for all conversation format parsers."""

    @abstractmethod
    def can_parse(self, raw_input: Any) -> bool:
        """Heuristic check to determine if the input matches this parser format."""
        pass

    @abstractmethod
    def parse(
        self,
        raw_input: Any,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CanonicalConversation:
        """Parse raw input into a validated CanonicalConversation."""
        pass

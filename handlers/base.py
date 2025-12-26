# handlers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

from core.yaml_parser import Stage


class BaseHandler(ABC):
    """Base interface for all stage handlers."""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the stage, reading and updating the context."""
        raise NotImplementedError

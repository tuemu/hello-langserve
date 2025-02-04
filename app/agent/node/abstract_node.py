from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.state import AgentState


class AbstractNode(ABC):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @abstractmethod
    def run(self, state: AgentState) -> dict[str, Any]:
        pass

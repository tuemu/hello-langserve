from typing import Any

from langchain_community.tools import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from app.agent.node.abstract_node import AbstractNode
from app.agent.state import AgentState


class Executor(AbstractNode):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]
        self.base_agent = create_react_agent(self.llm, self.tools)

    def run(self, state: AgentState) -> dict[str, Any]:
        print(f"Executor is run with state: {state}")
        current_task_index = state.current_task_index
        task = state.tasks[current_task_index]
        result = self.base_agent.invoke(
            {
                "messages": [
                    (
                        "system",
                        (
                            f"あなたは{task.role.name}です。\n"
                            f"説明: {task.role.description}\n"
                            f"主要なスキル: {', '.join(task.role.key_skills)}\n"
                            "あなたの役割に基づいて、与えられたタスクを最高の能力で遂行してください。\n"
                            f"但し、言語コード:<{state.language}>の言語で必ず回答すること。"
                        ),
                    ),
                    (
                        "human",
                        f"以下のタスクを実行してください：\n\n{task.description}",
                    ),
                ]
            }
        )
        result = result["messages"][-1].content
        return {
            "results": [result],
            "executed_task_numbers": [current_task_index]
        }

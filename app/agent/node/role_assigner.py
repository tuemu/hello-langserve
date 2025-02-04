from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.agent.node.abstract_node import AbstractNode
from app.agent.state import TasksWithRoles, AgentState


class RoleAssigner(AbstractNode):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.llm = llm.with_structured_output(TasksWithRoles)

    def run(self, state: AgentState) -> dict[str, Any]:
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "あなたは創造的な役割設計の専門家です。与えられたタスクに対して、ユニークで適切な役割を生成してください。"
                    ),
                ),
                (
                    "human",
                    (
                        "タスク:\n{tasks}\n\n"
                        "これらのタスクに対して、以下の指示に従って役割を割り当ててください：\n"
                        "1. 各タスクに対して、独自の創造的な役割を考案してください。既存の職業名や一般的な役割名にとらわれる必要はありません。\n"
                        "2. 役割名は、そのタスクの本質を反映した魅力的で記憶に残るものにしてください。\n"
                        "3. 各役割に対して、その役割がなぜそのタスクに最適なのかを説明する詳細な説明を提供してください。\n"
                        "4. その役割が効果的にタスクを遂行するために必要な主要なスキルやアトリビュートを3つ挙げてください。\n\n"
                        "創造性を発揮し、タスクの本質を捉えた革新的な役割を生成してください。"
                    ),
                ),
            ],
        )
        chain = prompt | self.llm
        tasks_with_roles = chain.invoke(
            {
                "tasks": "\n".join([task.description for task in state.tasks])
            }
        )

        return {
            "tasks": tasks_with_roles.tasks
        }



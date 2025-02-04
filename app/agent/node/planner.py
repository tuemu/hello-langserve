from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.node.abstract_node import AbstractNode
from app.agent.single_path_plan_generation.model.decomposed_tasks import DecomposedTasks
from app.agent.single_path_plan_generation.query_decomposer import QueryDecomposer
from app.agent.state import Task, AgentState


class Planner(AbstractNode):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.query_decomposer = QueryDecomposer(llm=llm)

    def run(self, state: AgentState) -> dict[str, Any]:
        print(f"Planner is run with state: {state}")

        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(query=state.query)
        tasks = [Task(description=task) for task in decomposed_tasks.values]
        return {
            "tasks": tasks
        }

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.node.executor import Executor
from app.agent.node.planner import Planner
from app.agent.node.reporter import Reporter
from app.agent.node.role_assigner import RoleAssigner
from app.agent.state import AgentState


class RoleBasedCooperation:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.planner = Planner(llm=llm)
        self.role_assigner = RoleAssigner(llm=llm)
        self.executor = Executor(llm=llm)
        self.reporter = Reporter(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self.planner.run)
        workflow.add_node("role_assigner", self.role_assigner.run)
        workflow.add_node("executor", self.executor.run)
        workflow.add_node("reporter", self.reporter.run)

        workflow.set_entry_point("planner")

        workflow.add_edge("planner", "role_assigner")
        workflow.add_edge("role_assigner", "executor")
        workflow.add_conditional_edges(
            "executor",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "executor", False: "reporter"},
        )

        workflow.add_edge("reporter", END)

        return workflow.compile()

    # def _plan_tasks(self, state: AgentState) -> dict[str, Any]:
    #     tasks = self.planner.run(query=state.query)
    #     return {"tasks": tasks}

    # def _assign_roles(self, state: AgentState) -> dict[str, Any]:
    #     tasks_with_roles = self.role_assigner.run(tasks=state.tasks)
    #     return {"tasks": tasks_with_roles}

    # def _execute_task(self, state: AgentState) -> dict[str, Any]:
    #     current_task = state.tasks[state.current_task_index]
    #     result = self.executor.run(task=current_task)
    #     return {
    #         "results": [result],
    #         "current_task_index": state.current_task_index + 1,
    #     }

    # def _generate_report(self, state: AgentState) -> dict[str, Any]:
    #     report = self.reporter.run(query=state.query, results=state.results)
    #     return {"final_report": report}

    # def run(self, query: str) -> str:
    #     initial_state = AgentState(query=query)
    #     final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
    #     return final_state["final_report"]

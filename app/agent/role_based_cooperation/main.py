import copy

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

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

    @staticmethod
    def _create_send_object(state: AgentState, index: int) -> Send:
        updated_state = copy.deepcopy(state)
        updated_state.current_task_index = index
        return Send('executor', updated_state)

    def _routing_parallel_node(self, state: AgentState) -> list[Send]:
        return [self._create_send_object(state, index) for index in range(len(state.tasks))]

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self.planner.run)
        workflow.add_node("role_assigner", self.role_assigner.run)
        workflow.add_node("executor", self.executor.run)
        workflow.add_node("reporter", self.reporter.run)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "role_assigner")
        workflow.add_conditional_edges('role_assigner', self._routing_parallel_node, ['executor'])
        workflow.add_edge("executor", "reporter")
        workflow.add_edge("reporter", END)

        return workflow.compile()

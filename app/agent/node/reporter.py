from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.node.abstract_node import AbstractNode
from app.agent.state import AgentState


class Reporter(AbstractNode):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.llm = llm

    def run(self, state: AgentState) -> dict[str, Any]:

        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "あなたは総合的なレポート作成の専門家です。複数の情報源からの結果を統合し、洞察力に富んだ包括的なレポートを作成する能力があります。"
                    ),
                ),
                (
                    "human",
                    (
                        "タスク: 以下の情報に基づいて、包括的で一貫性のある回答を作成してください。\n"
                        "要件:\n"
                        "1. 提供されたすべての情報を統合し、よく構成された回答にしてください。\n"
                        "2. 回答は元のクエリに直接応える形にしてください。\n"
                        "3. 各情報の重要なポイントや発見を含めてください。\n"
                        "4. 最後に結論や要約を提供してください。\n"
                        "5. 回答は詳細でありながら簡潔にし、250〜300語程度を目指してください。\n"
                        "6. 回答は日本語で行ってください。\n\n"
                        "ユーザーの依頼: {query}\n\n"
                        "収集した情報:\n{results}"
                    ),
                ),
            ],
        )
        chain = prompt | self.llm | StrOutputParser()
        report = chain.invoke(
            {
                "query": state.query,
                "results": "\n\n".join(
                    f"Info {i+1}:\n{result}" for i, result in enumerate(state.results)
                ),
            }
        )
        return {
            "final_report": report
        }

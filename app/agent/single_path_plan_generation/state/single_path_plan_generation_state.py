import operator
from typing import Annotated

from pydantic import BaseModel, Field


class SinglePathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(
        default="", description="最適化されたレスポンス定義"
    )
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")

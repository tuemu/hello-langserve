import operator
from typing import Annotated

from pydantic import BaseModel, Field


class Role(BaseModel):
    name: str = Field(..., description="役割の名前")
    description: str = Field(..., description="役割の詳細な説明")
    key_skills: list[str] = Field(..., description="この役割に必要な主要なスキルや属性")


class Task(BaseModel):
    description: str = Field(..., description="タスクの説明")
    role: Role = Field(default=None, description="タスクに割り当てられた役割")


class TasksWithRoles(BaseModel):
    tasks: list[Task] = Field(..., description="役割が割り当てられたタスクのリスト")


class AgentState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    language: str = Field(default="en", description="回答すべき言語の言語コード")
    tasks: list[Task] = Field(
        default_factory=list, description="実行するタスクのリスト"
    )
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    executed_task_numbers: Annotated[list[int], operator.add] = Field(
        default_factory=list, description="実行済みタスクのタスク番号"
    )
    final_report: str = Field(default="", description="最終的な出力結果")

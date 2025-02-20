from pydantic import BaseModel, Field


class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_items=1,
        max_items=3,
        description="1~3個に分解されたタスク",
    )
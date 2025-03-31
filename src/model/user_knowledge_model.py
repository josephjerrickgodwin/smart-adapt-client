from typing import Literal

from pydantic import BaseModel


class UserKnowledgeModel(BaseModel):
    user_role: Literal['admin', 'user']
    knowledge_ids: list

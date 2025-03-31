from typing import List, Dict, Optional

from pydantic import BaseModel


class CompletionRequest(BaseModel):
    user_id: str
    messages: List[Dict[str, str]]
    stream: bool
    knowledge_ids: Optional[List[str]] = None
    additional_kwargs: Optional[Dict] = None

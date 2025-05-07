from pydantic import BaseModel
from typing import Optional


class MsgPayload(BaseModel):
    msg_id: Optional[int]
    msg_name: str

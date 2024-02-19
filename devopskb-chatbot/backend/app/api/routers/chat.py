from typing import List

from llama_index.core.base.base_query_engine import BaseQueryEngine
from app.engine.index import get_index_and_query_engine
from fastapi import APIRouter, Depends, HTTPException, Request, status
from llama_index.core.llms import MessageRole
from pydantic import BaseModel

chat_router = r = APIRouter()


class _Message(BaseModel):
    role: MessageRole
    content: str


class _ChatData(BaseModel):
    messages: List[_Message]


@r.post("")
def chat(
    request: Request,
    data: _ChatData,
    query_engine: BaseQueryEngine = Depends(get_index_and_query_engine),
):
    # check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    lastMessage = data.messages.pop()
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )

    # query chat engine
    response = query_engine.query(lastMessage.content)

    return response.response

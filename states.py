# So now we are creating a graph

import os

#1 typed Dict

from typing import TypedDict

class mystate(TypedDict):
    topic : str
    summary : str
    score : int

#2 pydantic approach help to data validation

from pydantic import BaseModel ,field_validator

class mystate(BaseModel):
    topic :str
    score :int
    summary :str = ""

    @field_validator
    def score_positive(cls ,v):
        if v < 0 :
            raise ValueError ("score must be positive")


#python dataclassess
#standard python dataclass but it is used very rarely

from dataclasses import dataclass ,field

@dataclass
class mystate:
    topic : str =""
    score : int
    summary : str = ""
    messages : list =field(default_factory=list)

from langgraph.graph import MessagesState

class mystate(MessagesState):
    user_name : str
    language : str
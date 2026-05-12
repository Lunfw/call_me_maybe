from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Any, Optional


class GenerationState(Enum):
    START = 0
    NAME_KEY = 1
    NAME_VALUE = 2
    PARAMS_KEY = 3
    PARAMS_OPEN = 4
    ARG_KEY = 5
    ARG_VALUE = 6
    PARAMS_CLOSE = 7
    END = 8


class FunctionParameter(BaseModel):
    type: Optional[str] = Field(default=None)


class Prompt(BaseModel):
    prompt: Optional[str] = Field(default=None)


class FunctionCall(BaseModel):
    prompt: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class FunctionDefinition(BaseModel):
    name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    parameters: Dict[str, FunctionParameter] = Field(default_factory=dict)
    returns: Optional[FunctionParameter] = Field(default=None)

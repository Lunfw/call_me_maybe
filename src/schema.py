from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


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

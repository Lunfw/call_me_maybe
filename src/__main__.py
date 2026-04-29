from llm_sdk import Small_LLM_Model
from sys import argv, exit
from transformers import TextStreamer
from typing import List
from json import load
from pydantic import BaseModel, Field


class Parser:
    def __init__(self, receipt: str) -> None:
        self.parse(receipt)

    @staticmethod
    def parse(receipt: str) -> None:
        try:
            temp: List[str] = receipt.split(" ")


class Main(Basemodel):
    parser: Parser = Field(default_factory=Parser())


def main(argv: list[str]):
    if (len(argv) < 2 or argv[1] == ''):
        print("Usage: make run <prompt>")
        exit(1)
    Main().parser.parse(argv[:1])

if (__name__ == "__main__"):
    main(argv)

from llm_sdk import Small_LLM_Model
from sys import argv, exit, stderr
from typing import Dict, Any
from json import load
from argparse import ArgumentParser
from src.colors import Format
from os import path


class Parser:
    @staticmethod
    def parse(receipt: str) -> Dict[str, Any]:
        parser = ArgumentParser()
        parser.add_argument('--functions_definition', required=True)
        parser.add_argument('--input', required=True)
        parser.add_argument('--output', default='data/output/output.json')
        args = parser.parse_args()
        return (vars(args))

    @staticmethod
    def if_exist(receipt: Dict[str, str]) -> Dict[str, Any]:
        for i in receipt.keys():
            try:
                if (path.exists(receipt[i])):
                    continue
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                raise FileNotFoundError('file not found -> ' + receipt[i])
        return (receipt)


class Loader:
    @staticmethod
    def json_load(file: str) -> Dict[str, Any]:
        return (load(open(file)))

    @staticmethod
    def json_get(data: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in data.items():
            if (key != 'output'):
                data[key] = Loader().json_load(value)
        return (data)


class Main:
    def __init__(self, receipt: str) -> None:
        self.parsed: Dict[str, Any] = Loader().json_get(
                Parser.if_exist(Parser.parse(receipt))
                )
        for i in self.parsed:
            print(f'\n{i} -> {self.parsed[i]}')


if (__name__ == "__main__"):
    if (len(argv) < 2 or argv[1] == ''):
        print('Usage: make run <prompt>')
        exit(1)
    try:
        Main(' '.join(argv[1:]))
    except Exception as e:
        print(Format().colored(f'\nError: {e}', 'RED'), file=stderr)
    exit(0)

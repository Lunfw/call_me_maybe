from llm_sdk import Small_LLM_Model
from sys import argv, exit, stderr
from typing import Dict, Any, Iterator, List, Union
from json import load
from argparse import ArgumentParser
from src.schema import FunctionDefinition, Prompt
from src.translator import Translator
from src.colors import Format
from os import path
from pydantic import ValidationError


class Parser:
    @staticmethod
    def parse() -> Dict[str, Any]:
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
    def json_load(file: str) -> List[Any]:
        with open(file) as f:
            return load(f)

    @staticmethod
    def json_get(data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if (key != 'output'):
                yield Loader.json_parse(Loader.json_load(value), key)

    @staticmethod
    def json_parse(data: List[Any],
                   key: str) -> Union[List[FunctionDefinition], List[Prompt]]:
        result: List[Any] = []
        try:
            if (key == 'functions_definition'):
                result = [FunctionDefinition(**i) for i in data]
            elif (key == 'input'):
                result = [Prompt(**i) for i in data]
            else:
                raise ValueError('no such key -> ' + key)
        except (ValueError, KeyError, ValidationError) as e:
            raise ValueError(str(e))
        return (result)


class Main:
    def __init__(self) -> None:
        self.functions, self.prompts = Loader.json_get(
                Parser.if_exist(Parser.parse())
                )
        self.model = Small_LLM_Model()
        self.translated = Translator(self.model,
                                     self.model.get_path_to_vocab_file())

    def debug(self) -> None:
        print('')
        Format().draw_margin()
        for i in self.functions:
            print(Format.colored(f'\nFunction: {i.name}', 'GREEN'))
            print(i.description)
            print(f'Parameters: {i.parameters}')
        print('')
        for i in self.prompts:
            print(Format.colored(f'Prompt: {i.prompt}', 'GREEN'))


if (__name__ == "__main__"):
    if (len(argv) < 2 or argv[1] == ''):
        print('Usage: make run <prompt>')
        exit(1)
    try:
        Main()
    except Exception as e:
        print(Format.colored(f'\nError: {e}', 'RED'), file=stderr)
    exit(0)

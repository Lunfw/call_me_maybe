from llm_sdk import Small_LLM_Model
from sys import argv, exit, stderr
from typing import Dict, Any, Iterator, List, Union, Tuple
from json import load, dump, loads
from argparse import ArgumentParser
from src.schema import FunctionDefinition, Prompt
from src.translator import Translator
from src.colors import Format
from os import path, makedirs
from pydantic import ValidationError


class Parser:
    @staticmethod
    def parse() -> Dict[str, Any]:
        parser = ArgumentParser()
        parser.add_argument('--functions_definition', required=True)
        parser.add_argument('--input', required=True)
        parser.add_argument('--stdin', default=False)
        parser.add_argument(
                '--output',
                default='function_calling_results.json'
                )
        parser.add_argument('--max_token', default=200)
        args = parser.parse_args()
        return (vars(args))

    @staticmethod
    def if_exist(receipt: Dict[str, str]) -> Dict[str, Any]:
        exclude: Tuple[str] = ('max_token', 'output', 'stdin')
        for i in receipt.keys():
            try:
                if (path.exists(receipt[i]) or i in exclude):
                    continue
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                raise FileNotFoundError('File not found -> ' + receipt[i])
        return (receipt)


class Loader:
    @staticmethod
    def json_load(file: str) -> List[Any]:
        with open(file) as f:
            return load(f)

    @staticmethod
    def json_get(data: Dict[str, Any]) -> None:
        exclude = ('max_token', 'output', 'stdin')
        for key, value in data.items():
            if (key not in exclude):
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
                raise ValueError('No such key -> ' + key)
        except (ValueError, KeyError, ValidationError) as e:
            raise ValueError(str(e))
        return (result)


class Main:
    def __init__(self) -> None:
        self.functions, self.prompts = Loader.json_get(
                Parser.if_exist(Parser.parse())
                )
        self.model = Small_LLM_Model()
        self.translated = Translator(
                self.model.get_path_to_vocab_file(), self.functions)
        self.run()

    def run(self) -> None:
        Format().draw_margin()
        results: List[str] = []
        max_tokens: int = int(Parser.parse()['max_token'])
        for prompt in self.prompts:
            print(Format.colored('\n\n│ PROMPT: ' + prompt.prompt, 'CYAN'))
            llm_json = loads(self.translated.generate(prompt.prompt,
                                             self.model,
                                             max_tokens)
                             )
            result = {
                    "prompt": prompt.prompt,
                    "name": llm_json['name'],
                    "parameters": llm_json['parameters']
            }
            results.append(result)
        if (not path.exists('data/output')):
            makedirs('data/output')
        file_dump: str = 'data/output/' + Parser.parse()['output']
        dump(results, open(file_dump, 'w'), indent=2)

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
    print(Format.colored('\n\n│ DONE!', 'GOLD'))
    exit(0)

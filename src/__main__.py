from llm_sdk import Small_LLM_Model
from sys import argv, exit, stderr, stdin
from typing import Dict, Any, List, Union, Tuple, Iterator
from json import load, dump, loads
from argparse import ArgumentParser
from src.schema import FunctionDefinition, Prompt
from src.translator import Translator
from src.colors import Format
from os import path, makedirs
from time import perf_counter
from pydantic import ValidationError
from termios import tcgetattr, tcsetattr, TCSADRAIN
from tty import setraw


class Parser:
    @staticmethod
    def parse() -> Dict[str, Any]:
        parser = ArgumentParser()
        parser.add_argument('--functions_definition', required=True)
        parser.add_argument('--input', required=True)
        parser.add_argument('--model', default='Qwen/Qwen3-0.6B')
        parser.add_argument(
                '--output',
                default='function_calling_results.json'
                )
        parser.add_argument('--max_token', default=2048)
        args = parser.parse_args()
        return (vars(args))

    @staticmethod
    def if_exist(receipt: Dict[str, str]) -> Dict[str, Any]:
        exclude: Tuple[Any] = ('max_token',
                               'output',
                               'model'
                               )
        for i in receipt.keys():
            try:
                if (path.exists(receipt[i]) or i in exclude):
                    continue
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                raise FileNotFoundError('File not found -> ' + receipt[i])
        return (receipt)

    @staticmethod
    def model_allowed(receipt: str) -> str:
        permits: Tuple[Any] = ('Qwen/Qwen3-0.6B',
                               'Qwen/Qwen3-0.6B-Base',
                               'Qwen/Qwen2.5-0.5B',
                               'HuggingFaceTB/SmolLM2-135M',
                               'HuggingFaceTB/SmolLM2-135M-Instruct',
                               'facebook/opt-125M'
                               )
        if (receipt in permits):
            return (receipt)

        if (receipt == 'all'):
            print(Format.colored('\n│ Available models:\n', 'YELLOW'))
            for i in range(len(permits)):
                print(Format.colored(f"│ [{i}] -> {permits[i]}", 'GREY'))
            while True:
                fd = stdin.fileno()
                old = tcgetattr(fd)
                setraw(fd)
                selected = stdin.read(1)
                tcsetattr(fd, TCSADRAIN, old)
                if (selected in [str(i) for i in range(len(permits))]):
                    return (permits[int(selected)])
                elif (selected == '\x03'):
                    raise KeyboardInterrupt('Process interrupted')
                else:
                    continue

        candidates: List[str] = [p for p in permits
                                 if receipt.lower() in p.lower()]

        if (len(candidates) == 1):
            return (candidates[0])

        if (not len(candidates)):
            raise ValueError('Model not found/not permitted -> ' + receipt)

        print(Format.colored(
            f"\n│ WARNING: {receipt} matches several models.", 'YELLOW')
              )
        print(Format.colored(f"│ You can choose one to use.", 'YELLOW'))
        print(Format.colored(f"│ Available models:\n", 'YELLOW'))

        for i in range(len(candidates)):
            print(Format.colored(f"│ [{i}] -> {candidates[i]}", 'GREY'))

        while True:
            fd = stdin.fileno()
            old = tcgetattr(fd)
            setraw(fd)
            selected = stdin.read(1)
            tcsetattr(fd, TCSADRAIN, old)

            if (selected in [str(i) for i in range(len(candidates))]):
                return (candidates[int(selected)])
            elif (selected == '\x03'):
                raise KeyboardInterrupt('Process interrupted')
            else:
                continue

class Loader:
    @staticmethod
    def json_load(file: str) -> Any:
        with open(file) as f:
            return load(f)

    @staticmethod
    def json_get(data: Dict[str, Any]) -> Iterator[Any]:
        exclude = ('max_token', 'output', 'model')
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
        args = Parser.parse()
        args['model'] = Parser.model_allowed(args['model'])
        self.debug(args['model'], args['input'], args['functions_definition'])
        self.functions, self.prompts = Loader.json_get(
                Parser.if_exist(args)
                )
        self.model = Small_LLM_Model(model_name=args['model'])
        start = perf_counter()
        self.translated = Translator(
                self.model.get_path_to_vocab_file(), self.functions)

        print(Format.colored('\n\n│ DONE!! Written to ' + self.run(), 'GOLD'))
        print(Format.colored(
            f'│ > (in {perf_counter() - start:.2f}s)', 'GREY')
              )

    def run(self) -> str:
        Format().draw_margin()
        results: List[Dict[str, Any]] = []
        max_tokens: int = int(Parser.parse()['max_token'])
        for prompt in self.prompts:
            print(Format.colored('\n\n│ PROMPT: ' + prompt.prompt, 'CYAN'))
            llm_json = loads(self.translated.generate(prompt.prompt,
                                                      self.functions,
                                                      self.model,
                                                      max_tokens))
            result: Dict[str, Any] = {
                    "prompt": prompt.prompt,
                    "name": llm_json['name'],
                    "parameters": llm_json['parameters']
            }
            results.append(result)
        name: str = Parser.parse()['output']
        if (not path.exists('data/output')):
            makedirs('data/output')
        if (Parser.parse()['output'].rfind('/') != -1):
            name = Parser.parse()['output'].split('/')[-1]
        file_dump: str = 'data/output/' + name
        dump(results, open(file_dump, 'w'), indent=2)
        return (file_dump)

    def debug(self, model: str, input_path: str, func_path: str) -> None:
        print('')
        print(Format.colored('│ MODEL: ' + model, 'GREEN'))
        print(Format.colored('│ PROMPTS: ' + input_path, 'GREEN'))
        print(Format.colored('│ FUNCTIONS: ' + func_path, 'GREEN'))
        print('')


if (__name__ == "__main__"):
    if (len(argv) < 2 or argv[1] == ''):
        print('Usage: make run <prompt>')
        exit(1)
    try:
        Main()
    except Exception as e:
        print(Format.colored(f'\nError: {e}', 'RED'), file=stderr)
    exit(0)

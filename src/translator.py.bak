from typing import Dict, Any, List, Iterator
from json import load
from src.colors import Format
from src.schema import GenerationState
from numpy import argmax
from time import sleep, perf_counter


class Translator:
    def __init__(self, vocab_path: str, funcs: List[Any]):
        self.vocab: Dict[str, int] = self.load_vocab(vocab_path)
        self.context: str = Translator.load_context('context.md', funcs)
        self.id_to_token: Dict[int, str] = {
                    v: k for k, v in self.vocab.items()
                }

    @staticmethod
    def load_context(context_path: str, functions: List[Any]) -> str:
        text: str = 'Available functions:'
        for i in functions:
            params: str = ', '.join(
                        f'{k}: {v.type}' for k, v in i.parameters.items()
                    )
            text += '\n- ' + i.name + f'({params}): ' + i.description
        with open(context_path) as f:
            return (f.read() + text + '<|im_end|>')

    @staticmethod
    def load_vocab(vocab_path: str) -> Dict[str, int]:
        with open(vocab_path) as f:
            return (load(f))

    def get_prompt(self, prompt: str) -> str:
        temp: List[str] = [
                self.context,
                '<|im_start|>user\n/no_think\n',
                prompt,
                '<|im_end|>\n<|im_start|>assistant\n'
                ]
        prompt = ''.join(temp)
        return (prompt)

    def generate(self,
                 prompt: str,
                 functions: List[Any],
                 model,
                 max_tokens: int) -> Dict[str, Any]:
        state: GenerationState = GenerationState.START
        input_ids: List[int] = model.encode(prompt)
        prompt_len: int = len(input_ids)
        selected: Any = None
        partial: str = ''

        for i in range(max_tokens):
            pass
        return ({})

    def get_token(self, expected: str) -> Iterator[int]:
        for token_string, token_id in self.vocab.items():
            if (expected.startswith(token_string)):
                yield token_id

    @staticmethod
    def get_matched(partial: str, expected: str) -> int:
        for length in range(len(expected), 0, -1):
            if (partial[-length:] == expected[:length]):
                return (length)
        return (0)

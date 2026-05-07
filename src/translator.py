from typing import Dict, Any, Optional, List
from json import load, loads
from src.colors import Format
from numpy import argmax


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
            params: str = ', '.join(f'{k}: {v.type}' for k, v in i.parameters.items())
            text += '\n- ' + i.name + f'({params}): ' + i.description
        with open(context_path) as f:
            return (f.read() + text + '<|im_end|>')

    @staticmethod
    def load_vocab(vocab_path: str) -> Dict[str, int]:
        with open(vocab_path) as f:
            return (load(f))
    
    def generate(self, prompt: str, model, max_tokens: int) -> str:
        prompt: str = f'{self.context}<|im_start|>user\n/no_think\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        input_ids = model.encode(prompt).squeeze().tolist()
        prompt_len = len(input_ids)
        for i in range(max_tokens):
            logits = model.get_logits_from_input_ids(input_ids)
            next_token = argmax(logits)
            if (next_token == 151645):
                break
            input_ids.append(next_token)
        output = model.decode(input_ids[prompt_len:])
        start = output.find('{')
        end = output.rfind('}')
        if (start != -1 and end != -1):
            output = output[start:end + 1]
        return (output)

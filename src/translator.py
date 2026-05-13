from typing import Dict, Any, List, Iterator, Tuple
from json import load, dumps
from src.colors import Format
from numpy import argmax, array, float64
import numpy as np


class Translator:
    def __init__(self, vocab_path: str, funcs: List[Any]) -> None:
        self.vocab: Dict[str, int] = self.load_vocab(vocab_path)
        self.context: str = Translator.load_context('context.md', funcs)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self._funcs: List[Any] = funcs
        self._cache: Dict[str, List[int]] = {}
        self._num_tokens: Dict[int, str] = {}
        self._string_tokens: List[int] = []

    @staticmethod
    def load_context(context_path: str, functions: List[Any]) -> str:
        text: str = 'Available functions:'
        for i in functions:
            params: str = ', '.join(
                f'{k}: {v.type}' for k, v in i.parameters.items()
            )
            text += '\n- ' + i.name + f'({params}): ' + i.description
        with open(context_path) as f:
            return f.read() + text + '<|im_end|>'

    @staticmethod
    def load_vocab(vocab_path: str) -> Dict[str, int]:
        with open(vocab_path) as f:
            return load(f)

    def get_prompt(self, prompt: str) -> str:
        parts: List[str] = [
            self.context,
            '<|im_start|>user\n/no_think\n',
            prompt,
            '<|im_end|>\n<|im_start|>assistant\n',
        ]
        return ''.join(parts)

    def _enc(self, model: Any, text: str) -> List[int]:
        if text not in self._cache:
            self._cache[text] = [int(x) for x in model.encode(text).flatten()]
        return self._cache[text]

    def _filter_num_tokens(self) -> None:
        allowed_chars = set('0123456789.-')
        self._num_tokens = {}
        for tid, ts in self.id_to_token.items():
            clean = ts.replace('\u0120', '').replace('Ġ', '')
            if clean and all(c in allowed_chars for c in clean):
                self._num_tokens[tid] = clean

    def _filter_string_tokens(self, model: Any) -> None:
        quote_chars = {'"', '\u201c', '\u201d', '\u2018', '\u2019'}
        self._string_tokens = []
        for tid in self.id_to_token:
            decoded = model.decode([tid])
            if not any(c in decoded for c in quote_chars):
                self._string_tokens.append(tid)

    def _get_max_id(self, model: Any, input_ids: List[int],
                    allowed_ids: List[int]) -> int:
        raw = model.get_logits_from_input_ids(input_ids)
        if hasattr(raw, 'detach'):
            raw = raw.detach()
        if hasattr(raw, 'numpy'):
            raw = raw.numpy()
        scores = np.array(raw, dtype=float64).flatten()
        allowed = np.array(allowed_ids, dtype=np.int64)
        return int(allowed[np.argmax(scores[allowed])].item())

    def _generate_from_trie(self, model: Any, input_ids: List[int],
                             candidates: Dict[str, List[int]]) -> Tuple[str, List[int]]:
        names: List[str] = list(candidates.keys())
        i: int = 0
        while len(names) > 1:
            active = [n for n in names if i < len(candidates[n])]
            if not active:
                break
            names = active
            groups: Dict[int, List[str]] = {}
            for n in names:
                token = candidates[n][i]
                if token not in groups:
                    groups[token] = []
                groups[token].append(n)
            if len(groups) == 1:
                token = list(groups.keys())[0]
                input_ids = input_ids + [token]
            else:
                allowed = list(groups.keys())
                token = self._get_max_id(model, input_ids, allowed)
                input_ids = input_ids + [token]
                names = groups[token]
            i += 1
        choice: str = names[0]
        remaining: List[int] = candidates[choice][i:]
        input_ids = input_ids + remaining
        return choice, input_ids

    def _generate_numbers(self, model: Any,
                          input_ids: List[int],
                          terminator: str) -> Tuple[str, List[int]]:
        result: str = ''
        end_ids: List[int] = [
            self.vocab[t] for t in [',', '}', ', "', '}"']
            if t in self.vocab and terminator.startswith(t)
        ]
        if not end_ids:
            end_ids = [
                tid for ts, tid in self.vocab.items()
                if ts and terminator.startswith(ts)
            ]
        while True:
            allowed: List[int] = list(self._num_tokens.keys()) + end_ids
            if result:
                allowed = [
                    i for i in allowed
                    if '-' not in self._num_tokens.get(i, '-x')
                ]
            else:
                allowed = [i for i in allowed if i not in end_ids]
            best = self._get_max_id(model, input_ids, allowed)
            if best in end_ids:
                break
            result += self._num_tokens[best]
            input_ids = input_ids + [best]
        return result, input_ids

    def _generate_string(self, model: Any,
                         input_ids: List[int]) -> Tuple[str, List[int]]:
        result: str = ''
        end_ids: List[int] = [
            self.vocab[t] for t in ['"'] if t in self.vocab
        ]
        while True:
            allowed: List[int] = self._string_tokens + end_ids
            raw = model.get_logits_from_input_ids(input_ids)
            if hasattr(raw, 'detach'):
                raw = raw.detach()
            if hasattr(raw, 'numpy'):
                raw = raw.numpy()
            scores = np.array(raw, dtype=float64).flatten()
            for eid in end_ids:
                scores[eid] += 3.0
            token_id = int(
                np.array(allowed)[np.argmax(scores[np.array(allowed)])].item()
            )
            if token_id in end_ids:
                break
            result += model.decode([token_id])
            input_ids = input_ids + [token_id]
            if len(result) > 200:
                break
        return result, input_ids

    def generate(self,
                 prompt: str,
                 functions: List[Any],
                 model: Any,
                 max_tokens: int) -> str:
        self._filter_num_tokens()
        self._filter_string_tokens(model)

        fn_tokens: Dict[str, List[int]] = {
            f.name: self._enc(model, f.name)
            for f in functions if f.name is not None
        }
        param_key_tokens: Dict[str, Dict[str, List[int]]] = {
            f.name: {k: self._enc(model, k + '": ') for k in f.parameters}
            for f in functions if f.name is not None
        }

        input_ids: List[int] = (
            self._enc(model, self.get_prompt(prompt))
            + self._enc(model, '{"name": "')
        )

        result_name, input_ids = self._generate_from_trie(
            model, input_ids, fn_tokens
        )

        selected: Any = next(
            (f for f in functions if f.name == result_name), None
        )
        arg_keys: List[str] = list(selected.parameters.keys()) if selected else []
        result_params: Dict[str, Any] = {}

        input_ids += self._enc(model, '", "parameters": {')

        for arg_index, current_param in enumerate(arg_keys):
            is_last: bool = (arg_index == len(arg_keys) - 1)
            terminator: str = '}' if is_last else ', "'
            ptype: str = (
                selected.parameters[current_param].type or 'string'
            ) if selected else 'string'

            input_ids += self._enc(model, '"')
            input_ids += param_key_tokens[result_name][current_param]

            if ptype == 'number':
                raw_num, input_ids = self._generate_numbers(
                    model, input_ids, terminator
                )
                try:
                    result_params[current_param] = float(raw_num)
                except ValueError:
                    result_params[current_param] = 0.0
            else:
                input_ids += self._enc(model, '"')
                str_val, input_ids = self._generate_string(model, input_ids)
                result_params[current_param] = str_val

            if not is_last:
                input_ids += self._enc(model, ', ')

        input_ids += self._enc(model, '}}')

        dumped = dumps({'name': result_name, 'parameters': result_params})
        print(Format.colored(f'│ ANSWER: {dumped}', 'CYAN'))
        return dumped

    def get_token(self, expected: str) -> Iterator[int]:
        for token_string, token_id in self.vocab.items():
            if expected.startswith(token_string):
                yield token_id

    @staticmethod
    def get_matched(partial: str, expected: str) -> int:
        for length in range(len(expected), 0, -1):
            if partial[-length:] == expected[:length]:
                return length
        return 0

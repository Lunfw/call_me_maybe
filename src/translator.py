from typing import Dict, Any, List, Iterator, Tuple
from json import load, dumps
from src.colors import Format
from numpy import float64
from time import sleep, perf_counter
import numpy as np


class Translator:
    def __init__(self, vocab_path: str, funcs: List[Any]) -> None:
        self.vocab: Dict[str, int] = self.load_vocab(vocab_path)
        self.context: str = Translator.load_context('context.md', funcs)
        self.id_to_token: Dict[int, str] = {
                v: k for k, v in self.vocab.items()
                }
        self._funcs: List[Any] = funcs
        self._cache: Dict[str, List[int]] = {}
        self._num_tokens: Dict[int, str] = {}
        self._string_tokens: List[int] = []
        self._string_tokens_arr: Any = None
        self._ready: bool = False
        self._filter_num_tokens()

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

    def _filter_string_tokens(self) -> None:
        quote_chars = {'"', '\u201c', '\u201d', '\u2018', '\u2019'}
        self._string_tokens = [
            tid for tid, ts in self.id_to_token.items()
            if not any(c in ts for c in quote_chars)
        ]
        self._string_tokens_arr = np.array(self._string_tokens, dtype=np.int64)

    def _ensure_ready(self) -> None:
        if not self._ready:
            self._filter_string_tokens()
            self._ready = True

    def _scores(self, model: Any, input_ids: List[int]) -> Any:
        raw = model.get_logits_from_input_ids(input_ids)
        if hasattr(raw, 'detach'):
            raw = raw.detach()
        if hasattr(raw, 'numpy'):
            raw = raw.numpy()
        return np.array(raw, dtype=float64).flatten()

    def _get_max_id(self, model: Any, input_ids: List[int],
                    allowed_ids: Any) -> int:
        scores = self._scores(model, input_ids)
        arr = np.array(allowed_ids, dtype=np.int64)
        return int(arr[np.argmax(scores[arr])].item())

    def _generate_from_trie(self, model: Any, input_ids: List[int],
                            candidates: Dict[str, List[int]]
                            ) -> Tuple[str, List[int]]:
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
                token = self._get_max_id(model, input_ids, list(groups.keys()))
                input_ids = input_ids + [token]
                names = groups[token]
            i += 1
        choice: str = names[0]
        input_ids = input_ids + candidates[choice][i:]
        return choice, input_ids

    def _generate_numbers(self, model: Any, input_ids: List[int],
                          terminator: str) -> Tuple[str, List[int]]:
        end_ids: List[int] = [
            tid for ts, tid in self.vocab.items()
            if ts and terminator.startswith(ts)
        ]
        num_ids: List[int] = list(self._num_tokens.keys())
        end_arr = np.array(end_ids, dtype=np.int64)
        result: str = ''
        while True:
            if not result:
                allowed = np.array(
                    [i for i in num_ids
                     if '-' not in self._num_tokens[i] or result == ''],
                    dtype=np.int64
                )
            else:
                no_minus = np.array(
                    [i for i in num_ids if '-' not in self._num_tokens[i]],
                    dtype=np.int64
                )
                allowed = np.concatenate([no_minus, end_arr])
            scores = self._scores(model, input_ids)
            best = int(allowed[np.argmax(scores[allowed])].item())
            if best in end_ids:
                break
            result += self._num_tokens[best]
            input_ids = input_ids + [best]
        return result, input_ids

    def _generate_string(self, model: Any,
                         input_ids: List[int]) -> Tuple[str, List[int]]:
        end_ids: List[int] = [self.vocab['"']] if '"' in self.vocab else []
        end_arr = np.array(end_ids, dtype=np.int64)
        allowed_arr = np.concatenate([self._string_tokens_arr, end_arr])
        result: str = ''
        while True:
            scores = self._scores(model, input_ids)
            scores[end_arr] += 11.0
            token_id = int(allowed_arr[np.argmax(scores[allowed_arr])].item())
            if token_id in end_ids:
                break
            decoded = self.id_to_token.get(token_id, '')
            decoded = decoded.replace('\u0120', ' ').replace('Ġ', ' ')
            result += decoded
            input_ids = input_ids + [token_id]
            if len(result) > 200:
                break
        return result, input_ids

    def generate(self,
                 prompt: str,
                 functions: List[Any],
                 model: Any,
                 max_tokens: int) -> str:
        self._ensure_ready()

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

        if (selected):
            arg_keys: List[str] = list(selected.parameters.keys())
        else:
            arg_keys = []
        result_params: Dict[str, Any] = {}

        start = perf_counter()

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
        end = perf_counter()
        print(Format.colored('│ ANSWER:', 'CYAN'), end='', flush=True)
        for i in range(len(dumped)):
            print(Format.colored(dumped[i], 'CYAN'), end='', flush=True)
            sleep(0.01)
        print(Format.colored(f' ({end - start:.2f}s)', 'CYAN'), end='')
        return dumped

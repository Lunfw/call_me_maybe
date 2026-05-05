from typing import Dict, Any, Optional, List
from json import load


class Translator:
    def __init__(self, model, vocab_path: str):
        self.vocab: Dict[str, int] = self.load_vocab(vocab_path)
        self.id_to_token: Dict[int, str] = {}

    @staticmethod
    def load_vocab(vocab_path: str) -> Dict[str, int]:
        with open(vocab_path) as f:
            return (load(f))


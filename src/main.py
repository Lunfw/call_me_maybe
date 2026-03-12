from llm_sdk import Small_LLM_Model
from sys import argv, exit
from torch import no_grad


def process(prompt: str) -> str:
    model = Small_LLM_Model()
    input_ids = model.encode(prompt)

    with no_grad():
        output = model._model.generate(input_ids, max_length=200)

    new_text = model.decode(output[0])
    return (new_text)


def main(argv: list[str]):
    if (argv[1] == '' or argv[1] is None):
        print("Usage: make run <prompt>")
        exit(1);
    print(process(argv[1]));


from llm_sdk import Small_LLM_Model
from sys import argv
from torch import no_grad


def process(prompt: str) -> str:
    model = Small_LLM_Model()
    input_ids = model.encode(prompt)

    with no_grad():
        output = model._model.generate(input_ids, max_length=200)

    new_text = model.decode(output[0])
    return (new_text)


def main(argv: list[str]):
    print(process(argv[1]));


if (__name__ == '__main__'):
    main(argv)

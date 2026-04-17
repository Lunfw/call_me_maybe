from llm_sdk import Small_LLM_Model
from sys import argv, exit
from transformers import TextStreamer

def process(prompt: str) -> None:
    model = Small_LLM_Model()
    input_ids = model.encode(prompt)
    streamer = TextStreamer(model._tokenizer, skip_special_tokens=True)

    model._model.generate(
        input_ids,
        max_new_tokens=200,
        streamer=streamer,
    )

def main(argv: list[str]):
    if len(argv) < 2 or argv[1] == '':
        print("Usage: make run <prompt>")
        exit(1)
    process(argv[1])

if (__name__ == "__main__"):
    main(argv)

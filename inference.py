import argparse

from config import config
from utils import load_model
from scripts.Mini_LLM import Mini_LLM

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    help="Input prompt"
)

args = parser.parse_args()

model = Mini_LLM(config)

load_model(
    model,
    "models/mini_llm_checkpoint.pt"
)

text = model.predict(
    args.prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
)

print(text)
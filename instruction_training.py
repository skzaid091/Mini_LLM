import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from config import config
from utils import load_model
from utils import save_model

from scripts.Mini_LLM import Mini_LLM


class InstructionDataset(Dataset):

    def __init__(self, csv_path):

        df = pd.read_csv(csv_path)

        self.samples = []

        for _, row in df.iterrows():

            instruction = str(row["instruction"])

            input_text = str(row["input"]) if pd.notna(row["input"]) else ""
            response = str(row["response"])

            text = (
                "[INST]\n"
                f"{instruction}\n"
                f"{input_text}\n"
                "[/INST]\n\n"
                "[RESP]\n"
                f"{response}\n"
                "[/RESP]"
            )

            self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


dataset = InstructionDataset(
    "data/tiny_stories/tinystories_sft_data.csv"
)

train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True
)


model = Mini_LLM(config)
model.to(model.device)

load_model(
    model,
    "models/tiny_stories_mini_llm_checkpoint.pt"
)

for param_group in model.optimizer.param_groups:
    param_group["lr"] = 1e-5


epochs = 20

for epoch in range(epochs):

    running_loss = 0
    for batch in train_loader:

        loss = model.train_step(
            batch
        )

        running_loss += loss

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1} "
        f"| Loss: {avg_loss:.4f}"
    )

    save_model(
        epoch,
        model,
        "models/instruction_tuned_mini_llm.pt"
    )
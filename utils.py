import os
from config import config

import torch
from torch.utils.data import DataLoader

def save_model(epoch, model, model_save_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "scheduler_state_dict": model.scheduler.state_dict(),
        "config": config,
        "epoch": epoch
    }
    torch.save(checkpoint, model_save_path)
    print(f"Model saved at location: {model_save_path}")

def load_model(model, model_save_path):
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=model.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    print("No checkpoint found, starting from scratch")
    return 0

class TextDataset():

    def __init__(self, texts):

        self.texts = texts

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        return self.texts[idx]

def load_datasets():

    # Load train and val from txt files
    with open(config["train_dataset_path"], "r") as f:
        train_data = f.read()

    with open(config["test_dataset_path"], "r") as f:
        val_data = f.read()

    train_texts = [t for t in train_data.split('\n') if t.strip()]
    val_texts   = [t for t in val_data.split('\n') if t.strip()]

    train_dataset = TextDataset(train_texts)
    val_dataset   = TextDataset(val_texts)

    # No random_split needed — already split
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,   
        batch_size=config["batch_size"], 
        shuffle=False
    )

    return train_loader, val_loader
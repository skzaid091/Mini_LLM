import torch

from config import config
from utils import load_datasets, save_model, load_model

from scripts.Early_Stopping import EarlyStopping
es = EarlyStopping(
    config["patience"], 
    config["train_val_diff"], 
    config["val_improvement_threshold"],
    config["warm_up_epochs"],
    config["skip_train_val_overfit_check"]
)

from scripts.Mini_LLM import Mini_LLM
mini_llm = Mini_LLM(config)
mini_llm.to(mini_llm.device)
start_epoch = load_model(mini_llm, config["model_save_path"])

train_loader, val_loader = load_datasets()
for epoch in range(start_epoch, config["epochs"]):

    # ── Training ──────────────────────────────────────────
    mini_llm.train()
    total_train_loss = 0

    for train_batch in train_loader:

        train_loss = mini_llm.train_step(train_batch)
        total_train_loss += train_loss

    avg_train_loss = total_train_loss / len(train_loader)

    # ── Validation ────────────────────────────────────────
    mini_llm.eval()
    total_val_loss = 0

    with torch.no_grad():
        for val_batch in val_loader:

            val_loss = mini_llm.val_step(val_batch)
            total_val_loss += val_loss

    avg_val_loss = total_val_loss / len(val_loader)

    # ── Scheduler ─────────────────────────────────────────
    mini_llm.scheduler.step()

    # ── Logging ───────────────────────────────────────────
    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"LR: {mini_llm.scheduler.get_last_lr()[0]:.2e}"
    )


    stop_training, improvement = es.check_performance(epoch, avg_train_loss, avg_val_loss)
    if stop_training:
        break

    if improvement:
        save_model(epoch, mini_llm, config["model_save_path"])

    print("\n")
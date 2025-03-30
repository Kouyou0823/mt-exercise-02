import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_dir = "logs"

dropout_map = {
    "0.0": "log_d0.txt",
    "0.2": "log_d1.txt",
    "0.4": "log_d2.txt",
    "0.6": "log_d3.txt",
    "0.8": "log_d4.txt"
}

all_logs = {}

for dropout_val, filename in dropout_map.items():
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["epoch", "train_loss", "val_loss", "test_loss", "val_ppl", "test_ppl"])
    all_logs[dropout_val] = df

# === Perplexity Plots ===

# Training Perplexity plot
plt.figure(figsize=(10, 6))
for dropout_val, df in all_logs.items():
    train_ppl = np.exp(df["train_loss"])
    plt.plot(df["epoch"], train_ppl, label=f"dropout={dropout_val}")
plt.title("Training Perplexity vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.legend(title="Train", loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "train_ppl_curve.png"))
plt.close()

# Validation Perplexity plot
plt.figure(figsize=(10, 6))
for dropout_val, df in all_logs.items():
    plt.plot(df["epoch"], df["val_ppl"], label=f"dropout={dropout_val}")
plt.title("Validation Perplexity vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.legend(title="Valid", loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "valid_ppl_curve.png"))
plt.close()

# === Perplexity Tables  ===

epochs = all_logs["0.0"]["epoch"]

# Training
train_table = pd.DataFrame({"Epoch": epochs})
for dropout_val, df in all_logs.items():
    train_table[f"Dropout {dropout_val}"] = np.exp(df["train_loss"]).round(2)
train_table.to_csv(os.path.join(log_dir, "train_perplexity_table.csv"), index=False)

# Validation
valid_table = pd.DataFrame({"Epoch": epochs})
for dropout_val, df in all_logs.items():
    valid_table[f"Dropout {dropout_val}"] = df["val_ppl"].round(2)
valid_table.to_csv(os.path.join(log_dir, "valid_perplexity_table.csv"), index=False)

# Test
test_table = pd.DataFrame({"Epoch": epochs})
for dropout_val, df in all_logs.items():
    test_table[f"Dropout {dropout_val}"] = df["test_ppl"].round(2)
test_table.to_csv(os.path.join(log_dir, "test_perplexity_table.csv"), index=False)

print(" Saved plots and .csv tables in logs/")
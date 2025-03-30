import os
import pandas as pd
import matplotlib.pyplot as plt

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
        print(f"❌ Missing: {path}")
        continue
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["epoch", "train_loss", "val_loss", "test_loss", "val_ppl", "test_ppl"])
    all_logs[dropout_val] = df

plt.figure(figsize=(10, 6))
for dropout_val, df in all_logs.items():
    train_ppl = df["train_loss"].apply(lambda x: pow(2.71828, x))  # e^loss ≈ perplexity
    plt.plot(df["epoch"], train_ppl, label=f"dropout={dropout_val}")
plt.title("Training Perplexity vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.legend(title="Train", loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "train_ppl_curve.png"))
plt.close()

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

print("Saved logs/train_ppl_curve.png and logs/valid_ppl_curve.png")
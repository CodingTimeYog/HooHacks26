import subprocess
import json
import numpy as np
import os
import wandb
from dotenv import load_dotenv

load_dotenv("../.env" if not os.path.exists(".env") else ".env")

RUNS     = 50
META_PATH = "data/models/model_metadata.json"

t1_accs, t2_accs, t3_accs = [], [], []
t1_rmse,  t2_rmse,  t3_rmse  = [], [], []

print(f"🚀 Starting {RUNS} training loops...")

for i in range(RUNS):
    print(f"Run {i+1}/{RUNS}...", end=" ", flush=True)

    subprocess.run(
        ["python", "backend/train_models.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            t1_accs.append(meta.get("test_clf_acc_t1", 0))
            t2_accs.append(meta.get("test_clf_acc_t2", 0))
            t3_accs.append(meta.get("test_clf_acc_t3", 0))
            t1_rmse.append(meta.get("test_rmse_t1", 0))
            t2_rmse.append(meta.get("test_rmse_t2", 0))
            t3_rmse.append(meta.get("test_rmse_t3", 0))
        print("✅ Done")
    else:
        print("❌ Failed (metadata not found)")

# Log aggregate summary to W&B
wandb.init(
    project="foregast",
    name=f"experiment_avg_{RUNS}_runs",
    config={"total_runs": RUNS},
)
wandb.log({
    "t1_avg_clf_acc":  np.mean(t1_accs),
    "t2_avg_clf_acc":  np.mean(t2_accs),
    "t3_avg_clf_acc":  np.mean(t3_accs),
    "t1_avg_rmse":     np.mean(t1_rmse),
    "t2_avg_rmse":     np.mean(t2_rmse),
    "t3_avg_rmse":     np.mean(t3_rmse),
    "t1_std_rmse":     np.std(t1_rmse),
    "t2_std_rmse":     np.std(t2_rmse),
    "t3_std_rmse":     np.std(t3_rmse),
})
wandb.finish()

print("\n" + "="*40)
print(f"📊 AVERAGES OVER {RUNS} RUNS")
print("="*40)
print(f"t1 (30-Day) | Acc: {np.mean(t1_accs)*100:.1f}%  |  RMSE: ${np.mean(t1_rmse):.1f}")
print(f"t2 (60-Day) | Acc: {np.mean(t2_accs)*100:.1f}%  |  RMSE: ${np.mean(t2_rmse):.1f}")
print(f"t3 (90-Day) | Acc: {np.mean(t3_accs)*100:.1f}%  |  RMSE: ${np.mean(t3_rmse):.1f}")
print("="*40)
print("📈 Results logged to W&B project: foregast")
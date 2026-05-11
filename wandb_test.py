import random

import wandb
import time

# https://wandb.ai/hugo-demule-epfl/

# Start a new wandb run to track this script.
run = wandb.init(
    entity="hugo-demule-epfl",
    project="MNLP-Safety",
    name="benchmark-test",
    group="benchmark-demo",
    job_type="simulation",
    tags=["demo", "training", "benchmark"],
    config={
        "learning_rate": 0.02,
        "architecture": "Transformer",
        "dataset": "Synthetic-Safety",
        "epochs": 100,
        "benchmark_interval": 10,
        "seed": 7,
    },
)

epochs = 300
benchmark_interval = 10

wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("benchmark/*", step_metric="epoch")
wandb.define_metric("train/train_loss", summary="min")
wandb.define_metric("train/val_loss", summary="min")
wandb.define_metric("benchmark/pass_at_1", summary="max")
wandb.define_metric("benchmark/box_compliance", summary="max")
wandb.define_metric("benchmark/pass_at_1_among_box_compliant", summary="max")

base_loss = 1.8
base_accuracy = 0.38
base_benchmark = 0.32

offset = random.random() / 8
global_step = 0
best_train_loss = float("inf")
best_val_loss = float("inf")
best_train_accuracy = 0.0
best_benchmark_pass_at_1 = 0.0
best_benchmark_box_compliance = 0.0
best_benchmark_pass_at_1_among_box_compliant = 0.0

for epoch in range(2, epochs):
    global_step += 1
    time.sleep(0.5)

    progress = epoch / epochs
    decay = 2 ** (-progress * 5)
    noise = random.random() / max(epoch, 1)

    train_loss = max(0.02, base_loss * decay + noise + offset)
    val_loss = max(0.02, base_loss * decay + noise / 2 + offset) + 0.25 + 0.05 * random.random()
    accuracy = min(0.995, base_accuracy + (1 - decay) * 0.6 - noise / 2)
    learning_rate = 0.02 * (1 - 0.75 * progress)

    best_train_loss = min(best_train_loss, train_loss)
    best_train_accuracy = max(best_train_accuracy, accuracy)
    best_val_loss = min(best_val_loss, val_loss)

    log_payload = {
        "epoch": epoch,
        "train/train_loss": train_loss,
        "train/val_loss": val_loss,
        "train/learning_rate": learning_rate,
    }

    if epoch % benchmark_interval == 0:
        benchmark_score = min(0.995, base_benchmark + (1 - decay) * 0.62)
        bench_pass_at_1 = min(0.995, benchmark_score + random.random() * 0.03)
        bench_box_compliance = min(0.995, benchmark_score + 0.04 + random.random() * 0.02)
        bench_pass_at_1_among_box_compliant = min(
            0.995,
            max(0.0, bench_pass_at_1 - 0.03 + random.random() * 0.01),
        )

        best_benchmark_pass_at_1 = max(best_benchmark_pass_at_1, bench_pass_at_1)
        best_benchmark_box_compliance = max(best_benchmark_box_compliance, bench_box_compliance)
        best_benchmark_pass_at_1_among_box_compliant = max(
            best_benchmark_pass_at_1_among_box_compliant,
            bench_pass_at_1_among_box_compliant,
        )

        log_payload.update({
            "benchmark/pass_at_1": bench_pass_at_1,
            "benchmark/box_compliance": bench_box_compliance,
            "benchmark/pass_at_1_among_box_compliant": bench_pass_at_1_among_box_compliant,
        })

    run.log(log_payload, commit=True)

run.summary["best_train_loss"] = best_train_loss
run.summary["best_val_loss"] = best_val_loss
run.summary["best_benchmark_pass_at_1"] = best_benchmark_pass_at_1
run.summary["best_benchmark_box_compliance"] = best_benchmark_box_compliance
run.summary["best_benchmark_pass_at_1_among_box_compliant"] = best_benchmark_pass_at_1_among_box_compliant
run.summary["final_epoch"] = epochs - 1

run.finish()
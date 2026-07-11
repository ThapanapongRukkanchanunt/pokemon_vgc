import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from torch_alpha_model import (
    ValueDataset,
    binary_metrics,
    collate_value,
    create_model_config,
    load_jsonl,
    model_from_config,
    resolve_device,
    save_json,
    seed_to_int,
    split_examples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an AlphaStar-like action-value scorer.")
    parser.add_argument("--dataset", default="data/datasets/value/phase4_mixed_q.jsonl")
    parser.add_argument("--out-dir", default="models/torch_value/phase4_mixed_q")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--group-validation-by", default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", default="torch_value")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-state-tokens", type=int, default=384)
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 0:
        raise SystemExit("--epochs must be >= 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.validation_split < 0 or args.validation_split >= 1:
        raise SystemExit("--validation-split must be >= 0 and < 1")
    return args


def load_checkpoint(path, device):
    if not path:
        return None
    return torch.load(path, map_location=device)


def data_loader(examples, model_config, batch_size, shuffle, seed, num_workers):
    dataset = ValueDataset(
        examples,
        vocab_size=model_config["vocab_size"],
        max_state_tokens=model_config["max_state_tokens"],
        max_action_tokens=model_config["max_action_tokens"],
    )
    generator = torch.Generator()
    generator.manual_seed(seed_to_int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_value,
        generator=generator,
    )


def split_grouped(examples, validation_split, seed, key):
    if not key:
        return split_examples(examples, validation_split, seed)
    groups = {}
    for index, example in enumerate(examples):
        group = str(example.get(key, f"missing:{index}"))
        groups.setdefault(group, []).append(example)
    group_rows = [{"group": group, "rows": rows} for group, rows in groups.items()]
    train_groups, validation_groups = split_examples(group_rows, validation_split, seed)
    train_examples = [row for group in train_groups for row in group["rows"]]
    validation_examples = [row for group in validation_groups for row in group["rows"]]
    return train_examples, validation_examples


def move_batch(batch, device):
    batch["state_ids"] = batch["state_ids"].to(device)
    batch["action_ids"] = batch["action_ids"].to(device)
    batch["owners"] = batch["owners"].to(device)
    batch["targets"] = batch["targets"].to(device)
    return batch


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_brier = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["state_ids"], batch["action_ids"], batch["owners"])
            loss, correct, count, brier = binary_metrics(logits, batch["targets"])
            total_loss += float(loss.item()) * count
            total_correct += correct
            total_examples += count
            total_brier += brier * count
    return {
        "examples": total_examples,
        "avg_loss": total_loss / total_examples if total_examples else None,
        "accuracy": total_correct / total_examples if total_examples else None,
        "avg_brier": total_brier / total_examples if total_examples else None,
        "correct": total_correct,
    }


def train(args):
    out_dir = Path(args.out_dir)
    checkpoint_path = out_dir / "checkpoint.pt"
    metrics_path = out_dir / "metrics.json"
    if checkpoint_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists at {checkpoint_path}; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"device={device}")
    torch.manual_seed(seed_to_int(args.seed))

    checkpoint = load_checkpoint(args.init_checkpoint, device)
    if checkpoint:
        model_config = checkpoint["model_config"]
        global_step = int(checkpoint.get("global_step", 0))
        print(f"Loaded init checkpoint: {args.init_checkpoint} global_step={global_step}")
    else:
        model_config = create_model_config(args)
        global_step = 0

    examples = load_jsonl(args.dataset, args.limit)
    train_examples, validation_examples = split_grouped(
        examples, args.validation_split, args.seed, args.group_validation_by
    )
    print(f"Loaded {len(examples)} examples; train={len(train_examples)} validation={len(validation_examples)}")

    model = model_from_config(model_config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if checkpoint and args.resume_optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay

    train_loader = data_loader(train_examples, model_config, args.batch_size, True, args.seed, args.num_workers)
    validation_loader = data_loader(validation_examples, model_config, args.batch_size, False, args.seed, args.num_workers)
    history = []

    def record(epoch):
        train_metrics = evaluate(model, train_loader, device)
        validation_metrics = evaluate(model, validation_loader, device) if validation_examples else None
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        text = (
            f"epoch={epoch} train_loss={train_metrics['avg_loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} train_brier={train_metrics['avg_brier']:.4f}"
        )
        if validation_metrics:
            text += (
                f" val_loss={validation_metrics['avg_loss']:.4f} "
                f"val_acc={validation_metrics['accuracy']:.3f} val_brier={validation_metrics['avg_brier']:.4f}"
            )
        print(text)

    record(0)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["state_ids"], batch["action_ids"], batch["owners"])
            loss, _, count, _ = binary_metrics(logits, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            seen += count
            if args.progress_every > 0 and batch_index % args.progress_every == 0:
                elapsed = time.time() - started
                print(f"epoch={epoch} batch={batch_index} seen={seen}/{len(train_examples)} loss={loss.item():.4f} elapsed={elapsed:.1f}s")
        if epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0:
            record(epoch)

    final_train = evaluate(model, train_loader, device)
    final_validation = evaluate(model, validation_loader, device) if validation_examples else None
    metrics = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": args.dataset,
        "out_dir": args.out_dir,
        "init_checkpoint": args.init_checkpoint,
        "resume_optimizer": args.resume_optimizer,
        "iteration": args.iteration,
        "global_step": global_step,
        "config": vars(args),
        "model_config": model_config,
        "examples": len(examples),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "history": history,
        "final_train": final_train,
        "final_validation": final_validation,
    }

    torch.save({
        "checkpoint_type": "alphastar_like_value",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "training_config": vars(args),
        "iteration": args.iteration,
        "global_step": global_step,
        "metrics_path": str(metrics_path),
    }, checkpoint_path)
    save_json(metrics_path, metrics)
    print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote metrics: {metrics_path}")


def main():
    train(parse_args())


if __name__ == "__main__":
    main()

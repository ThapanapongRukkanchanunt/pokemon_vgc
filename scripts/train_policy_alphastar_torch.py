import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from torch_alpha_model import (
    actor_critic_from_config,
    PolicyDataset,
    collate_policy,
    create_actor_critic_config,
    create_model_config,
    load_jsonl,
    model_from_config,
    policy_loss_and_accuracy,
    resolve_device,
    save_json,
    seed_to_int,
    split_examples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an AlphaStar-like legal-action policy scorer.")
    parser.add_argument("--dataset", default="data/datasets/search/phase6_search_improved.jsonl")
    parser.add_argument("--out-dir", default="models/torch_policy/phase6_search_policy")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-coef", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--train-value-head", action="store_true")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--group-validation-by", default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", default="torch_policy")
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


class PolicyValueDataset(PolicyDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        target = self.examples[index].get("win_target")
        if target not in (0, 1):
            raise ValueError(f"{self.examples[index].get('example_id', index)} win_target must be 0 or 1")
        item["value_target"] = 1.0 if target == 1 else -1.0
        return item


def collate_policy_value(batch):
    collated = collate_policy(batch)
    collated["value_targets"] = torch.tensor([item["value_target"] for item in batch], dtype=torch.float32)
    return collated


def split_training_examples(examples, validation_split, seed, group_field=None):
    if not group_field:
        return split_examples(examples, validation_split, seed)
    groups = {}
    for example in examples:
        group = str(example.get(group_field, example.get("example_id", "unknown")))
        groups.setdefault(group, []).append(example)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    validation_groups = set(keys[:int(len(keys) * validation_split)])
    train = []
    validation = []
    for key, rows in groups.items():
        (validation if key in validation_groups else train).extend(rows)
    return train, validation


def data_loader(examples, model_config, batch_size, shuffle, seed, num_workers, train_value_head=False):
    dataset_type = PolicyValueDataset if train_value_head else PolicyDataset
    dataset = dataset_type(
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
        collate_fn=collate_policy_value if train_value_head else collate_policy,
        generator=generator,
    )


def move_batch(batch, device):
    batch["state_ids"] = batch["state_ids"].to(device)
    batch["action_ids"] = batch["action_ids"].to(device)
    batch["owners"] = batch["owners"].to(device)
    if "value_targets" in batch:
        batch["value_targets"] = batch["value_targets"].to(device)
    return batch


def evaluate(model, loader, device, train_value_head=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_value_loss = 0.0
    total_value_correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["state_ids"], batch["action_ids"], batch["owners"])
            loss, correct, count = policy_loss_and_accuracy(logits, batch["offsets"], batch["labels"])
            total_loss += float(loss.item()) * count
            total_correct += correct
            total_examples += count
            if train_value_head:
                values = model.value(batch["state_ids"])
                value_loss = torch.nn.functional.mse_loss(values, batch["value_targets"])
                total_value_loss += float(value_loss.item()) * count
                total_value_correct += int(((values >= 0) == (batch["value_targets"] >= 0)).sum().item())
    metrics = {
        "examples": total_examples,
        "avg_loss": total_loss / total_examples if total_examples else None,
        "accuracy": total_correct / total_examples if total_examples else None,
        "correct": total_correct,
    }
    if train_value_head:
        metrics["value_loss"] = total_value_loss / total_examples if total_examples else None
        metrics["value_accuracy"] = total_value_correct / total_examples if total_examples else None
    return metrics


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
        model_config = dict(checkpoint["model_config"])
        if args.train_value_head:
            model_config["architecture"] = "alphastar_like_actor_critic_v1"
        global_step = int(checkpoint.get("global_step", 0))
        print(f"Loaded init checkpoint: {args.init_checkpoint} global_step={global_step}")
    else:
        model_config = create_actor_critic_config(args) if args.train_value_head else create_model_config(args)
        global_step = 0

    examples = load_jsonl(args.dataset, args.limit)
    if args.train_value_head:
        examples = [example for example in examples if example.get("win_target") in (0, 1)]
    train_examples, validation_examples = split_training_examples(
        examples,
        args.validation_split,
        args.seed,
        args.group_validation_by,
    )
    print(f"Loaded {len(examples)} examples; train={len(train_examples)} validation={len(validation_examples)}")

    model = (actor_critic_from_config(model_config) if args.train_value_head else model_from_config(model_config)).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=not args.train_value_head)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if checkpoint and args.resume_optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay

    train_loader = data_loader(
        train_examples, model_config, args.batch_size, True, args.seed, args.num_workers, args.train_value_head
    )
    validation_loader = data_loader(
        validation_examples, model_config, args.batch_size, False, args.seed, args.num_workers, args.train_value_head
    )
    history = []

    def record(epoch):
        train_metrics = evaluate(model, train_loader, device, args.train_value_head)
        validation_metrics = evaluate(model, validation_loader, device, args.train_value_head) if validation_examples else None
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        text = f"epoch={epoch} train_loss={train_metrics['avg_loss']:.4f} train_acc={train_metrics['accuracy']:.3f}"
        if validation_metrics:
            text += f" val_loss={validation_metrics['avg_loss']:.4f} val_acc={validation_metrics['accuracy']:.3f}"
        if args.train_value_head:
            text += f" train_value_loss={train_metrics['value_loss']:.4f} train_value_acc={train_metrics['value_accuracy']:.3f}"
            if validation_metrics:
                text += f" val_value_loss={validation_metrics['value_loss']:.4f} val_value_acc={validation_metrics['value_accuracy']:.3f}"
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
            policy_loss, _, count = policy_loss_and_accuracy(logits, batch["offsets"], batch["labels"])
            loss = args.policy_coef * policy_loss
            if args.train_value_head:
                values = model.value(batch["state_ids"])
                value_loss = torch.nn.functional.mse_loss(values, batch["value_targets"])
                loss = loss + args.value_coef * value_loss
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

    final_train = evaluate(model, train_loader, device, args.train_value_head)
    final_validation = evaluate(model, validation_loader, device, args.train_value_head) if validation_examples else None
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
        "checkpoint_type": "alphastar_like_actor_critic_bootstrap" if args.train_value_head else "alphastar_like_policy",
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

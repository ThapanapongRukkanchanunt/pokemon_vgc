import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from torch_alpha_model import (
    actor_critic_from_config,
    collate_policy,
    create_actor_critic_config,
    encode_tokens,
    load_jsonl,
    pad_sequences,
    resolve_device,
    save_json,
    seed_to_int,
    state_tokens,
    action_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a warm-started PPO actor-critic from rollout JSONL.")
    parser.add_argument("--rollouts", default="data/datasets/rl/phase8_ppo_rollouts_rollouts.jsonl")
    parser.add_argument("--out-dir", default="models/torch/phase8_ppo")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exclude-request-types", default="")
    parser.add_argument("--seed", default="phase8_ppo")
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 0:
        raise SystemExit("--epochs must be >= 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.clip_epsilon <= 0:
        raise SystemExit("--clip-epsilon must be > 0")
    return args


def load_checkpoint(path, device):
    if not path:
        return None
    return torch.load(path, map_location=device)


def model_config_from_args_or_checkpoint(args, checkpoint):
    if checkpoint:
        config = dict(checkpoint["model_config"])
        config["architecture"] = "alphastar_like_actor_critic_v1"
        return config
    return create_actor_critic_config(args)


def load_model(args, device):
    checkpoint = load_checkpoint(args.init_checkpoint, device)
    model_config = model_config_from_args_or_checkpoint(args, checkpoint)
    model = actor_critic_from_config(model_config).to(device)
    global_step = 0
    if checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        global_step = int(checkpoint.get("global_step", 0))
        print(f"Loaded init checkpoint: {args.init_checkpoint} global_step={global_step}")
        if missing:
            print(f"Warm-start missing keys: {','.join(missing)}")
        if unexpected:
            print(f"Warm-start unexpected keys: {','.join(unexpected)}")
    return model, model_config, checkpoint, global_step


def trajectory_sort_key(row):
    return (row.get("battle_id", ""), row.get("side", ""), int(row.get("step_index", 0)))


def add_advantages(rows, gamma, gae_lambda):
    grouped = defaultdict(list)
    for row in sorted(rows, key=trajectory_sort_key):
        grouped[row["trajectory_id"]].append(row)

    enriched = []
    for trajectory_rows in grouped.values():
        gae = 0.0
        next_value = 0.0
        for row in reversed(trajectory_rows):
            reward = float(row.get("reward", 0.0))
            value = float(row.get("value_prediction", 0.0))
            nonterminal = 0.0 if row.get("done") else 1.0
            delta = reward + gamma * next_value * nonterminal - value
            gae = delta + gamma * gae_lambda * nonterminal * gae
            row["advantage"] = gae
            row["return"] = gae + value
            next_value = value
        enriched.extend(trajectory_rows)

    advantages = torch.tensor([float(row["advantage"]) for row in enriched], dtype=torch.float32)
    mean = float(advantages.mean().item()) if len(advantages) else 0.0
    std = float(advantages.std(unbiased=False).item()) if len(advantages) else 1.0
    if std <= 1e-8:
        std = 1.0
    for row in enriched:
        row["normalized_advantage"] = (float(row["advantage"]) - mean) / std
    return enriched


class PpoRolloutDataset(torch.utils.data.Dataset):
    def __init__(self, rows, model_config):
        self.rows = rows
        self.model_config = model_config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        legal_actions = row.get("legal_actions") or []
        action_index = int(row.get("action_index"))
        if action_index < 0 or action_index >= len(legal_actions):
            raise ValueError(f"{row.get('trajectory_id')} action_index out of range")
        example = {
            "request_type": row.get("request_type", "unknown"),
            "side": row.get("side", "unknown"),
            "team": row.get("team", "unknown"),
            "lead": row.get("lead", "unknown"),
            "turn": row.get("turn", 0),
            "state": row.get("state") or {},
            "agent_diagnostics": (row.get("state") or {}).get("agent_diagnostics") or {},
        }
        return {
            "state": encode_tokens(
                state_tokens(example),
                self.model_config["vocab_size"],
                self.model_config["max_state_tokens"],
            ),
            "actions": [
                encode_tokens(
                    action_tokens(action),
                    self.model_config["vocab_size"],
                    self.model_config["max_action_tokens"],
                )
                for action in legal_actions
            ],
            "action_index": action_index,
            "old_log_prob": float(row.get("log_prob")),
            "return": float(row.get("return")),
            "advantage": float(row.get("normalized_advantage")),
        }


def collate_ppo(batch):
    policy_like = []
    for item in batch:
        policy_like.append({
            "state": item["state"],
            "actions": item["actions"],
            "label_index": item["action_index"],
        })
    collated = collate_policy(policy_like)
    collated["old_log_probs"] = torch.tensor([item["old_log_prob"] for item in batch], dtype=torch.float32)
    collated["returns"] = torch.tensor([item["return"] for item in batch], dtype=torch.float32)
    collated["advantages"] = torch.tensor([item["advantage"] for item in batch], dtype=torch.float32)
    return collated


def move_batch(batch, device):
    for key in ["state_ids", "action_ids", "owners", "old_log_probs", "returns", "advantages"]:
        batch[key] = batch[key].to(device)
    return batch


def selected_log_probs(logits, offsets, labels):
    log_prob_rows = []
    entropy_rows = []
    for (start, end), label in zip(offsets, labels):
        row_logits = logits[start:end]
        row_log_probs = torch.nn.functional.log_softmax(row_logits, dim=0)
        row_probs = torch.exp(row_log_probs)
        log_prob_rows.append(row_log_probs[label])
        entropy_rows.append(-(row_probs * row_log_probs).sum())
    return torch.stack(log_prob_rows), torch.stack(entropy_rows)


def train(args):
    out_dir = Path(args.out_dir)
    checkpoint_path = out_dir / "checkpoint.pt"
    metrics_path = out_dir / "metrics.json"
    if checkpoint_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists at {checkpoint_path}; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"device={device}")
    torch.manual_seed(seed_to_int(args.seed))

    model, model_config, checkpoint, global_step = load_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if checkpoint and args.resume_optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay

    excluded_request_types = {
        item.strip() for item in args.exclude_request_types.split(",") if item.strip()
    }
    raw_rows = [
        row for row in load_jsonl(args.rollouts, args.limit)
        if row.get("request_type", "unknown") not in excluded_request_types
    ]
    if not raw_rows:
        raise SystemExit("No PPO rollout rows remain after request-type filtering")
    rows = add_advantages(raw_rows, args.gamma, args.gae_lambda)
    dataset = PpoRolloutDataset(rows, model_config)
    generator = torch.Generator()
    generator.manual_seed(seed_to_int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ppo,
        generator=generator,
    )

    history = []
    early_stop = False
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "examples": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "loss": 0.0,
        }
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["state_ids"], batch["action_ids"], batch["owners"])
            new_log_probs, entropy = selected_log_probs(logits, batch["offsets"], batch["labels"])
            values = model.value(batch["state_ids"])

            ratio = torch.exp(new_log_probs - batch["old_log_probs"])
            unclipped = ratio * batch["advantages"]
            clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * batch["advantages"]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch["returns"])
            entropy_mean = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            approx_kl = (batch["old_log_probs"] - new_log_probs).mean().detach()
            count = len(batch["labels"])
            totals["examples"] += count
            totals["policy_loss"] += float(policy_loss.item()) * count
            totals["value_loss"] += float(value_loss.item()) * count
            totals["entropy"] += float(entropy_mean.item()) * count
            totals["approx_kl"] += float(approx_kl.item()) * count
            totals["loss"] += float(loss.item()) * count
            if args.target_kl > 0 and float(approx_kl.item()) > args.target_kl:
                early_stop = True
                break

        count = max(1, totals["examples"])
        epoch_metrics = {
            "epoch": epoch,
            "examples": totals["examples"],
            "policy_loss": totals["policy_loss"] / count,
            "value_loss": totals["value_loss"] / count,
            "entropy": totals["entropy"] / count,
            "approx_kl": totals["approx_kl"] / count,
            "loss": totals["loss"] / count,
            "global_step": global_step,
            "elapsed_seconds": time.time() - started,
            "early_stop": early_stop,
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch} loss={epoch_metrics['loss']:.4f} "
            f"policy={epoch_metrics['policy_loss']:.4f} value={epoch_metrics['value_loss']:.4f} "
            f"entropy={epoch_metrics['entropy']:.4f} kl={epoch_metrics['approx_kl']:.5f}"
        )
        if early_stop:
            break

    metrics = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rollouts_path": args.rollouts,
        "out_dir": args.out_dir,
        "init_checkpoint": args.init_checkpoint,
        "resume_optimizer": args.resume_optimizer,
        "iteration": args.iteration,
        "global_step": global_step,
        "config": vars(args),
        "model_config": model_config,
        "examples": len(rows),
        "excluded_request_types": sorted(excluded_request_types),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "history": history,
    }

    torch.save({
        "checkpoint_type": "ppo_actor_critic",
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

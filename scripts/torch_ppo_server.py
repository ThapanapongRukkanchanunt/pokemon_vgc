import argparse
import json
import sys

import torch

from torch_alpha_model import (
    action_tokens,
    actor_critic_from_config,
    encode_tokens,
    resolve_device,
    state_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(description="JSONL scorer for PPO actor-critic checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_actor_critic(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = dict(checkpoint["model_config"])
    config["architecture"] = "alphastar_like_actor_critic_v1"
    model = actor_critic_from_config(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, config, checkpoint.get("checkpoint_type"), missing, unexpected


def tensor_from_ids(items, device):
    max_length = max(len(item) for item in items)
    result = torch.zeros((len(items), max_length), dtype=torch.long, device=device)
    for index, item in enumerate(items):
        result[index, :len(item)] = torch.tensor(item, dtype=torch.long, device=device)
    return result


def score_example(model, config, example, legal_actions, device):
    state_ids = encode_tokens(
        state_tokens(example),
        config["vocab_size"],
        config["max_state_tokens"],
    )
    action_ids = [
        encode_tokens(
            action_tokens(action),
            config["vocab_size"],
            config["max_action_tokens"],
        )
        for action in legal_actions
    ]
    if not action_ids:
        return {"scores": [], "log_probs": [], "probabilities": [], "value": 0.0, "entropy": 0.0}

    state_tensor = tensor_from_ids([state_ids], device)
    action_tensor = tensor_from_ids(action_ids, device)
    owners = torch.zeros(len(action_ids), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(state_tensor, action_tensor, owners)
        log_probs = torch.nn.functional.log_softmax(logits, dim=0)
        probabilities = torch.exp(log_probs)
        entropy = -(probabilities * log_probs).sum()
        value = model.value(state_tensor)[0]
    return {
        "scores": [float(value) for value in logits.detach().cpu().tolist()],
        "log_probs": [float(value) for value in log_probs.detach().cpu().tolist()],
        "probabilities": [float(value) for value in probabilities.detach().cpu().tolist()],
        "value": float(value.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, config, checkpoint_type, missing, unexpected = load_actor_critic(args.checkpoint, device)
    print(json.dumps({
        "ready": True,
        "device": str(device),
        "checkpoint": args.checkpoint,
        "checkpoint_type": checkpoint_type,
        "architecture": config.get("architecture"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            legal_actions = request.get("legal_actions") or []
            result = score_example(model, config, request.get("example") or {}, legal_actions, device)
            print(json.dumps({
                "id": request.get("id"),
                "ok": True,
                **result,
            }), flush=True)
        except Exception as error:
            print(json.dumps({
                "id": request.get("id") if "request" in locals() else None,
                "ok": False,
                "error": str(error),
            }), flush=True)


if __name__ == "__main__":
    main()

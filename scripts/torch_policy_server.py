import argparse
import json
import sys

import torch

from torch_alpha_model import (
    action_tokens,
    encode_tokens,
    model_from_config,
    resolve_device,
    state_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(description="JSONL scorer for PyTorch AlphaStar-like policy checkpoints.")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_policy(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["model_config"]
    model = model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


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
        return []
    state_tensor = tensor_from_ids([state_ids], device)
    action_tensor = tensor_from_ids(action_ids, device)
    owners = torch.zeros(len(action_ids), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(state_tensor, action_tensor, owners)
    return [float(value) for value in logits.detach().cpu().tolist()]


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, config = load_policy(args.policy_checkpoint, device)
    print(json.dumps({
        "ready": True,
        "device": str(device),
        "checkpoint": args.policy_checkpoint,
        "architecture": config.get("architecture"),
    }), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            legal_actions = request.get("legal_actions") or []
            scores = score_example(model, config, request.get("example") or {}, legal_actions, device)
            print(json.dumps({
                "id": request.get("id"),
                "ok": True,
                "scores": scores,
            }), flush=True)
        except Exception as error:
            print(json.dumps({
                "id": request.get("id") if "request" in locals() else None,
                "ok": False,
                "error": str(error),
            }), flush=True)


if __name__ == "__main__":
    main()

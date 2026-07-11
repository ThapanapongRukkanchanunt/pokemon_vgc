import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn

PAD_ID = 0


def to_id(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def stable_hash(text):
    value = 2166136261
    for char in str(text):
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def seed_to_int(seed):
    return stable_hash(seed) % (2 ** 31)


def token_id(token, vocab_size):
    return stable_hash(token) % (vocab_size - 1) + 1


def hp_bucket(condition):
    text = str(condition or "")
    if "fnt" in text:
        return "fnt"
    match = re.search(r"(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)", text)
    if not match:
        return "unknown"
    current = float(match.group(1))
    maximum = float(match.group(2))
    if maximum <= 0:
        return "unknown"
    ratio = current / maximum
    if ratio <= 0.25:
        return "0_25"
    if ratio <= 0.5:
        return "25_50"
    if ratio <= 0.75:
        return "50_75"
    return "75_100"


def species_from_details(details):
    if not details:
        return ""
    return str(details).split(",")[0].strip()


def turn_bucket(turn):
    try:
        value = float(turn)
    except (TypeError, ValueError):
        return "0"
    if value <= 0:
        return "0"
    if value <= 2:
        return "1_2"
    if value <= 5:
        return "3_5"
    if value <= 10:
        return "6_10"
    return "11_plus"


def add_token(tokens, token):
    if token:
        tokens.append(str(token))


def state_tokens(example):
    request = ((example.get("state") or {}).get("request") or {})
    public_state = ((example.get("state") or {}).get("public_state") or {})
    team_context = ((example.get("state") or {}).get("team_context") or {})
    diagnostics = example.get("agent_diagnostics") or ((example.get("state") or {}).get("agent_diagnostics") or {})
    tokens = []

    add_token(tokens, f"request_type:{example.get('request_type', 'unknown')}")
    add_token(tokens, f"side:{example.get('side', 'unknown')}")
    add_token(tokens, f"team:{example.get('team', 'unknown')}")
    add_token(tokens, f"lead:{example.get('lead', 'unknown')}")
    add_token(tokens, f"turn:{turn_bucket(example.get('turn'))}")

    own_pokemon = ((request.get("side") or {}).get("pokemon") or [])
    add_token(tokens, f"own_team_size:{len(own_pokemon)}")
    alive_count = 0
    for index, pokemon in enumerate(own_pokemon):
        species = to_id(species_from_details(pokemon.get("details")) or pokemon.get("species") or pokemon.get("ident"))
        if not species:
            continue
        condition = pokemon.get("condition")
        if "fnt" not in str(condition or ""):
            alive_count += 1
        add_token(tokens, f"own_slot:{index}:species:{species}")
        add_token(tokens, f"own_species:{species}")
        add_token(tokens, f"own_slot:{index}:hp:{hp_bucket(condition)}")
        if pokemon.get("active"):
            add_token(tokens, f"own_active:{species}")
        for move in pokemon.get("moves") or []:
            add_token(tokens, f"own_known_move:{to_id(move)}")
    add_token(tokens, f"own_alive:{alive_count}")

    for relation, key in [("own_roster", "own_team"), ("foe_roster", "opponent_team")]:
        team = team_context.get(key) if isinstance(team_context, dict) else None
        if not isinstance(team, dict):
            continue
        add_token(tokens, f"{relation}:id:{to_id(team.get('id'))}")
        add_token(tokens, f"{relation}:representative_mega:{to_id(team.get('representative_mega'))}")
        for mega in team.get("primary_megas") or []:
            add_token(tokens, f"{relation}:mega:{to_id(mega)}")
        for set_data in team.get("sets") or []:
            if not isinstance(set_data, dict):
                continue
            slot = set_data.get("slot", "unknown")
            species = to_id(set_data.get("species"))
            if not species:
                continue
            add_token(tokens, f"{relation}:slot:{slot}:species:{species}")
            add_token(tokens, f"{relation}:species:{species}")
            add_token(tokens, f"{relation}:slot:{slot}:item:{to_id(set_data.get('item'))}")
            add_token(tokens, f"{relation}:slot:{slot}:ability:{to_id(set_data.get('ability'))}")
            add_token(tokens, f"{relation}:slot:{slot}:nature:{to_id(set_data.get('nature'))}")
            evs = str(set_data.get("evs") or "")
            for ev_part in evs.split("/"):
                normalized = to_id(ev_part)
                if normalized:
                    add_token(tokens, f"{relation}:slot:{slot}:ev:{normalized}")

    predicted_back = team_context.get("predicted_opponent_back") if isinstance(team_context, dict) else None
    if isinstance(predicted_back, list):
        add_token(tokens, f"foe_predicted_back_count:{len(predicted_back)}")
        for set_data in predicted_back:
            if not isinstance(set_data, dict):
                continue
            slot = set_data.get("slot", "unknown")
            species = to_id(set_data.get("species"))
            if not species:
                continue
            add_token(tokens, f"foe_predicted_back:slot:{slot}:species:{species}")
            add_token(tokens, f"foe_predicted_back:species:{species}")
            add_token(tokens, f"foe_predicted_back:slot:{slot}:item:{to_id(set_data.get('item'))}")
            add_token(tokens, f"foe_predicted_back:slot:{slot}:nature:{to_id(set_data.get('nature'))}")
            evs = str(set_data.get("evs") or "")
            for ev_part in evs.split("/"):
                normalized = to_id(ev_part)
                if normalized:
                    add_token(tokens, f"foe_predicted_back:slot:{slot}:ev:{normalized}")

    for active_index, active in enumerate(request.get("active") or []):
        if active.get("canMegaEvo"):
            add_token(tokens, f"active:{active_index}:can_mega")
        for move_index, move in enumerate(active.get("moves") or []):
            move_id = to_id(move.get("id") or move.get("move"))
            if not move_id:
                continue
            add_token(tokens, f"active:{active_index}:move:{move_index}:{move_id}")
            add_token(tokens, f"available_move:{move_id}")
            add_token(tokens, f"move_target:{move_id}:{move.get('target', 'unknown')}")
            if move.get("disabled"):
                add_token(tokens, f"active:{active_index}:move_disabled:{move_id}")

    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list):
        add_token(tokens, "force_switch:" + "_".join("1" if item else "0" for item in force_switch))

    if request.get("teamPreview"):
        add_token(tokens, "team_preview")
        add_token(tokens, f"team_preview_size:{request.get('maxChosenTeamSize', len(own_pokemon))}")

    active_by_side = (public_state.get("active") or {})
    own_side = example.get("side")
    foe_side = "p2" if own_side == "p1" else "p1"
    for side, actives in active_by_side.items():
        relation = "own_public" if side == own_side else ("foe_public" if side == foe_side else f"public_{side}")
        for slot, active in enumerate(actives or []):
            if not active:
                continue
            species = to_id(active.get("species"))
            if not species:
                continue
            add_token(tokens, f"{relation}:slot:{slot}:species:{species}")
            add_token(tokens, f"{relation}:species:{species}")
            add_token(tokens, f"{relation}:slot:{slot}:hp:{hp_bucket(active.get('condition'))}")
            if active.get("fainted"):
                add_token(tokens, f"{relation}:slot:{slot}:fainted")

    hmm_belief = diagnostics.get("hmm_belief") if isinstance(diagnostics, dict) else None
    if isinstance(hmm_belief, dict):
        top_state = to_id(hmm_belief.get("top_state"))
        if top_state:
            add_token(tokens, f"hmm_top_state:{top_state}")
        probabilities = hmm_belief.get("probabilities") or {}
        if isinstance(probabilities, dict):
            for state, probability in sorted(probabilities.items()):
                try:
                    bucket = int(max(0, min(1, float(probability))) * 10)
                except (TypeError, ValueError):
                    continue
                state_id = to_id(state)
                add_token(tokens, f"hmm_state:{state_id}:p{bucket}")
                if bucket >= 4:
                    add_token(tokens, f"hmm_likely:{state_id}")

    return tokens


def action_tokens(choice):
    text = re.sub(r"\s+", " ", str(choice or "").strip().lower())
    team_match = re.fullmatch(r"team (\d+)", text)
    if team_match:
        slots = list(team_match.group(1))
        leads = sorted(slots[:2])
        backs = sorted(slots[2:])
        text = "team " + "".join(leads + backs)
    tokens = [f"choice:{text}"]
    commands = [part.strip() for part in text.split(",") if part.strip()]
    kinds = []
    for command_index, command in enumerate(commands):
        parts = command.split()
        if not parts:
            continue
        kind = parts[0]
        kinds.append(kind)
        add_token(tokens, f"cmd:{command_index}:kind:{kind}")
        add_token(tokens, f"kind:{kind}")
        if kind == "move":
            add_token(tokens, f"move_index:{parts[1] if len(parts) > 1 else 'unknown'}")
            if "mega" in parts:
                add_token(tokens, "mega")
            for part in parts[2:]:
                if re.fullmatch(r"-?\d+", part):
                    add_token(tokens, f"target:{part}")
        elif kind == "switch":
            add_token(tokens, f"switch_slot:{parts[1] if len(parts) > 1 else 'unknown'}")
        elif kind == "team":
            spec = parts[1] if len(parts) > 1 else ""
            add_token(tokens, f"team_spec:{spec}")
            add_token(tokens, f"team_lead_pair:{'+'.join(spec[:2])}")
            add_token(tokens, f"team_back_pair:{'+'.join(spec[2:])}")
            for index, slot in enumerate(spec):
                add_token(tokens, f"team_pick:{index}:{slot}")
                if index < 2:
                    add_token(tokens, f"team_lead:{index}:{slot}")
    if kinds:
        add_token(tokens, "command_kinds:" + "+".join(kinds))
    return tokens


def encode_tokens(tokens, vocab_size, max_tokens):
    ids = [token_id(token, vocab_size) for token in tokens[:max_tokens]]
    if not ids:
        ids = [token_id("empty", vocab_size)]
    return ids


def load_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_examples(examples, validation_split, seed):
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    validation_count = int(len(shuffled) * validation_split)
    return shuffled[validation_count:], shuffled[:validation_count]


def pad_sequences(sequences, pad_value=PAD_ID):
    max_length = max(len(item) for item in sequences)
    result = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)
    for index, item in enumerate(sequences):
        result[index, :len(item)] = torch.tensor(item, dtype=torch.long)
    return result


class PolicyDataset(torch.utils.data.Dataset):
    def __init__(self, examples, vocab_size, max_state_tokens, max_action_tokens):
        self.examples = examples
        self.vocab_size = vocab_size
        self.max_state_tokens = max_state_tokens
        self.max_action_tokens = max_action_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        legal_actions = example.get("legal_actions") or []
        label = example.get("label_action")
        label_index = example.get("label_action_index")
        if not isinstance(label_index, int) or label_index < 0 or label_index >= len(legal_actions) or legal_actions[label_index] != label:
            label_index = legal_actions.index(label)
        return {
            "state": encode_tokens(state_tokens(example), self.vocab_size, self.max_state_tokens),
            "actions": [
                encode_tokens(action_tokens(action), self.vocab_size, self.max_action_tokens)
                for action in legal_actions
            ],
            "label_index": label_index,
            "request_type": example.get("request_type", "unknown"),
        }


class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, examples, vocab_size, max_state_tokens, max_action_tokens):
        self.examples = examples
        self.vocab_size = vocab_size
        self.max_state_tokens = max_state_tokens
        self.max_action_tokens = max_action_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        action = example.get("action") or example.get("label_action") or example.get("chosen_action")
        target = example.get("target", example.get("win_target"))
        if target not in (0, 1):
            raise ValueError(f"{example.get('example_id', index)} target must be 0 or 1")
        return {
            "state": encode_tokens(state_tokens(example), self.vocab_size, self.max_state_tokens),
            "action": encode_tokens(action_tokens(action), self.vocab_size, self.max_action_tokens),
            "target": float(target),
            "request_type": example.get("request_type", "unknown"),
        }


def collate_policy(batch):
    state_ids = pad_sequences([item["state"] for item in batch])
    action_sequences = []
    owners = []
    offsets = []
    labels = []
    cursor = 0
    for owner, item in enumerate(batch):
        offsets.append((cursor, cursor + len(item["actions"])))
        labels.append(item["label_index"])
        for action in item["actions"]:
            action_sequences.append(action)
            owners.append(owner)
        cursor += len(item["actions"])
    return {
        "state_ids": state_ids,
        "action_ids": pad_sequences(action_sequences),
        "owners": torch.tensor(owners, dtype=torch.long),
        "offsets": offsets,
        "labels": labels,
    }


def collate_value(batch):
    return {
        "state_ids": pad_sequences([item["state"] for item in batch]),
        "action_ids": pad_sequences([item["action"] for item in batch]),
        "owners": torch.arange(len(batch), dtype=torch.long),
        "targets": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
    }


class AlphaStarLikeScorer(nn.Module):
    def __init__(self, vocab_size=65536, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def masked_mean(self, values, ids):
        mask = (ids != PAD_ID).unsqueeze(-1)
        summed = (values * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1)
        return summed / counts

    def encode_state(self, state_ids):
        state_emb = self.embedding(state_ids)
        key_padding_mask = state_ids == PAD_ID
        encoded = self.state_encoder(state_emb, src_key_padding_mask=key_padding_mask)
        return self.masked_mean(encoded, state_ids)

    def encode_action(self, action_ids):
        action_emb = self.embedding(action_ids)
        return self.action_projection(self.masked_mean(action_emb, action_ids))

    def forward(self, state_ids, action_ids, owners):
        state_vecs = self.encode_state(state_ids)
        action_vecs = self.encode_action(action_ids)
        selected_state = state_vecs[owners]
        features = torch.cat([
            selected_state,
            action_vecs,
            selected_state * action_vecs,
            torch.abs(selected_state - action_vecs),
        ], dim=-1)
        return self.score_head(features).squeeze(-1)


class AlphaStarLikeActorCritic(AlphaStarLikeScorer):
    def __init__(self, vocab_size=65536, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def value(self, state_ids):
        return self.value_head(self.encode_state(state_ids)).squeeze(-1)


def create_model_config(args):
    return {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "max_state_tokens": args.max_state_tokens,
        "max_action_tokens": args.max_action_tokens,
        "architecture": "alphastar_like_legal_action_scorer_v1",
    }


def create_actor_critic_config(args):
    config = create_model_config(args)
    config["architecture"] = "alphastar_like_actor_critic_v1"
    return config


def model_from_config(config):
    return AlphaStarLikeScorer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
    )


def actor_critic_from_config(config):
    return AlphaStarLikeActorCritic(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
    )


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return resolved


def policy_loss_and_accuracy(logits, offsets, labels):
    losses = []
    correct = 0
    for (start, end), label in zip(offsets, labels):
        row = logits[start:end].unsqueeze(0)
        target = torch.tensor([label], dtype=torch.long, device=logits.device)
        losses.append(torch.nn.functional.cross_entropy(row, target))
        if int(torch.argmax(row, dim=1).item()) == label:
            correct += 1
    return torch.stack(losses).mean(), correct, len(labels)


def binary_metrics(logits, targets):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    predictions = torch.sigmoid(logits)
    correct = ((predictions >= 0.5).float() == targets).sum().item()
    brier = torch.mean((predictions - targets) ** 2).item()
    return loss, correct, targets.numel(), brier


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")

import math

import torch

from train_ppo_torch import selected_log_probs


logits = torch.log(torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32))
offsets = [(0, 3)]
labels = torch.tensor([1], dtype=torch.long)

mixed_log_prob, mixed_entropy = selected_log_probs(
    logits,
    offsets,
    labels,
    torch.tensor([0.2], dtype=torch.float32),
)
expected_probability = 0.8 * 0.2 + 0.2 / 3
assert math.isclose(math.exp(float(mixed_log_prob[0])), expected_probability, rel_tol=1e-6)
assert float(mixed_entropy[0]) > 0

plain_log_prob, _ = selected_log_probs(
    logits,
    offsets,
    labels,
    torch.tensor([0.0], dtype=torch.float32),
)
assert math.isclose(math.exp(float(plain_log_prob[0])), 0.2, rel_tol=1e-6)

print("PASS PPO learner recomputes the epsilon-softmax behavior policy")

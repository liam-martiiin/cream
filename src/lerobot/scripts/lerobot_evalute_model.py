import torch
import random
from lerobot.datasets import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load policy
policy = ACTPolicy.from_pretrained(
    "/home/russelbenliam/lerobot/lerobot-so101-lab/output/tracingACTpressure2/checkpoints/070000/pretrained_model"
)
policy = policy.to(device)
policy.eval()

# 2. Load dataset
dataset = LeRobotDataset("CelerySticks/TracingLineDataHeightVarried1")
batch_size = 8

# After: dataset = LeRobotDataset("CelerySticks/TracingLineDataHeightVarried1")
state_names = dataset.features["observation.state"]["names"]
print("State component names (in order):", state_names)

# Helper to build a batch from random indices
def get_random_batch(n):
    indices = random.sample(range(len(dataset)), n)
    samples = [dataset[i] for i in indices]
    obs_keys = [k for k in samples[0].keys() if k.startswith("observation")]
    batch_input = {k: torch.stack([s[k] for s in samples]).to(device) for k in obs_keys}
    return batch_input, obs_keys

# Build a random batch
batch_input, obs_keys = get_random_batch(batch_size)

# 3. Baseline actions with a FRESH policy state
policy.reset()   # clear history
with torch.no_grad():
    baseline_actions = policy.select_action(batch_input)

print("Baseline action shape:", baseline_actions.shape)
print("Sample baseline actions (first 2):")
print(baseline_actions[:2])

# 4. Ablate each component of observation.state (with per-call reset)
state_tensor = batch_input["observation.state"]
state_dim = state_tensor.shape[1]

component_deltas = {}
for i, name in enumerate(state_names):
    ablated_input = {k: v.clone() for k, v in batch_input.items()}
    ablated_input["observation.state"][:, i] = 0.0   # or use mean

    policy.reset()   # CRITICAL: reset history before this call
    with torch.no_grad():
        ablated_actions = policy.select_action(ablated_input)

    diff = torch.norm(baseline_actions - ablated_actions, dim=1)
    avg_delta = diff.mean().item()
    component_deltas[name] = avg_delta

print("\nPer‑component importance (L2 change):")
for k, v in sorted(component_deltas.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v:.6f}")

# 5. Ablate each whole observation key (with per-call reset)
print("\nOverall observation‑key importances:")
key_deltas = {}
for key in obs_keys:
    ablated_input = {k: v.clone() for k, v in batch_input.items()}
    ablated_input[key] = torch.zeros_like(ablated_input[key])

    policy.reset()   # CRITICAL: reset before this call
    with torch.no_grad():
        ablated_actions = policy.select_action(ablated_input)

    diff = torch.norm(baseline_actions - ablated_actions, dim=1)
    avg_delta = diff.mean().item()
    key_deltas[key] = avg_delta

for k, v in sorted(key_deltas.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v:.6f}")
import numpy as np
import torch

# ----- helpers -----
relu = torch.nn.ReLU()
def forward(h, W, b):
    h_next = W @ h + b
    s = relu(h_next).sum()
    return h_next, s

def pretty(name, h_next, s):
    print(f"{name}:")
    print("  h^{l+1} =", torch.round(h_next,decimals=6))
    print("  s =", float(torch.round(s,decimals=6)))
    print()

# ----- define circuit -----
W = torch.tensor([
    [0.75, 0.25,  0.0,  0.5],
    [0.0, 1.0,  0.0,  0.0],
    [1.0, 1.0, -1.0, -1.0],
]).float()
b = torch.tensor([-0.5, -0.5, 0.0]).float()

print("Weights:", W)
print("Bias:", b)

# Class A (two cases)
hA1 = torch.tensor([1, 0, 1, 0]).float()
hA2 = torch.tensor([0, 1, 1, 0]).float()

# Class B (two cases)
hB1 = torch.tensor([0, 0, 1, 0]).float()
hB2 = torch.tensor([0, 0, 1, 1]).float()

# ----- native runs (A, B) -----
hA1_next, sA1 = forward(hA1, W, b)   # expect [0.25, -0.5,  0], s=0.25
hA2_next, sA2 = forward(hA2, W, b)   # expect [-0.25, 0.5,  0], s=0.5
hB1_next, sB1 = forward(hB1, W, b)   # expect [-0.5,-0.5, -1],  s=0
hB2_next, sB2 = forward(hB2, W, b)   # expect [0.0,-0.5, -2],   s=0

assert sA1 > 0
assert sA2 > 0
assert sB1 <= 0
assert sB2 <= 0

print("A1:", hA1)
pretty("A case 1 (native)", hA1_next, sA1)
print("A2:", hA2)
pretty("A case 2 (native)", hA2_next, sA2)
print("B1:", hB1)
pretty("B case 1 (native)", hB1_next, sB1)
print("B2:", hB2)
pretty("B case 2 (native)", hB2_next, sB2)

# ----- mean-difference delta_{B->A} over all four (equal weight) -----
# (1/4) * sum_i sum_j (hA_i - hB_j)
A_cases = [hA1, hA2]
B_cases = [hB1, hB2]

delta = sum((a - b_) for a in A_cases for b_ in B_cases) / 4.0
print("delta_{B->A} =", delta, "\n")  # expect [0.5, 0.5, 0.0, -0.5]

# ----- intervene B cases with the mean-diff vector -----
hhat_B1 = hB1 + delta   # expect [0.5,0.5,1,-0.5]
hhat_B2 = hB2 + delta   # expect [0.5,0.5,1, 0.5]

hhat_B1_next, shat_B1 = forward(hhat_B1, W, b)  # expect [-0.25, 0.0,  0.5], s=0.5
hhat_B2_next, shat_B2 = forward(hhat_B2, W, b)  # expect [0.25,  0.0, -0.5], s=0.25

print("B1+delta", hhat_B1)
pretty("Intervened B case 1 -> A", hhat_B1_next, shat_B1)
print("B2+delta", hhat_B2)
pretty("Intervened B case 2 -> A", hhat_B2_next, shat_B2)

assert shat_B1 > 0
assert shat_B2 > 0

# ----- intervene A cases with the mean-diff vector -----
hhat_A1 = hA1 - delta   # expect [0.5,-0.5,0,0.5]
hhat_A2 = hA2 - delta   # expect [-0.5,0.5,0,0.5]

hhat_A1_next, shat_A1 = forward(hhat_A1, W, b)  # expect [0, -1,  -1.5],     s=0
hhat_A2_next, shat_A2 = forward(hhat_A2, W, b)  # expect [-0.5,  0.0, -1.5], s=0

print("A1-delta", hhat_A1)
pretty("Intervened A case 1 -> B", hhat_A1_next, shat_A1)
print("A2-delta", hhat_A2)
pretty("Intervened A case 2 -> B", hhat_A2_next, shat_A2)

assert shat_A1 <= 0
assert shat_A2 <= 0

# ----- sanity check noted in text: if B had only a single case [0,0,1,0] -----
# Then delta' = average_A - that_single_B
A_mean = (hA1 + hA2) / 2.0
B_single = hB1.clone()  # [0,0,1,0]
delta_single = A_mean - B_single          # expect [0.5,0.5,0.0,0.0]
hhat_single = B_single + delta_single     # [0.5,0.5,1.0,0.0]
hhat_single_next, shat_single = forward(hhat_single, W, b)  # expect [0,0,0], y=0

print("If B consisted of only B case 1", B_single)
print("single-B Delta", delta_single)
pretty("Intervened single-B -> A", hhat_single_next, shat_single)

## ----- optional: assert the key equalities numerically -----
assert torch.allclose(delta, torch.tensor([0.5, 0.5, 0.0, -0.5]))
assert torch.allclose(hA1_next, torch.tensor([ 0.25, -0.5,  0.0]))
assert torch.allclose(hA2_next, torch.tensor([-0.25,  0.5,  0.0]))
assert torch.allclose(hB1_next, torch.tensor([-0.5, -0.5, -1.0]))
assert torch.allclose(hB2_next, torch.tensor([ 0.0, -0.5, -2.0]))
assert np.isclose(sA1.numpy(), 0.25) and np.isclose(sA2.numpy(), 0.5)
assert np.isclose(sB1.numpy(), 0.0)  and np.isclose(sB2.numpy(), 0.0)
assert torch.allclose(hhat_B1, torch.tensor([0.5,0.5,1.0,-0.5]))
assert torch.allclose(hhat_B2, torch.tensor([0.5,0.5,1.0, 0.5]))
# Note: first coord for B1 pre-ReLU is -0.25 (becomes 0 after ReLU), matching the writeup's ReLUed display
assert np.isclose(shat_B1.numpy(), 0.5) and np.isclose(shat_B2.numpy(), 0.25)
assert np.isclose(shat_single.numpy(), 0.0)
print("All checks passed.")


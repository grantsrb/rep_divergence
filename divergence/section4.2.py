import numpy as np

# ----- helpers -----
relu = lambda z: np.maximum(z, 0.0)
def forward(h, W, b):
    h_next = W @ h + b
    y = relu(h_next).sum()
    return h_next, y

def pretty(name, h_next, y):
    print(f"{name}:")
    print("  h^{l+1} =", h_next.round(6))
    print("  y =", float(np.round(y, 6)))
    print()

# ----- define circuit -----
W = np.array([
    [1.0, 0.0,  0.0,  0.5],
    [0.0, 1.0,  0.0,  0.0],
    [1.0, 1.0, -1.0, -1.0],
])
b = np.array([-0.5, -0.5, 0.0])

# Class A (two cases)
hA1 = np.array([1, 0, 1, 0], dtype=float)
hA2 = np.array([0, 1, 1, 0], dtype=float)

# Class B (two cases)
hB1 = np.array([0, 0, 1, 0], dtype=float)
hB2 = np.array([0, 0, 1, 1], dtype=float)

# ----- native runs (A, B) -----
hA1_next, yA1 = forward(hA1, W, b)   # expect [0.5, -0.5,  0], y=0.5
hA2_next, yA2 = forward(hA2, W, b)   # expect [-0.5, 0.5,  0], y=0.5
hB1_next, yB1 = forward(hB1, W, b)   # expect [-0.5,-0.5, -1], y=0
hB2_next, yB2 = forward(hB2, W, b)   # expect [0.0,-0.5, -2], y=0

pretty("A case 1 (native)", hA1_next, yA1)
pretty("A case 2 (native)", hA2_next, yA2)
pretty("B case 1 (native)", hB1_next, yB1)
pretty("B case 2 (native)", hB2_next, yB2)

# ----- mean-difference delta_{B->A} over all four (equal weight) -----
# (1/4) * sum_i sum_j (hA_i - hB_j)
A_cases = [hA1, hA2]
B_cases = [hB1, hB2]

delta = sum((a - b_) for a in A_cases for b_ in B_cases) / 4.0
print("delta_{B->A} =", delta, "\n")  # expect [0.5, 0.5, 0.0, -0.5]

# ----- intervene B cases with the mean-diff vector -----
hhat_B1 = hB1 + delta   # expect [0.5,0.5,1,-0.5]
hhat_B2 = hB2 + delta   # expect [0.5,0.5,1, 0.5]

hhat_B1_next, yhat_B1 = forward(hhat_B1, W, b)  # expect pre-ReLU [-0.25, 0.0,  0.5], y=0.5
hhat_B2_next, yhat_B2 = forward(hhat_B2, W, b)  # expect [0.25,  0.0, -0.5], y=0.25

pretty("Intervened B case 1 -> A", hhat_B1_next, yhat_B1)
pretty("Intervened B case 2 -> A", hhat_B2_next, yhat_B2)

# ----- sanity check noted in text: if B had only a single case [0,0,1,0] -----
# Then delta' = average_A - that_single_B
A_mean = (hA1 + hA2) / 2.0
B_single = hB1.copy()  # [0,0,1,0]
delta_single = A_mean - B_single          # expect [0.5,0.5,0.0,0.0]
hhat_single = B_single + delta_single     # [0.5,0.5,1.0,0.0]
hhat_single_next, yhat_single = forward(hhat_single, W, b)  # expect [0,0,0], y=0

pretty("Intervened with single-B baseline", hhat_single_next, yhat_single)

# ----- optional: assert the key equalities numerically -----
assert np.allclose(delta, np.array([0.5, 0.5, 0.0, -0.5]))
assert np.allclose(hA1_next, np.array([ 0.5, -0.5,  0.0]))
assert np.allclose(hA2_next, np.array([-0.5,  0.5,  0.0]))
assert np.allclose(hB1_next, np.array([-0.5, -0.5, -1.0]))
assert np.allclose(hB2_next, np.array([ 0.0, -0.5, -2.0]))
assert np.isclose(yA1, 0.5) and np.isclose(yA2, 0.5)
assert np.isclose(yB1, 0.0) and np.isclose(yB2, 0.0)
assert np.allclose(hhat_B1, np.array([0.5,0.5,1.0,-0.5]))
assert np.allclose(hhat_B2, np.array([0.5,0.5,1.0, 0.5]))
# Note: first coord for B1 pre-ReLU is -0.25 (becomes 0 after ReLU), matching the writeup's ReLUed display
assert np.isclose(yhat_B1, 0.5) and np.isclose(yhat_B2, 0.25)
assert np.isclose(yhat_single, 0.0)
print("All checks passed.")

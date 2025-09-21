import numpy as np

# ---------- helpers ----------
relu = lambda z: np.maximum(z, 0.0)

def forward_precursor(h, W, b):
    return relu(W @ h + b)

def forward_output(h_next, W_out, b_out):
    return relu(W_out @ h_next + b_out)

def pretty(name, vec):
    print(f"{name} =", np.round(vec, 6))

# ---------- precursor circuit (Appendix sup) ----------
W_prec = np.array([
    [1, 0, 0, 0.5, 0],
    [0, 1, 0, 0,   0],
    [1, 1,-1,-1,   1],
    [0, 0,0, 0,    1],
], dtype=float)
b_prec = np.array([-0.5, -0.5, 0, 0], dtype=float)

# h^ell cases
hA1 = np.array([1,0,1,0,0], dtype=float)
hA2 = np.array([0,1,1,0,0], dtype=float)

hB1 = np.array([0,0,1,0,0], dtype=float)
hB2 = np.array([0,0,1,0,1], dtype=float)
hB3 = np.array([0,0,1,1,0], dtype=float)

hC  = np.array([0,0,0,0,1], dtype=float)

print("=== Native precursor outputs ===")
for name,h in [
    ("A case1", hA1), ("A case2", hA2),
    ("B case1", hB1), ("B case2", hB2), ("B case3", hB3),
    ("C", hC)
]:
    pretty(name, forward_precursor(h, W_prec, b_prec))
print()

# ---------- mean-difference delta_{B->A} ----------
# Exclude B case2 (low probability); include B1, B3.
A_cases = [hA1, hA2]
B_cases = [hB1, hB3]
delta = sum((a - b) for a in A_cases for b in B_cases) / (len(A_cases)*len(B_cases))
pretty("delta_{B->A}", delta)
print()

# ---------- intervene on B cases ----------
print("=== Intervened precursor outputs (B+delta) ===")
for name,h in [("B case1", hB1), ("B case2", hB2), ("B case3", hB3)]:
    h_hat = h + delta
    pretty(f"{name} h_hat", h_hat)
print()

# ---------- result of intervened B cases ----------
print("=== Intervened next layer outputs (B+delta) ===")
for name,h in [("B case1", hB1), ("B case2", hB2), ("B case3", hB3)]:
    h_hat = h + delta
    pretty(f"{name} h^{'{'}l+1{'}'}_hat", forward_precursor(h_hat, W_prec, b_prec))
print()


# ---------- downstream classifier circuit (Section hidden behavior) ----------
W_out = np.array([
    [ 1.25,  1,  1, 0],
    [-1, -1, -1, 0],
    [ 0,  0,  1, 1],
], dtype=float)
b_out = np.array([-0.25, 0.25, -1.0])

print("=== Downstream output classification ===")
# Use native A, B, C states (from precursor layer)
for name,h in [
    ("A case1", forward_precursor(hA1, W_prec, b_prec)),
    ("A case2", forward_precursor(hA2, W_prec, b_prec)),
    ("B case1", forward_precursor(hB1, W_prec, b_prec)),
    ("B case2", forward_precursor(hB2, W_prec, b_prec)),
    ("B case3", forward_precursor(hB3, W_prec, b_prec)),
    ("C",       forward_precursor(hC,  W_prec, b_prec)),
]:
    y = forward_output(h, W_out, b_out)
    pretty(name+" y", y)
    print("\tClass:", np.argmax(y))
print()

print("=== Downstream outputs h_next for intervened B→A ===")
for name,h in [("B case1", hB1), ("B case2", hB2), ("B case3", hB3)]:
    h_hat = h + delta
    h_next_hat = forward_precursor(h_hat, W_prec, b_prec)
    y_hat = forward_output(h_next_hat, W_out, b_out)
    pretty(name+" h_next_hat", h_next_hat)

print()
print("=== Downstream outputs for intervened B→A ===")
for name,h in [("B case1", hB1), ("B case2", hB2), ("B case3", hB3)]:
    h_hat = h + delta
    h_next_hat = forward_precursor(h_hat, W_prec, b_prec)
    y_hat = forward_output(h_next_hat, W_out, b_out)
    pretty(name+" y_hat", y_hat)
    print("\tClass:", np.argmax(y_hat))

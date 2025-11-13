import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# ========== User settings ==========
DATA_PATH = "data.csv"   # replace if you have your own file
learning_rate = 0.1
epochs = 100
random_seed = 42
np.random.seed(random_seed)
# ===================================

# --- Load data (or generate synthetic) ---
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 3:
        x_cols = num_cols[:2]
        y_col = num_cols[2]
        X = df[x_cols].values
        y = df[y_col].values
    else:
        # fallback to named columns
        try:
            X = df[['x1','x2']].values
            y = df['y'].values
        except Exception:
            raise ValueError("data.csv found but unable to parse. Provide two numeric features + one label column.")
else:
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=random_seed, cluster_std=1.2)
    y = (y > 0).astype(int)

y = np.array(y).astype(int)
if set(np.unique(y)) - {0,1}:
    raise ValueError("Labels must be 0 or 1.")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize random weights and bias
n_features = X_scaled.shape[1]
weights = np.random.uniform(-0.5, 0.5, size=n_features)
bias = np.random.uniform(-0.5, 0.5)

def predict_step(x_row, w, b):
    return 1 if (np.dot(w, x_row) + b) >= 0 else 0

def decision_line(w, b, scaler, x_range=None, n=200):
    if x_range is None:
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    else:
        x_min, x_max = x_range
    xs = np.linspace(x_min, x_max, n)
    xs_scaled = scaler.transform(np.column_stack((xs, np.zeros_like(xs))))[:,0]
    if abs(w[1]) < 1e-8:
        ys = np.full_like(xs, X[:,1].mean())
    else:
        ys_scaled = -(w[0] * xs_scaled + b) / w[1]
        pts_scaled = np.column_stack((xs_scaled, ys_scaled))
        pts_orig = scaler.inverse_transform(pts_scaled)
        ys = pts_orig[:,1]
    return xs, ys

loss_history = []
boundaries = []
boundaries.append((weights.copy(), bias))  # initial boundary

for epoch in range(1, epochs+1):
    for xi, yi in zip(X_scaled, y):
        y_hat = predict_step(xi, weights, bias)
        error = yi - y_hat
        if error != 0:
            bias += learning_rate * error
            weights += learning_rate * error * xi
    logits = X_scaled.dot(weights) + bias
    probs = 1 / (1 + np.exp(-logits))
    eps = 1e-15
    probs = np.clip(probs, eps, 1-eps)
    epoch_loss = log_loss(y, probs)
    loss_history.append(epoch_loss)
    boundaries.append((weights.copy(), bias))

# --- Plot separation lines ---
plt.figure(figsize=(9,6))
pos = y == 1
neg = y == 0
plt.scatter(X[pos,0], X[pos,1], marker='o', label='class 1')
plt.scatter(X[neg,0], X[neg,1], marker='x', label='class 0')

# initial (red)
w0,b0 = boundaries[0]
xs, ys = decision_line(w0,b0,scaler)
plt.plot(xs, ys, color='red', linewidth=2, label='initial boundary (red)')

# dashed green for intermediate epochs (sample to avoid clutter)
indices = list(range(1, len(boundaries)-1))
max_plot = 20
if len(indices) > max_plot:
    step = max(1, len(indices)//max_plot)
    indices = indices[::step]
for idx in indices:
    wi, bi = boundaries[idx]
    xs, ys = decision_line(wi, bi, scaler)
    plt.plot(xs, ys, linestyle='--', linewidth=1, alpha=0.7, color='green')

# final (black)
wf, bf = boundaries[-1]
xs, ys = decision_line(wf,bf,scaler)
plt.plot(xs, ys, color='black', linewidth=2, label='final boundary (black)')

plt.xlabel('x1'); plt.ylabel('x2')
plt.title('Perceptron decision boundaries (initial red, intermediate dashed green, final black)')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
sep_path = "separation_plot.png"
plt.savefig(sep_path, dpi=150)
plt.show()

# --- Plot loss per epoch, highlight every 10th epoch ---
plt.figure(figsize=(8,4.5))
ep_range = np.arange(1, epochs+1)
plt.plot(ep_range, loss_history, linewidth=1.5)
every = 10
plt.scatter(ep_range[every-1::every], np.array(loss_history)[every-1::every], s=50, marker='x', label=f'every {every} epoch')
plt.xlabel('Epoch'); plt.ylabel('Log Loss')
plt.title('Log Loss per Epoch (every 10th epoch highlighted)')
plt.grid(True); plt.legend()
plt.tight_layout()
loss_path = "loss_plot.png"
plt.savefig(loss_path, dpi=150)
plt.show()

print("Saved plots:", sep_path, "and", loss_path)
print("Final weights:", weights, "bias:", bias)

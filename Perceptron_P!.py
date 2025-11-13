import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
rng = np.random.default_rng(7)  # change seed if you want different initial weights

# --------- parameters to play with ----------
LEARNING_RATE = 0.1   # try values like 0.01, 0.05, 0.1, 0.5
EPOCHS = 15            # "enough number of times" from the instructions
CSV_PATH = "data.csv"  # expects two feature columns and one label column
# -------------------------------------------

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Try common names; otherwise assume first two columns are features and last is label
    cols = [c.lower() for c in df.columns]
    if set(["x1", "x2", "y"]).issubset(cols):
        x1 = df[df.columns[cols.index("x1")]].to_numpy()
        x2 = df[df.columns[cols.index("x2")]].to_numpy()
        y  = df[df.columns[cols.index("y")]].to_numpy().astype(int)
    else:
        if df.shape[1] < 3:
            raise ValueError("data.csv must have at least 3 columns: two features and a binary label.")
        x1 = df.iloc[:, 0].to_numpy()
        x2 = df.iloc[:, 1].to_numpy()
        y  = df.iloc[:, -1].to_numpy().astype(int)
    X = np.column_stack([x1, x2])
    # Ensure labels are 0/1
    uniq = np.unique(y)
    assert set(uniq).issubset({0, 1}), "Labels must be 0/1."
    return X, y

def predict(X, w, b):
    # classification is 1 if w·x + b >= 0; else 0
    return ((X @ w + b) >= 0).astype(int)

def plot_points_and_line(ax, X, y, w, b, color, linestyle="-", linewidth=2, label=None):
    # Scatter points
    ax.scatter(X[y==0, 0], X[y==0, 1], c="tab:blue", marker="o", label="class 0")
    ax.scatter(X[y==1, 0], X[y==1, 1], c="tab:orange", marker="^", label="class 1")
    # Decision boundary: w1*x + w2*y + b = 0 -> y = -(w1*x + b)/w2 (unless w2==0)
     x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    if abs(w[1]) > 1e-12:
        xs = np.linspace(x_min, x_max, 200)
        ys = -(w[0]*xs + b) / w[1]
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    else:
        # vertical line x = -b/w1
        if abs(w[0]) > 1e-12:
            x_vert = -b / w[0]
            ax.axvline(x_vert, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Perceptron (heuristic updates)")
    ax.legend(loc="best")

def train_perceptron_heuristic(X, y, lr=0.1, epochs=10, plot_each_epoch=True):
    n_features = X.shape[1]
    # Random initial weights and bias (small values)
    w = rng.normal(scale=0.5, size=n_features)
    b = rng.normal(scale=0.5)
    history = [(w.copy(), b)]  # store initial for plotting

    fig, ax = plt.subplots(figsize=(7, 6))
    # Initial line in red
    plot_points_and_line(ax, X, y, w, b, color="red", linestyle="-", linewidth=2, label="initial")

    for ep in range(epochs):
        # Go through each point once (online update)
        for i in range(X.shape[0]):
            xi, yi = X[i], y[i]
            pred = 1 if (np.dot(w, xi) + b) >= 0 else 0
            if pred != yi:
                # Heuristic from your box:
                # if classification == 0 (pred=0) -> add lr and lr*xi
                # if classification == 1 (pred=1) -> subtract lr and lr*xi
                if pred == 0:
                    b += lr
                    w += lr * xi
                else:
                    b -= lr
                    w -= lr * xi
                    history.append((w.copy(), b))
        if plot_each_epoch:
            # dashed green for intermediate lines
            plot_points_and_line(ax, X, y, w, b, color="green", linestyle="--", linewidth=1.5,
                                 label=f"epoch {ep+1}")

    # Final line in black on top
    plot_points_and_line(ax, X, y, w, b, color="black", linestyle="-", linewidth=2.5, label="final")
    plt.tight_layout()
    return w, b, history, fig, ax

def main():
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"{CSV_PATH} not found.")
    X, y = load_data(CSV_PATH)
    w, b, history, fig, ax = train_perceptron_heuristic(
        X, y, lr=LEARNING_RATE, epochs=EPOCHS, plot_each_epoch=True
    )
    # Report simple metrics
    preds = predict(X, w, b)
    acc = (preds == y).mean()
    print(f"Final weights: {w}, bias: {b:.4f}, training accuracy: {acc:.3f}")
    plt.show()

if __name__ == "__main__":
    main()

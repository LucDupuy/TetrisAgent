import numpy as np
from main import run

np.set_printoptions(precision=6, suppress=True, floatmode='fixed', linewidth=np.inf)


def test_weights(episodes, weights, gamma=0.9):
    print("Testing", episodes, "with weights:", weights)
    Q_values = []  # List of all best_Qvalues
    Q_features = []  # List of all feature vectors of the best_Qvalues
    for i in range(1, episodes+1):
        _, _, _, _ = run(i, Q_values, Q_features, weights, gamma, False)


if __name__ == "__main__":
    weights = -np.ones(22)
    test_weights(100, weights)


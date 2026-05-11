# Write a python program to illustrate ART neural network.

import numpy as np

class ART1:

    def __init__(self, n_features, n_clusters, vigilance=0.5):

        self.n_features = n_features
        self.n_clusters = n_clusters
        self.vigilance = vigilance

        # Initialize weights
        self.bottom_up = np.ones((n_clusters, n_features)) / (1 + n_features)

        self.top_down = np.ones((n_clusters, n_features))

    def _choice_function(self, x):

        return np.dot(self.bottom_up, x)

    def _match_function(self, x, w):

        return np.sum(np.minimum(x, w)) / np.sum(x)

    def train(self, data):

        for i, x in enumerate(data):

            print(f"\nInput Pattern {i+1}: {x}")

            active = list(range(self.n_clusters))

            while active:

                T = self._choice_function(x)

                j = np.argmax(T)

                match = self._match_function(x, self.top_down[j])

                print(f"Checking cluster {j}, Match = {match:.2f}")

                if match >= self.vigilance:

                    print(f"Pattern assigned to cluster {j}")

                    # Update weights
                    self.top_down[j] = np.minimum(
                        x,
                        self.top_down[j]
                    )

                    self.bottom_up[j] = (
                        self.top_down[j]
                        / (0.5 + np.sum(self.top_down[j]))
                    )

                    break

                else:

                    active.remove(j)

                    self.bottom_up[j] = 0   # inhibit

            if not active:

                print("No suitable cluster found!")

# Example usage
data = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

art = ART1(
    n_features=4,
    n_clusters=3,
    vigilance=0.6
)

art.train(data)

import numpy as np

class Kohonen:
    def __init__(self, grid_size_k, input_dim, learning_rate = 0.5, radius = None):
        self.k = grid_size_k
        self.input_dim = input_dim
        self.lr0 = learning_rate
        self.r0 = radius if radius is not None else grid_size_k / 2
        self.weights = np.random.rand(self.k, self.k, self.input_dim)

    def _initialize_weights_from_data(self, data):
        indices = np.random.choice(data.shape[0], self.k ** 2, replace = False)
        self.weights = data[indices].reshape(self.k, self.k, self.input_dim)

    # Returns (row, col)  meaning the idx :)
    def _get_best_match(self, x):
        # Eucledian Distance
        distances = np.sum((self.weights - x) ** 2, axis = 2)
        best_match_idx = np.unravel_index(np.argmin(distances, axis = None), distances.shape)
        return best_match_idx  

    def _update_weights(self, x, best_match_idx, epoch, max_epochs):
        t = epoch / max_epochs
        current_learning_rate = self.lr0 * (1 - t)
        current_r = self.r0 * (1 - t)

        xx, yy = np.meshgrid(np.arange(self.k), np.arange(self.k))
        grid_coords = np.stack([yy, xx], axis=-1)

        dist_to_best_match_sq = np.sum((grid_coords - np.array(best_match_idx)) ** 2, axis = 2)

        # Will only affect those within reach of the radius current_r
        neighborhood_func = np.exp(-dist_to_best_match_sq / ( 2 * current_r ** 2))

        delta_w = current_learning_rate * neighborhood_func[..., np.newaxis] * ( x - self.weights )
        self.weights += delta_w

    def train(self, data, epochs):
        self._initialize_weights_from_data(data)

        for epoch in range(epochs):
            shuffled_data = data[np.random.permutation(data.shape[0])]

            for x in shuffled_data:
                best_match_idx = self._get_best_match(x)

                self._update_weights(x, best_match_idx, epoch, epochs)
    
    # Avg distance between each neuron and its neighbors
    def get_unified_matrix(self):
        u_matrix = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                w = self.weights[i, j]
                distances = []

                if i > 0: distances.append(np.linalg.norm( w - self.weights[i-1, j]))
                if i < self.k - 1: distances.append(np.linalg.norm( w - self.weights[i+1, j]))
                if j > 0: distances.append(np.linalg.norm( w - self.weights[i, j - 1]))
                if j < self.k - 1: distances.append(np.linalg.norm(w - self.weights[i, j + 1]))

                u_matrix[i, j] = np.mean(distances)
        
        return u_matrix

    # Amount of registers paired to each neuron
    def get_activation_map(self, data):
        activation_map = np.zeros((self.k, self.k))

        for x in data:
            best_match_idx = self._get_best_match(x)
            activation_map[best_match_idx] += 1

        return activation_map

    # Binds data to the best match
    def map_data(self, data):
        mapped_data = []
        for x in data:
            best_match_idx = self._get_best_match(x)
            mapped_data.append(best_match_idx)
        
        return mapped_data
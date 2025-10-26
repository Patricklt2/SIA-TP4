import numpy as np 

class Oja:
    def __init__(self, input_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.lr = learning_rate

        self.w = np.random.uniform(0, 1, input_dim)
        self.w = self.w / np.linalg.norm(self.w)
        self.convergence_history = []
        self.y_mean_history = []

    def train(self, data, epochs):
        n_samples = data.shape[0]
        y_mean_per_epoch = []
        
        for e in range(epochs):
            shuffled_data = data[np.random.permutation(n_samples)]
            w_prev = self.w.copy()
            y_epoch = []

            for x in shuffled_data:
                # y = w**T * x
                y = np.dot(x, self.w)

                # d_w = learning_Rate * y * (x - y * w)
                delta_w = self.lr * y * (x - y * self.w)
                self.w += delta_w

                self.w = self.w / np.linalg.norm(self.w)
                y_epoch.append(y)

            y_mean_per_epoch.append(np.mean(np.abs(y_epoch)))
            diff = np.linalg.norm(self.w - w_prev)

            self.convergence_history.append(diff)

        self.y_mean_history = y_mean_per_epoch

    
    def get_pc1_loads(self):
        return self.w

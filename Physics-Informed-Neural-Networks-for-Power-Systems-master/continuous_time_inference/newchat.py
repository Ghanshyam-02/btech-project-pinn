# Code developed based on the initial code by
# Mazziar Raissi: https://maziarraissi.github.io/PINNs
# @author: Georgios Misyris

# When publishing results based on this data/code, please cite:
# G. Misyris, A. Venzke, S. Chatzivasileiadis, "Physics-Informed
# Neural Networks for Power Systems", 2019. Available online:
# https://arxiv.org/abs/1911.03737

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
import time
import scipy.optimize
import os

np.random.seed(1234)
tf.random.set_seed(1234)


class PhysicsInformedNN(tf.Module):
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        super().__init__()
        self.lb = tf.convert_to_tensor(lb, dtype=tf.float32)
        self.ub = tf.convert_to_tensor(ub, dtype=tf.float32)

        self.x_u = tf.convert_to_tensor(X_u[:, 0:1], dtype=tf.float32)
        self.t_u = tf.convert_to_tensor(X_u[:, 1:2], dtype=tf.float32)
        self.u = tf.convert_to_tensor(u, dtype=tf.float32)

        self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
        self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)

        self.layers = layers
        self.nu = nu

        self.weights, self.biases = self.initialize_NN(layers)
        self.trainables = self.weights + self.biases

        self.optimizer = tf.keras.optimizers.Adam()

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=stddev), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = self.xavier_init([layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(len(self.layers) - 2):
            H = tf.tanh(tf.add(tf.matmul(H, self.weights[l]), self.biases[l]))
        return tf.add(tf.matmul(H, self.weights[-1]), self.biases[-1])

    def net_u(self, x, t):
        X = tf.concat([x, t], axis=1)
        return self.neural_net(X)

    def net_f(self, x, t):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, t])
                u = self.net_u(x, t)
            u_t = tape1.gradient(u, t)
        u_tt = tape2.gradient(u_t, t)
        del tape1, tape2
        f = 0.4 * u_tt + 0.15 * u_t + self.nu * tf.sin(u) - x
        return f

    def loss_fn(self):
        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss = tf.reduce_mean(tf.square(self.u - u_pred)) + tf.reduce_mean(tf.square(f_pred))
        return loss

    def train(self, epochs=10000):
        for epoch in range(epochs): 
            with tf.GradientTape() as tape:
                loss = self.loss_fn()
            grads = tape.gradient(loss, self.trainables)
            self.optimizer.apply_gradients(zip(grads, self.trainables))
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.numpy():.5e}')

    def predict(self, X_star):
        x = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        u_pred = self.net_u(x, t).numpy()
        f_pred = self.net_f(x, t).numpy()
        return u_pred, f_pred

    def get_weights(self):
        return tf.concat([tf.reshape(w, [-1]) for w in self.trainables], axis=0).numpy()

    def set_weights(self, flat_weights):
        pointer = 0
        for var in self.trainables:
            shape = var.shape
            size = tf.size(var).numpy()
            new_val = tf.convert_to_tensor(flat_weights[pointer:pointer + size].reshape(shape), dtype=tf.float32)
            var.assign(new_val)
            pointer += size

    def loss_and_grad(self, flat_weights):
        self.set_weights(flat_weights)
        with tf.GradientTape() as tape:
            loss_value = self.loss_fn()
        grads = tape.gradient(loss_value, self.trainables)
        grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0).numpy()
        return loss_value.numpy().astype(np.float64), grads_flat.astype(np.float64)

    def train_LBFGS(self):
        print("Starting L-BFGS-B optimization...")
        init_params = self.get_weights()

        result = scipy.optimize.minimize(fun=self.loss_and_grad,
                                         x0=init_params,
                                         jac=True,
                                         method='L-BFGS-B',
                                         options={'maxiter': 50000,
                                                  'maxfun': 50000,
                                                  'maxcor': 50,
                                                  'maxls': 50,
                                                  'ftol': 1e-12,
                                                  'gtol': 1e-12,
                                                  'eps': 1e-8,
                                                  'disp': True})
        self.set_weights(result.x)
        print("L-BFGS-B optimization complete.")


if __name__ == "__main__":
    nu = 0.2
    noise = 0.0

    N_u = 40
    N_f = 8000
    layers = [2, 10, 10, 10, 10, 10, 1]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.abspath(os.path.join(base_dir, '..', 'Data', 'swingEquation_inference.mat'))
    print(f"Loading .mat file from: {mat_path}")

    data = scipy.io.loadmat(mat_path)

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    lb = np.array([0.08, 0.0])
    ub = np.array([0.18, 20.0])

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)

    start_time = time.time()
    model.train(epochs=10000)
    elapsed = time.time() - start_time
    print('Adam training time: %.4f seconds' % elapsed)

    model.train_LBFGS()
    model.save_model('trained_pinn_swing_eq.npz')

    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Relative L2 Error in u: %e' % error_u)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    # Plot predicted solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, U_pred, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u_pred')
    plt.title('Predicted Solution')
    plt.show()

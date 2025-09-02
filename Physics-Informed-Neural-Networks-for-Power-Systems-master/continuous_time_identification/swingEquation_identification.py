#	Code developed based on the initial code by 
#	Mazziar Raissi: https://maziarraissi.github.io/PINNs

# 	@author: Georgios Misyris

#When publishing results based on this data/code, please cite:
#	G. Misyris, A. Venzke, S. Chatzivasileiadis, " Physics-Informed 
#	Neural Networks for Power Systems", 2019. Available online: 
#	https://arxiv.org/abs/1911.03737	

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import os

np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, u, layers, lb, ub, nu):
        self.lb = lb
        self.ub = ub

        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.u = u

        self.nu = nu
        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)

        self.x_tf = tf.convert_to_tensor(self.x, dtype=tf.float32)
        self.t_tf = tf.convert_to_tensor(self.t, dtype=tf.float32)
        self.u_tf = tf.convert_to_tensor(self.u, dtype=tf.float32)

        self.optimizer_Adam = tf.keras.optimizers.Adam()

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        lambda_1 = self.lambda_1  
        lambda_2 = self.lambda_2     

        u = self.net_u(x,t)

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, t])
                u = self.net_u(x, t)
            u_t = tape1.gradient(u, t)
        u_tt = tape2.gradient(u_t, t)
        f = lambda_1*u_tt + lambda_2*u_t + self.nu*tf.math.sin(u) - x
        return f

    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, lambda_2))

    def get_loss(self):
        u_pred = self.net_u(self.x_tf, self.t_tf)
        f_pred = self.net_f(self.x_tf, self.t_tf)
        loss = tf.reduce_mean(tf.square(self.u_tf - u_pred)) + tf.reduce_mean(tf.square(f_pred))
        return loss

    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            loss = self.get_loss()
        gradients = tape.gradient(loss, self.weights + self.biases + [self.lambda_1, self.lambda_2])
        self.optimizer_Adam.apply_gradients(zip(gradients, self.weights + self.biases + [self.lambda_1, self.lambda_2]))
        return loss

    def train(self, nIter):
        start_time = time.time()
        for it in range(nIter):
            loss_value = self.train_step()
            if it % 10 == 0:
                elapsed = time.time() - start_time
                l1 = self.lambda_1.numpy()[0]
                l2 = self.lambda_2.numpy()[0]
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' % 
                      (it, loss_value, l1, l2, elapsed))
                start_time = time.time()

    def predict(self, X_star):
        x = tf.convert_to_tensor(X_star[:,0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(X_star[:,1:2], dtype=tf.float32)
        u_star = self.net_u(x, t)
        f_star = self.net_f(x, t)
        return u_star.numpy(), f_star.numpy()

if __name__ == "__main__": 
    nu=0.2
    N_u=100
    layers = [2, 30, 30, 30, 30, 30,  1]

    # Dynamically get the current file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute path to .mat file
    mat_path = os.path.join(base_dir, '..', 'Data', 'swingEquation_identification.mat')
    mat_path = os.path.abspath(mat_path)

    # Optional debug print
    print(f"Loading .mat file from: {mat_path}")

    # Load the .mat file
    data = scipy.io.loadmat(mat_path)

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    lb=np.array([0.08 ,  0.        ])
    ub=np.array([0.18,  20.        ])

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    noise = 0.0            

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
    start_time = time.time()
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub, nu)
    model.train(10000)

    u_pred, f_pred = model.predict(X_star)
    elapsed = time.time() - start_time                

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    lambda_1_value = model.lambda_1.numpy()[0]
    lambda_2_value = model.lambda_2.numpy()[0]

    error_lambda_1 = np.abs(lambda_1_value - 0.25)/0.25*100
    error_lambda_2 = np.abs(lambda_2_value - 0.15)/0.15*100

    print('Error u: %e' % (error_u))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))

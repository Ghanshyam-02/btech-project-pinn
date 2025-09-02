# -----------------------------------------------------------------------------
#   Code developed based on the initial code by
#   Mazziar Raissi: https://maziarraissi.github.io/PINNs
#
#   @author: Georgios Misyris
#   TF2.x Conversion and Debugging by: Gemini (Google)
#
#   When publishing results based on this data/code, please cite:
#   G. Misyris, A. Venzke, S. Chatzivasileiadis, " Physics-Informed
#   Neural Networks for Power Systems", 2019. Available online:
#   https://arxiv.org/abs/1911.03737
# -----------------------------------------------------------------------------

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyDOE import lhs

# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN(tf.keras.Model):
    # Initialize the class
    def __init__(self, layers, lb, ub, nu):
        super(PhysicsInformedNN, self).__init__()
        
        self.lb = tf.constant(lb, dtype=tf.float32)
        self.ub = tf.constant(ub, dtype=tf.float32)
        self.nu = nu
        self.layers = layers
        
        # Initialize the neural network
        self.model = self.initialize_NN(layers)
    
    def initialize_NN(self, layers):
        """Initializes a sequential Keras model."""
        model = tf.keras.Sequential()
        # Input layer is handled by the input shape in the first Dense layer
        # Hidden layers with 'tanh' activation
        for i in range(len(layers) - 2):
            model.add(tf.keras.layers.Dense(layers[i+1],
                                            activation='tanh',
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            input_shape=(layers[i],) if i == 0 else ()))
        # Output layer with linear activation
        model.add(tf.keras.layers.Dense(layers[-1],
                                        activation=None,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal()))
        return model

    @tf.function
    def call(self, x, t):
        """Forward pass for u(x,t). Equivalent to net_u."""
        X = tf.concat([x, t], 1)
        # Normalize inputs
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        u = self.model(H)
        return u

    @tf.function
    def net_f(self, x, t):
        """Computes the residual of the PDE.
           f = 0.4*u_tt + 0.15*u_t + nu*sin(u) - x
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(t)
                u = self.call(x, t)
            u_t = tape1.gradient(u, t)
        u_tt = tape2.gradient(u_t, t)
        
        del tape1, tape2
        
        f = 0.4 * u_tt + 0.15 * u_t + self.nu * tf.math.sin(u) - x
        return f

    @tf.function
    def loss_fn(self, x_u, t_u, u_data, x_f, t_f):
        """Computes the total loss."""
        u_pred = self.call(x_u, t_u)
        f_pred = self.net_f(x_f, t_f)
        
        loss_u = tf.reduce_mean(tf.square(u_data - u_pred))
        loss_f = tf.reduce_mean(tf.square(f_pred))
        
        return loss_u + loss_f

    def _get_weights(self):
        """Returns the current weights of the model as a single flat tensor."""
        weights = self.trainable_variables
        weights_flat = tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)
        return weights_flat

    def _set_weights(self, weights_flat):
        """Sets the model weights from a single flat tensor."""
        shapes = [w.shape for w in self.trainable_variables]
        split_sizes = [tf.size(w) for w in self.trainable_variables]
        
        new_weights = tf.split(weights_flat, split_sizes)
        
        for i, (variable, new_weight) in enumerate(zip(self.trainable_variables, new_weights)):
            variable.assign(tf.reshape(new_weight, shapes[i]))
            
    def _value_and_gradients(self, x_u, t_u, u_data, x_f, t_f):
        """A factory function to create the value_and_gradients function for L-BFGS."""
        @tf.function
        def func(weights_flat):
            self._set_weights(weights_flat)
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss = self.loss_fn(x_u, t_u, u_data, x_f, t_f)
            
            grads = tape.gradient(loss, self.trainable_variables)
            grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            
            tf.print("Loss:", loss)
            
            return loss, grads_flat
        return func

    def train(self, X_u, u, X_f):
        """Trains the model using the L-BFGS optimizer."""
        x_u_tf = tf.convert_to_tensor(X_u[:, 0:1], dtype=tf.float32)
        t_u_tf = tf.convert_to_tensor(X_u[:, 1:2], dtype=tf.float32)
        u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
        x_f_tf = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
        t_f_tf = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)

        val_and_grad_fn = self._value_and_gradients(x_u_tf, t_u_tf, u_tf, x_f_tf, t_f_tf)

        init_params = self._get_weights()
        
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=val_and_grad_fn,
            initial_position=init_params,
            max_iterations=50000,
            num_correction_pairs=50,
            tolerance=1e-15, # Corresponds to ftol
            gradient_tolerance=1e-8 # Corresponds to gtol
        )
        
        self._set_weights(results.position)

    def predict(self, X_star):
        """Makes predictions at the given points."""
        x_star_tf = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        t_star_tf = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        
        u_star = self.call(x_star_tf, t_star_tf)
        f_star = self.net_f(x_star_tf, t_star_tf)
        
        return u_star.numpy(), f_star.numpy()

if __name__ == "__main__":
    
    nu = 0.2
    noise = 0.0

    N_u = 40
    N_f = 8000
    layers = [2, 10, 10, 10, 10, 10, 1]

    # Load data from .mat file
    try:
        data = scipy.io.loadmat('swingEquation_inference.mat')
    except FileNotFoundError:
        print("Error: 'swingEquation_inference.mat' not found.")
        print("Please ensure the data file is in the correct directory.")
        # As a fallback for demonstration, create dummy data.
        print("Creating dummy data to proceed...")
        t_dummy = np.linspace(0, 20, 201)
        x_dummy = np.linspace(0.08, 0.18, 101)
        X_dummy, T_dummy = np.meshgrid(x_dummy, t_dummy)
        usol_dummy = np.sin(np.pi * X_dummy) * np.cos(2 * np.pi * T_dummy/20)
        data = {'t': t_dummy, 'x': x_dummy, 'usol': usol_dummy}

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Domain bounds
    lb = np.array([0.08, 0.0])
    ub = np.array([0.18, 20.0])
    
    # Boundary and initial condition points
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T)) # t=0
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1])) # x=lb
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:])) # x=ub
    uu3 = Exact[:, -1:]
    
    X_u_train_all = np.vstack([xx1, xx2, xx3])
    u_train_all = np.vstack([uu1, uu2, uu3])

    # Collocation points for physics loss
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train_all))
    
    # --- CORRECTED SAMPLING LOGIC TO PREVENT INDEXERROR ---
    # To prevent potential indexing errors from shape mismatches, it is safer to
    # stack the input (X) and output (u) data together, perform sampling on the
    # combined array, and then split them. This ensures the indices are valid
    # for both and that the correspondence between X_u_train and u_train is
    # correctly maintained.
    
    # Combine the boundary/initial condition data
    all_u_data = np.hstack((X_u_train_all, u_train_all))
    
    # Randomly sample N_u points for training from the combined data
    idx = np.random.choice(all_u_data.shape[0], N_u, replace=False)
    sampled_u_data = all_u_data[idx, :]
    
    # Split the sampled data back into X_u_train and u_train
    X_u_train = sampled_u_data[:, 0:2]
    u_train = sampled_u_data[:, 2:3]
    # --- END OF CORRECTION ---
    
    # Initialize model
    model = PhysicsInformedNN(layers, lb, ub, nu)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    model.train(X_u_train, u_train, X_f_train)
    elapsed = time.time() - start_time
    print(f'Training finished. Time: {elapsed:.4f}s')
    
    # Make predictions
    print("\nMaking predictions...")
    start_time = time.time()
    u_pred, f_pred = model.predict(X_star)
    elapsed = time.time() - start_time
    print(f'Prediction time: {elapsed:.4f}s')
    
    # Calculate error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print(f'L2 Error for u: {error_u:e}')

    # Grid the prediction data for plotting
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    
    # --- Plotting ---
    print("Generating plots...")
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3)

    # Plot Exact Solution
    ax = plt.subplot(gs[0])
    im = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_title('Exact $u(x,t)$', fontsize=15)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')

    # Plot Predicted Solution
    ax = plt.subplot(gs[1])
    im = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_title('Predicted $u(x,t)$', fontsize=15)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')

    # Plot Absolute Error
    ax = plt.subplot(gs[2])
    im = ax.imshow(np.abs(Exact.T - U_pred.T), interpolation='nearest', cmap='rainbow',
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin='lower', aspect='auto', vmin=0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_title('Absolute Error', fontsize=15)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')

    plt.tight_layout()
    plt.show()
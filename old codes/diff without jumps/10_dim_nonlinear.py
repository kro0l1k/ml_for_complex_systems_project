import time
import numpy as np
import tensorflow as tf
import math
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt

LAMBDA = 0.05

tf.keras.backend.set_floatx('float64')

class Solver(object):
    def __init__(self,):
        self.valid_size = 512
        self.batch_size = 256
        self.num_iterations = 5000
        self.logging_frequency = 200
        self.lr_values = [5e-3, 5e-3, 5e-3]
        
        self.lr_boundaries = [2000, 4000]
        self.config = Config()

        self.model = WholeNet()
        self.y_init = self.model.y_init
        print("y_intial: ", self.y_init.numpy())
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.lr_boundaries, self.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        """Training the model"""
        start_time = time.time()
        training_history = []
        dW = self.config.sample(self.valid_size)
        print("dW shape: ", dW.shape) # (512, 100, 25) (M = valid size, d,T)
        valid_data = dW
        for step in range(self.num_iterations+1):
            if step % self.logging_frequency == 0:
                loss, cost, ratio = self.model(valid_data, training=True)
                y_init = self.y_init.numpy()[0][0]
                elapsed_time = time.time() - start_time
                training_history.append([step, cost, y_init, loss, ratio])
                print("step: %5u, loss: %.4e, Y0: %.4e, cost: %.4e,  elapsed time: %3u, ratio: %.4e" % (step, loss, y_init, cost, elapsed_time, ratio))
            self.train_step(self.config.sample(self.batch_size))
        # print('Y0_true: %.4e' % y_init)
        self.training_history = training_history

    @tf.function
    def train_step(self, train_data):
        """Updating the gradients"""
        with tf.GradientTape(persistent=True) as tape:
            loss, cost, ratio = self.model(train_data, training = True)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

class WholeNet(tf.keras.Model):
    """Building the neural network architecture"""
    def __init__(self):
        super(WholeNet, self).__init__()
        self.config = Config()
        self.y_init = self.add_weight(
            name = 'y_init',
            shape = (1, self.config.dim_y),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
            dtype=tf.dtypes.float64
        )
        self.z_net = FNNetZ()
        self.u_net = FNNetU()

    def call(self, dw, training):
        x_init = tf.ones([1, self.config.dim_x], dtype=tf.dtypes.float64) * 0.0
        time_stamp = np.arange(0, self.config.num_time_interval) * self.config.delta_t
        all_one_vec = tf.ones([tf.shape(dw)[0], 1], dtype=tf.dtypes.float64)
        x = tf.matmul(all_one_vec, x_init)
        y = tf.matmul(all_one_vec, self.y_init)
        l = 0.0 # The cost functional
        H = 0.0 # The constraint term
        for t in range(0, self.config.num_time_interval):
            data = time_stamp[t], x
            z = self.z_net(data, training=training)
            u = self.u_net(data, training=training)
            l = l + self.config.f_fn(time_stamp[t], x, u) * self.config.delta_t
            H = H + self.config.Hu_fn(time_stamp[t], x, y, z, u)
            b_ = self.config.b_fn(time_stamp[t], x, u)
            sigma_ = self.config.sigma_fn(time_stamp[t], x, u)
            f_ = self.config.Hx_fn(time_stamp[t], x, u, y, z)

            x = x + b_ * self.config.delta_t + sigma_ * dw[:, :, t]
            y = y - f_ * self.config.delta_t + z * dw[:, :, t]

        delta = y + self.config.hx_tf(self.config.total_T, x)
        loss = tf.reduce_mean(tf.reduce_sum(delta**2, 1, keepdims=True) + LAMBDA * H)
        ratio = tf.reduce_mean(tf.reduce_sum(delta**2, 1, keepdims=True)) / tf.reduce_mean(H)

        l = l + self.config.h_fn(self.config.total_T, x)
        cost = tf.reduce_mean(l)

        return loss, cost, ratio

class FNNetZ(tf.keras.Model):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetZ, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim_z
        self.dense_layers.append(tf.keras.layers.Dense(self.config.dim_z, activation=None))

    def call(self, inputs, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        t, x = inputs
        ts = tf.ones([tf.shape(x)[0], 1], dtype=tf.dtypes.float64) * t
        x = tf.concat([ts, x], axis=1)
        x = self.bn_layers[0](x, training=training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training=training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)
        return x

class FNNetU(tf.keras.Model):
    """ Define the feedforward neural network """
    def __init__(self):
        super(FNNetU, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim_u
        self.dense_layers.append(tf.keras.layers.Dense(self.config.dim_u, activation=None))

    def call(self, inputs, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        t, x = inputs
        # print(" t shape, x shape: ", t.shape, x.get_shape())
        ts = tf.ones([x.get_shape()[0], 1], dtype=tf.dtypes.float64) * t
        x = tf.concat([ts, x], axis=1)
        x = self.bn_layers[0](x, training=training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training=training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)
        # print("Final output shape: ", x.shape)
        return x

class Config(object):
    """Define the configs in the systems"""
    def __init__(self):
        super(Config, self).__init__()
        self.dim_x = 10
        self.dim_y = 10
        self.dim_z = 10
        self.dim_u = 10
        
        self.num_time_interval = 25
        self.total_T = 1.0
        self.delta_t = (self.total_T + 0.0) / self.num_time_interval
        self.sqrth = np.sqrt(self.delta_t)
        self.t_stamp = np.arange(0, self.num_time_interval) * self.delta_t

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.dim_x, self.num_time_interval]) * self.sqrth
        return dw_sample

    def f_fn(self, t, x, u):
        return tf.reduce_sum(u ** 2, 1, keepdims=True)

    def h_fn(self, t, x):
        return tf.math.log(0.5*(1 + tf.reduce_sum(x**2, 1, keepdims=True)))

    def b_fn(self, t, x, u):
        return 2 * u

    def sigma_fn(self, t, x, u):
        return np.sqrt(2)

    def Hx_fn(self, t, x, u, y, z):
        return 0

    def hx_tf(self, t, x):
        a = 1 + tf.reduce_sum(x ** 2, 1, keepdims=True)
        return 2 * x / a

    def Hu_fn(self, t, x, y, z, u):
        a = 2 * y - 2 * u
        return tf.reduce_sum(a**2, 1, keepdims=True)
        

def main():
    print('Training time 1:')
    solver = Solver()
    solver.train()
    k = 10
    ratio = '001'
    data = np.array(solver.training_history)
    output = np.zeros((len(data[:, 0]), 4 + k))
    output[:, 0] = data[:, 0] # step
    output[:, 1] = data[:, 2] # y_init
    output[:, 2] = data[:, 3] # loss
    output[:, 3] = data[:, 4] # ratio
    output[:, 4] = data[:, 1] # cost

    for i in range(k - 1):
        print('Training time %3u:' % (i + 2))
        solver = Solver()
        solver.train()
        data = np.array(solver.training_history)
        output[:, 5 + i] = data[:, 1]

    a = ['%d', '%.5e', '%.5e', '%.5e']
    for i in range(k):
        a.append('%.5e')
    np.savetxt('./LQ_data_'+ratio+'_d100.csv', output, fmt=a, delimiter=',')

    print('Solving is done!')

if __name__ == '__main__':
    main()
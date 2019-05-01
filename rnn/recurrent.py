import numpy as np
import time

from sklearn.utils import check_random_state
from scipy.special import expit

def sigmoid_prime(z):
    p = expit(z)
    return p*(1 - p)

sigmoid = expit
random_state = check_random_state(None)

class RecurrentNetwork:

    def __init__(self, n_hidden_neurons, learning_rate=1.0, n_epochs=20,
                 lmbda=1.0, mu=0.5, output_activation='softmax',
                 random_state=None, verbose=0):
        self.n_hidden_neurons = n_hidden_neurons
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.lmbda = lmbda
        self.mu = mu
        self.output_activation = output_activation
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(np.concatenate(y))
        classmap = {c: i for (i, c) in enumerate(self.classes_)}

        Y = []
        for yi in y:
            Yi = np.zeros((len(yi), len(self.classes_)))

            for t, yit in np.ndenumerate(yi):
                c = classmap[yit]
                Yi[t, c] = 1

            Y.append(Yi)
        Y = np.asarray(Y)

        self.n_neurons_ = [X[0].shape[1]] + [self.n_hidden_neurons] + \
                          [Y[0].shape[1]]

        self.init_parameters()
        self.online_gradient_descent(X, Y)

        return self

    def init_parameters(self):
        sdev = 1.0 / np.sqrt(self.n_neurons_[0] + self.n_neurons_[1])
        dim = (self.n_neurons_[1], self.n_neurons_[0])
        self.Wh_ = self.random_state_.normal(0, sdev, size=dim)
        self.bh_ = self.random_state_.randn(self.n_neurons_[1])

        dim = (self.n_neurons_[1], self.n_neurons_[1])
        self.Wr_ = self.random_state_.normal(0, sdev, size=dim)

        dim = (self.n_neurons_[2], self.n_neurons_[1])
        sdev = 1.0 / np.sqrt(self.n_neurons_[1])
        self.Wo_ = self.random_state_.normal(0, sdev, size=dim)
        self.bo_ = self.random_state_.randn(self.n_neurons_[2])

        # Velocities (momentum-based online gradient descent)
        self.Vwh_ = np.zeros(self.Wh_.shape)
        self.Vbh_ = np.zeros(self.bh_.shape)

        self.Vwr_ = np.zeros(self.Wr_.shape)

        self.Vwo_ = np.zeros(self.Wo_.shape)
        self.Vbo_ = np.zeros(self.bo_.shape)

    def online_gradient_descent(self, X, Y):
        for epoch in range(self.n_epochs):
            now = time.time()

            s = self.random_state_.permutation(X.shape[0])
            Xs, Ys = X[s], Y[s]

            for Xi, Yi in zip(Xs, Ys):
                self.gradient_descent_step(Xi, Yi, len(X))

            if self.verbose > 0:
                now, last = time.time(), now
                print('Epoch {0} ({1:.01f}s).'.format(epoch + 1, now - last))

    def gradient_descent_step(self, Xi, Yi, n):
        T = len(Xi)

        # Forward pass
        Z, A = [], []
        prev_h_a = np.zeros(self.n_neurons_[1])
        for t in range(T):
            x = Xi[t]

            z, a = self.forward_pass(x, prev_h_a)
            prev_h_a = a[1]

            Z.append(z)
            A.append(a)

        # Error for output neurons at each time step
        output_err = np.zeros((T, self.n_neurons_[2]))
        for t in range(T):
            output_err[t] = (A[t][2] - Yi[t]) / T

        # Error for hidden neurons at each time step
        hidden_err = np.zeros((T + 1, self.n_neurons_[1]))
        for t in reversed(range(T)):
            err = self.Wo_.T.dot(output_err[t]) + self.Wr_.T.dot(hidden_err[t + 1])
            hidden_err[t] = err * sigmoid_prime(Z[t][0])

        # Partial derivatives of the cost c wrt each parameter
        partial_bh = hidden_err.sum(axis=0)
        partial_Wh = np.zeros((self.n_neurons_[1], self.n_neurons_[0]))
        for t in range(T):
            partial_Wh += hidden_err[t].reshape(-1, 1) * Xi[t].reshape(1, -1)

        partial_Wr = np.zeros((self.n_neurons_[1], self.n_neurons_[1]))
        for t in range(1, T):
            partial_Wr += hidden_err[t].reshape(-1, 1) * A[t - 1][1].reshape(1, -1)

        partial_bo = output_err.sum(axis=0)
        partial_Wo = np.zeros((self.n_neurons_[2], self.n_neurons_[1]))
        for t in range(T):
            partial_Wo += output_err[t].reshape(-1, 1) * A[t][1].reshape(1, -1)

        self.move_parameters(partial_bh, partial_Wh, partial_Wr, partial_bo,
                             partial_Wo, n)


    def move_parameters(self, partial_bh, partial_Wh, partial_Wr, partial_bo,
                        partial_Wo, n):
        lmbda_n = float(self.lmbda) / n

        self.Vbh_ = self.mu * self.Vbh_ - self.learning_rate * partial_bh
        self.Vwh_ = self.mu * self.Vwh_ - \
                    self.learning_rate * (partial_Wh + lmbda_n * self.Wh_)

        self.bh_ += self.Vbh_
        self.Wh_ += self.Vwh_

        self.Vwr_ = self.mu * self.Vwr_ - \
                    self.learning_rate * (partial_Wr + lmbda_n * self.Wr_)

        self.Wr_ += self.Vwr_

        self.Vbo_ = self.mu * self.Vbo_ - self.learning_rate * partial_bo
        self.Vwo_ = self.mu * self.Vwo_ - \
                    self.learning_rate * (partial_Wo + lmbda_n * self.Wo_)

        self.bo_ += self.Vbo_
        self.Wo_ += self.Vwo_


    def forward_pass(self, x, prev_h_a):
        z, a = [], [x]

        z.append(self.bh_ + self.Wh_.dot(x) + self.Wr_.dot(prev_h_a))
        a.append(sigmoid(z[-1]))

        z.append(self.bo_ + self.Wo_.dot(a[-1]))
        if self.output_activation == 'sigmoid':
            a.append(sigmoid(z[-1]))
        elif self.output_activation == 'softmax':
            Z = np.exp(z[-1])
            a.append(Z / Z.sum())

        return z, a

    def predict_proba(self, X):
        Yp = []
        for Xi in X:
            Yip = np.zeros((Xi.shape[0], len(self.classes_)))

            prev_h_a = np.zeros(self.n_neurons_[1])
            for t in range(len(Xi)):
                _, a = self.forward_pass(Xi[t], prev_h_a)
                prev_h_a = a[1]

                Yip[t] = a[-1] / a[-1].sum()

            Yp.append(Yip)

        Yp = np.asarray(Yp)

        return Yp

    def predict(self, X):
        ypred = []

        Yp = self.predict_proba(X)
        for Yip in Yp:
            Yiargmax = Yip.argmax(axis=1)
            ypred.append(self.classes_[Yiargmax])

        ypred = np.asarray(ypred)

        return ypred

    def score(self, X, y):
        acc = 0.0

        ypred = self.predict(X)
        for yipred, yi in zip(ypred, y):
            acc += float((yipred == yi).sum()) / len(yi)

        return acc / len(y)


def nback(n, k, length):
    """Random n-back targets given n, number of digits k and sequence length"""
    Xi = random_state.randint(k, size=length)
    yi = np.zeros(length, dtype=int)

    for t in range(n, length):
        yi[t] = (Xi[t - n] == Xi[t])

    return Xi, yi


def one_of_k(Xi_, k):
    Xi = np.zeros((len(Xi_), k))
    for t, Xit in np.ndenumerate(Xi_):
        Xi[t, Xit] = 1

    return Xi


def nback_dataset(n_sequences, mean_length, std_length, n, k):
    X, y = [], []

    for _ in range(n_sequences):
        length = random_state.normal(loc=mean_length, scale=std_length)
        length = int(max(n + 1, length))

        Xi_, yi = nback(n, k, length)
        Xi = one_of_k(Xi_, k)

        X.append(Xi)
        y.append(yi)

    return X, y


def nback_example():
    # Input dimension
    k = 4
    # n-back
    n = 3

    n_sequences = 100
    mean_length = 20
    std_length = 5

    # Training
    Xtrain, ytrain = nback_dataset(n_sequences, mean_length, std_length, n, k)

    rnn = RecurrentNetwork(64, learning_rate=2.0, n_epochs=30,
                           lmbda=0.0, mu=0.2, output_activation='softmax',
                           random_state=None, verbose=1)

    rnn.fit(Xtrain, ytrain)

    # Evaluating
    Xtest, ytest = nback_dataset(5 * n_sequences, 5 * mean_length, 5 * std_length, n, k)

    print('Average accuracy: {0:.3f}'.format(rnn.score(Xtest, ytest)))

    acc_zeros = 0.0
    for yi in ytest:
        acc_zeros += float((yi == 0).sum()) / len(yi)
    acc_zeros /= len(ytest)
    print('Negative guess accuracy: {0:.3f}'.format(acc_zeros))

    # Example
    Xi_ = [3, 2, 1, 3, 2, 1, 3, 2, 2, 1, 2, 3, 1, 2, 0, 0, 2, 0]
    print('\nExample sequence: {0}'.format(Xi_))
    yi = np.zeros(len(Xi_), dtype=int)
    for t in range(n, len(Xi_)):
        yi[t] = (Xi_[t - n] == Xi_[t])

    Xi = one_of_k(Xi_, k)

    yipred = rnn.predict([Xi])[0]
    print('Correct: \t{0}'.format(yi))
    print('Predicted: \t{0}'.format(yipred))
    print('Accuracy: {0:.3f}'.format(float((yi == yipred).sum()) / len(yi)))


if __name__ == "__main__":
    nback_example()


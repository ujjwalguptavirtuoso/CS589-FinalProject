import numpy as np
import contextlib

class NeuralNet():

    def __init__(self, layers, reg_lambda, instances, labels, alpha, weights=None):
        self.layers = layers
        self.l = len(layers)
        self.reg_lambda = reg_lambda
        self.instances = instances
        self.labels = labels
        self.alpha = alpha
        if weights == None:
            self.weights = []
            for i in range(self.l - 1):
                n = layers[i]
                m = layers[i + 1]
                layer_weights = np.random.standard_normal(size=(m, n + 1))
                self.weights.append(layer_weights)
        else:
            self.weights = weights

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def one_minus(self, a):
        return 1 - a

    def forward_propagation(self, x):
        x_vector = [[1]]
        for i in x:
            x_vector.append([i])
        x_vector = np.array(x_vector)
        activations = [x_vector]
        z_list = [x_vector]
        vsigmoid = np.vectorize(self.sigmoid)
        for k in range(1, self.l - 1):
            current_weights = self.weights[k - 1]
            z = current_weights.dot(activations[k - 1])
            a = vsigmoid(z)
            a = np.insert(a, 0, 1.0)
            length = a.size
            a = a.reshape(length, 1)
            activations.append(a)
            z_list.append(z)

        z = (self.weights[self.l - 2]).dot(activations[-1])
        a = vsigmoid(z)
        length = a.size
        a = a.reshape(length, 1)
        activations.append(a)
        z_list.append(z)
        return activations, z_list


    def error_util(self, y, f):
        # print(y,f)
        one_minus_v = np.vectorize(self.one_minus)
        y = np.array(y).reshape(len(y), 1)
        f = np.array(f).reshape(len(f), 1)
        one_minus_f = one_minus_v(f).reshape(len(f), 1)
        one_minus_y = one_minus_v(y).reshape(len(y), 1)
        j = (-y * np.log(f) - one_minus_y * np.log(one_minus_f))

        j_scalar = 0
        for i in j:
            j_scalar += sum(i)
        return j_scalar

    def get_error(self, predicted_labels, true_labels=None):
        if true_labels is None:
            true_labels = self.labels  # Default to training labels if not provided

        errors = []
        for i in range(len(predicted_labels)):
            f = predicted_labels[i]
            y = true_labels[i]
            errors.append(self.error_util(y, f))

        total_error = sum(errors)

        # Regularization penalty
        s = 0
        for a in self.weights:
            for b in range(len(a)):
                for c in range(1, len(a[b])):  # Skipping bias weights
                    s += (a[b][c] ** 2)

        s *= (self.reg_lambda / 2)
        total_error += s
        total_error /= len(predicted_labels)

        return errors, total_error

    def back_propagation_util(self, x, y, d):
        activations, z_list = self.forward_propagation(x)
        predicted = activations[-1]
        true_label = np.array([y]).transpose()
        delta = [np.array(predicted - true_label)]
        vminus_one = np.vectorize(self.one_minus)
        for k in range(self.l - 2, 0, -1):
            current_weights = self.weights[k]
            # print(current_weights, "current")
            current_delta = current_weights.transpose().dot(delta[0])
            current_activations = activations[k]
            current_delta *= current_activations
            one_minus_activations = vminus_one(current_activations)
            length = one_minus_activations.size
            one_minus_activations = one_minus_activations.reshape(length, 1)

            current_delta *= (one_minus_activations)
            current_delta = current_delta[1:]
            delta.insert(0, current_delta)
        delta.insert(0, [0])
        for k in range(self.l - 2, -1, -1):
            current_activations = activations[k]
            d[k] = d[k] + delta[k + 1].dot(current_activations.transpose())
            p = self.reg_lambda * (self.weights[k])
            # d[k] = d[k] + p

        return d, delta

    def get_d(self):
        d = []
        for i in range(self.l - 1):
            d.append(np.zeros(shape=(self.layers[i + 1], self.layers[i] + 1)))

        d.append(np.zeros(shape=1))
        return d

    def back_propagation(self):
        n = len(self.instances)

        delta_list = []
        d = self.get_d()
        # d.append(np.zeros(shape=(self.layers[-2]+1, self.layers[-1])))

        '''
        for i in range(self.l-1):
        '''
        for i in range(n):
            x = self.instances[i]
            y = self.labels[i]
            d, delta = self.back_propagation_util(x, y, d)
            delta_list.append(delta)

        for k in range(self.l - 2, -1, -1):
            p = self.reg_lambda * (self.weights[k])
            for i in range(len(p)):
                p[i][0] = 0
            d[k] = (1 / n) * (p + d[k])

        for k in range(self.l - 2, -1, -1):
            # print(self.weights[k].shape, d[k].shape, "weomwef")
            self.weights[k] -= (self.alpha * d[k])

        return d

def verify_code(nn: NeuralNet):
    x = nn.instances
    y = nn.labels
    print("Regularization parameter lamba: " + str(nn.reg_lambda) + "\n")
    print("Initializing the network with the following structure (number of neurons per layer): " + str(nn.layers) + "\n")
    print("Forward Propagation:\n")
    errors = []
    predicted = []
    for i in range(len(x)):
        print("For instance %d, parameters are:" % (i + 1))
        activations, z_list = nn.forward_propagation(x[i])
        predicted_label = activations[-1].transpose()
        predicted.append(activations[-1])
        current_error = nn.error_util(y[i], predicted[-1])
        print("a1 :", activations[0].transpose(), "\n")
        for j in range(1, len(activations)):
            print("z%d : %s" % (j + 1, str(z_list[j].transpose())))
            print("a%d : %s\n" % (j + 1, str(activations[j].transpose())))
        print("Predicted output for instance %d is : %s\n" % (i + 1, str(predicted_label)))
        print("Cost J, assosiated with instance %d : %f\n" % (i + 1, current_error))

    errors, total_error = nn.get_error(predicted)
    print("Final (regularized) cost, J, based on the complete training set: %f" % total_error + "\n")
    print("------------------------\n")
    print("Running Back Propagation\n")

    for i in range(len(x)):
        d = nn.get_d()
        d, delta = nn.back_propagation_util(x[i], y[i], d)
        print("Computing gradients based on training instance %d" % (i + 1))
        for j in range(nn.l - 1, 0, -1):
            print("delta%d: %s\n" % (j + 1, str(delta[j].transpose())))

        for j in range(nn.l - 2, -1, -1):
            print("Gradients of Theta%d based on training instance %d:\n" % (j + 1, i + 1))
            print(d[j])

    d = nn.back_propagation()
    print("The entire training set has been processes. Computing the average (regularized) gradients:\n")
    for j in range(0, nn.l - 1):
        print("Final Regularized gradients of Theta%d:\n" % (j + 1))
        print(d[j])


def test_network_1():
    x = [[0.13000], [0.42000]]
    y = [[0.90000], [0.23000]]

    w1 = np.array([[0.40000, 0.10000], [0.30000, 0.20000]])
    w2 = np.array([[0.70000, 0.50000, 0.60000]])
    layers = [1, 2, 1]
    weights = [w1, w2]
    nn = NeuralNet(layers, 0, x, y, 0.1, weights)
    verify_code(nn)


def test_network_2():
    x = [[0.32000, 0.68000], [0.83000, 0.02000]]
    y = [[0.75000, 0.98000], [0.75000, 0.28000]]
    layers = [2, 4, 3, 2]
    w1 = np.array([[0.42000, 0.15000, 0.40000], [0.72000, 0.10000, 0.54000], [0.01000, 0.19000, 0.42000],
                   [0.30000, 0.35000, 0.68000]])
    w2 = np.array([[0.21000, 0.67000, 0.14000, 0.96000, 0.87000], [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
                   [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]])
    w3 = np.array([[0.04000, 0.87000, 0.42000, 0.53000], [0.17000, 0.10000, 0.95000, 0.69000]])
    weights = [w1, w2, w3]
    nn = NeuralNet(layers, 0.25, x, y, 0.1, weights)
    verify_code(nn)


if __name__ == "__main__":
    with open("test_network_1_output.txt", "w") as f1:
        with contextlib.redirect_stdout(f1):
            test_network_1()

    with open("test_network_2_output.txt", "w") as f2:
        with contextlib.redirect_stdout(f2):
            test_network_2()

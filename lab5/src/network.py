import numpy as np
from layer import Layer
import matplotlib.pyplot as plt


class Network(object):
    def __init__(
        self,
        hidden_count,
        hidden_size,
        hidden_act_fun,
        output_act_fun,
        learning_rate,
        momentum,
    ):
        self.hidden_count = hidden_count
        self.hidden_size = hidden_size
        self.hidden_act_fun = hidden_act_fun
        self.output_act_fun = output_act_fun
        self.learning_rate = learning_rate
        self.momentum = momentum

    def init_layers(self, input_size, output_size):
        hidden_layers = []
        for i in range(self.hidden_count):
            if i == 0:
                layer = Layer(self.hidden_size, input_size)
            else:
                layer = Layer(self.hidden_size, self.hidden_size)
            hidden_layers.append(layer)

        output_layer = Layer(output_size, self.hidden_size)
        return hidden_layers, output_layer

    def forward_prop(self, batch):
        """
        batch: size = image_vector x batch_size (rows x cols)
        layer.weights: size = neurons x input_size
        hidden activation: size = neurons x batch_size
        output: size = output_size x batch_size
        """

        pre_activations = []
        activations = []

        curr_data = batch
        for i in range(self.hidden_count):
            data = (
                self.hidden_layers[i].weights.dot(curr_data)
                + self.hidden_layers[i].biases
            )
            pre_activations.append(data)
            curr_data = self.hidden_act_fun(data)
            activations.append(curr_data)

        data = self.output_layer.weights.dot(curr_data) + self.output_layer.biases
        output = self.output_act_fun(data)

        return pre_activations, activations, output, data

    def backward_prop(
        self, input, pre_activations, activations, output_error, pre_output
    ):
        batch_size = output_error.shape[1]

        dz_out = output_error / batch_size * self.output_act_fun(pre_output, True)
        dw_out = 1 / batch_size * dz_out.dot(activations[-1].T)
        db_out = 1 / batch_size * np.sum(dz_out, 1, keepdims=True)

        dz_layers = []
        dw_layers = []
        db_layers = []

        for layer in range(self.hidden_count - 1, -1, -1):
            if layer == self.hidden_count - 1:
                dz_layer = self.output_layer.weights.T.dot(
                    dz_out
                ) * self.hidden_act_fun(pre_activations[layer], True)
            else:
                dz_layer = self.hidden_layers[layer + 1].weights.T.dot(
                    dz_layers[0]
                ) * self.hidden_act_fun(pre_activations[layer], True)

            if layer != 0:
                dw_layer = 1 / batch_size * dz_layer.dot(pre_activations[layer].T)
            else:
                dw_layer = 1 / batch_size * dz_layer.dot(input.T)

            db_layer = 1 / batch_size * np.sum(dz_layer, 1, keepdims=True)
            dz_layers.insert(0, dz_layer)
            dw_layers.insert(0, dw_layer)
            db_layers.insert(0, db_layer)

        return dw_out, db_out, dw_layers, db_layers

    def update_params(self, layers_dWs, layers_dBs, output_dWs, output_dBs):
        """A method to update the parameters of the model.

        Args:
            layers_dWs (list): list of layers weights.
            layers_dBs (list): list of layers biases.
            output_dWs (np.array): array of output layer weights.
            output_dBs (np.array): array of output layer biases.
        """
        for i in range(self.hidden_count):

            self.hidden_layers[i].vd_weights = (
                self.momentum * self.hidden_layers[i].vd_weights
                + (1.0 - self.momentum) * layers_dWs[i]
            )
            self.hidden_layers[i].weights -= (
                self.learning_rate * self.hidden_layers[i].vd_weights
            )

            self.hidden_layers[i].vd_biases = self.momentum * self.hidden_layers[
                i
            ].vd_biases + (1.0 - self.momentum) * np.reshape(
                layers_dBs[i], (self.hidden_layers[i].neurons_count, 1)
            )
            self.hidden_layers[i].biases -= (
                self.learning_rate * self.hidden_layers[i].vd_biases
            )

        self.output_layer.vd_weights = (
            self.momentum * self.output_layer.vd_weights
            + (1.0 - self.momentum) * output_dWs
        )
        self.output_layer.weights -= self.learning_rate * self.output_layer.vd_weights

        self.output_layer.vd_biases = self.momentum * self.output_layer.vd_biases + (
            1.0 - self.momentum
        ) * np.reshape(output_dBs, (self.output_layer.neurons_count, 1))
        self.output_layer.biases -= self.learning_rate * self.output_layer.vd_biases

    def fit(
        self, epochs, batch_size, train_data, train_labels, valid_data, valid_labels
    ):
        """A method to train the model.

        Args:
            epochs (int): number of epochs.
            batch_size (int): size of the batch.
            train_data (np.array): training data.
            train_labels (np.array): training labels.
            valid_data (np.array): validation data.
            valid_labels (np.array): validation labels.

        Returns:
            lists: lists of accuracies and losses for each epoch.
        """

        input_size = train_data.shape[1]
        output_size = train_labels.shape[1]

        self.hidden_layers, self.output_layer = self.init_layers(
            input_size, output_size
        )

        train_accs, train_losses = [], []
        valid_accs, valid_losses = [], []
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))

            permutation = np.random.permutation(train_data.shape[0])
            train_data = train_data[permutation, :]
            train_labels = train_labels[permutation, :]

            for i in range(batch_size, len(train_data), batch_size):
                data_sample = train_data[i - batch_size : i].T
                sample_labels = train_labels[i - batch_size : i].T
                pre_activations, activations, output, pre_output = self.forward_prop(
                    data_sample
                )
                output_dws, output_dbs, layers_dws, layers_dbs = self.backward_prop(
                    data_sample,
                    pre_activations,
                    activations,
                    2 * (output - sample_labels),
                    pre_output,
                )
                self.update_params(layers_dws, layers_dbs, output_dws, output_dbs)

            train_acc, train_loss = self.evaluate(train_data, train_labels, "train")
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            valid_acc, valid_loss = self.evaluate(valid_data, valid_labels, "valid")
            valid_accs.append(valid_acc)
            valid_losses.append(valid_loss)
            print("---------------------------------------------------------")
        return train_accs, train_losses, valid_accs, valid_losses

    def predict(self, data):
        _, _, output, _ = self.forward_prop(data.T)
        return output

    def get_predicted_labels(self, network_output):
        return np.argmax(network_output, axis=0)

    def get_accuracy(self, data, labels):
        predictions = self.get_predicted_labels(self.predict(data))
        return np.sum(predictions == np.argmax(labels, axis=1)) / labels.shape[0]

    def get_loss(self, data, labels):
        network_output = self.predict(data)
        batch_size = labels.shape[0]
        error = np.power(labels.T - network_output, 2)
        mean_error = np.sum(error) / batch_size
        return mean_error

    def evaluate(self, data, labels, data_part):
        accuracy = self.get_accuracy(data, labels)
        print(
            "{} data accuracy: {:.2f}%".format(data_part.capitalize(), accuracy * 100)
        )
        loss = self.get_loss(data, labels)
        print("{} loss: {:.4f}".format(data_part, loss))
        return accuracy, loss

    def test_prediction(self, data, labels, index):
        current_image = data[index, :]
        prediction = self.get_predicted_labels(
            self.predict(np.reshape(current_image, (1, current_image.size)))
        )
        label = np.argmax(labels[index, :])
        print("Prediction: ", prediction)
        print(labels[index, :])
        print("Label: ", label)

        current_image = (
            np.reshape(
                current_image,
                (int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1]))),
            )
            * 16
        )
        plt.gray()
        plt.imshow(current_image, interpolation="nearest")
        plt.show()

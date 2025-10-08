import torch
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np


class ClassificationRBM():

    def __init__(self, num_visible, num_hidden, k, num_classes=2, learning_rate=0.05, sparse_constant=0.00,
                 use_cuda=False, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.use_cuda = False
        self.num_classes = num_classes
        self.loss = torch.nn.CrossEntropyLoss()
        self.sparse_constant = sparse_constant

        self.weights = torch.randn(num_visible, num_hidden, ) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)
        self.class_bias = torch.zeros(num_classes)
        self.class_weights = torch.zeros(num_classes, num_hidden) * 0.5

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()
            self.class_weights = self.class_weights.cuda()
            self.class_bias = self.class_bias.cuda()

        self.acc_per_epoch_list = []
        self.auc_per_epoch_list = []


    def sample_hidden(self, visible_activations, class_activations):
        hidden_activations = torch.matmul(visible_activations, self.weights) + self.hidden_bias + torch.matmul(
            class_activations, self.class_weights)
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_activations):
        visible_activations = torch.matmul(hidden_activations, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def sample_class(self, hidden_activations):

        class_probablities = torch.exp(torch.matmul(hidden_activations, self.class_weights.t()) + self.class_bias)
        # print(torch.sum(class_probablities, dim = 1).shape)
        class_probablities = torch.nn.functional.normalize(class_probablities, p=1, dim=1)
        # print(class_probablities.shape)
        return class_probablities

    def sample_class_given_x(self, input_data):
        """Sampling the label given input data in time O(num_hidden * num_visible + num_classes * num_classes) for each example"""

        precomputed_factor = torch.matmul(input_data, self.weights) + self.hidden_bias
        class_probabilities = torch.zeros((input_data.shape[0], self.num_classes))  # .cuda()

        for y in range(self.num_classes):
            prod = torch.zeros(input_data.shape[0], device=input_data.device)
            prod += self.class_bias[y]
            for j in range(self.num_hidden):
                prod += torch.log(1 + torch.exp(precomputed_factor[:, j] + self.class_weights[y, j]))
            # print(prod)
            class_probabilities[:, y] = prod

        copy_probabilities = torch.zeros(class_probabilities.shape, device=input_data.device)

        for c in range(self.num_classes):
            for d in range(self.num_classes):
                copy_probabilities[:, c] += torch.exp(-1 * class_probabilities[:, c] + class_probabilities[:, d])

        copy_probabilities = 1 / copy_probabilities

        class_probabilities = copy_probabilities

        return class_probabilities

    def update_weights(self, batch_size, factor=1):

        self.weights += factor * self.weights_grad * self.learning_rate / batch_size
        self.visible_bias += factor * self.visible_bias_grad * self.learning_rate / batch_size
        self.hidden_bias += factor * self.hidden_bias_grad * self.learning_rate / batch_size
        self.class_bias += factor * self.class_bias_grad * self.learning_rate / batch_size
        self.class_weights += factor * self.class_weights_grad * self.learning_rate / batch_size

        self.visible_bias -= self.sparse_constant

        self.hidden_bias -= self.sparse_constant
        self.class_bias -= self.sparse_constant

    def discriminative_training(self, input_data, class_label, factor=1):

        batch_size = input_data.size(0)

        class_one_hot = torch.nn.functional.one_hot(class_label.to(torch.int64), num_classes=self.num_classes).float()
        o_y_j = self._sigmoid((torch.matmul(input_data, self.weights) + self.hidden_bias).unsqueeze_(-1).expand(-1, -1,
                                                                                                                self.num_classes) + self.class_weights.t())
        class_probabilities = self.sample_class_given_x(input_data)

        positive_sum = torch.zeros(batch_size, self.num_hidden, device=input_data.device)
        class_weight_grad = torch.zeros(self.num_classes, self.num_hidden, device=input_data.device)

        for i, c in enumerate(class_label):
            positive_sum[i] += o_y_j[i, :, c]
            class_weight_grad[c, :] += positive_sum[i]

        # print(positive_sum)
        unfolded_input = input_data.unsqueeze(-1).expand(-1, -1, self.num_hidden)
        positive_associations = torch.sum(torch.mul(unfolded_input, positive_sum.unsqueeze_(1)), dim=0)
        # print(positive_associations.shape)

        negetive_sum = torch.zeros(batch_size, self.num_hidden, device=input_data.device)

        for c in range(self.num_classes):
            class_weight_grad[c, :] -= torch.sum(o_y_j[:, :, c] * class_probabilities[:, c].unsqueeze_(-1), dim=0)
            negetive_sum += o_y_j[:, :, c] * class_probabilities[:, c].unsqueeze(-1)

        negetive_associations = torch.sum(torch.mul(unfolded_input, negetive_sum.unsqueeze_(1)), dim=0)

        self.weights_grad = (positive_associations - negetive_associations)

        self.class_weights_grad = (class_weight_grad)

        self.hidden_bias_grad = torch.sum(positive_sum.squeeze_() - negetive_sum.squeeze_(), dim=0)

        self.class_bias_grad = torch.sum(class_one_hot - class_probabilities, dim=0)

        self.visible_bias_grad = 0

        self.update_weights(batch_size, factor)

        error = self.loss(class_probabilities, class_label)

        _, predicted = torch.max(class_probabilities, 1)

        return error, predicted, class_probabilities

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities

    def train_rbm(self, train_loader, epochs, cuda=False, validation_loader=None, test_loader=None, method='discriminative',
                  generative_factor=None, discriminative_factor=1):
        print('Training RBM...')
        loss_list = []
        nll_list = []
        best_validation_acc = 0
        best_validation_model = None
        patience_count = 0
        for epoch in range(epochs):
            epoch_error = 0.0
            epoch_nll = 0.0
            total_samples = 0

            for batch, labels in tqdm(train_loader, total=len(train_loader), leave=False):
                batch = batch.view(len(batch), self.num_visible)  # flatten input data

                if cuda:
                    batch = batch.float().cuda()
                    labels = labels.cuda()

                if method == 'discriminative':
                    batch_error, _, class_probs = self.discriminative_training(batch, labels)
                else:
                    raise NotImplementedError

                epoch_error += batch_error.item()
                total_samples += batch.size(0)

                # 🔥 Compute NLL for this batch
                log_probs = torch.log(class_probs + 1e-8)
                nll = -log_probs[range(batch.size(0)), labels]
                batch_nll = nll.mean()
                epoch_nll += batch_nll.item()

            loss_list.append(epoch_error / len(train_loader))
            nll_list.append(epoch_nll / len(train_loader))

            acc, auc = self.run_test_set(test_loader)
            self.acc_per_epoch_list.append(acc)
            self.auc_per_epoch_list.append(auc)
            print(
                f'Epoch {epoch} | Error: {epoch_error / len(train_loader):.4f} | NLL: {epoch_nll / len(train_loader):.4f}')


            best_validation_model = self

        return loss_list, best_validation_model, nll_list

    def test_rbm_model(self, rbm_model, test_loader, args):
        correct = 0
        total = 0

        for batch, labels in tqdm(test_loader):
            batch = batch.view(len(batch), args.visible_units)  # flatten input data

            if args.cuda:
                batch = batch.float().cuda()
                labels = labels.cuda()

            predicted_probabilities = rbm_model.sample_class_given_x(batch)

            _, predicted = torch.max(predicted_probabilities, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

        return correct / total

    def get_device(self, use_gpu: bool) -> torch.device:
        '''
            Gets the device on which the neural networks lives. If use_gpu is true,
            a GPU (cuda or mps) is returned, if found.

            Parameters
            ----------
            use_gpu: bool
                if true then the device will be a GPU if one is available

            Return
            ----------
            torch.device:
                GPU if it should be used and is available and CPU otherwise
        '''
        # if torch.cuda.is_available() and use_gpu:
        #    return torch.device("cuda")
        # if torch.backends.mps.is_available() and use_gpu:
        #    return torch.device("mps")
        return torch.device("cpu")

    def run_test_set(self, test_loader):
        correct = 0
        total = 0

        # Lists to store predictions and true labels
        all_preds = []
        all_labels = []

        for batch, labels in tqdm(test_loader):
            batch = batch.view(len(batch), self.num_visible)  # flatten input data

            # Get predicted probabilities
            predicted_probabilities = self.sample_class_given_x(batch)

            # Get predicted class
            _, predicted = torch.max(predicted_probabilities, 1)

            # Append predictions and true labels for confusion embedding_matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print Accuracy
        accuracy = correct / total
        print('Accuracy of the network on the test images: ', accuracy)

        auc_score = roc_auc_score(all_labels, all_preds)
        print('AUC Score: %.4f' % auc_score)



        # # Compute and plot confusion embedding_matrix
        # cm = confusion_matrix(all_labels, all_preds)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #
        # # Display the confusion embedding_matrix
        # disp.plot(cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix')
        # plt.show()

        return accuracy, auc_score


    def get_num_params(self):
        """
        Returns the number of parameters in the RBM model.
        """
        num_params = 0
        num_params += self.weights.numel()
        num_params += self.visible_bias.numel()
        num_params += self.hidden_bias.numel()
        num_params += self.class_weights.numel()
        num_params += self.class_bias.numel()
        return num_params

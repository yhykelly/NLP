
import torch
import torch.nn as nn
import copy
import json
from sklearn.metrics import f1_score
from constants import *


class FeedForwardNet(nn.Module):
    def __init__(self, n_dims, hidden_size, n_classes):
        """
        A feedforward network for multi-class classification with 1 input layer,
        1 hidden layer, and 1 output layer.

        :param n_dims: number of dimensions in input
        :param hidden_size: number of neurons in the hidden layer
        :param n_classes: number of classes
        """
        super(FeedForwardNet, self).__init__()
        # model layers
        self.linear1 = nn.Linear(n_dims, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        The forward pass applies the network layers to x.

        :param x: the input data as a tensor of size (n_samples, embedding_dim)
        :return: the output of the last layer
        """
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class Trainer:
    def __init__(self):
        """
        Class for training a Feedforward Network.

        Public methods include:
            - load_data()
            - train()
            - save_best_model()
        """

        # class variables that will be set later
        self.X_train = None  # train X tensors
        self.y_train = None  # train y tensors
        self.X_dev = None    # dev X tensors
        self.y_dev = None    # dev y tensors
        self.label_map = None  # {label:class_code} dictionary
        self.n_dims = None # number of dimensions in training data
        self.n_classes = None  # number of classes in training data
        self.best_model = None

    def _load_train_tensors(self, train_tensor_file):
        """
        Private method to load the training tensors in train_tensor_file
        into class variables self.X_train, self.y_train, and self.label_map.
        Assumes that the file uses keys X_KEY, Y_KEY, MAP_KEY,
        as defined in constants.py.
        Helper method for load_data().

        Note: You can use torch.load() here.

        :param train_tensor_file: file containing training tensors
        """

        temp = torch.load(train_tensor_file)
        self.X_train = temp[X_KEY]
        self.y_train = temp[Y_KEY]
        self.label_map = temp[MAP_KEY]
        self.n_dims = temp[X_KEY].shape[1]
        self.n_classes = len(temp[MAP_KEY])


    def _load_dev_tensors(self, dev_tensor_file):
        """
        Private method to load the dev tensors in dev_tensor_file
        into class variables self.X_dev, self.y_dev, and self.label_map.
        Assumes that the file uses keys X_KEY, Y_KEY, MAP_KEY,
        as defined in constants.py.
        Helper method for load_data().

        Note: You can use torch.load() here.

        :param dev_tensor_file: file containing dev tensors
        """
        temp = torch.load(dev_tensor_file)
        self.X_dev = temp[X_KEY]
        self.y_dev = temp[Y_KEY]


    def load_data(self, train_tensor_file, dev_tensor_file):
        """
        Public method to load train and dev tensors from files,
        as saved in the Preprocessor class.

        Also sets self.n_dims, and self.n_classes.

        :param train_tensor_file: file containing training tensors
        :param dev_tensor_file: file containing dev tensors
        """
        self._load_train_tensors(train_tensor_file)
        self._load_dev_tensors(dev_tensor_file)


    def _macro_f1(self, model):
        """
        Private method to calculate the macro f1 score of the given model
        on the dev data.
        Helper method for _training_loop().

        Note that the predictions on the dev data is the output of the forward pass,
        with shape (n_samples, n_classes).
        This means that you need to get the index of the highest value in each
        row of predictions (which is the class code of the predicted class).
        You can use torch.argmax() for that.
        Use sklearn.metrics.f1_score to calculate the macro-averaged F1 score.

        Note: It is important that gradient calculation is turned off here,
        which can be done by putting the code for this function in
        a **with torch.no_grad():** block.

        :param model: the model to test on the dev data
        :return: float - macro F1 score
        """
        with torch.no_grad():
            model.eval()
            temp = model.forward(self.X_dev)
            # retrieve the index of the maximum value of each row 
            dev_predict = torch.argmax(temp, dim=1)
            f1score = f1_score(self.y_dev, dev_predict, average='macro')

        return f1score


    def _training_loop(self, model, loss_fn, optimizer, n_epochs):
        """
        This is where the actual training takes place.
        Private method to train model using the given loss function,
        optimizer, and n_epochs.
        Helper method for train().

        Training and dev data are stored as class variables.

        At each epoch, evaluate the model on the dev data.
        If the macro-averaged F1 score is better than the current best score,
        update the current best score, best epoch, and best model (you must make
        a deep copy of the model state).

        Returns a dictionary containing information about the best model (the one
        with the highest macro-averaged F1 score, not the last model).
        The returned dictionary should contain the following keys:

        - MODEL_STATE_KEY: make a deep copy: copy.deepcopy(model.state_dict())
        - F1_MACRO_KEY: F1 score of the best model
        - BEST_EPOCH_KEY: epoch of the best model

        :param model: the model to train
        :param loss_fn: the loss function
        :param optimizer: the optimizer
        :param n_epochs: number of training epochs
        :return: dictionary containing model state, F1 score, and epoch of the best model
        """

        best_state = None
        best_f1 = float(0)
        best_epoch = 0

        for epoch in range(n_epochs):
            # 1. forward pass
            y_hat = model.forward(self.X_train)

            # 2. compute loss
            loss = loss_fn(y_hat, self.y_train)
            loss.backward()

            # 3. update the model
            optimizer.step()
            optimizer.zero_grad()

            # 4. measure the dev_data f1 score for that epoch
            f1score = self._macro_f1(model)

            #4a renew
            if f1score > best_f1:
                best_epoch = epoch + 1
                best_f1 = f1score
                best_state = copy.deepcopy(model.state_dict())

        current_model_dict = {
            MODEL_STATE_KEY: best_state,
            F1_MACRO_KEY: best_f1,
            BEST_EPOCH_KEY: best_epoch
        }

        return current_model_dict

    def train(self, hidden_size, n_epochs, learning_rate):
        """
        Public method to train a model.

        - Create a model (an instance of FeedForwardNet). The parameters of the
          model are stored in class variables, and hyperparameters are passed in
          to this method.
        - Set the loss function (CrossEntropyLoss) and optimizer (AdamW)
        - Train the model and add the following keys to the dictionary returned
          by _training_loop():
            - HIDDEN_SIZE_KEY
            - N_DIMS_KEY
            - N_CLASSES_KEY
            - LEARNING_RATE_KEY
            - N_EPOCHS_KEY
            - OPTIMIZER_NAME_KEY get with optimizer.__class__.__name__
            - LOSS_FN_NAME_KEY get with loss_fn.__class__.__name__
        - Store the updated dictionary in self.best_model
        - Return updated dictionary

        :param n_epochs: number of epochs
        :param hidden_size: hidden_size
        :param learning_rate: learning rate
        :return: best model dictionary containing the model state and all metadata
        """
        # Use a seed to make sure that results are reproducible.
        # Please do not remove or change the seed.
        torch.manual_seed(42)

        # initialise an ffn model, loss function, optimizer
        ffn_model = FeedForwardNet(self.n_dims, hidden_size, self.n_classes)
        loss_ce = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(ffn_model.parameters(),lr=learning_rate)

        # start the training loop to retrieve the best model info
        best_model_dict = self._training_loop(ffn_model, loss_ce, optimizer, n_epochs)
        best_model_dict[HIDDEN_SIZE_KEY] = hidden_size
        best_model_dict[N_DIMS_KEY]= self.n_dims
        best_model_dict[N_CLASSES_KEY]= self.n_classes
        best_model_dict[LEARNING_RATE_KEY]= learning_rate
        best_model_dict[N_EPOCHS_KEY]= n_epochs
        best_model_dict[OPTIMIZER_NAME_KEY]= optimizer.__class__.__name__
        best_model_dict[LOSS_FN_NAME_KEY]= loss_ce.__class__.__name__
        self.best_model = best_model_dict

        return best_model_dict
    

    def save_best_model(self, base_filename):
        """
        Save the trained model in self.best_model, as well as its metadata.

        2 dictionaries are saved:

        - base_filename.pt (use torch.save())
          The model and all information required to load it:
                - MODEL_STATE_KEY
                - N_DIMS_KEY
                - N_CLASSES_KEY
                - HIDDEN_SIZE_KEY

        - base_filename-info.json (use json library)
          Metadata about the model (all keys except MODEL_STATE_KEY):

                - HIDDEN_SIZE_KEY
                - N_DIMS_KEY
                - N_CLASSES_KEY
                - LEARNING_RATE_KEY
                - N_EPOCHS_KEY
                - BEST_EPOCH_KEY
                - F1_MACRO_KEY
                - OPTIMIZER_NAME_KEY
                - LOSS_FN_NAME_KEY

        :param base_filename: path and base name to save files (e.g. "Models/best")
        """

        # save the keys for .pt file
        base = {
            MODEL_STATE_KEY: self.best_model[MODEL_STATE_KEY],
            N_DIMS_KEY: self.best_model[N_DIMS_KEY],
            N_CLASSES_KEY: self.best_model[N_CLASSES_KEY],
            HIDDEN_SIZE_KEY: self.best_model[HIDDEN_SIZE_KEY]
        }
        pt_name = base_filename + ".pt"
        torch.save(base, pt_name)

        # save the keys for .json file
        meta_dict = {
            F1_MACRO_KEY: self.best_model[F1_MACRO_KEY],
            BEST_EPOCH_KEY: self.best_model[BEST_EPOCH_KEY],
            HIDDEN_SIZE_KEY: self.best_model[HIDDEN_SIZE_KEY],
            N_DIMS_KEY: self.best_model[N_DIMS_KEY],
            N_CLASSES_KEY: self.best_model[N_CLASSES_KEY],
            LEARNING_RATE_KEY: self.best_model[LEARNING_RATE_KEY],
            N_EPOCHS_KEY: self.best_model[N_EPOCHS_KEY],
            OPTIMIZER_NAME_KEY: self.best_model[OPTIMIZER_NAME_KEY],
            LOSS_FN_NAME_KEY: self.best_model[LOSS_FN_NAME_KEY],
        }
        json_name = base_filename + "-info" + ".json"
        with open (json_name, "w") as w:
            temp = json.dumps(meta_dict, indent=4)
            w.write(temp)
            w.close()


if __name__ == '__main__':
    """
    Try out your Trainer here.
    If you train and save a model using the same hyperparameters as
    Models/baseline-model-given, you should get the same results.
    """
    trainee = Trainer()
    trainee.load_data("./Data/my-train-tensor.pt", "./Data/my-dev-tensor.pt")
    dictionary = trainee.train(8,200,0.01)
    savename = "./Data/my-trained-fnn-model-2"
    trainee.save_best_model(savename)


    pass

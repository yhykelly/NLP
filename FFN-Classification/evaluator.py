import torch
from sklearn.metrics import classification_report
from trainer import FeedForwardNet
from constants import *


class Evaluator:
    def __init__(self):
        """
        Class for loading a Feedforward Network from file, and evaluating
        on test data also loaded from file.

        Public methods include:
            - load_model()
            - load_data()
            - evaluate()
        """
        self.model = None
        self.X_test = None
        self.y_test = None
        self.label_map = None

    def load_model(self, model_dict):
        """
        Loads the model in model_file and saves it in self.model.

        - Load model_file, which contains a dictionary
          as saved with Trainer.save_best_model()
        - Instantiate the model
        - Set model to evaluation mode.

        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        :param model_file: file containing the model
        """
        # Load model_file
        model_dict = torch.load(model_dict)
        n_dims = model_dict[N_DIMS_KEY]
        hidden_size = model_dict[HIDDEN_SIZE_KEY]
        n_class = model_dict[N_CLASSES_KEY]

        # Instantiate the model
        eval_model = FeedForwardNet(n_dims, hidden_size, n_class)
        eval_model.load_state_dict(model_dict[MODEL_STATE_KEY])

        # set to evaluation mode
        eval_model.eval()
        self.model = eval_model

        pass

    def load_data(self, data_file):
        """
        Load the evaluation tensors in data_file, where data_file
        is as generated by Preprocessor.save_tensors().
        Set self.X_test, self.y_test, and self.label_map.

        Note: You can use torch.load() here.

        :param data_file: file containing test data as tensors
        """
        test_data = torch.load(data_file)
        self.X_test = test_data[X_KEY]
        self.y_test = test_data[Y_KEY]
        self.label_map = test_data[MAP_KEY]

        pass

    def evaluate_model(self):
        """
        Evaluate the model loaded in load_model using the evaluation
        data loaded by load_data().
        Return two reports generated by the
        sklearn.metrics.classification_report package:

        - report dictionary
        - report as a string

        You will need the label_map to generate the *target_names* for the
        classification_report() function.

        Note: It is important that gradient calculation is turned off during
        evaluation. This can be done by getting the model predictions
        in a **with torch.no_grad():** block.

        :return dict, str: evaluation report as dictionary and as string
        """
        with torch.no_grad():
            # compute the forward pass result of the input and get the predicted class
            output = self.model.forward(self.X_test)
            test_predict = torch.argmax(output, dim=1)
            classes = list(self.label_map.keys())
            # compute the reports from the sklearn package
            report_dict = classification_report(self.y_test,test_predict,target_names=classes, output_dict=True)
            report_str = classification_report(self.y_test,test_predict,target_names=classes, output_dict=False)

            return report_dict, report_str


if __name__ == '__main__':
    """
    Evaluate the following:
    
    - baseline model on the dev data
    - baseline model on the test data
    - Your best model, generated by grid search, on dev data
    - Your best model, generated by grid search, on test data
    """
    tester = Evaluator()
    tester.load_data("./Data/my-test-tensor.pt")
    tester.load_model("./Models/my-best-trained-model-by-grid.pt")
    dict, string_ver = tester.evaluate_model()
    print(dict)
    print(string_ver)

    pass

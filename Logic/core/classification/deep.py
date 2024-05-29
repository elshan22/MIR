import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.classification.basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embeddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        train_dataset = ReviewDataSet(x, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        best_f1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            loss = 0
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                l = self.criterion(outputs, batch_y)
                l.backward()
                self.optimizer.step()
                loss += l.item()
            f1_score = self._eval_epoch(self.test_loader, self.model)[3]
            if f1_score > best_f1:
                best_f1 = f1_score
                self.best_model = self.model.state_dict()
        self.model.load_state_dict(self.best_model)
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        test_dataset = ReviewDataSet(x, np.zeros(len(x)))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch_x, _ in tqdm(test_loader):
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predicted = torch.max(outputs, 1)[1]
                predictions.extend(predicted.cpu().numpy())
        return np.array(predictions)

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        losses = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                losses.append(loss.item())
                predicted = torch.max(outputs, 1)[1]
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_y.cpu().numpy())
        avg_loss = np.mean(losses)
        f1 = f1_score(true_labels, predictions, average='macro')
        return avg_loss, predictions, true_labels, f1

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        return classification_report(y, self.predict(x))


# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader('../IMDB Dataset.csv')
    loader.load_data()
    loader.embeddings = np.load('embeddings.npy')
    x_train, x_test, y_train, y_test = loader.split_data()
    in_features = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    batch_size = 64
    num_epochs = 50
    deep_model = DeepModelClassifier(in_features, num_classes, batch_size, num_epochs)
    deep_model.set_test_dataloader(x_test, y_test)
    deep_model.fit(x_train, y_train)
    y_pred = deep_model.predict(x_test)
    print(classification_report(y_test, y_pred))

from typing import List
from collections import defaultdict
import numpy as np
import wandb

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0
        for i in range(len(actual)):
            a, p = set(actual[i]), set(predicted[i])
            precision += len(a & p) / (len(p) * len(predicted))
        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0
        for i in range(len(actual)):
            a, p = set(actual[i]), set(predicted[i])
            recall += len(a & p) / (len(a) * len(predicted))
        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        p = self.calculate_precision(actual, predicted)
        r = self.calculate_recall(actual, predicted)
        return 2 * p * r / (p + r)
    
    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = []
        counter = 0
        for i, data in enumerate(predicted):
            if data in actual:
                counter += 1
                AP.append(counter / (i + 1))
        return sum(AP) / len(AP)
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        return sum([self.calculate_AP(actual[i], predicted[i]) for i in range(len(predicted))]) / len(predicted)
    
    def cacluate_DCG(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0
        coefficients = defaultdict(int)
        for i in range(len(actual)):
            coefficients[actual[i]] = len(actual) - i
        for i in range(len(predicted)):
            DCG += coefficients[predicted[i]] / np.log2(i + 1)
        return DCG
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0
        for i in range(len(predicted)):
            NDCG += self.cacluate_DCG(actual[i], predicted[i]) / self.cacluate_DCG(actual[i], actual[i])
        return NDCG / len(predicted)
    
    def cacluate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        return 1 / (predicted.index(actual[0]) + 1) if actual[0] in predicted else 0
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        return sum([self.cacluate_RR(actual[i], predicted[i]) for i in range(len(predicted))]) / len(predicted)

    def print_evaluation(self, precision, recall, f1, map, ndcg, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Mean Average Precision (MAP): {map}')
        print(f'Normalized Discounted Cumulative Gain (NDCG): {ndcg}')
        print(f'Mean Reciprocal Rank (MRR): {mrr}')

    def log_evaluation(self, precision, recall, f1, map, ndcg, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        wandb.init()
        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MAP": map,
            "NDCG": ndcg,
            "MRR": mrr
        })
        wandb.finish()

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, map_score, ndcg, mrr)
        self.log_evaluation(precision, recall, f1, map_score, ndcg, mrr)

import json
import torch
import torch.utils.data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.model = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as f:
            self.df = pd.DataFrame(json.load(f))[['first_page_summary', 'genres']]

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres.
        """
        self.df.dropna(inplace=True)
        top_genres = self.df['genres'].explode().value_counts().nlargest(self.top_n_genres).index
        self.df = self.df[self.df['genres'].apply(lambda genres: any(genre in top_genres for genre in genres))]
        self.df['label'] = LabelEncoder().fit_transform(self.df['genres'].apply(lambda genres: next(genre for genre in genres if genre in top_genres)))

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        train_val_df, self.test_df = train_test_split(self.df, test_size=test_size, stratify=self.df['label'])
        self.train_df, self.val_df = train_test_split(train_val_df, test_size=val_size/(1 - test_size), stratify=train_val_df['label'])

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def tokenize_data(self, texts):
        """
        Tokenize the data using the BERT tokenizer.

        Args:
            texts (list): A list of texts to tokenize.

        Returns:
            dict: The tokenized input encodings.
        """
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        train_encodings = self.tokenize_data(self.train_df['first_page_summary'].tolist())
        val_encodings = self.tokenize_data(self.val_df['first_page_summary'].tolist())
        self.train_df = self.create_dataset(train_encodings, self.train_df['label'].tolist())
        self.val_df = self.create_dataset(val_encodings, self.val_df['label'].tolist())
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.top_n_genres, from_tf=True)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_df,
            eval_dataset=self.val_df,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall}

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        test_encodings = self.tokenize_data(self.test_df['first_page_summary'].tolist())
        self.test_df = self.create_dataset(test_encodings, self.test_df['label'].tolist())
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=16,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=self.test_df,
            compute_metrics=self.compute_metrics
        )

        results = trainer.evaluate(eval_dataset=self.test_df)
        print(f"Test Set Evaluation Results:\n{results}")

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.push_to_hub(model_name)
        self.tokenizer.push_to_hub(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)

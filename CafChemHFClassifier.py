import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import huggingface_hub
import transformers
import datasets
import numpy as np
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

def make_dataset(filename: str, labels: list, sep = ",", valid_percent = 0.15, push = False, setname = None):
  '''
    Takes a CSV file and returns a HuggingFace dataset. optionally pushes the dataset to the HuggingFace Hub.

      Args:
        filename: name of CSV file
        labels: labels to use for the columns in the CSV file
        sep: separator used in the CSV file
        valid_percent: percentage of the dataset to use for validation
        push: whether to push the dataset to the HuggingFace Hub
        setname: name of the dataset in the HuggingFace Hub
      Returns:
        user_data: HuggingFace dataset
  '''
  user_data = load_dataset("csv", data_files=filename, sep=sep, names=labels)
  user_data = user_data["train"].train_test_split(test_size=valid_percent)

  if push:
    user_data.push_to_hub(setname)
    print("Dataset pushed to HuggingFace Hub")
  return user_data

def setup_foundation(model_name: str, num_labels: int, class_labels: dict, add_tokens = False, new_tokens = None):
  '''
    Sets up the foundation model for fine-tuning.

      Args:
        model_name: name of the model to use
        num_labels: number of labels in the dataset
        class_labels: dictionary of class labels
        add_tokens: whether to add new tokens to the tokenizer
        new_tokens: list of new tokens to add
      Returns:
        ft_model: fine-tuned model
        tokenizer: tokenizer for the model
        device: device to use for training
  '''
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  ft_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                            id2label=class_labels).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  if add_tokens == True and len(new_tokens) != 0:
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))
    ft_model.resize_token_embeddings(len(tokenizer))
  
  return ft_model, tokenizer,device

def encode_dataset(dataset, tokenizer):
  '''
    Encodes the dataset using the tokenizer.

      Args:
        dataset: dataset to encode
        tokenizer: tokenizer to use
      Returns:
        encoded_dataset: encoded dataset
  '''
  encoded_dataset = dataset.map(lambda examples: tokenizer(examples["text"], 
                                padding=True, truncation=True), 
                                batched=True, batch_size=None)
  encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
  return encoded_dataset
  
class classify():
  '''
  '''
  def __init__(self, ft_model_name: str, ft_model, tokenizer, encoded_dataset, epochs: int, learning_rate = 2e-5,push_hub = False, batch_size = 32):
    '''
      Trains the model.

        Args:
          ft_model_name: name of the fine-tuned model
          ft_model: fine-tuned model
          tokenizer: tokenizer to use
          encoded_dataset: encoded dataset to use for training
          epochs: number of epochs to train for
          learning_rate: learning rate to use for training
          push_hub: whether to push the model to the HuggingFace Hub
    '''
    self.ft_model = ft_model
    self.ft_model_name = ft_model_name
    self.tokenizer = tokenizer
    self.encoded_dataset = encoded_dataset
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.push_hub = push_hub
    self.batch_size = batch_size

    print("Defining training model")

  def compute_metrics(self,pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

  def train(self, weight_decay: float):
    '''
      Trains the model.

        Args:
          weight_decay: weight decay to use for training
        Returns:
          trainer: trainer object
    '''
    logging_steps = len(self.encoded_dataset["train"]) // self.batch_size
    
    training_args = TrainingArguments(output_dir=self.ft_model_name,
                                    num_train_epochs=self.epochs,
                                    learning_rate=self.learning_rate,
                                    #optim="adafactor",
                                    lr_scheduler_type="constant",
                                    per_device_train_batch_size=self.batch_size,
                                    per_device_eval_batch_size=self.batch_size,
                                    weight_decay=weight_decay,
                                    eval_strategy="epoch",
                                    save_strategy="epoch",
                                    logging_strategy="epoch",
                                    load_best_model_at_end=True,
                                    metric_for_best_model="f1",
                                    save_total_limit=1,
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=self.push_hub,
                                    log_level="error",
                                    report_to='none')
    
    trainer = Trainer(model=self.ft_model, args=training_args, 
                      compute_metrics=self.compute_metrics, 
                      train_dataset=self.encoded_dataset["train"], 
                      eval_dataset=self.encoded_dataset["test"], 
                      tokenizer=self.tokenizer)
    
    trainer.train()

    return trainer

class evaluate():
  '''
  '''
  def __init__(self, trainer, labels: list, encoded_dataset):
    '''
    '''
    self.trainer = trainer
    self.labels = labels
    self.encoded_dataset = encoded_dataset
 
  def plot_confusion_matrix(self, y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

  def confusion(self):
    '''
    '''
    preds_output = self.trainer.predict(self.encoded_dataset["test"])
    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_true = np.array(self.encoded_dataset["test"]["label"])
    self.plot_confusion_matrix(y_preds, y_true, labels=self.labels)

def push_trainer(trainer, message: str):
  '''
    Pushes the trainer to the HuggingFace Hub.

      Args:
        trainer: trainer object to push
        message: commit message to use
  '''
  trainer.push_to_hub(commit_message=message)

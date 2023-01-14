import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

class SentimentClassifier:
    def __init__(self, dataset, batch_size, truncation_length, num_epochs=1):
        checkpoint = "trituenhantaoio/bert-base-vietnamese-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        def tokenize_function(example):
            # truncation_length = 512
            return tokenizer(example["Review"], truncation=True, max_length=truncation_length)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Drop unused columns
        tokenized_datasets = tokenized_datasets.remove_columns(['Rate', 'Review', 'Index'])
        # Rename label column
        tokenized_datasets = tokenized_datasets.rename_column('Label', 'label')

        # Load the dataset into separate splits
        self.train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
        )
        self.eval_dataloader = DataLoader(
            tokenized_datasets["validation"], batch_size = batch_size, collate_fn = data_collator
        )
        self.test_dataloader = DataLoader(
            tokenized_datasets["test"], batch_size = batch_size, collate_fn = data_collator
        )

        self.num_epochs = num_epochs

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
        self.model.to(self.device)
        
    
    def optimizer(self):
        return AdamW(self.model.parameters(), lr=5e-6)

    def lr_scheduler(self, optimizer):
        num_training_steps = self.num_epochs*len(self.train_dataloader)
        return get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    def training_loop(self):

        optimizer = self.optimizer()
        lr_scheduler = self.lr_scheduler(optimizer)

        for epoch in range(self.num_epochs):
            print(f"Training Epoch {epoch+1}/{self.num_epochs} ")
            
            # Training Loop
            print("Training...")
            train_accum_loss = 0
            self.model.train()
            train_preds_list = []
            train_labels_list = []
            for batch in tqdm(self.train_dataloader, position=0, leave=True):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                train_accum_loss += loss
                loss.backward()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                train_preds_list.extend(predictions)
                train_labels_list.extend(batch["labels"])
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Evaluation loop
            print("Evaluating...")
            eval_accum_loss = 0
            self.model.eval()
            eval_preds_list = []
            eval_labels_list = []
            for batch in tqdm(self.eval_dataloader, position=0, leave=True):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    eval_accum_loss += loss

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                eval_preds_list.extend(predictions)
                eval_labels_list.extend(batch["labels"])

            # Train metrics
            self.model_metrics(train_preds_list, train_labels_list, train_accum_loss, len(self.train_dataloader), "Train")    
            # Eval metrics
            self.model_metrics(eval_preds_list, eval_labels_list, eval_accum_loss, len(self.eval_dataloader), "Evaluation")

        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, "model.pt")

    def model_metrics(self, preds_list, labels_list, accum_loss, dataloader_length, current_stage):
        accuracy = self.get_accuracy(preds_list, labels_list)
        loss = self.get_loss(accum_loss, dataloader_length)
        print(f"{current_stage} accuracy: {accuracy}, {current_stage} loss: {loss}")

    def get_accuracy(self, preds_list, labels_list):
        matches = sum(1 for pred, label in zip(preds_list, labels_list) if pred == label)
        accuracy = matches / len(preds_list) * 100
        return accuracy
        
    def get_loss(self, accum_loss, dataloader_length):
        return accum_loss/dataloader_length
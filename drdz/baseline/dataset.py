from torch.utils.data import Dataset, DataLoader, random_split
import lightning as pl
from .preprocess_data import preprocess_data
from transformers import AutoTokenizer
import torch

class LJPDataset(Dataset):
    def __init__(self, dataset_path, max_length=512, load_num=None) -> None:
        super().__init__()
        self.facts, self.charge_labels, self.article_labels, self.penalty_labels, self.defendant_nums = preprocess_data(dataset_path, data_num=load_num)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        # inputs, targets, defendant_nums
        return self.facts[index], {
            'charge_label': self.charge_labels[index],
            'article_label': self.article_labels[index],
            'penalty_label': self.penalty_labels[index],
        }, self.defendant_nums[index]

class LJPDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_path, train_dataset_path=None, val_dataset_path=None, test_dataset_path=None, batch_size=8, num_workers=0, max_length=512, load_num=None, train_val_split=[0.9, 0.1]) -> None:
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_length = max_length
        self.load_num = load_num
        self.train_val_split = train_val_split
    
    def setup(self, stage=None):
        if stage == "fit":
            if self.val_dataset_path is not None:
                self.train_dataset = LJPDataset(self.train_dataset_path, self.max_length, self.load_num)
                self.val_dataset = LJPDataset(self.val_dataset_path, self.max_length, self.load_num//2)
            else:
                # randomly split train dataset into train and val first
                print("spliting training and validation dataset.")
                total_dataset = LJPDataset(self.train_dataset_path, self.max_length, self.load_num)
                total_dataset_len = len(total_dataset)
                train_size = int(total_dataset_len * self.train_val_split[0])
                val_size = total_dataset_len - train_size
                self.train_dataset, self.val_dataset = random_split(total_dataset, [train_size, val_size])
        elif stage == "test":
            self.test_dataset = LJPDataset(self.test_dataset_path, self.max_length, self.load_num)
    
    def collate(self, data):
        facts, targets, defendant_nums = zip(*data)
        # tokenize facts
        inputs = self.tokenizer(list(facts), return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        targets = {
            "charge_labels": torch.tensor([target["charge_label"] for target in targets], dtype=torch.float32),
            "article_labels": torch.tensor([target["article_label"] for target in targets], dtype=torch.float32),
            "penalty_labels": torch.tensor([target["penalty_label"] for target in targets], dtype=torch.float32)
        }
        return inputs, targets, defendant_nums

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate)
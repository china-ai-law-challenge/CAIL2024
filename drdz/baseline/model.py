import lightning as pl
import torch
from transformers import BertModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
from .case_level_metrics import get_case_level_metrics_by_per_defendant_metrics, calculate_final_score
from .preprocess_data import charge_num, article_num, penalty_num


class LJPBertModule(pl.LightningModule):
    def __init__(self, model_path, lr=1e-4):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.charge_classifier = torch.nn.Linear(self.bert.config.hidden_size, charge_num)
        self.article_classifer = torch.nn.Linear(self.bert.config.hidden_size, article_num)
        self.penalty_classifier = torch.nn.Linear(self.bert.config.hidden_size, penalty_num)
        self.lr = lr
        self.save_hyperparameters("model_path", "lr")
    
    def forward(self, input_ids, attention_mask, charge_labels=None, article_labels=None, penalty_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # (batch_size, hidden_size)
        outputs = self.dropout(outputs.pooler_output)
        # (batch_size, charge_label_num)
        charge_logits = self.charge_classifier(outputs)
        # (batch_size, article_label_num)
        article_logits = self.article_classifer(outputs)
        # (batch_size, penalty_label_num)
        penalty_logits = self.penalty_classifier(outputs)

        loss = None
        if charge_labels is not None and article_labels is not None and penalty_labels is not None:
            ce_loss = CrossEntropyLoss()
            bce_loss = BCEWithLogitsLoss()
            charge_loss = bce_loss(charge_logits, charge_labels.type_as(charge_logits))
            article_loss = bce_loss(article_logits, article_labels.type_as(article_logits))
            penalty_loss = ce_loss(penalty_logits, penalty_labels)
            # compute loss as the sum of three losses
            loss = charge_loss + article_loss + penalty_loss
        
        # processing into one-hot vector. threshold is 0.5 for multi-label classification
        charge_logits = (torch.sigmoid(charge_logits) > 0.5).int()
        article_logits = (torch.sigmoid(article_logits) > 0.5).int() 
        penalty_indices = torch.argmax(penalty_logits, dim=1)
        penalty_logits = torch.zeros_like(penalty_logits).scatter_(1, penalty_indices.unsqueeze(1), 1)
        return loss, charge_logits, article_logits, penalty_logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        loss, _, _, _ = self(input_ids=inputs["input_ids"], 
                                attention_mask=inputs["attention_mask"], 
                                charge_labels=targets["charge_labels"], 
                                article_labels=targets["article_labels"], 
                                penalty_labels=targets["penalty_labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.all_val_pred = {"charge": [], "article": [], "penalty": []}
        self.all_val_target = {"charge": [], "article": [], "penalty": []}
        self.all_val_defendant_nums = []

    def validation_step(self, batch, batch_idx):
        inputs, targets, defendant_nums = batch
        loss, charge_logits, article_logits, penalty_logits = self(input_ids=inputs["input_ids"], 
                                   attention_mask=inputs["attention_mask"], 
                                   charge_labels=targets["charge_labels"], 
                                   article_labels=targets["article_labels"], 
                                   penalty_labels=targets["penalty_labels"])
        self.all_val_pred["charge"].append(charge_logits)
        self.all_val_pred["article"].append(article_logits)
        self.all_val_pred["penalty"].append(penalty_logits)
        self.all_val_target["charge"].append(targets["charge_labels"])
        self.all_val_target["article"].append(targets["article_labels"])
        self.all_val_target["penalty"].append(targets["penalty_labels"])
        self.all_val_defendant_nums.extend(defendant_nums)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        # calculate scores
        self.all_val_pred["charge"] = torch.cat(self.all_val_pred["charge"], dim=0)
        self.all_val_pred["article"] = torch.cat(self.all_val_pred["article"], dim=0)
        self.all_val_pred["penalty"] = torch.cat(self.all_val_pred["penalty"], dim=0)
        self.all_val_target["charge"] = torch.cat(self.all_val_target["charge"], dim=0)
        self.all_val_target["article"] = torch.cat(self.all_val_target["article"], dim=0)
        self.all_val_target["penalty"] = torch.cat(self.all_val_target["penalty"], dim=0)
        case_level_metrics = get_case_level_metrics_by_per_defendant_metrics(self.all_val_pred, self.all_val_target, self.all_val_defendant_nums)
        final_score = calculate_final_score(case_level_metrics)
        # use final score as early stopping metric
        self.log("final_score", final_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.all_test_pred = {"charge": [], "article": [], "penalty": []}
        self.all_test_target = {"charge": [], "article": [], "penalty": []}
        self.all_test_defendant_nums = []
    
    def test_step(self, batch, batch_idx):
        inputs, targets, defendant_nums = batch
        _, charge_logits, article_logits, penalty_logits = self(input_ids=inputs["input_ids"], 
                                   attention_mask=inputs["attention_mask"], 
                                   charge_labels=targets["charge_labels"], 
                                   article_labels=targets["article_labels"], 
                                   penalty_labels=targets["penalty_labels"])
        self.all_test_pred["charge"].append(charge_logits)
        self.all_test_pred["article"].append(article_logits)
        self.all_test_pred["penalty"].append(penalty_logits)
        self.all_test_target["charge"].append(targets["charge_labels"])
        self.all_test_target["article"].append(targets["article_labels"])
        self.all_test_target["penalty"].append(targets["penalty_labels"])
        self.all_test_defendant_nums.extend(defendant_nums)

    def on_test_epoch_end(self) -> None:
        # calculate scores
        self.all_test_pred["charge"] = torch.cat(self.all_test_pred["charge"], dim=0)
        self.all_test_pred["article"] = torch.cat(self.all_test_pred["article"], dim=0)
        self.all_test_pred["penalty"] = torch.cat(self.all_test_pred["penalty"], dim=0)
        self.all_test_target["charge"] = torch.cat(self.all_test_target["charge"], dim=0)
        self.all_test_target["article"] = torch.cat(self.all_test_target["article"], dim=0)
        self.all_test_target["penalty"] = torch.cat(self.all_test_target["penalty"], dim=0)
        case_level_metrics = get_case_level_metrics_by_per_defendant_metrics(self.all_test_pred, self.all_test_target, self.all_test_defendant_nums)
        final_score = calculate_final_score(case_level_metrics)
        self.log("final_score", final_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
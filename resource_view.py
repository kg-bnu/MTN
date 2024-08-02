import os
import warnings

import transformers
from transformers import BertTokenizer, BertModel, BertForMaskedLM, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore", category=FutureWarning)

class MLMDataset(Dataset):
    def __init__(self, args, tokenizer):
        with open(args.data_path + "resources.tsv", 'r') as f:
            lines = [i.strip().split('\t')[1:] for i in f.readlines()]
            self.sequences = []
            for line in lines:
                self.sequences.append(', '.join(line))
        #self.sequences = self.sequences[:10]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

def mlm(args):        
    print('Resource view pre-training...')
    tokenizer = BertTokenizer.from_pretrained(args.bert_type)
    model = BertForMaskedLM.from_pretrained(args.bert_type)
    dataset = MLMDataset(args, tokenizer)
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=args.rs_mask_prob)
    training_args = TrainingArguments(
        f"temp",
        evaluation_strategy="epoch",
        learning_rate=args.rs_mlm_lr,
        weight_decay=0.01,
        num_train_epochs=args.rs_mlm_epoch,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    output_dir = '%soutput/seq_bert_%d_%s_%s_%s'%(args.data_path, args.fold_id, str(args.rs_mlm_epoch), str(args.rs_mlm_lr), str(args.rs_mask_prob))
    if not os.path.exists('%soutput/'%args.data_path):
        os.mkdir('%soutput/'%args.data_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    trainer.save_model(output_dir)
    return output_dir

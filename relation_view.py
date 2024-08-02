import os
import random
import numpy as np
import warnings

import transformers
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

transformers.logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KGEDataset(Dataset):
    def __init__(self, args):
        self.bert_type = args.bert_type
        
        with open('%sfold_%d/train.tsv'%(args.data_path, args.fold_id), 'r') as f:
            self.prerequisite_data = [line.strip().split('\t') for line in f.readlines()]
        with open(args.data_path+'entities.tsv', 'r') as f:
            self.id2entity = {line.strip().split('\t')[1]:line.strip().split('\t')[0] for line in f.readlines()}
        with open(args.data_path+'relations.tsv', 'r') as f:
            self.id2relation = {line.strip().split('\t')[1]:line.strip().split('\t')[0] for line in f.readlines()}
        with open(args.data_path+'entity_description.txt', 'r') as f:
            self.entity2description = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in f.readlines()}
        
        with open(args.data_path+'triples.tsv', 'r') as f:
            self.triples = [line.strip().split('\t') for line in f.readlines()]
        for line in self.prerequisite_data:
            if line[0] == '1':
                self.triples.append([self.id2entity[line[1]], 'prerequisite of', self.id2entity[line[2]]])
        self.labels = [1] * len(self.triples)
        self.negative_sample(negative_sampling_num = args.rel_negative_num)

        if args.use_description:
            for triple in self.triples:
                triple[0] = self.entity2description[triple[0]] if triple[0] in self.entity2description else triple[0]
                triple[2] = self.entity2description[triple[2]] if triple[2] in self.entity2description else triple[2]

        self.data = ['[CLS] %s [SEP] %s [SEP] %s [SEP]'%(triple[0], triple[1], triple[2]) for triple in self.triples]
        #self.data = self.data[:10]

    def __len__(self):
        return len(self.data)
    
    def negative_sample(self, negative_sampling_num):
        data_len = len(self.triples)
        for t in tqdm(range(data_len)):
            triple = self.triples[t]
            for i in range(negative_sampling_num):
                flag = random.randint(0, 1)
                if flag == 0:
                    negative_head = np.random.choice(list(self.id2entity.values()))
                    while [negative_head, triple[1], triple[2]] in self.triples:
                        negative_head = np.random.choice(list(self.id2entity.values()))
                    self.triples.append([negative_head, triple[1], triple[2]])
                    self.labels.append(0)
                else:
                    negative_tail = np.random.choice(list(self.id2entity.values()))
                    while [triple[0], triple[1], negative_tail] in self.triples:
                        negative_tail = np.random.choice(list(self.id2entity.values()))
                    self.triples.append([triple[0], triple[1], negative_tail])
                    self.labels.append(0)

    def __getitem__(self, i):
        return self.labels[i], self.data[i]

class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_type)
        self.bert_model = BertModel.from_pretrained(args.bert_type)
        self.linear = nn.Linear(768, 1)
    
    def forward(self, data):
        encoding = self.tokenizer(data, return_tensors='pt', padding='max_length', truncation=True, max_length=200)
        bert_encode = self.bert_model(**encoding.to(device))

        bert_encode = bert_encode.last_hidden_state
        bert_encode = bert_encode[:, 0, :]
        output = torch.sigmoid(self.linear(bert_encode))
        return output

def kge(args):
    print('Relation view pre-training...')
    train_dataset = KGEDataset(args)
    model = KGEModel(args).to(device)
    params = [{'params': model.parameters(), 'lr': args.rel_lr}]
    optimizer = torch.optim.Adam(params)
    crit = nn.BCELoss()
    output_dir = '%soutput/kge_bert_%d_%s_%s_%s/'%(args.data_path, args.fold_id, str(args.rel_lr), str(args.rel_negative_num), str(args.rel_epoch))
    
    if not os.path.exists('%soutput/'%args.data_path):
        os.mkdir('%soutput/'%args.data_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    max_f1 = 0
    max_i_m = 0
    print('output_dir: %s'%(output_dir))
    for epoch_num in range(args.rel_epoch):
        model.train()
        train_loss = 0
        train_acc = 0
        train_loader = DataLoader(train_dataset, batch_size=args.rel_batch_size, shuffle=True)
        for i, (labels, sentences) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = model(sentences)
            output = output.squeeze(1)
            labels = labels.to(device).float()
            loss = crit(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += ((output > 0.5).int() == labels.int()).sum().item()
    model.bert_model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    return max_i_m, max_f1, output_dir

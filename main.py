import argparse
import os
import random
import numpy as np
import warnings

import transformers
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from resource_view import mlm 
from relation_view import kge
from text_view import text_train

transformers.logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MTNDataset(Dataset):
    def __init__(self, args, mode):
        with open('%sfold_%d/%s.tsv'%(args.data_path, args.fold_id, mode), 'r') as f:
            self.data = [line.strip().split('\t') for line in f.readlines()]
        
        with open(args.data_path+'concepts.tsv', 'r') as f:
            self.concept = {line.strip().split('\t')[1]:line.strip().split('\t')[0] for line in f.readlines()}
        with open(args.data_path+'descriptions.tsv', 'r') as f:
            self.description = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in f.readlines()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = int(self.data[i][0])

        id1 = self.data[i][1]
        id2 = self.data[i][2]

        sen1 = self.description[id1] if id1 in self.description else self.concept[id1]
        sen2 = self.description[id2] if id2 in self.description else self.concept[id2]

        text_sentence = "[CLS] " + sen1 + " [SEP] " + sen2 + " [SEP]"
        resource_sentence = "[CLS] %s, %s [SEP]"%(self.concept[id1], self.concept[id2])
        kg_sentence = "[CLS] %s [SEP] prerequisite of [SEP] %s [SEP]"%(self.concept[id1], self.concept[id2])

        return label, text_sentence, resource_sentence, kg_sentence

class MTNModel(nn.Module):
    def __init__(self, args):
        super(MTNModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_type)

        masked_lm_model_path = '%soutput/seq_bert_%d_%s_%s_%s'%(args.data_path, args.fold_id, str(args.rs_mlm_epoch), str(args.rs_mlm_lr), str(args.rs_mask_prob))
        self.resource_bert_model = BertModel.from_pretrained(masked_lm_model_path)
        
        kge_model_path = '%soutput/kge_bert_%d_%s_%s_%s/'%(args.data_path, args.fold_id, str(args.rel_lr), str(args.rel_negative_num), str(args.rel_epoch))
        self.kg_bert_model = BertModel.from_pretrained(kge_model_path)

        text_model_path = '%soutput/text_bert_%d_%s_%s/'%(args.data_path, args.fold_id, str(args.text_lr), str(args.text_epoch))
        self.text_bert_model = BertModel.from_pretrained(text_model_path)

        self.att_linear = nn.Linear(768, 1)
        self.linear = nn.Linear(768, 1)
        self.att = torch.zeros(8, 3, dtype = float)

    def forward(self, text_sentences, resource_sentences, kg_sentences):
        resource_encoding = self.tokenizer(resource_sentences, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        resource_bert_encode = self.resource_bert_model(**resource_encoding.to(device))
        resource_bert_encode = resource_bert_encode.last_hidden_state
        resource_bert_encode = resource_bert_encode[:, 0, :]

        kg_encoding = self.tokenizer(kg_sentences, return_tensors='pt', padding='max_length', truncation=True, max_length=200)
        kg_bert_encode = self.kg_bert_model(**kg_encoding.to(device))
        kg_bert_encode = kg_bert_encode.last_hidden_state
        kg_bert_encode = kg_bert_encode[:, 0, :]

        text_encoding = self.tokenizer(text_sentences, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        text_bert_encode = self.text_bert_model(**text_encoding.to(device))
        text_bert_encode = text_bert_encode.last_hidden_state
        text_bert_encode = text_bert_encode[:, 0, :]

        att_text = F.leaky_relu(self.att_linear(text_bert_encode))
        att_resource = F.leaky_relu(self.att_linear(resource_bert_encode))
        att_kg = F.leaky_relu(self.att_linear(kg_bert_encode))

        concat_att = F.softmax(torch.stack([att_text, att_resource, att_kg], dim = 2), dim = 2)
        concat_embeds = torch.stack([text_bert_encode, resource_bert_encode, kg_bert_encode], dim = 1)

        self.att = concat_att

        embeds = torch.matmul(concat_att, concat_embeds)

        out = torch.sigmoid(self.linear(embeds))

        return out

def train(dataset, model, optimizer, criterion, batch_size, epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i, (labels, text_sentence, resource_sentence, kg_sentence) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        output = model(text_sentence, resource_sentence, kg_sentence)
        output = output.squeeze()
        labels = labels.to(device).float()
        labels = labels.squeeze()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += ((output > 0.5).int() == labels.int()).sum().item()  
    print('Epoch %d, Train Loss: %.4f, Train Acc: %.4f'%(epoch, train_loss/len(train_loader), train_acc/len(train_loader.dataset)))

def valid(dataset, model, criterion, batch_size, epoch):
    model.eval()
    valid_loss = 0
    valid_acc = np.array([0] * 8)
    prediction_all = np.empty((8,0))
    output_all = np.array([])
    label_all = np.array([])
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for _, (labels, text_sentence, resource_sentence, kg_sentence) in enumerate(tqdm(valid_loader)):
        output = model(text_sentence, resource_sentence, kg_sentence)
        output = output.squeeze()
        labels = labels.to(device).float()
        labels = labels.squeeze()
        loss = criterion(output, labels)
        valid_loss += loss.item()

        valid_acc += np.array([((output > (i*0.1)).int() == labels.int()).sum().item() for i in range(1,9)])
        prediction = np.array([(output > (i*0.1)).int().cpu().numpy().reshape(-1) for i in range(1,9)])
        prediction_all = np.concatenate((prediction_all, prediction), axis=1)
        output_all = np.concatenate((output_all, output.cpu().detach().numpy().reshape(-1)))
        label_all = np.concatenate((label_all, labels.cpu().numpy().reshape(-1)))
    
    precision = np.array([precision_score(label_all, prediction_all[i]) for i in range(8)])
    recall =  np.array([recall_score(label_all, prediction_all[i]) for i in range(8)])
    F_score = np.array([f1_score(label_all, prediction_all[i]) for i in range(8)])
    AUC = roc_auc_score(label_all, output_all)
    i_m = F_score.argmax(axis = 0)

    return i_m, precision[i_m], recall[i_m], F_score[i_m], AUC

def test(dataset, model, criterion, batch_size, epoch, border):
    model.eval()
    valid_loss = 0
    valid_acc = 0
    prediction_all = np.array([])
    output_all = np.array([])
    label_all = np.array([])
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    result_mtn_raw = []
    for _, (labels, text_sentence, resource_sentence, kg_sentence) in enumerate(tqdm(valid_loader)):
        output = model(text_sentence, resource_sentence, kg_sentence)
        output = output.squeeze()
        labels = labels.to(device).float()
        labels = labels.squeeze()
        predicts = (output > border).int().cpu().numpy()
        loss = criterion(output, labels)
        valid_loss += loss.item()

        valid_acc += ((output > border).int() == labels.int()).sum().item()  
        
        prediction_all = np.append(prediction_all, (output > border).int().cpu().numpy())
        output_all = np.append(output_all, output.cpu().detach().numpy())
        label_all = np.append(label_all, labels.cpu().detach().numpy())
        try:
            for i in range(len(predicts)):
                result_mtn_raw.append('%d\t%lf\n'%(int(labels[i]), float(output[i])))
        except:
            continue

    precision = precision_score(label_all, prediction_all)
    recall =  recall_score(label_all, prediction_all)
    F_score = f1_score(label_all, prediction_all)
    AUC = roc_auc_score(label_all, output_all)

    print('Epoch %d, Valid Loss: %.4f, Valid Acc: %.4f'%(epoch, valid_loss/len(valid_loader), valid_acc/len(valid_loader.dataset)))
    print('Precision: %.4f, Recall: %.4f, F_score: %.4f, AUC: %.4f'%(precision, recall, F_score, AUC))
    return precision, recall, F_score, AUC, result_mtn_raw

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True

def mtn_valid(args):
    train_dataset = MTNDataset(args, 'train')
    valid_dataset = MTNDataset(args, 'valid')

    model = MTNModel(args).to(device)
    params = [{'params': model.parameters(), 'lr': args.mtn_lr}]
    optimizer = torch.optim.Adam(params)
    crit = nn.BCELoss()    
    max_f1 = 0
    for epoch_num in range(args.mtn_epoch):
        train(train_dataset, model, optimizer, crit, args.mtn_batch_size, epoch_num)
        i_m, p, r, f1, auc = valid(valid_dataset, model, crit, args.mtn_batch_size, epoch_num)
        
        if max_f1 < f1:
            max_f1 = f1
            precision, recall, F_score, AUC = p, r, f1, auc
    return precision, recall, F_score, AUC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="benchmarks/hard_setting/LectureBank/", help="Path of the dataset.")
    parser.add_argument("--fold_id", type=int, default=0, help="Fold ID of the dataset.")
    parser.add_argument("--use_description", type=bool, default=False, help="Use entity description or not.")
    parser.add_argument("--bert_type", type=str, default="bert-base-uncased", help="Type of BERT model.")
    parser.add_argument("--rel_negative_num", type=int, default=1, help="Ratio of negative sampling in relation-view.")
    parser.add_argument("--rel_batch_size", type=int, default=8, help="Batch size of relation-view pre-training.")
    parser.add_argument("--rel_epoch", type=int, default=3, help="Epoch number of relation-view pre-training.")
    parser.add_argument("--rel_lr", type=float, default=1e-5, help="Learning rate of relation-view pre-training.")
    parser.add_argument("--rs_mask_prob", type=float, default=0.15, help="Randomly select 15% of the input tokens for masking")
    parser.add_argument("--rs_mlm_epoch", type=int, default=3, help="Learning rate of MLM in resource module.")
    parser.add_argument("--rs_mlm_lr", type=float, default=3e-6, help="Learning rate of MLM in resource module.")
    parser.add_argument("--text_epoch", type=int, default=3, help="Learning rate of text module.")
    parser.add_argument("--text_lr", type=float, default=1e-5, help="Learning rate of text module.")
    parser.add_argument("--mtn_batch_size", type=int, default=8, help="Batch size of MTN training.")
    parser.add_argument("--mtn_epoch", type=int, default=10, help="Epoch number of MTN training.")
    parser.add_argument("--mtn_lr", type=float, default=1e-5, help="Learning rate of MTN.")
    args = parser.parse_args()
    seed_torch()

    kge_model_path = '%soutput/kge_bert_%d_%s_%s_%s/'%(args.data_path, args.fold_id, str(args.rel_lr), str(args.rel_negative_num), str(args.rel_epoch))
    if not os.path.exists(kge_model_path):
        kge(args)
    masked_lm_model_path = '%soutput/seq_bert_%d_%s_%s_%s'%(args.data_path, args.fold_id, str(args.rs_mlm_epoch), str(args.rs_mlm_lr), str(args.rs_mask_prob))
    if not os.path.exists(masked_lm_model_path):
        mlm(args)
    text_model_path = '%soutput/text_bert_%d_%s_%s/'%(args.data_path, args.fold_id, str(args.text_lr), str(args.text_epoch))
    if not os.path.exists(text_model_path):
        text_train(args)

    train_dataset = MTNDataset(args, 'train')
    valid_dataset = MTNDataset(args, 'valid')
    test_dataset = MTNDataset(args, 'test')

    model = MTNModel(args).to(device)
    params = [{'params': model.parameters(), 'lr': args.mtn_lr}]
    optimizer = torch.optim.Adam(params)
    crit = nn.BCELoss()
    max_f1 = 0
    
    for epoch_num in range(args.mtn_epoch):
        train(train_dataset, model, optimizer, crit, args.mtn_batch_size, epoch_num)
        i_m, precision, recall, F_score, AUC = valid(valid_dataset, model, crit, args.mtn_batch_size, epoch_num)
        
        if max_f1 < F_score:
            max_f1 = F_score
            precision, recall, F_score, AUC, result_mtn_raw = test(test_dataset, model, crit, args.mtn_batch_size, epoch_num, i_m*0.1+0.1)
            print('Epoch %d:\nPrecision: %.4f\nRecall: %.4f\nF_score: %.4f\nAUC: %.4f\nThreshold: %.2f\n\n'%(epoch_num, precision, recall, F_score, AUC, i_m*0.1+0.1)) 

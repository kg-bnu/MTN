import os
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

class TextDataset(Dataset):
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

        sentence = "[CLS] " + sen1 + " [SEP] " + sen2 + " [SEP]"
        
        return label, sentence


class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_type)
        self.bert_model = BertModel.from_pretrained(args.bert_type)
        self.linear = nn.Linear(768, 1)

    def forward(self, sentences):
        encoding = self.tokenizer(sentences, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        bert_encode = self.bert_model(**encoding.to(device))
        bert_encode = bert_encode.last_hidden_state
        bert_encode = bert_encode[:, 0, :]

        out = torch.sigmoid(self.linear(bert_encode))

        return out

#def text_train(dataset, model, optimizer, criterion, batch_size, epoch):
def text_train(args):
    print('Text view pre-training...')
    dataset = TextDataset(args, 'train')
    model = TextModel(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.text_lr)
    criterion = nn.BCELoss()
    epoch = args.text_epoch

    model.train()
    train_loss = 0
    train_acc = 0
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for i, (labels, sentences) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        output = model(sentences)
        output = output.squeeze(1)
        labels = labels.to(device).float()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += ((output > 0.5).int() == labels.int()).sum().item()                                                                                                                                                                                                                                                                                                                                                                                                           
    print('Epoch %d, Train Loss: %.4f, Train Acc: %.4f'%(epoch, train_loss/len(train_loader), train_acc/len(train_loader.dataset)))

    output_dir = '%soutput/text_bert_%d_%s_%s/'%(args.data_path, args.fold_id, str(args.text_lr), str(args.text_epoch))
    
    if not os.path.exists('%soutput/'%args.data_path):
        os.mkdir('%soutput/'%args.data_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model.bert_model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
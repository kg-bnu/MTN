import pandas as pd
import random
import re
import os
import json
import argparse
from tqdm import tqdm

from wiki_query import kg_process

parser = argparse.ArgumentParser()

parser.add_argument("--wiki_query", action='store_true', help="Whether to query the concept descriptions from Wikipedia.")
args = parser.parse_args()

def extract_concepts(text, concepts):
    pattern = r'\b(?:' + r'|'.join([re.escape(c) for c in concepts]) + r')\b'
    matches = re.findall(pattern, text, re.IGNORECASE)

    return matches

# LectureBank
raw_data_path = 'benchmarks/hard_setting/LectureBank/raw_data/'

with open(raw_data_path + '208topics.csv', 'r') as f:
    concepts = []
    raw_data_dict = {}
    for line in f.readlines():
        line = line.strip().split(',')
        raw_data_dict[line[0]] = line[1].replace(';', ',')
        concepts.append(line[1].replace(';', ','))

with open('benchmarks/hard_setting/LectureBank/concepts.tsv', 'w') as f:
    for index, concept in enumerate(concepts):
        f.write('%s\t%d\n'%(concept.lower(), index))

with open(raw_data_path + 'prerequisite_annotation.csv', 'r') as f:
    prerequisites = []
    for line in f.readlines():
        line = line.strip().split(',')
        if int(line[0]) > 209 or int(line[1]) > 209:
            continue
        if line[-1] == '1':
            prerequisites.append('%s\tprerequisite_of\t%s\n'%(raw_data_dict[line[0]].lower(), raw_data_dict[line[1]].lower()))

with open('benchmarks/hard_setting/LectureBank/prerequisites.tsv', 'w') as f:
    f.writelines(prerequisites)

concepts = []
with open('benchmarks/hard_setting/LectureBank/concepts.tsv', 'r') as f:
    for line in f.readlines():
        concepts.append(line.strip().split('\t')[0])
resources_text = []
file_path = 'benchmarks/hard_setting/LectureBank/raw_data/lecturebank_text_files/'
for file in os.listdir(file_path):
    with open(file_path+file, 'r') as f:
        resources_text.append(f.read().lower())
sequences = []
for text in tqdm(resources_text):
    sequences.append(extract_concepts(text, concepts))
with open('benchmarks/hard_setting/LectureBank/resources.tsv', 'w') as f:
    for i in range(len(sequences)):
        if len(sequences[i]) < 2:
            continue
        seqs = []
        st = 0
        Len = len(sequences[i])
        while st < Len:
            seqs.append(sequences[i][st])
            st += 1
            while st < Len and sequences[i][st-1] == sequences[i][st]:
                st += 1
        f.write('%s\t%s\n'%(i, '\t'.join(seqs)))

# MOOC DSA&ML
raw_data_path = ['benchmarks/hard_setting/MOOC_DSA/raw_data/', 'benchmarks/hard_setting/MOOC_ML/raw_data/']
output_path = ['benchmarks/hard_setting/MOOC_DSA/', 'benchmarks/hard_setting/MOOC_ML/']
dataset_name = ['DSA', 'ML']
for i in range(2):
    with open(raw_data_path[i] + dataset_name[i] + '_LabeledFile', 'r') as f:
        concepts = []
        prerequisites = []
        for line in f.readlines():
            line = line.strip().split('\t\t')
            if line[0] not in concepts:
                concepts.append(line[0])
            if line[1] not in concepts:
                concepts.append(line[1])
            if line[2] == '1-':
                prerequisites.append('%s\tprerequisite_of\t%s\n'%(line[1].lower(), line[0].lower()))
            elif line[2] == '-1':
                prerequisites.append('%s\tprerequisite_of\t%s\n'%(line[0].lower(), line[1].lower()))

    with open(output_path[i] + 'concepts.tsv', 'w') as f:
        for index, concept in enumerate(concepts):
            f.write('%s\t%d\n'%(concept.lower(), index))
    
    with open(output_path[i] + 'prerequisites.tsv', 'w') as f:
        f.writelines(prerequisites)

def load_data(data_path):
    with open(data_path, 'r') as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data

raw_resources_path = [['Captions_algorithms_Princeton.json', 'Captions_algorithms_Stanford.json', 'Captions_data-structure-and-algorithm_UC-San-Diego.json'], ['Captions_machine-learning_Stanford.json', 'Captions_machine-learning_Washington.json']]
raw_data_path = ['benchmarks/hard_setting/MOOC_DSA/raw_data/', 'benchmarks/hard_setting/MOOC_ML/raw_data/']
output_path = ['benchmarks/hard_setting/MOOC_DSA/', 'benchmarks/hard_setting/MOOC_ML/']
for i in range(2):
    with open(output_path[i] + 'concepts.tsv') as f:
        concepts = [line.strip().split('\t')[0] for line in f.readlines()]
    resources_raw = []
    for j in range(len(raw_resources_path[i])):
        resources_raw += load_data(raw_data_path[i] + raw_resources_path[i][j])
    resources_text = [i['text'].lower() for i in resources_raw]
    sequences = []
    for line in tqdm(resources_text):
        sequences.append(extract_concepts(line, concepts))
    with open(output_path[i] + 'resources.tsv', 'w') as f:
        for i in range(len(sequences)):
            if len(sequences[i]) < 2:
                continue
            seqs = []
            st = 0
            Len = len(sequences[i])
            while st < Len:
                while 0 < st < Len and sequences[i][st-1] == sequences[i][st]:
                    st += 1
                if st >= Len:
                    break
                seqs.append(sequences[i][st])
                st += 1
            f.write('%s\t%s\n'%(i, '\t'.join(seqs)))

raw_resources_path = [['Captions_algorithms_Princeton.json', 'Captions_algorithms_Stanford.json', 'Captions_data-structure-and-algorithm_UC-San-Diego.json'], ['Captions_machine-learning_Stanford.json', 'Captions_machine-learning_Washington.json']]
raw_data_path = ['benchmarks/hard_setting/MOOC_DSA/raw_data/', 'benchmarks/hard_setting/MOOC_ML/raw_data/']
output_path = ['benchmarks/hard_setting/MOOC_DSA/', 'benchmarks/hard_setting/MOOC_ML/']
for i in range(2):
    with open(output_path[i] + 'concepts.tsv') as f:
        concepts = [line.strip().split('\t')[0] for line in f.readlines()]
    resources_raw = []
    for j in range(len(raw_resources_path[i])):
        resources_raw += load_data(raw_data_path[i] + raw_resources_path[i][j])
    #print(len(resources_raw))

# University Course
raw_data_path = 'benchmarks/hard_setting/UC/raw_data/'

with open(raw_data_path + 'cs_preqs.csv', 'r') as f:
    concepts = []
    prerequisites = []
    for line in f.readlines():
        line = line.strip().split(',')
        line[0] = line[0].replace('_', ' ').replace(';', ',').lower()
        line[1] = line[1].replace('_', ' ').replace(';', ',').lower()
        if line[0] not in concepts:
            concepts.append(line[0])
        if line[1] not in concepts:
            concepts.append(line[1])
        prerequisites.append('%s\tprerequisite_of\t%s\n'%(line[1], line[0]))

with open('benchmarks/hard_setting/UC/concepts.tsv', 'w') as f:
    for index, concept in enumerate(concepts):
        f.write('%s\t%d\n'%(concept, index))

with open('benchmarks/hard_setting/UC/prerequisites.tsv', 'w') as f:
    f.writelines(prerequisites)

concepts = []
with open('benchmarks/hard_setting/UC/concepts.tsv', 'r') as f:
    for line in f.readlines():
        concepts.append(line.strip().split('\t')[0])
resources_text = []
with open('benchmarks/hard_setting/UC/raw_data/cs_courses.csv', 'r') as f:
    for line in f.readlines():
        line = line.strip().split(',', 1)
        resources_text.append(line[1].lower())
sequences = []
for line in tqdm(resources_text):
    sequences.append(extract_concepts(line, concepts))
with open('benchmarks/hard_setting/UC/resources.tsv', 'w') as f:
    for i in range(len(sequences)):
        if len(sequences[i]) < 2:
            continue
        seqs = []
        st = 0
        Len = len(sequences[i])
        while st < Len:
            while 0 < st < Len and sequences[i][st-1] == sequences[i][st]:
                st += 1
            if st >= Len:
                break
            seqs.append(sequences[i][st])
            st += 1
        f.write('%s\t%s\n'%(i, '\t'.join(seqs)))

# concept graph building
if args.wiki_query:
    concepts = []
    with open('benchmarks/hard_setting/LectureBank/concepts.tsv', 'r') as f:
        for line in f.readlines():
            concepts.append(line.strip().split('\t')[0])
    with open('benchmarks/hard_setting/MOOC_DSA/concepts.tsv', 'r') as f:
        for line in f.readlines():
            concepts.append(line.strip().split('\t')[0])
    with open('benchmarks/hard_setting/MOOC_ML/concepts.tsv', 'r') as f:
        for line in f.readlines():
            concepts.append(line.strip().split('\t')[0])
    with open('benchmarks/hard_setting/UC/concepts.tsv', 'r') as f:
        for line in f.readlines():
            concepts.append(line.strip().split('\t')[0])
    concepts = list(set(concepts))
    triples, defs = kg_process(concepts[:10])

dataset_path = ['benchmarks/hard_setting/LectureBank/', 'benchmarks/hard_setting/MOOC_DSA/', 'benchmarks/hard_setting/MOOC_ML/', 'benchmarks/hard_setting/UC/']
descriptions_all = {}
with open('benchmarks/hard_setting/wiki_description.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        descriptions_all[line[0]] = line[1]
# 读取数据集
for path in dataset_path:
    concepts = []
    with open(path + 'concepts.tsv', 'r') as f:
        for line in f.readlines():
            concepts.append(line.strip().split('\t')[0])
    descriptions = {}
    for concept in concepts:
        if concept in descriptions_all:
            descriptions[concept] = descriptions_all[concept]
    with open(path + 'descriptions.tsv', 'w') as f:
        for concept in descriptions:
            f.write(concept + '\t' + descriptions[concept] + '\n')

from collections import defaultdict
dataset_path = ['benchmarks/hard_setting/LectureBank/', 'benchmarks/hard_setting/MOOC_DSA/', 'benchmarks/hard_setting/MOOC_ML/', 'benchmarks/hard_setting/UC/']
triples_all = []
with open('benchmarks/hard_setting/triples.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        triples_all.append(line)
# 读取数据集
for path in dataset_path:
    concepts = []
    with open(path + 'concepts.tsv', 'r') as f:
        for line in f.readlines():
            concepts.append(line.strip().split('\t')[0])
    entities = set()
    cunt = 0
    for triple in triples_all:
        if triple[0] in concepts or triple[2] in concepts:
            entities.add(triple[0])
            entities.add(triple[2])
            cunt += 1
    triples = []
    for triple in triples_all:
        if triple[0] in entities or triple[2] in entities:
            triples.append(triple)
    with open(path + 'triples.tsv', 'w') as f:
        for triple in triples:
            f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')

    # KG simplify
    neighbors = defaultdict(list)
    for triple in triples:
        neighbors[triple[0]].append(triple[2])
        neighbors[triple[2]].append(triple[0])
    triples_new = []
    for triple in triples:
        if triple[0] not in concepts and len(neighbors[triple[0]]) == 1:
            continue
        if triple[2] not in concepts and len(neighbors[triple[2]]) == 1:
            continue
        triples_new.append(triple)
        
    entities = {}
    relations = {}
    with open(path + 'concepts.tsv', 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            entities[line[0]] = int(line[1])
    entity_id = len(entities)
    relation_id = 0
    for triple in triples_new:    
        if triple[0] not in entities:
            entities[triple[0]] = entity_id
            entity_id += 1
        if triple[2] not in entities:
            entities[triple[2]] = entity_id
            entity_id += 1
        if triple[1] not in relations:
            relations[triple[1]] = relation_id
            relation_id += 1
    with open(path + 'entities.tsv', 'w') as f:
        for entity in entities:
            f.write(entity + '\t' + str(entities[entity]) + '\n')
    with open(path + 'relations.tsv', 'w') as f:
        for relation in relations:
            f.write(relation + '\t' + str(relations[relation]) + '\n')
    with open(path + 'triples.tsv', 'w') as f:
        for triple in triples_new:
            f.write(triple[0] + '\t' + triple[1].replace('_', ' ') + '\t' + triple[2] + '\n')

dataset_path = ['benchmarks/hard_setting/LectureBank/', 'benchmarks/hard_setting/MOOC_DSA/', 'benchmarks/hard_setting/MOOC_ML/', 'benchmarks/hard_setting/UC/']
entities = set()
for path in dataset_path:
    with open(path + 'entities.tsv', 'r') as f:
        for line in f.readlines():
            entities.add(line.strip().split('\t')[0])
with open('benchmarks/hard_setting/entities.tsv', 'w') as f:
    f.write('\n'.join(list(entities))+'\n')

for path in dataset_path:
    e_description = {}
    with open(path + 'entities.tsv', 'r') as f:
        for line in f.readlines():
            entities.add(line.strip().split('\t')[0])

def data_split(triples, fold_num = 5):
    random.shuffle(triples)
    fold_size = len(triples) // fold_num
    folds = []
    for i in range(fold_num):
        folds.append(triples[i*fold_size:(i+1)*fold_size])
    return folds

def negative_sample(concepts, data, triples, negative_sampling_ratio):
    ns_num = len(data) * negative_sampling_ratio
    for i in range(len(data)):
        data.append(['0', data[i][2], data[i][1]])
    num = 0
    while num < ns_num:
        data_index = random.randint(0, len(data)-1)
        flag = random.randint(0, 1)
        if flag == 0:
            c1 = data[data_index][1]
            c2 = random.randint(0, len(concepts)-1)
        else:
            c1 = random.randint(0, len(concepts)-1)
            c2 = data[data_index][2]
        if c1 == c2 or ['0', str(c1), str(c2)] in data or ['1', str(c1), str(c2)] in triples:
            continue
        data.append(['0', str(c1), str(c2)])
        num += 1
    return data

def data_process(dataset_path, negative_sampling_ratio = 7, fold_num = 5, is_inference = False):
    concepts = {}
    triples = []
    f = open(dataset_path + 'concepts.tsv', 'r', encoding = 'utf-8')
    text = f.readlines()
    f.close()
    for i in text:
        temp = i.strip().split('\t')
        concepts[temp[0]] = temp[1]
    #print(concepts)
    f = open(dataset_path + 'prerequisites.tsv', 'r', encoding = 'utf-8')
    text = f.readlines()
    f.close()
    random.shuffle(text)
    for i in text:
        temp = i.strip().split('\t')
        #print(temp)
        triples.append(['1', concepts[temp[0]], concepts[temp[2]]])
    
    data = data_split(triples, fold_num=fold_num)
    for fold_id in range(fold_num):
        #print('fold_id: ', fold_id)
        valid = list(data[fold_id])
        test = list(data[(fold_id+1)%fold_num])
        train = []
        for i in range(fold_num):
            if i != fold_id and i != (fold_id+1)%fold_num:
                train += list(data[i])
        
        #print(len(train), len(valid), len(test))
        train = negative_sample(concepts, train, triples, negative_sampling_ratio)
        valid = negative_sample(concepts, valid, triples, negative_sampling_ratio)
        test = negative_sample(concepts, test, triples, negative_sampling_ratio)

        output_dir = '%sfold_%d/'%(dataset_path, fold_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        f = open(output_dir + 'train.tsv', 'w', encoding = 'utf')
        for i in train:
            f.write('\t'.join(i) + '\n')
        f.close()
        f = open(output_dir + 'test.tsv', 'w', encoding = 'utf')
        for i in test:
            f.write('\t'.join(i) + '\n')
        f.close()
        f = open(output_dir + 'valid.tsv', 'w', encoding = 'utf')
        for i in valid:
            f.write('\t'.join(i) + '\n')
        f.close()
        #print(len(train), len(valid), len(test))

dataset_path = ['benchmarks/hard_setting/LectureBank/', 'benchmarks/hard_setting/MOOC_DSA/', 'benchmarks/hard_setting/MOOC_ML/', 'benchmarks/hard_setting/UC/']
for path in dataset_path:
    data_process(dataset_path=path, negative_sampling_ratio = 7, fold_num=5, is_inference=False)
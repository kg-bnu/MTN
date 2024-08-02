import wikipedia
from SPARQLWrapper import SPARQLWrapper, JSON

from tqdm import tqdm
import requests
import sys
import re
import time
import random
from urllib.request import unquote
from bs4 import BeautifulSoup
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.63'}

query_1 = """SELECT ?wdt ?item ?itemLabel
WHERE
{
  wd:%s ?wdt ?item.
  VALUES ?wdt { wdt:P279 wdt:P527 wdt:P31 wdt:P361 wdt:P1269 wdt:P1552 wdt:P2670 }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

query_2 = """SELECT ?wdt ?item ?itemLabel
WHERE
{
  ?item ?wdt wd:%s.
  VALUES ?wdt { wdt:P279 wdt:P527 wdt:P31 wdt:P361 wdt:P1269 wdt:P1552 wdt:P2670 }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

property_ids = {
    "P279": "subclass_of",
    "P527": "has_part",
    "P31": "instance_of",
    "P361": "part_of",
    "P1269": "facet_of",
    "P1552": "has_quality",
    "P2670": "has_part_of"
}

def get_concept_description(concepts):
    defs = {}
    for concept in tqdm(concepts):
        try:
            defs[concept] = wikipedia.summary(concept, sentences=1).split('\n')[0]
            if defs[concept] != '.':
                text = wikipedia.summary(concept).replace('\n', '').replace('\r', '').replace('  ', '').split('.')[0] + '.'
                defs[concept] = text
        except:
            continue
    return defs

def get_wiki_id(concepts):
    concept2Qid = {}
    for concept in tqdm(concepts):
        try:
            url = wikipedia.page(concept).url
            r = requests.get(url, headers = headers)
            r.encoding = r.apparent_encoding
            html = r.text
            soup = BeautifulSoup(html, 'html.parser')
            temp2 = soup.find_all('li', {'id':'t-wikibase'})
            tempID=temp2[0].select('a')[0].get('href').split('/')[-1]
            concept2Qid[concept] = tempID
        except:
            continue
    return concept2Qid

def get_triples(endpoint_url, concept2Qid):
  triples = []
  Qid_list = []
  user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
  sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
  # TODO adjust user agent; see https://w.wiki/CX6
  for concept in tqdm(concept2Qid):
    Qid = concept2Qid[concept]
    try:
      sparql.setQuery(query_1%Qid)
      sparql.setReturnFormat(JSON)
      results = sparql.query().convert()
      for line in results["results"]["bindings"]:
        rel = property_ids[line['wdt']['value'].split('/')[-1]]
        entity = line['itemLabel']['value']
        entity_id = line['item']['value'].split('/')[-1]
        Qid_list.append([entity, entity_id])
        triples.append([Qid, rel, entity])
    except:
      print('query_1 failed: %s'%(concept))
    try:
      sparql.setQuery(query_2%Qid)
      sparql.setReturnFormat(JSON)
      results = sparql.query().convert()
      for line in results["results"]["bindings"]:
        rel = property_ids[line['wdt']['value'].split('/')[-1]]
        entity = line['itemLabel']['value']
        entity_id = line['item']['value'].split('/')[-1]
        Qid_list.append([entity, entity_id])
        triples.append([entity, rel, Qid])
    except:
      print('query_2 failed: %s'%(concept)) 
  return triples, Qid_list

def kg_process(concepts):
    defs = get_concept_description(concepts)
    concept2Qid = get_wiki_id(concepts)
    triples_v1, core_concept_list = get_triples("https://query.wikidata.org/sparql", concept2Qid)
    core_conept2Qid = {x[0]:x[1] for x in core_concept_list}
    triples_v2, qid_list = get_triples("https://query.wikidata.org/sparql", core_conept2Qid)
    core_concepts = set([x[0] for x in core_concept_list])
    triples = []
    for triple in triples_v2:
        if triple[0] in core_concepts and triple[2] in core_concepts:
            triples.append(triple)
    with open('benchmarks/concepts.tsv', 'w') as f:
        for i in range(len(concepts)):
            f.write('%s\t%d\n'%(concepts[i], i))
    with open('benchmarks/wiki_description.txt', 'w') as f:
        for concept in defs:
            f.write('%s\t%s\n'%(concept, defs[concept]))
    with open('benchmarks/triples.txt', 'w') as f:
        for triple in triples:
            f.write('%s\t%s\t%s\n'%(triple[0], triple[1], triple[2]))
    return triples, defs
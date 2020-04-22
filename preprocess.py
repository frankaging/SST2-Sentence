######################################################
#
# SST dataset, the sentences are encoded with latin1,
# and the rest is in utf-8. You have to decode and 
# encode properly to construct the dataset.
#
######################################################


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, copy, itertools
import pandas as pd
import numpy as np
import csv
import bs4 as bs
from unicodedata import normalize
from functools import partial


#################
#
# id to rating
#
#################
sentence_rr = []
sentence_id = dict()
sentence_file = "./root/datasetSentences.txt"
f = open(sentence_file, "r")
head = True
for x in f:
	if head:
		head = False
		continue
	s_p = " ".join([t.encode('latin1').decode('utf-8').lower() for t in x.split("\t")[-1].strip().split(" ")])
	sentence_rr.append(s_p)
	# this is to prevent collisions
	if s_p in sentence_id.keys():
		sentence_id[s_p].append(int(x.split("\t")[0]))
	else:
		sentence_id[s_p] = [int(x.split("\t")[0])]
print("Finish loading sentences ...")
# Load phrase mapping
phrase = dict()
phrase_file = "./root/dictionary.txt"
f = open(phrase_file, "r")
for x in f:
	seq = " ".join([t.lower() for t in x.split("|")[0].strip().split(" ")])
	phrase_id = int(x.split("|")[1])
	phrase[seq] = phrase_id
# Load phrase rate mapping
phrase_rate = dict()
phrase_rate_file = "./root/sentiment_labels.txt"
f = open(phrase_rate_file, "r")
head = True
for x in f:
	if head:
		head = False
		continue
	p_id = int(x.split("|")[0])
	rr = float(x.split("|")[1])
	phrase_rate[p_id] = rr
# Loading sentence to phrase id mapping
sentence_rate = dict()
count = 0
miss_count = 0

# have a separate count matrics to avoid collisions
sentence_count = dict()
for seq in sentence_rr:
	sentence_count[seq] = 0

for seq in sentence_rr:
	if seq not in phrase.keys():
		# it menas it contain tokenized parenthese
		seq_l = re.sub("-lrb-", "(", seq)
		seq_lr = re.sub("-rrb-", ")", seq_l)
		sentence_rate[sentence_id[seq][0]] = phrase_rate[phrase[seq_lr]] # no collision
		if seq_lr not in phrase.keys():
			assert(False)
	else:
		sentence_rate[sentence_id[seq][sentence_count[seq]]] = phrase_rate[phrase[seq]]
		sentence_count[seq] = sentence_count[seq] + 1

#################
#
# id to partition
#
#################
# Load partition info
partition = dict()
partition['Train'] = []
partition['Test'] = []
partition['Valid'] = []
partition_file = "./root/datasetSplit.txt"
f = open(partition_file, "r")
head = True
for x in f:
	if head:
		head = False
		continue
	p = int(x.split(",")[1])
	fi = int(x.split(",")[0])
	if p == 1:
		partition['Train'].append(fi)
	elif p == 2:
		partition['Test'].append(fi)
	elif p == 3:
		partition['Valid'].append(fi)
print("Finish loading partition ...")


#################
#
# id to embeddings
#
#################
# Load tokens
token_file = "./glove/dict.txt"
token_list = []
f = open(token_file, "r")
for x in f:
	token_list += [x.strip()]
print("Finish loading tokens ...")
# Load embeddings for those tokens
glove_file = "./glove/sentiment_glove_300.txt"
token_glove = dict()
f = open(glove_file, "r")
index = 0
for x in f:
	token_glove[token_list[index]] = \
		[float(dim) for dim in x.split(" ")]
	index += 1
print("Finish loading glove dictionary ...")
# Load sentences
sentence = dict()
sentence_file = "./root/datasetSentences.txt"
f = open(sentence_file, "r")
head = True
for x in f:
	if head:
		head = False
		continue
	sentence[int(x.split("\t")[0])] = [t.encode('latin1').decode('utf-8').lower() for t in x.split("\t")[-1].strip().split(" ")]
print("Finish loading sentences ...")
# embedding and sentence
na_vec = 300 * [0.0]
header = ['word']
header.extend(['glove'+str(i) for i in range(300)])
sentence_id_embed = dict()
for seq in sentence.keys():
	# print("Creating feature files for sentence id: " + str(seq) + " ...")
	sentence_id_embed[seq] = []
	for token in sentence[seq]:
		if token not in token_list:
			# we should not have any words in here
			assert(True)
		else:
			sentence_id_embed[seq].append(token_glove[token])


#################
#
# save
#
#################
import pickle
# id to sentence
pickle.dump( sentence, open("id_sentence.p", "wb") )

# partition - rating and embedding
train_final = []
val_final = []
test_final = []
for sss in sentence_rate.keys():
	if sss in partition['Train']:
		train_final.append(sss)
	elif sss in partition['Valid']:
		val_final.append(sss)
	elif sss in partition['Test']:
		test_final.append(sss)

train_embed = dict()
train_rr = dict()
for sss in train_final:
	train_embed[sss] = sentence_id_embed[sss]
	train_rr[sss] = sentence_rate[sss]
pickle.dump( train_embed, open("id_embed_train.p", "wb") )
pickle.dump( train_rr, open("id_rating_train.p", "wb") )

valid_embed = dict()
valid_rr = dict()
for sss in val_final:
	valid_embed[sss] = sentence_id_embed[sss]
	valid_rr[sss] = sentence_rate[sss]
pickle.dump( valid_embed, open("id_embed_valid.p", "wb") )
pickle.dump( valid_rr, open("id_rating_valid.p", "wb") )

test_embed = dict()
test_rr = dict()
for sss in test_final:
	test_embed[sss] = sentence_id_embed[sss]
	test_rr[sss] = sentence_rate[sss]
pickle.dump( test_embed, open("id_embed_test.p", "wb") )
pickle.dump( test_rr, open("id_rating_test.p", "wb") )



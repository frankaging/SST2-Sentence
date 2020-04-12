# Ready-to-use parsed sentence level dataset of Stanford Sentiment Treebank Dataset

This folder contains raw SST-2 data as well as our parsed script. And we also provided ready-to-use pickle files that directly loads everything for you for your own training purposes. If you consider us this folder, please cite the original paper.

## Data Partition

This folder contains original partitioned datasets, Train Set (8190 sentences), Valid Set (2131 sentences) and Test Set. Each sentence will have labels indicating its valence rating. Note that you can recover the 5 classes by mapping the positivity probability using the following cut-offs: [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]. You can change it to binary label as well by ignoring the neutral sentences.

## Format of Each File

*id_embed_[Specific Set]*: map between the original sentence id and embeddings. {'id' : [[glove300], [glove300], ..., [glove300]]}. Each word is embedded using standard GloVe 300 vector training by wikipedia. The raw embeddings are attached in the Glove folder as well.

*id_rating_[Specific Set]*: map between the original sentence id and its rating. {'id': float}.

*id_sentence_[Specific Set]*: map between the original sentence id and the original sentece in text encoded in utf-8. {'id': string}.

## Simple Load Using Pickle
You can directly load all those files by using pickle module avaliable in python.

**import pickle**

**id_embed_train = pickle.load( open( "id_embed_train.p", "rb" ) )**

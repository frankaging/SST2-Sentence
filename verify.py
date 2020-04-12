import pickle
id_embed_train = pickle.load( open( "id_embed_train.p", "rb" ) )
# print(id_embed_train.keys()) # 11853, 11854, 11855
print(len(id_embed_train[11855]))
print(len(id_embed_train[11853]))

id_sentence = pickle.load( open( "id_sentence.p", "rb" ) )
print(id_sentence[11855])
print(id_sentence[11854])


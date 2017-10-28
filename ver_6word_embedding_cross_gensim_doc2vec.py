import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pickle
import os
from collections import namedtuple
import numpy as np
import random

current_path = os.getcwd()
os.chdir(current_path + "\\transcript_file")
cross_val = 6

#emotionList = ['anger', 'disgust', 'joy', 'shame', 'fear', 'sadness', 'guilt']
#emotionList = ["anger","boredom","disgust","empty","enthusiasm","fear","fun","guilt","happiness","hate","joy","love","neutral","relief","sadness","surprise","worry"]
emotionList = ["happiness", "joy"]
#f = open("total_transcript_twitter_ISEAR.txt", mode="r", encoding="utf-8")
fp = open("combined_be_final.pkl", "rb")
#flabel = open("total_emotion.txt", mode="r", encoding="utf-8")

# Labeled Sentence가 필요
combine = pickle.load(fp)
mixing = []
for key in combine:
    tmp = combine[key]
    for i in range(0,len(tmp)):
        mixing.append((tmp[i],key))

random.shuffle(mixing)
item = []

for i in range(0,len(mixing)):
    item.append(mixing[i][1])

# Hyperparameter
min_count = 0
size = 50
window = 4
sample = 1e-4
alpha = 0.025
min_alpha = 0.025
words, sentences = [], []
count = 0
rate = 0.7

result_base_path = current_path + "\\gensim_result"
os.chdir(result_base_path)
fw = open("doc2vec.txt", mode="w", encoding="utf-8")

docs = []
analyzedDocument = namedtuple("AnalyzedDocument", "words tags")
raw = []
j = 0
for i in range(0, len(item)):
    words = " ".join(item[i]).split()
    tag = [j]
    docs.append(analyzedDocument(words, tag))
    j += 1

model = doc2vec.Doc2Vec(docs, size=size, window=window, min_count=min_count, workers=4)
for i in range(0, len(model.docvecs)):
    fw.write(str(model.docvecs[i]) + "\n")




convert_to_numpy = np.asarray(model.docvecs)
length = int(convert_to_numpy.shape[0]/6)
index = 0

os.chdir(result_base_path+"\\CrossValidation")
for i in range(0,6):
    transcript = convert_to_numpy[index:index+length,]
    np.save("transcriptIndex%d"%i,transcript)
    index = index + length




######################Convert the emotion Index
indexEmotion = np.zeros([convert_to_numpy.shape[0], len(emotionList)], dtype=np.int32)
i = 0
tIndex = 0

for i in range(0,len(mixing)):
    indexList = []
    if mixing[i][1] in emotionList:
        tIndex = emotionList.index(mixing[i][1])
    for k in range(0,tIndex):
        indexList.append("0")
    indexList.append("1")
    if tIndex<len(emotionList):
        while tIndex != len(emotionList)-1:
            indexList.append("0")
            tIndex += 1
    indexEmotion[i] = np.array(indexList)




index = 0
for i in range(0,cross_val):
    label = indexEmotion[index:index+length,]
    np.save("emotionIndex%d"%i,label)
    index = index + length




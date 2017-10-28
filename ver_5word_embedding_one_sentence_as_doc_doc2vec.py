import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pickle
import os
from collections import namedtuple
import random

#emotionList = ['anger', 'disgust', 'joy', 'shame', 'fear', 'sadness', 'guilt']
emotionList = ['relief','shame','hate','fun','disgust','anger']
#emotionList = ['happiness','worry']
#emotionList = ["anger","boredom","disgust","empty","enthusiasm","fear","fun","guilt","happiness","hate","joy","love","neutral","relief","sadness","surprise","worry"]
current_path = os.getcwd()
os.chdir(current_path+"\\transcript_file")


fp = open("combined_be_final.pkl","rb")
breakpoint = pickle.load(fp)

#######################
middle_dict = {}
for k in breakpoint:
    if k in emotionList:
        tmp = breakpoint[k]
        middle_dict[k] = tmp
########################

mixing = []
print(os.getcwd())
for key in breakpoint:
    if key == "joy":
        key = "happiness"
    if key in emotionList:
        tmp = breakpoint[key]
        f = open(key + '.txt', mode='w', encoding="utf-8")
        if key == "anger":
            anger = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "disgust":
            digust = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "happiness":
            hapiness = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "fun":
            fun = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "neutral":
            neutral = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "hate":
            hate = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "relief":
            relief = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "sadness":
            sadness = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        elif key == "shame":
            shame = tmp
            #for i in range(1, len(tmp)):
            #    f.write(" ".join(tmp[i]) + '\n')
        for i in range(0,len(tmp)):
            mixing.append((tmp[i],key))
    else:
        pass

item = []
random.shuffle(mixing)
for i in range(0,len(mixing)):
    item.append(mixing[i][1])
print(len(mixing))
result_base_path = current_path+"\\gensim_result"
os.chdir(result_base_path)

#Hyperparameter
min_count = 0
size = 50
window = 4
sample = 1e-4
alpha = 0.025
min_alpha = 0.025
words, sentences = [], []
count = 0


os.chdir(current_path)

docs = []
j=0
analyzedDocument = namedtuple("AnalyzedDocument","words tags")
raw = []
item = []

for k in range(0,len(mixing)):
    item.append(mixing[k][0])


'''
training, test = [], []

for_test = int(len(mixing) *0.8)
training = item[0:for_test]
test = item[for_test:]
'''
for i in range(0,len(item)):
    words = " ".join(item[i]).split()
    tag = [i]
    docs.append(analyzedDocument(words,tag))
    i += 1


model = doc2vec.Doc2Vec(docs, size=size, window=window, min_count=min_count, workers=4)

word_vectors = model.wv
print(word_vectors.similar_by_word("happy"))
print(word_vectors.similarity("woman","man"))

'''
for i in range(0,len(test)):
    list = test[i][0].split()
    for element in list:
        try:
            print(element + " " +str(word_vectors.similar_by_word(element))+"\n")
        except KeyError:
            print("KeyError")
'''

convert_to_numpy = np.asarray(model.docvecs)
seperate_key = int(convert_to_numpy.shape[0] * 0.8)

trainTranscript = convert_to_numpy[:seperate_key,]
testTranscript = convert_to_numpy[seperate_key:,]

os.chdir(result_base_path+"\\Training")
np.save("transcriptIndex.npy",trainTranscript)
os.chdir(result_base_path+"\\Test")
np.save("transcriptIndex.npy",testTranscript)


'''
#Not one-hot
indexEmotion = np.zeros([convert_to_numpy.shape[0], 1], dtype=np.int32)
i = 0
seperate_key = int(indexEmotion.shape[0]*0.8)

for i in range(0,len(mixing)):
    label = mixing[i][1]
    indexEmotion[i] = emotionList.index(label)


'''
#One-hot
tmpIndex = 0
indexEmotion = np.zeros([convert_to_numpy.shape[0], len(emotionList)], dtype=np.int32)
for i in range(0,len(mixing)):
    indexList = []
    if mixing[i][1] in emotionList:
        tmpIndex = emotionList.index(mixing[i][1])
    for k in range(0,tmpIndex):
        indexList.append("0")
    indexList.append("1")
    if tmpIndex<len(emotionList):
        while tmpIndex !=  len(emotionList)-1:
            indexList.append("0")
            tmpIndex += 1
    indexEmotion[i] = np.array(indexList)


trainLabel = indexEmotion[:seperate_key,]
testLabel = indexEmotion[seperate_key:,]

os.chdir(result_base_path + "\\Training")
np.save("emotionIndex.npy", trainLabel)
os.chdir(result_base_path + "\\Test")
np.save("emotionIndex.npy", testLabel)
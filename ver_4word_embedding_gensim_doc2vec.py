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
os.chdir(current_path+"\\transcript_file")

emotionList = ['anger', 'disgust', 'joy', 'shame', 'fear', 'sadness', 'guilt']

f = open("total_transcript_twitter_ISEAR.txt",mode="r",encoding="utf-8")
fp = open("combined.pkl","rb")
flabel = open("total_emotion.txt",mode="r",encoding="utf-8")

#Labeled Sentence가 필요
combine = pickle.load(fp)
tuple_list = []
for key in combine:
    tuple_list.append(combine[key],key)
random.shuffle(tuple_list)


#Hyperparameter
min_count = 0
size = 50
window = 4
sample = 1e-4
alpha = 0.025
min_alpha = 0.025
words, sentences = [], []
count = 0
rate = 0.7

result_base_path = current_path+"\\gensim_result"
os.chdir(result_base_path)
fw = open("doc2vec.txt",mode="w",encoding="utf-8")


docs = []
analyzedDocument = namedtuple("AnalyzedDocument","words tags")
raw = []
j=0
for key in combine:
    item = combine[key]
    raw += item
    for i in range(0,len(item)):
        words = " ".join(item[i]).split()
        tag = [j]
        docs.append(analyzedDocument(words,tag))
        j += 1

model = doc2vec.Doc2Vec(docs, size=size, window=window, min_count=min_count, workers=4)
for i in range(0,len(model.docvecs)):
    fw.write(str(model.docvecs[i])+"\n")





convert_to_numpy = np.asarray(model.docvecs)
seperate_key = int(convert_to_numpy.shape[0] * 0.7)

trainTranscript = convert_to_numpy[:seperate_key,]
testTranscript = convert_to_numpy[seperate_key:,]

os.chdir(result_base_path+"\\Training")
np.save("transcriptIndex.npy",trainTranscript)
os.chdir(result_base_path+"\\Test")
np.save("transcriptIndex.npy",testTranscript)




######################Convert the emotion Index
indexEmotion = np.zeros([convert_to_numpy.shape[0], 1], dtype=np.int32)
i = 0
seperate_key = int(indexEmotion.shape[0]*0.7)

while True:
    label = flabel.readline().strip()
    indexList = []
    if not label: break
    if label in emotionList:
        tmpIndex = emotionList.index(label)
    indexEmotion[i] = tmpIndex
    '''
    for k in range(0,tmpIndex):
        indexList.append("0")
    indexList.append("1")
    if tmpIndex < 7:
        while tmpIndex != 6:
            indexList.append("0")
            tmpIndex += 1
    indexEmotion[i] = np.array(indexList)
    '''
    i += 1



trainLabel = indexEmotion[:seperate_key,]
testLabel = indexEmotion[seperate_key:,]

os.chdir(result_base_path + "\\Training")
np.save("emotionIndex.npy", trainTranscript)
os.chdir(result_base_path + "\\Test")
np.save("emotionIndex.npy", testLabel)

#np.save("emotionIndex.npy",indexEmotion)







'''
document = []
documents = {}
for key in combine:
    documents[key+".txt"]=key

    sentence = combine[key]
    for i in range(0,len(combine[key])):
        document.append(TaggedDocument(" ".join(sentence[i]).split(),key))

########################
#model = doc2vec(size=size, window= window, min_count=min_count, workers =4)
model = doc2vec(alpha=alpha, min_alpha=min_alpha)
model.build_vocab(document)
model.train(document)
#########################

input = TaggedDocument(documents)

models = doc2vec(size=size, window= window, min_count=min_count, workers =4)
#models = doc2vec(alpha=alpha, min_alpha=min_alpha)
#models = doc2vec(documents)
models.build_vocab(input.to_array())
models.train(input.sentence_perm())

print(".")



while True:
    line = f.readline().strip()
    if len(line.split()) < 50:
        words.append(line.split())
    sentences.append(line)
    if not line : break

result = np.zeros((len(words),50))
model = gensim.models.Doc2Vec(words, min_count=min_count, size=size, window = window)

for i in range(0,len(words)):
    try:
        result[i,0:len(words[i])] = model[words[i]].transpose()[0]
    except ValueError:
        print(str(i))
        print(str(words[i]))

for i in range(0,len(words)):
    fw.write(str(result[i])+'\n')

pickle.dump(result,open("pickle_result_word_embedding_gensim.pkl","wb"))
'''
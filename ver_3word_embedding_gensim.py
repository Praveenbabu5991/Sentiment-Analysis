import gensim
import numpy as np
import pickle
import os

current_path = os.getcwd()
os.chdir(current_path+"\\transcript_file")

f = open("total_transcript.txt",mode="r",encoding="utf-8")
min_count = 0
size = 20
window = 4
words, sentences = [], []
count = 0
os.chdir(current_path)
fw = open("result_word_embedding_gensim.txt",mode="w",encoding="utf-8")


while True:
    line = f.readline().strip()
    if len(line.split()) < 50:
        words.append(line.split())
    sentences.append(line)
    if not line : break

result = np.zeros((len(words),50))
model = gensim.models.Word2Vec(words, min_count=min_count, size=size, window = window)

for i in range(0,len(words)):
    try:
        result[i,0:len(words[i])] = model[words[i]].transpose()[0]
    except ValueError:
        print(str(i))
        print(str(words[i]))

for i in range(0,len(words)):
    fw.write(str(result[i])+'\n')

pickle.dump(result,open("pickle_result_word_embedding_gensim.pkl","wb"))



print("/")
'''
for i in range(0,len(sentences)):
    try:
        fw.write(sentences[i]+"\n")
        for j in range(0,len(sentences[i])):
            fw.write(str(model[sentences[i]])+"\n")
    except KeyError:
        print("Error")
'''
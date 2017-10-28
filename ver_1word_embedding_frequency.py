'''
Word to Vector Representation Code
Based on the emotion classes, Each sentences that belongs to the emotion class
'''

from collections import Counter
import itertools
import numpy as np
import pickle
import os
import random
limit = { "maxt" : 50  }
#emotionList = ["neu" , "hap", "fru", "ang", "sur" , "fea", "exc", "xxx", "dis", "sad", "oth"]
UNK = "unk"
cross_val = 6

VOCAB_SIZE = 8000

# Extract the emotion classes list
f = open("stop_rm_Dict.txt", mode="r")
emotionList = f.readline()
emotionList = emotionList.replace("\\","")
emotionList = emotionList.replace("\'","")
emotionList = emotionList.replace("[","")
emotionList = emotionList.replace("]","")
emotionList = emotionList.replace(",","")
emotionList = emotionList.split()

#f = open("EmotionCollection.txt", mode="r",encoding = "utf-8")
#femotion =  f.readline()
#ft = open("emotion_transcript.txt",mode ="r",encoding="utf-8")
#ftemotion = ft.readline()

mappingDict = {}
twmappingDict = {}
transcript = []
key = ""

while True:
    line = f.readline().strip()
    if not line:
        mappingDict[priorKey] = transcript
        break
    if line.split()[0] == "Emotion":
        key = line.split()[2]
        if transcript != []:
            mappingDict[priorKey] = transcript
            transcript = []
    else:
        transcript.append(line)
        priorKey = key

'''
transcript = []
priorKey = ""
key = ""
twkey = ""
while True:
    twline = ft.readline().strip()
    twline = twline.replace("[","")
    twline = twline.replace("]","")
    if not twline:
        mappingDict[priorKey] = transcript
        break
    if twline.split()[0] == "Emotion":
        twkey = twline.split()[2]
        if transcript !=[]:
            mappingDict[priorKey] = transcript
            transcript = []
    else:
        if twline in transcript: pass
        else:
            transcript.append(twline)
        priorKey = twkey
'''

global_index = 0
dataset = []
set = {}
for i in range(0,cross_val):
    for k in mappingDict:
        length = int(len(mappingDict[k]) / cross_val)
        if len(mappingDict[k]) == 1:
            set[k] = mappingDict[k]
        else:
            set[k] = mappingDict[k][global_index:global_index+int(length)]
    dataset.append(set)
    set = {}
    global_index += length


def tweet_refine(word):
    word = word.lower()
    word = word.replace(" ","")
    word = word.replace(",","")
    word = word.replace("\\", "")
    word = word.replace("?", "")
    word = word.replace("!", "")
    word = word.replace(".","")
    word = word.replace("-", "")
    word = word.replace(":", "")
    word = word.replace("#","")
    word = word.replace("&", "")
    word = word.replace(".","")
    word = word.replace("\'", "")
    word = word.replace("\"", "")
    return word

def all_counter_unigram(dict_):
    all_transcript = []
    all_count = 0
    tmp3 = []
    for key in dict_:
        tmp = dict_[key]
        tmp.pop(0)
        for element in tmp:
            tmp2 = element.split()
            #Refining the word of each element
            for e in tmp2:
                refine_e = tweet_refine(e)
                tmp3.append(refine_e)
            all_transcript.append(tmp3)
            tmp3,tmp2 = [],[]
    all_counter = Counter(itertools.chain(*all_transcript))
    for element in all_counter:
        all_count += all_counter[element]
    for element in all_counter:
        all_counter[element] /= all_count
    return all_counter

def all_counter_bigram(dict_):
    all_transcript = []
    all_count = 0
    tmp3 = []
    for key in dict_:
        tmp = dict_[key]
        tmp.pop(0)
        for element in tmp:
            tmp2 = element.split()
            for i in range(0,len(tmp2)-1):
                first_word = tweet_refine(tmp2[i])
                second_word = tweet_refine(tmp2[i+1])
                if first_word=="" or second_word =="": pass
                else:
                    combined_word = first_word +" " + second_word
                    tmp3.append(combined_word)
            all_transcript.append(tmp3)
    all_counter = Counter(itertools.chain(*all_transcript))
    for element in all_counter:
        all_count += all_counter[element]
    for element in all_counter:
        all_counter[element] /= all_count
    return all_counter


dict_unigram = all_counter_unigram(mappingDict)
dict_bigram = all_counter_bigram(mappingDict)


def refining(word):
    word = word.lower()
    word = word.replace(",","")
    word = word.replace("\\", "")
    word = word.replace("?", "")
    word = word.replace("!", "")
    word = word.replace(".","")
    word = word.replace("-", "")
    return word

def emotionList_return():
    global emotionList
    print(str(emotionList))
    return emotionList


# Extract the emotion dictionary based on the frequency.
def index_emotion_(dictionary, vocabSize):
    prob_dict, probDict = {}, {}
    word_list = []
    tmp_list = []
    seperate_list = []
    total_count, probCount = 0, 0
    for key in dictionary:
        list = dictionary[key]
        for e in list:
            word_list.append(e.split())
        wordDist = Counter(itertools.chain(*word_list))
        #wordDist = nltk.FreqDist(itertools.chain(*word_list))
        for e in list:
            rawData = e.split()
            for i in range(0,len(rawData)-1):
                rawData[i] = tweet_refine(rawData[i])
                rawData[i+1] = tweet_refine(rawData[i+1])
                if rawData[i] == "" or rawData[i+1] == "": pass
                else:
                    word = rawData[i] + " " +rawData[i+1]
                    seperate_list.append(word)
            tmp_list.append(seperate_list)
            seperate_list = []
        probDist = Counter(itertools.chain(*tmp_list))
        #probDist = nltk.FreqDist(itertools.chain(*tmp_list))
        probVocab = probDist.most_common(vocabSize)

        for probItem in probVocab:
            if probItem[0].split()[0] in wordDist:
                probCount = wordDist[probItem[0].split()[0]]
            else:
                probCount = 10000
            try:
                probDict[probItem[0]] = (float(probItem[1])/float(probCount))
            except ZeroDivisionError:
                probDict[probItem[0]] = (float(probItem[1])/float(1.0))

            if probDict[probItem[0]] > 1: probDict[probItem[0]] = 1

        tmp_list = []
    probDict["UNK"] = 0.000000001
    probDict["_"] = 0.000000001
    return probDict

def index_(tokenizedSentence,vocabSize):
    total_count = 0
    #freqDist = nltk.FreqDist(itertools.chain(*tokenizedSentence))
    freqDist = Counter(itertools.chain(*tokenizedSentence))
    vocab = freqDist.most_common(vocabSize)
    '''
    for key in freqDist:
        total_count += freqDist[key]

    for key in freqDist:
        freqDist[key] = freqDist[key]/total_count

    for key in dictionary:
        list = dictionary[key]
        for e in list:
            rawData = e.split()
            for i in range(0,len(rawData)-1):
                word = rawData[i] + " " +rawData[i+1]
                seperate_list.append(word)
            tmp_list.append(seperate_list)
        probDist = nltk.FreqDist(itertools.chain(*tmp_list))
        probVocab = probDist.most_common(vocabSize)
        for probItem in probVocab:
            probCount += probItem[1]
        for probItem in probVocab:
            probDict[probItem[0]] = float(probItem[1])/float(probCount)
        probCount = 0
        seperate_list = []
        tmp_list = []

    --------------------------
    for item in vocab:
        total_count  += item[1]
    for item in vocab:
        prob_dict[item[0]] = float(item[1]/total_count)

    prob_dict["_"] = 0.11
    prob_dict["UNK"] = 0.11
    '''
    index2word = ['_'] + ['UNK'] + [x[0] for x in vocab]
    word2index = dict([(w,i) for i,w in enumerate(index2word)])
    return index2word, word2index, freqDist


def emotionIndex_(emotionTokenized):
    dataLength = len(emotionTokenized)
    indexList = []
    indexEmotion = np.zeros([dataLength,len(emotionList)], dtype = np.int32)
    tempIndex = 0

    for e in range(0, len(emotionTokenized)):
        #for k in range(0,len(emotionList)):
        if emotionTokenized[e] in emotionList:
            tempIndex = emotionList.index(emotionTokenized[e])
                #indexList.append("1")
            #else:
            #    indexList.append("0")
        for k in range(0,tempIndex):
            indexList.append("0")
        indexList.append("1")
        if tempIndex < 7 :
            while tempIndex != 6:
                indexList.append("0")
                tempIndex += 1


        indexEmotion[e] = np.array(indexList)
        indexList = []
    return indexEmotion

def rawEmotion(emotionTokenized):
    return emotionTokenized

def zeroPadding(trasnTokenized, dict):
    dataLength = len(trasnTokenized)
    indexText = np.zeros([dataLength,limit['maxt']], dtype = np.float32)

    for i in range(dataLength):
        textIndices = padSeq(trasnTokenized[i],dict,limit['maxt'])
        if len(textIndices) > 50: pass
        else:
            indexText[i] = np.array(textIndices)
    return indexText

def padSeq(sequence,lookup,maxLength):
    indices = []
    for i in range(0,len(sequence)):
        if sequence[i] in lookup:
            indices.append(lookup[sequence[i]])
        else:
            try:
                indices.append(lookup['UNK'])
            except KeyError:
                print(str(sequence))
    return indices + [0]*(maxLength-len(sequence))

def processData(dictionary,crossIndex):
    emotionToken = []
    transcriptToken = []
    mix_transciptToken = []
    mix_emotion= []
    temp_list = []
    for_append = ""
    frawdata = open("rawData%d.txt"%crossIndex,"w")
    mixing = []

    for k in dictionary:
        for item in dictionary[k]:
            item_token = item.split()
            for i in range(0,len(item_token)-1):
                prior = for_append
                item_token[i] = tweet_refine(item_token[i])
                item_token[i+1] = tweet_refine(item_token[i+1])
                if item_token[i] == "" or item_token[i+1] == "": pass
                else:
                    for_append = item_token[i] + " " + item_token[i+1]
                    if prior == for_append : pass
                    else:
                        temp_list.append(for_append)
            #temp_list.append(k)
            mixing.append((temp_list,k))
            #transcriptToken.append(temp_list)
            temp_list = []
            #emotionToken.append(k)
    random.shuffle(mixing)
    for i in range(0,len(mixing)):
        transcriptToken.append(mixing[i][0])
        emotionToken.append(mixing[i][1])
    index2word, word2Index, freqDist = index_(transcriptToken, vocabSize = VOCAB_SIZE)
    print("index2word of " + str(index2word)+"\n")
    print("word2Index of " + str(word2Index)+"\n")
    print("freqDist of " + str(freqDist))
    with open("freqDist%d.pkl"%crossIndex,"wb") as f:
        pickle.dump(freqDist, f)

    prob_dictionary = index_emotion_(dictionary, vocabSize=VOCAB_SIZE)
    #Represent in One vector, but how can I revise?

    print("Zero Padding\n")
    transcriptIndex = zeroPadding(transcriptToken, dict_bigram)

    print("Each emotion of the transcript\n")
    indexEmotion = emotionIndex_(emotionToken)


    ###########The printing transcript data#################
    for i in range(0,len(transcriptToken)):
        if len(transcriptToken[i]) == 0: pass
        else:
            frawdata.write("Emotion = " + emotionToken[i] + " ")
            write_token = ""
            for k in range(0,len(transcriptToken[i])):
                if k != len(transcriptToken[i])-1:
                    try:
                        write_token = write_token + str(" "+ transcriptToken[i][k].split()[0])
                    except IndexError:
                        print(str(transcriptToken[i]))
                else:
                    write_token = write_token + " " + str(transcriptToken[i][k])
            frawdata.write(write_token + "\n")
            frawdata.write(str(transcriptIndex[i])+'\n')
            frawdata.write(str(indexEmotion[i])+'\n')



    #Save numpy file to the disk
    np.save("transcriptIndex%d.npy"%crossIndex, transcriptIndex)
    np.save("emotionIndex%d.npy"%crossIndex, indexEmotion)

    metadata = {
                'word2Index' : word2Index,
                'index2word' : index2word,
                'limit' : limit,
                'freqDist' : freqDist
                }

    with open('metadata%d.pkl'%crossIndex, "wb")as f:
        pickle.dump(metadata,f)

if __name__ == "__main__":
    currentPath = os.getcwd()
    os.chdir(currentPath+"\\CrossValidation")
    for i in range(0,cross_val):
        processData(dataset[i], i)
    '''
    os.chdir(currentPath + "\\Training")
    processData(trainSet)
    os.chdir(currentPath  + "\\Test")
    processData(testSet)
    os.chdir(currentPath + "\\Validation")
    processData(validationSet)
    '''




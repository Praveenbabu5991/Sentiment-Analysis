import os
import numpy as np
from sklearn.svm import SVC

emotionList = ["anger","boredom","disgust","empty","enthusiasm","fear","fun","guilt","happiness","hate","joy","love","neutral","relief","sadness","surprise","worry"]
cross_val = 6
currentPath = os.getcwd() + "\\gensim_result"
os.chdir(currentPath + "\\Training")
x = np.load("transcriptIndex.npy")
y = np.load("emotionIndex.npy")

os.chdir(currentPath+"\\Test")
test_x = np.load("transcriptIndex.npy")
test_y = np.load("emotionIndex.npy")

clf = SVC()
clf.fit(x,y)
pred_y = clf.predict(test_x)
count = 0
for i in range(0,test_y.shape[0]):
    if test_y[i] == pred_y[i]:
        count += 1
    else:
        pass
print(str(count)+'\n')
print(str(test_y.shape[0]))




import numpy as np
import os

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit

#emotionList = ['anger', 'disgust', 'joy', 'shame', 'fear', 'sadness', 'guilt']
#emotionList = ["hapiness","worry"]
emotionList = ["anger","boredom","disgust","empty","enthusiasm","fear","fun","guilt","happiness","hate","joy","love","neutral","relief","sadness","surprise","worry"]

cross_val = 6
currentPath = os.getcwd() + "\\gensim_result"
os.chdir(currentPath + "\\Training")
x = np.load("transcriptIndex.npy")
y = np.load("emotionIndex.npy")

os.chdir(currentPath+"\\Test")
test_x = np.load("transcriptIndex.npy")
test_y = np.load("emotionIndex.npy")

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1)
lr_cv = LogisticRegressionCV(
    solver='newton-cg'
    ,multi_class='multinomial'
    ,cv=cv
)

count = 0

trained_lr_cv = lr_cv.fit(x, y)
pred_y = trained_lr_cv.predict(test_x)
for i in range(0,test_y.shape[0]):
    if test_y[i] == pred_y[i]:
        count += 1
    else:
        pass
print(str(count)+'\n')
print(str(test_y.shape[0]))



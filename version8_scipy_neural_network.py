from sklearn.neural_network import MLPClassifier
import os
import numpy as np

emotionList  = ["happiness","worry"]
count = 0
currentPath = os.getcwd()+"\\gensim_result"
os.chdir(currentPath  + "\\Training")
Xtraining = np.load("transcriptIndex.npy")
Ytraining = np.load("emotionIndex.npy")

os.chdir(currentPath  + "\\Test")
Xtest = np.load("transcriptIndex.npy")
Ytest = np.load("emotionIndex.npy")


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(len(Xtraining[0]), 2), random_state=1)
clf.fit(Xtraining,Ytraining)

result = clf.predict(Xtest)
print(str(result))


#Layer normalization with dropout
#5-cross-validation

import tensorflow as tf
import numpy as np
import os
#from without_nltk import emotionList_return
from tensorflow.python.framework import dtypes

emotionList = ['anger', 'disgust', 'joy', 'shame', 'fear', 'sadness', 'guilt']
cross_val = 6

count = 0
currentPath = os.getcwd()
os.chdir(currentPath+"\\CrossValidation")
transcript = []
emotion = []
for i in range(0,cross_val):
    transcript.append(np.load("transcriptIndex%d.npy"%i))
    emotion.append(np.load("emotionIndex%d.npy"%i))


####HyperParameter
num_classes = len(emotionList)
input_dimension = 50    #len(Xtraining[0])
hidden_size = len(emotionList)
batch_size = 50
line_length = 50        #len(Xtraining[0])
sequence_length = 50    #len(Xtraining[0])
learning_rate = 0.001
num_layer = 4
dropout = tf.constant(1.0)

num_epoch = 100
num_batch = (len(transcript[0])+len(transcript[1])+len(transcript[2])+len(transcript[3])+len(transcript[4]))/batch_size

def generate_batch(batch_size, train, label):
    data_index = 0
    batch, labels = [] , []
    iterate_num = int(len(train)/batch_size)
    for i in range(iterate_num):
        batch.append(train[data_index:data_index+batch_size])
        labels.append(label[data_index:data_index+batch_size])
        data_index = (data_index + batch_size)
    return batch, labels


def training(Xt, Yt, Xte, Yte,index):
    '''
    all_training = tf.convert_to_tensor(Xt,dtype = dtypes.float32)
    all_label = tf.convert_to_tensor(Yt, dtype = dtypes.int32)
    all_test = tf.convert_to_tensor(Xte,dtype = dtypes.float32)
    all_test_label = tf.convert_to_tensor(Yte, dtype = dtypes.int32)

    train_input_queue = tf.train.slice_input_producer([Xt, Yt], shuffle=False)
    test_input_queue = tf.train.slice_input_producer([Xte,Yte], shuffle=False)

    train_transcript = train_input_queue[0]
    train_label = train_input_queue[1]
    test_trancript = test_input_queue[0]
    test_label = test_input_queue[1]


    train_transcript_batch, train_label_batch = tf.train.batch([train_transcript,train_label], batch_size = batch_size)
    #print("This is the batch of x" +str(train_transcript_batch))
    test_transcript_batch, test_label_batch = tf.train.batch([test_trancript,test_label],batch_size = batch_size)
    #또 내가 궁금한건, weight variable을 내가 직접 초기화 안해도 잘 작동하는건가?
    inputs = []

    for i in range(0,num_epoch):
        inputs.append(tf.placeholder(tf.float32,shape = [batch_size,line_length,line_length]))
    '''
    X = tf.placeholder(tf.float32,shape = [None,line_length,input_dimension])
    Y = tf.placeholder(tf.int32,shape = [None,line_length,num_classes])

    #Generate batch for training and test
    batch_, label_= generate_batch(batch_size, Xt, Yt)
    test_batch_, test_label_ = generate_batch(batch_size,Xte,Yte)
    print("the training Batch for %d th iteration is "%index, len(batch_))
    print("the test Batch for %d th iteration is "%index, len(test_batch_))

    '''
    finput = open("input.txt","w")
    for i in range(0,len(batch_)):
        finput.write(str(batch_[i])+"\n")

    output_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_classes)
    '''
    cells = []
    with tf.variable_scope("RNN%d"%index):
        for _ in range(num_layer):
            cell = tf.nn.rnn_cell.BasicRNNCell(num_units = sequence_length,
                                               reuse=tf.get_variable_scope().reuse)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)


        output, _state = tf.nn.dynamic_rnn(cell,X,dtype = tf.float32)
        #print(str(tf.shape(Y)))
        #print(str(output_shape))

        ###Softmax Fully-connected
        X_for_softmax = tf.reshape(output,[-1,sequence_length])
        output = tf.contrib.layers.fully_connected(X_for_softmax,num_classes,activation_fn=None)
        output = tf.reshape(output, [-1, sequence_length, num_classes])

        sequence_loss = tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = Y)
        loss = tf.reduce_mean(sequence_loss)
        train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

        #결과값중 가장 큰 값을 1로 설정하는 함수가 argmax인 것.
        prediction = tf.argmax(output,axis=2)
        correct_prediction = tf.equal(tf.argmax(output,axis=2), tf.argmax(Y,axis=2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #Check with the combined batch is needed.....헷갈려


    training_total_=[]
    avg_cost = 0
    ftrain = open("without_cross_validation.txt","w")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord= coord)
        print("from the train set")
        for epoch in range(num_epoch):
            trainingTotal = 0
            for i in range(len(batch_)):
                trainingCount = 0
                t_emotion, tg_emotion = [], []
                try:
                    x_batch, y_batch = [batch_[i]], [label_[i]]
                    l, _ = sess.run([loss, train], feed_dict={X: x_batch, Y: y_batch})
                    avg_cost += l / batch_size

                    ###############Prediction about the training data, which shows almost 98.6% correctness for prediction
                    training_result = sess.run(prediction, feed_dict={X: x_batch})
                    for j in range(0, training_result.shape[1]):
                        t_emotion.append(emotionList[training_result[0][j]])
                    for j in range(0, len(label_[i])):
                        tg_emotion.append(emotionList[np.ndarray.tolist(label_[i][j]).index(1)])
                    ftrain.write("Compare with %d th number of the training batch" % i + "\n")
                    ftrain.write("The predicted label is " + str(t_emotion) + "\n" + "The True label   " + str(
                        tg_emotion) + "\n")
                    for j in range(0, len(t_emotion)):
                        if t_emotion[j] == tg_emotion[j]: trainingCount += 1
                    trainingTotal += trainingCount
                    training_total_.append(trainingTotal)
                except IndexError:
                    pass

            print("epoch of " + str(epoch) + " :" + str(avg_cost))

            avg_cost = 0
        ftrain.write("The max value of the matched count %d"%max(training_total_))
        total_accuracy = 0
        fafterTraining = open("without_cross_validation_test_result.txt","w")
        afterCount = 0
        afterb_emotion, afterbg_emotion = [], []
        for i in range(0,len(test_batch_)):
            print("Accuracy of each in a row %d;"%i, sess.run(accuracy, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]}))
            total_accuracy += sess.run(accuracy, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]})
            result = sess.run(prediction, feed_dict={X: [test_batch_[i]]})
            for j in range(0, result.shape[1]):
                afterb_emotion.append(emotionList[result[0][j]])
            for j in range(0, len(test_label_[i])):
                afterbg_emotion.append(emotionList[np.ndarray.tolist(test_label_[i][j]).index(1)])
            for j in range(0,len(afterb_emotion)):
                if afterb_emotion[j] == afterbg_emotion[j]: afterCount +=1
            fafterTraining.write("Compare with the %d th number of the test batch" % i + "\n")
            fafterTraining.write("the predicted label is " + str(afterb_emotion) + "\n" + "true label is  " + str(afterbg_emotion) + "\n")
            fafterTraining.write("the matched emotion until now is %d"%afterCount+"\n")
            afterb_emotion, afterbg_emotion = [], []


        print("Accuracy of total %f"%(total_accuracy/(i+1)))


        coord.request_stop()
        coord.join(threads)

Xtest = transcript[5]
Ytest = emotion[5]
transcriptLength = 0
emotionLength = 0
for i in range(0,cross_val-1):
    transcriptLength += len(transcript[i])
    emotionLength += len(emotion[i])
forX= np.empty((50,transcriptLength), float)
forY = np.empty((7,emotionLength), int)

for i in range(0,cross_val-1):

    Xtraining = transcript[0:4]
    Ytraining = emotion[0:4]
    Xvalid = transcript[i]
    Yvalid = emotion[i]
    Xtraining.pop(i)
    Ytraining.pop(i)

    for j in range(0,len(Xtraining)):
        if j == 0:
            forX = Xtraining[j]
            forY = Ytraining[j]
        else:
            np.append(forX,Xtraining[j],axis=1)
            np.append(forY,Ytraining[j], axis=1)


    print("Cross validation with %d as Validation Set"%i)
    training(forX,forY,Xvalid,Yvalid,i)
    forX= np.empty((50,transcriptLength), float)
    forY = np.empty((7,emotionLength), int)
    Xvalid = np.empty((Xvalid.shape[0], Xvalid.shape[1]))
    Yvalid = np.empty((Yvalid.shape[0],Yvalid.shape[1]))
    Xtraining, Ytraining, = [], []



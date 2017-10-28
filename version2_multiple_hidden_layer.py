#Layer normalization with dropout

import tensorflow as tf
import numpy as np
import os
from ver_1word_embedding_frequency import emotionList_return
from tensorflow.python.framework import dtypes

emotionList = emotionList_return()

count = 0
currentPath = os.getcwd()
os.chdir(currentPath  + "/Training")
Xtraining = np.load("transcriptIndex.npy")
Ytraining = np.load("emotionIndex.npy")

os.chdir(currentPath  + "/Test")
Xtest = np.load("transcriptIndex.npy")
Ytest = np.load("emotionIndex.npy")



num_classes = len(emotionList)
input_dimension = 50 #len(Xtraining[0])
hidden_size = len(emotionList)
batch_size = 50
line_length = len(Xtraining[0])
sequence_length = len(Xtraining[0])
learning_rate = 0.001
num_layer = 3
dropout = tf.constant(1.0)
#dropout = tf.placeholder(tf.float32)

num_epoch = 100

num_batch = len(Xtraining)//batch_size
def generate_batch(batch_size, train, label):
    data_index = 0
    batch, labels = [] , []
    iterate_num = int(len(train)/batch_size)
    for i in range(iterate_num):
        batch.append(train[data_index:data_index+batch_size])
        labels.append(label[data_index:data_index+batch_size])
        data_index = (data_index + batch_size)
    return batch, labels



all_training = tf.convert_to_tensor(Xtraining,dtype = dtypes.float32)
all_label = tf.convert_to_tensor(Ytraining, dtype = dtypes.int32)
all_test = tf.convert_to_tensor(Xtest,dtype = dtypes.float32)
all_test_label = tf.convert_to_tensor(Ytest, dtype = dtypes.int32)

train_input_queue = tf.train.slice_input_producer([Xtraining, Ytraining], shuffle=False)
test_input_queue = tf.train.slice_input_producer([Xtest,Ytest], shuffle=False)

train_transcript = train_input_queue[0]
train_label = train_input_queue[1]
test_trancript = test_input_queue[0]
test_label = test_input_queue[1]


train_transcript_batch, train_label_batch = tf.train.batch([train_transcript,train_label], batch_size = batch_size)
print("This is the batch of x" +str(train_transcript_batch))
test_transcript_batch, test_label_batch = tf.train.batch([test_trancript,test_label],batch_size = batch_size)
#######################이 위에는 batch를 tesor로 만든 버전 아래는 batch를 list로 만든 버전.

inputs = []

for i in range(0,num_epoch):
    inputs.append(tf.placeholder(tf.float32,shape = [batch_size,len(Xtraining[0]),line_length]))

X = tf.placeholder(tf.float32,shape = [None,len(Xtraining[0]),input_dimension])
Y = tf.placeholder(tf.int32,shape = [None,len(Xtraining[0]),num_classes])


batch_, label_= generate_batch(batch_size, Xtraining, Ytraining)
test_batch_, test_label_ = generate_batch(batch_size,Xtest,Ytest)


finput = open("input.txt","w")
for i in range(0,len(batch_)):
    finput.write(str(batch_[i])+"\n")

output_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_classes)
cells = []
for _ in range(num_layer):
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units = sequence_length)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
    cells.append(cell)
#cells.append(output_cell)
cell = tf.nn.rnn_cell.MultiRNNCell(cells)


#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = hidden_size, state_is_tuple=True)
#multicells = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple = True)   #The number of stack is multipled by cell
initial_state = cell.zero_state(batch_size = batch_size, dtype = tf.float32)
#The initial State can be affected to the multiple layer stacked RNN neural network.
####If we have only one hidden layer, the percentage of the correctness is 20 percent. loss 1.7 around with learning rate 0.001
#WOOOOO HOOOOO when use mulitple layer with 3, the loss became 0.4 with learning rate 0.001 woooohooooo~~~~~~


#What is output shape?

output, _state = tf.nn.dynamic_rnn(cell,X,dtype = tf.float32)
#The argument of the rnn.dynamic_rnn
#  should be tensor and the static rnn argument input must be format of list, string or other python data types.
output_shape = tf.shape(output)
print(str(tf.shape(Y)))
print(str(output_shape))

###Softmax Fully-connected
X_for_softmax = tf.reshape(output,[-1,sequence_length])
output = tf.contrib.layers.fully_connected(X_for_softmax,num_classes,activation_fn=None)
#output = tf.reshape(output, [batch_size, sequence_length,num_classes])
#softmax_w = tf.get_variable("softmax_w",[hidden_size,num_classes])
#softmax_v = tf.get_variable("softmax_b",[num_classes])
output = tf.reshape(output, [-1,sequence_length, num_classes])


#weight = tf.Variable(tf.random_normal([batch_size,num_classes]))
sequence_loss = tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = Y)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

#결과값중 가장 큰 값을 1로 설정하는 함수가 argmax인 것.
prediction = tf.argmax(output,axis=2)
#true_val = tf.argmax(Y,axis=2)
#prediction = tf.equal(tf.argmax(Y,1), tf.argmax(output,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))




#fon =open("training.txt","w")
result = np.empty([Ytest.shape[0], Ytest.shape[1]])
avg_cost = 0
b_emotion, bg_emotion = [], []
fcheck = open("check_for_multi_layer.txt","w")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord= coord)
    print("from the train set")
    for epoch in range(num_epoch):
        for i in range(len(batch_)):
            b_emotion, bg_emotion = [], []
            try:
                x_batch, y_batch = [batch_[i]], [label_[i]]
            except IndexError:
                x_batch, y_batch = train_transcript_batch, train_label_batch
            l,_ = sess.run([loss,train], feed_dict = {X: x_batch, Y:y_batch})
            avg_cost += l / batch_size
            try:
                result = sess.run(prediction, feed_dict={X:[test_batch_[i]]})
            except IndexError:
                pass
            for j in range(0,result.shape[1]):
                b_emotion.append(emotionList[result[0][j]])
            for j in range(0,len(label_[i])):
                bg_emotion.append(emotionList[np.ndarray.tolist(label_[i][j]).index(1)])

            fcheck.write("the predicted label is " + str(b_emotion) + "\n" + "true label is  " + str(bg_emotion) + "\n")
            #fon.write(str(result) + "\n")
            #avg_cost += l /len(batch_)
            '''
            print("from the test set")
            print("from the test set")
            for i in range(1000):
                fopen = open("output.txt","w")
                fopen.write(str(sess.run(test_label_batch)) + "\n")
                #print sess.run(test_label_batch)
            '''
        print("epoch of " + str(epoch) + " :" + str(avg_cost))
        avg_cost = 0
        ###To see the result after each epoch
        #print("epoch;","%d" % (epoch +1), "cost= ","{:.9f}".format(avg_cost))

    #print("Accuracy of total;", accuracy.eval(session=sess, feed_dict={X: test_batch_, Y: test_label_}))
    foutput=open("after_finish_training_prediction.txt","w")
    one = 1
    count = 0
    for i in range(0,len(test_batch_)):
        #test_batch_[i] = np.ndarray.tolist(test_batch_[i])
        #test_label_[i] = np.ndarray.tolist(test_label_[i])
        predict_emotion = []
        ground_emotion = []
        result = sess.run(prediction, feed_dict={X: [test_batch_[i]]})
        #print(str(len(test_label_[i])))
        # result = sess.run(Xtest)
        # print(str(max(result[i])))
        for j in range(0,result.shape[1]):
            predict_emotion.append(emotionList[result[0][j]])
        for j in range(0,len(test_label_[i])):
            ground_emotion.append(emotionList[np.ndarray.tolist(test_label_[i][j]).index(one)])
        foutput.write("the predicted label is "+ str(predict_emotion) + "\n" + "true label is  " + str(ground_emotion)+"\n" )
        for i in range(0,len(predict_emotion)):
            if predict_emotion[i] == ground_emotion[i]: count += 1

        #print(str(i) + " The loss is " + str(l) + " " + "prediction= " + str(result) + " " + "trueY: " + str(test_label_[i]) + "\n")
        #foutput.write("Accuracy for each;" , accuracy.eval(session=sess, feed_dict = {X: [test_batch_[i]], Y: [test_label_[i]]}))
        foutput.write("\n")

    foutput.write("The total same emotion label is " + str(count))
    coord.request_stop()
    coord.join(threads)

'''
emotion_result = []
result = np.ndarray.tolist(result)
fon.write(str(result)+"\n")
for i in range(0,len(result)):
    line = result[i]
    max_val = max(result[i])
    index = line.index(max_val)
    try:
        emotion_result.append(emotionList[index])
    except IndexError:
        emotion_result.append("UNK")


ground_truth = []
one = 1
Ytest = np.ndarray.tolist(Ytest)
for i in range(0,len(Ytest)):
    line = Ytest[i]
    index = Ytest[i].index(one)
    ground_truth.append(emotionList[index])

fresult = open("result.txt","w")
count = 0
for i in range(0,len(emotion_result)):
    if emotion_result[i] == ground_truth[i] :  count += 1
    fresult.write(emotion_result[i]+ " " + ground_truth[i] + "\n")

print(str(count))



'''


#The loss are not decreasing.......even it is increasing.......

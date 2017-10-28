#Layer normalization with dropout

import tensorflow as tf
import numpy as np
import os

#emotionList = emotionList_return()
#emotionList = ["anger","boredom","disgust","empty","enthusiasm","fear","fun","guilt","happiness","hate","joy","love","neutral","relief","sadness","surprise","worry"]
emotionList = ['relief','shame','hate','fun','disgust','anger']
#emotionList  = ["happiness","worry"]
count = 0
currentPath = os.getcwd()+"\\gensim_result"
os.chdir(currentPath  + "\\Training")
Xtraining = np.load("transcriptIndex.npy")
Ytraining = np.load("emotionIndex.npy")

os.chdir(currentPath  + "\\Test")
Xtest = np.load("transcriptIndex.npy")
Ytest = np.load("emotionIndex.npy")



num_classes = len(emotionList)
input_dimension = 50
hidden_size = len(emotionList)
batch_size = 50
line_length = len(Xtraining[0])
sequence_length = len(Xtraining[0])
learning_rate = 0.01
num_layer = 2
dropout = tf.constant(0.8)
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

######################################################################################################################
all_training = tf.convert_to_tensor(Xtraining,dtype = tf.float32)
all_label = tf.convert_to_tensor(Ytraining, dtype = tf.int32)
all_test = tf.convert_to_tensor(Xtest,dtype = tf.float32)
all_test_label = tf.convert_to_tensor(Ytest, dtype = tf.int32)

train_input_queue = tf.train.slice_input_producer([Xtraining, Ytraining], shuffle=False)
test_input_queue = tf.train.slice_input_producer([Xtest,Ytest], shuffle=False)

train_transcript = train_input_queue[0]
train_label = train_input_queue[1]
test_trancript = test_input_queue[0]
test_label = test_input_queue[1]


train_transcript_batch, train_label_batch = tf.train.batch([train_transcript,train_label], batch_size = batch_size)
print("This is the batch of x" +str(train_transcript_batch))
test_transcript_batch, test_label_batch = tf.train.batch([test_trancript,test_label],batch_size = batch_size)

inputs = []

for i in range(0,num_epoch):
    inputs.append(tf.placeholder(tf.float32,shape = [batch_size,len(Xtraining[0]),line_length]))
#######################################################################################################################


X = tf.placeholder(tf.float32,shape = [None,len(Xtraining[0]),input_dimension])
Y = tf.placeholder(tf.int32,shape = [None,len(Xtraining[0]),num_classes])


batch_, label_= generate_batch(batch_size, Xtraining, Ytraining)
test_batch_, test_label_ = generate_batch(batch_size,Xtest,Ytest)


finput = open("input.txt","w")
for i in range(0,len(batch_)):
    finput.write(str(batch_[i])+"\n")

cells = []
with tf.variable_scope("RNN"):
    for i in range(num_layer):
        cell = tf.contrib.rnn.GRUCell(num_units=sequence_length,
                                               reuse=tf.get_variable_scope().reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
        cells.append(cell)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    output, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

###Softmax Fully-connected
X_for_softmax = tf.reshape(output,[-1,sequence_length])
output = tf.contrib.layers.fully_connected(X_for_softmax,num_classes,activation_fn=None)
#output = tf.reshape(output, [batch_size, sequence_length,num_classes])
#softmax_w = tf.get_variable("softmax_w",[hidden_size,num_classes])
#softmax_v = tf.get_variable("softmax_b",[num_classes])
output = tf.reshape(output, [-1, sequence_length, num_classes])

#weight = tf.ones([batch_size,num_classes])
sequence_loss = tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = Y)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(loss)


#결과값중 가장 큰 값을 1로 설정하는 함수가 argmax인 것.
prediction = tf.argmax(output,axis=2)
#여기서 prediction이 감정 수 만큼 형성되어 있나?
top_k = tf.nn.top_k(output,k=3,sorted=True)


correct_prediction = tf.equal(tf.argmax(output,axis=2), tf.argmax(Y,axis=2))
#true_val = tf.argmax(Y,axis=2)
#prediction = tf.equal(tf.argmax(Y,1), tf.argmax(output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



result = np.empty([Ytest.shape[0], Ytest.shape[1]])
count = 0
totalCount = 0
total_ = []
acc = []
trainingCount = 0
trainingTotal = 0
training_total_=[]
#for_total_match_count = 0
avg_cost = 0
b_emotion, bg_emotion = [], []
t_emotion, tg_emotion = [], []
fcheck = open("training_label_check.txt","w")
ftrain = open("without_cross_validation_train_result.txt","w")
fcheck.write("The Result is based on every test batch with length %d"%len(test_batch_)+"\n")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("from the train set")
    #print("This is the value of indices"+str(sess.run(indices).indices()))

    for epoch in range(num_epoch):
        trainingTotal = 0
        for i in range(len(batch_)):
            count = 0
            trainingCount = 0
            b_emotion, bg_emotion = [], []
            t_emotion, tg_emotion = [], []
            try:
                x_batch, y_batch = [batch_[i]], [label_[i]]
            except IndexError:
                x_batch, y_batch = train_transcript_batch, train_label_batch


            l,_ = sess.run([loss,train], feed_dict = {X: x_batch, Y:y_batch})
            avg_cost += l / batch_size

            #if i%5 == 0 :
            #    print(sess.run(accuracy,feed_dict={X:x_batch,Y:y_batch}))



            ###############Prediction about the training data, which shows almost 98.6% correctness for prediction
            training_result = sess.run(prediction, feed_dict={X:x_batch})
            for j in range(0,training_result.shape[1]):
                t_emotion.append(emotionList[training_result[0][j]])
            for j in range(0,len(label_[i])):
                tg_emotion.append(emotionList[np.ndarray.tolist(label_[i][j]).index(1)])
            ftrain.write("Compare with %d th numb er of the training batch"%i+"\n" )
            ftrain.write("The predicted label is " + str(t_emotion) + "\n" + "The True label   " + str(tg_emotion) + "\n")
            for j in range(0,len(t_emotion)):
                if t_emotion[j] == tg_emotion[j]: trainingCount += 1
            trainingTotal += trainingCount
            training_total_.append(trainingTotal)

            '''
            ################Prediction about the test data, only return 20% prediction
            try:
                result = sess.run(prediction, feed_dict={X:[test_batch_[i]]})
                for j in range(0, result.shape[1]):
                    b_emotion.append(emotionList[result[0][j]])
                for j in range(0, len(test_label_[i])):
                    bg_emotion.append(emotionList[np.ndarray.tolist(test_label_[i][j]).index(1)])
                fcheck.write("Compare with the %d th number of the test batch"%i+ "\n")
                fcheck.write("the predicted label is " + str(b_emotion) + "\n" + "true label is  " + str(bg_emotion) + "\n")

                for i in range(0,len(b_emotion)):
                    if b_emotion[i] == bg_emotion[i]:
                        count += 1
                totalCount += count
                fcheck.write("The matched emotion label was " + str(count) +"\n")
                fcheck.write("The Total emotion label matched was " + str(totalCount) +"\n")
                #print("The total matched emotion label is " +str(totalCount))
                #fcheck.write("The total emotion matched was " +str(for_total_match_count) + "\n" )

                #for_total_match_count += count
                #if i == 0 : for_total_match_count = 0
            except IndexError:
                total_.append(totalCount)
                totalCount = 0
                pass
            '''
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

        #fcheck.write("The max value of the matched count = %d"%max(total_))
        print("epoch of " + str(epoch) + " :" + str(avg_cost))


        #train_accuracy = sess.run(accuracy(feed_dict={X: x_batch, Y: y_batch}))
        #print("the training accuracy: ", train_accuracy)

        avg_cost = 0
        ###To see the result after each epoch
        #print("epoch;","%d" % (epoch +1), "cost= ","{:.9f}".format(avg_cost))
        ftrain.write("The max value of the matched count %d"%max(training_total_))
        total_accuracy = 0
        fafterTraining = open("without_cross_validation_test_result.txt","w")
        afterCount = 0
        afterb_emotion, afterbg_emotion = [], []
        for i in range(0,len(test_batch_)):
            #print("Accuracy of each in a row %d;"%i, sess.run(accuracy, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]}))
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


            # Checking the top3 label of the emotion
            check_top_emotion_label_file = open("chcek_top_emotion_label_file.txt", mode="w", encoding="utf-8")
            top_k_values, top_k_indice = sess.run(top_k, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]})
            for m in range(0, len(top_k_values)):
                top_label = []
                for j in range(0, len(top_k_values[m])):
                    for k in range(0,len(top_k_values[m][j])):
                        index = int(top_k_indice[m][j][k])
                        top_emotion = emotionList[index]
                        top_label.append(top_emotion)
                    check_top_emotion_label_file.write(" ".join(top_label) + " ")
                    check_top_emotion_label_file.write(" ***The Truth label is " + afterbg_emotion[i] +"\n")
                    top_label = []
            afterb_emotion, afterbg_emotion = [], []

        print("Accuracy of total %f"%(total_accuracy/(i+1)))
        acc.append(total_accuracy/(i+1))


    '''
    ###The prediction of total in once
    total_emotion, total_ground_truth = [], []
    test_result = sess.run(prediction, feed_dict={X:Xtest})
    for j in range(0,test_result.shape[1]):
        total_emotion.append(emotionList[test_result[0][j]])
    #for j in range(0,len(test_label_)):
    #    total_ground_truth.append(emotionList[test_label_[j]])
    print("The total emotion is " + str(total_emotion) +"\n")


    '''
    coord.request_stop()
    coord.join(threads)
    print("the maximum of the accuracy is %d"%max(acc))
    '''
    #print("Accuracy of total;", accuracy.eval(session=sess, feed_dict={X: test_batch_, Y: test_label_}))
    foutput=open("label_compare.txt","w")
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
    '''

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


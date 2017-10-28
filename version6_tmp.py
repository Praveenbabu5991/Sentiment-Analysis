for epoch in range(0, num_epoch):
    for i in range(len(batch_)):
        try:
            x_batch, y_batch = [batch_[i]], [label_[i]]
            l, _ = sess.run(activation, feed_dict={X: x_batch, Y: y_batch})
            avg_cost += l / batch_size

            ###############Prediction about the training data, which shows almost 98.6% correctness for prediction
            training_result = sess.run(prediction, feed_dict={X: x_batch})
            for j in range(0, training_result.shape[1]):
                t_emotion.append(emotionList[training_result[0][j]])
            for j in range(0, len(label_[i])):
                tg_emotion.append(emotionList[np.ndarray.tolist(label_[i][j]).index(1)])
        except IndexError:
            pass
    # print("the length of the prediction label "+str(len(t_emotion)))
    ftrain.write("The is epoch %d" % epoch + "\n")
    ftrain.write("The predicted label with length %d\n" % len(t_emotion) + str(t_emotion) + "\n")
    ftrain.write("The true label with length %d\n" % len(tg_emotion) + str(tg_emotion) + "\n")

    for j in range(0, len(t_emotion)):
        if t_emotion[j] == tg_emotion[j]: trainingCount += 1
    ftrain.write("Total number of match is " + str(trainingCount) + '\n')

    print("epoch of " + str(epoch) + " :" + str(avg_cost))
    avg_cost = 0
    if i == len(batch_) - 1:
        t_emotion, tg_emotion = [], []
        trainingCount = 0
    saver.save(sess, ckpt_path + "/my_model", global_step=index)
    saver.export_meta_graph(ckpt_path + "/my_model.meta")

total_accuracy = 0
fafterTraining = open("5-cross_test_label_result.txt", "w")
afterCount = 0
afterb_emotion, afterbg_emotion = [], []
for i in range(0, len(test_batch_)):
    print("Accuracy of each in a row %d;" % i,
          sess.run(accuracy, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]}))
    total_accuracy += sess.run(accuracy, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]})
    result = sess.run(prediction, feed_dict={X: [test_batch_[i]]})
    # fafterTraining.write("The length that will going to predict is %d"%len(test_batch_[i]))
    for j in range(0, result.shape[1]):
        afterb_emotion.append(emotionList[result[0][j]])
    for j in range(0, len(test_label_[i])):
        afterbg_emotion.append(emotionList[np.ndarray.tolist(test_label_[i][j]).index(1)])
    for j in range(0, len(afterb_emotion)):
        if afterb_emotion[j] == afterbg_emotion[j]: afterCount += 1
    # fafterTraining.write("Compare with the %d th number of the test batch" % i + "\n")
    print("The current match of the test data is %d" % afterCount)

fafterTraining.write(
    "the predicted label is " + str(afterb_emotion) + "\n" + "true label is  " + str(afterbg_emotion) + "\n")
fafterTraining.write("the matched emotion until now is %d" % afterCount + "\n")
fafterTraining.write("The total number of label is %d" % len(afterb_emotion) + '\n')

print("Accuracy of total %f" % (total_accuracy / (i + 1)))

with tf.Graph().as_default() as g:
    for epoch in range(0, num_epoch):
        for i in range(len(batch_)):
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
                # ftrain.write("Compare with %d th number of the training batch" % i + "\n")
                # ftrain.write("The predicted label is " + str(t_emotion) + "\n" + "The True label   " + str(
                #    tg_emotion) + "\n")
                '''
             for j in range(0, len(t_emotion)):
                if t_emotion[j] == tg_emotion[j]: trainingCount += 1
                    trainingTotal += trainingCount
             training_total_.append(trainingTotal)
             '''
            except IndexError:
                pass
                # print("the length of the prediction label "+str(len(t_emotion)))
        ftrain.write("The is epoch %d" % epoch + "\n")
        ftrain.write("The predicted label with length %d\n" % len(t_emotion) + str(t_emotion) + "\n")
        ftrain.write("The true label with length %d\n" % len(tg_emotion) + str(tg_emotion) + "\n")

        for j in range(0, len(t_emotion)):
            if t_emotion[j] == tg_emotion[j]: trainingCount += 1
        ftrain.write("Total number of match is " + str(trainingCount) + '\n')

        print("epoch of " + str(epoch) + " :" + str(avg_cost))
        avg_cost = 0
        if i == len(batch_) - 1:
            t_emotion, tg_emotion = [], []
            trainingCount = 0

    total_accuracy = 0
    fafterTraining = open("5-cross_test_label_result.txt", "w")
    afterCount = 0
    afterb_emotion, afterbg_emotion = [], []
    for i in range(0, len(test_batch_)):
        print("Accuracy of each in a row %d;" % i,
              sess.run(accuracy, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]}))
        total_accuracy += sess.run(accuracy, feed_dict={X: [test_batch_[i]], Y: [test_label_[i]]})
        result = sess.run(prediction, feed_dict={X: [test_batch_[i]]})
        # fafterTraining.write("The length that will going to predict is %d"%len(test_batch_[i]))
        for j in range(0, result.shape[1]):
            afterb_emotion.append(emotionList[result[0][j]])
        for j in range(0, len(test_label_[i])):
            afterbg_emotion.append(emotionList[np.ndarray.tolist(test_label_[i][j]).index(1)])
        for j in range(0, len(afterb_emotion)):
            if afterb_emotion[j] == afterbg_emotion[j]: afterCount += 1
        # fafterTraining.write("Compare with the %d th number of the test batch" % i + "\n")
        print("The current match of the test data is %d" % afterCount)
    fafterTraining.write(
        "the predicted label is " + str(afterb_emotion) + "\n" + "true label is  " + str(afterbg_emotion) + "\n")
    fafterTraining.write("the matched emotion until now is %d" % afterCount + "\n")
    fafterTraining.write("The total number of label is %d" % len(afterb_emotion) + '\n')

    print("Accuracy of total %f" % (total_accuracy / (i + 1)))

    meta_graph_def = tf.train.export_meta_graph(filename=ckpt_path + "/my-model.meta")
    print("Model saved in file: %s" % ckpt_path + "/my_model")
    # coord.request_stop()
    # coord.join(threads)


if index == 0:
        graph = tf.get_default_graph()
        sess = tf.Session(graph=graph)
        activation = [loss, train]
        graph.add_to_collection("activation", activation)
        graph.add_to_collection("prediction", prediction)
        graph.add_to_collection("accuracy", accuracy)
    else:
        new_graph = tf.Graph()
        new_graph.as_default()
        sess = tf.Session(graph=new_graph)
        new_saver = tf.train.import_meta_graph(ckpt_path + "/my_model.meta", clear_devices=True)

        ckpt = tf.train.get_checkpoint_state(ckpt_path + "/my_model.ckpt")
        if ckpt and ckpt.model_checkpoint_path:
            new_saver.restore(sess, ckpt.model_checkpoint_path)
        new_saver.restore(sess, ckpt_path + "/my_model")
        activation = tf.get_collection("activation")
        prediction = tf.get_collection("predcition")
        accuracy = tf.get_collection("accuracy")












# Layer normalization with dropout
# 5-cross-validation

import tensorflow as tf
import numpy as np
import os

emotionList = ['anger', 'disgust', 'joy', 'shame', 'fear', 'sadness', 'guilt']
cross_val = 6
count = 0
currentPath = os.getcwd() + "\\gensim_result"
os.chdir(currentPath + "\\CrossValidation")
ckpt_path = "./tmp"
transcript = []
emotion = []
for i in range(0, cross_val):
    transcript.append(np.load("transcriptIndex%d.npy" % i))
    emotion.append(np.load("emotionIndex%d.npy" % i))

####HyperParameter
num_classes = len(emotionList)
input_dimension = 50
hidden_size = len(emotionList)
batch_size = 50
line_length = 50  # len(Xtraining[0])
sequence_length = 50  # len(Xtraining[0])
learning_rate = 0.002
num_layer = 3
dropout = tf.constant(0.5)

num_epoch = 1
num_batch = (len(transcript[0]) + len(transcript[1]) + len(transcript[2]) + len(transcript[3]) + len(
    transcript[4])) / batch_size


# Loss 3.5-2.7

def generate_batch(batch_size, train, label):
    data_index = 0
    batch, labels = [], []
    iterate_num = int(len(train) / batch_size)
    for i in range(iterate_num):
        batch.append(train[data_index:data_index + batch_size])
        labels.append(label[data_index:data_index + batch_size])
        data_index = (data_index + batch_size)
    return batch, labels


def restore_last_session():
    saver = tf.train.Saver()
    sess = tf.Session()
    ckpt = tf.train.import_meta_graph(ckpt_path+"/model.ckpt.meta")

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return sess

def declaration_variables(Xt,Yt,Xte,Yte,index):

    X = tf.placeholder(tf.float32, shape=[None, line_length, input_dimension], name="X")
    Y = tf.placeholder(tf.int32, shape=[None, line_length, num_classes],name="Y")

    # Generate batch for training and test
    batch_, label_ = generate_batch(batch_size, Xt, Yt)
    test_batch_, test_label_ = generate_batch(batch_size, Xte, Yte)
    print("the training Batch for %d th iteration is " % index, len(batch_))
    print("the length of total training data is %d" % len(Xt))
    print("the test Batch for %d th iteration is " % index, len(test_batch_))

    initializer = tf.random_uniform_initializer(-0.01, 0.01)
    cells = []
    with tf.variable_scope("RNN%d" % index):
        for i in range(num_layer):
            cell = tf.contrib.rnn.LSTMCell(num_units=sequence_length,
                                          reuse=tf.get_variable_scope().reuse, initializer=initializer, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
            cells.append(cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        output, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


        ###Softmax Fully-connected
        X_for_softmax = tf.reshape(output, [-1, sequence_length])
        output = tf.contrib.layers.fully_connected(X_for_softmax, num_classes, activation_fn=None)
        output = tf.reshape(output, [-1, sequence_length, num_classes])

        sequence_loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y)
        loss = tf.reduce_mean(sequence_loss,name="loss")
        train = tf.train.AdamOptimizer(learning_rate=learning_rate,name="train").minimize(loss)

        # 결과값중 가장 큰 값을 1로 설정하는 함수가 argmax인 것
        prediction = tf.argmax(output, axis=2, name="prediction")
        correct_prediction = tf.equal(tf.argmax(output, axis=2), tf.argmax(Y, axis=2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

        tf.add_to_collection("vars",X)
        tf.add_to_collection("vars",Y)
        tf.add_to_collection("vars",dropout)
        tf.add_to_collection("variables",cell)
        tf.add_to_collection("vars",output)
        #merged = tf.summary.merge([X,Y,dropout])
    return X,Y,loss, train, batch_,label_,prediction,test_batch_,test_label_,accuracy


def training(Xt, Yt, Xte, Yte, index):

    #graph = tf.Graph()
    if index == 0:
        graph = tf.get_default_graph()
        variable_list = declaration_variables(Xt,Yt,Xte,Yte,index)
    else:
        tf.reset_default_graph()
        new_graph = tf.Graph()
        with new_graph.as_default():
            variable_list = declaration_variables(Xt,Yt,Xte,Yte,index)


    X,Y,loss, train, batch_,label_,prediction,test_batch_,test_label_,accuracy = variable_list
    activation = [loss,train]

    training_total_ = []
    avg_cost = 0
    ftrain = open("5-cross_train_label_result.txt", "w")
    t_emotion, tg_emotion = [], []
    trainingCount = 0



    if index == 0:
        sess = tf.Session(graph=graph)
        #sess.run(tf.global_variables_initializer())
        activation = [loss,train]
    else:
        sess = tf.Session(graph=new_graph)
        #new_saver = tf.train.Saver()
        new_saver = tf.train.import_meta_graph(ckpt_path+"/my_model.meta")

        ckpt = tf.train.get_checkpoint_state(ckpt_path+"/my_model.ckpt")
        if ckpt and ckpt.model_checkpoint_path:
            new_saver.restore(sess,ckpt.model_checkpoint_path)
        #new_saver.restore(sess, ckpt.model_checkpoint_path)
        #new_saver.restore(sess,ckpt_path+"/my_model.ckpt")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord=coord,start=True)

    saver = tf.train.Saver()

    print("from the train set")

    for epoch in range(0, num_epoch):
        for i in range(len(batch_)):
            try:
                x_batch, y_batch = [batch_[i]], [label_[i]]
                l, _ = sess.run(activation, feed_dict={"X:0": x_batch, "Y:0": y_batch})
                avg_cost += l / batch_size

                ###############Prediction about the training data, which shows almost 98.6% correctness for prediction
                training_result = sess.run(prediction, feed_dict={"X:0": x_batch})
                for j in range(0, training_result.shape[1]):
                    t_emotion.append(emotionList[training_result[0][j]])
                for j in range(0, len(label_[i])):
                    tg_emotion.append(emotionList[np.ndarray.tolist(label_[i][j]).index(1)])
            except IndexError:
                pass
        # print("the length of the prediction label "+str(len(t_emotion)))
        ftrain.write("The is epoch %d" % epoch + "\n")
        ftrain.write("The predicted label with length %d\n" % len(t_emotion) + str(t_emotion) + "\n")
        ftrain.write("The true label with length %d\n" % len(tg_emotion) + str(tg_emotion) + "\n")

        for j in range(0, len(t_emotion)):
            if t_emotion[j] == tg_emotion[j]: trainingCount += 1
        ftrain.write("Total number of match is " + str(trainingCount) + '\n')

        print("epoch of " + str(epoch) + " :" + str(avg_cost))
        avg_cost = 0
        if i == len(batch_) - 1:
            t_emotion, tg_emotion = [], []
            trainingCount = 0
        #tf.train.write_graph(sess.graph_def,"./tmp","test.pb",False)
        #saver.export_meta_graph(ckpt_path + "/my_model.meta")
        saver.save(sess, ckpt_path + "/my_model.ckpt", global_step=index, meta_graph_suffix="meta", write_meta_graph=True)

    total_accuracy = 0
    fafterTraining = open("5-cross_test_label_result.txt", "w")
    afterCount = 0
    afterb_emotion, afterbg_emotion = [], []
    for i in range(0, len(test_batch_)):
        print("Accuracy of each in a row %d;" % i,
              sess.run(accuracy, feed_dict={"X:0": [test_batch_[i]], "Y:0": [test_label_[i]]}))
        total_accuracy += sess.run(accuracy, feed_dict={"X:0": [test_batch_[i]], "Y:0": [test_label_[i]]})
        result = sess.run(prediction, feed_dict={"X:0": [test_batch_[i]]})
        # fafterTraining.write("The length that will going to predict is %d"%len(test_batch_[i]))
        for j in range(0, result.shape[1]):
            afterb_emotion.append(emotionList[result[0][j]])
        for j in range(0, len(test_label_[i])):
            afterbg_emotion.append(emotionList[np.ndarray.tolist(test_label_[i][j]).index(1)])
        for j in range(0, len(afterb_emotion)):
            if afterb_emotion[j] == afterbg_emotion[j]: afterCount += 1
        # fafterTraining.write("Compare with the %d th number of the test batch" % i + "\n")
        print("The current match of the test data is %d" % afterCount)

    fafterTraining.write(
        "the predicted label is " + str(afterb_emotion) + "\n" + "true label is  " + str(afterbg_emotion) + "\n")
    fafterTraining.write("the matched emotion until now is %d" % afterCount + "\n")
    fafterTraining.write("The total number of label is %d" % len(afterb_emotion) + '\n')

    print("Accuracy of total %f" % (total_accuracy / (i + 1)))
    coord.request_stop()
    coord.join(threads)

    '''
    saver.save(sess,ckpt_path+"/my_model-10000")
    saver.export_meta_graph(ckpt_path+"/my_model-10000.meta")
    #meta_graph_def = tf.train.export_meta_graph(filename=ckpt_path + "/my-model.meta")
    '''



############Applying cross-validation################
Xtest = transcript[5]
Ytest = emotion[5]
transcriptLength = 0
emotionLength = 0
for i in range(0, cross_val - 2):
    transcriptLength += len(transcript[i])
    emotionLength += len(emotion[i])
forX = np.empty((50, transcriptLength), float)
forY = np.empty((7, emotionLength), int)
finfo = open("information.txt", "w")

for i in range(0, cross_val - 1):
    Xtraining = transcript[0:5]
    Ytraining = emotion[0:5]
    Xvalid = transcript[i]
    Yvalid = emotion[i]
    Xtraining.pop(i)
    Ytraining.pop(i)

    for j in range(0, len(Xtraining)):
        if j == 0:
            forX = Xtraining[j]
            forY = Ytraining[j]
        else:
            forX = np.append(forX, Xtraining[j], axis=0)
            forY = np.append(forY, Ytraining[j], axis=0)

    print("Cross validation with %d as Validation Set" % i)

    finfo.write("Cross validation with %d as Validation Set" % i + "\n")
    finfo.write("The length of Xtraining Data is %d" % transcriptLength + "\n")
    finfo.write("The length of Ytraining Data is %d" % emotionLength + "\n")
    finfo.write("The length of the validation data is " + str(Xvalid.shape[0]) + str(Xvalid.shape[1]) + '\n')

    training(forX, forY, Xvalid, Yvalid, i)

    forX = np.empty((50, transcriptLength), float)
    forY = np.empty((7, emotionLength), int)
    Xvalid = np.empty((Xvalid.shape[0], Xvalid.shape[1]))
    Yvalid = np.empty((Yvalid.shape[0], Yvalid.shape[1]))
    Xtraining, Ytraining, = [], []





































# Layer normalization with dropout
# 5-cross-validation

import tensorflow as tf
import tflean
import numpy as np
import os

emotionList = ['anger', 'disgust', 'joy', 'shame', 'fear', 'sadness', 'guilt']
cross_val = 6
count = 0
currentPath = os.getcwd() + "\\gensim_result"
os.chdir(currentPath + "\\CrossValidation")
ckpt_path = "./tmp"
transcript = []
emotion = []
for i in range(0, cross_val):
    transcript.append(np.load("transcriptIndex%d.npy" % i))
    emotion.append(np.load("emotionIndex%d.npy" % i))

####HyperParameter
num_classes = len(emotionList)
input_dimension = 50
hidden_size = len(emotionList)
batch_size = 50
line_length = 50  # len(Xtraining[0])
sequence_length = 50  # len(Xtraining[0])
learning_rate = 0.002
num_layer = 3
dropout = tf.constant(0.5, name="drop_out")

num_epoch = 1
num_batch = (len(transcript[0]) + len(transcript[1]) + len(transcript[2]) + len(transcript[3]) + len(
    transcript[4])) / batch_size


# Loss 3.5-2.7

def generate_batch(batch_size, train, label):
    data_index = 0
    batch, labels = [], []
    iterate_num = int(len(train) / batch_size)
    for i in range(iterate_num):
        batch.append(train[data_index:data_index + batch_size])
        labels.append(label[data_index:data_index + batch_size])
        data_index = (data_index + batch_size)
    return batch, labels


def restore_last_session():
    saver = tf.train.Saver()
    sess = tf.Session()
    ckpt = tf.train.import_meta_graph(ckpt_path+"/model.ckpt.meta")

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return sess

def declaration_variables(Xt,Yt,Xte,Yte,index):

    X = tf.placeholder(tf.float32, shape=[None, line_length, input_dimension], name="X")
    Y = tf.placeholder(tf.int32, shape=[None, line_length, num_classes],name="Y")

    # Generate batch for training and test
    batch_, label_ = generate_batch(batch_size, Xt, Yt)
    test_batch_, test_label_ = generate_batch(batch_size, Xte, Yte)
    print("the training Batch for %d th iteration is " % index, len(batch_))
    print("the length of total training data is %d" % len(Xt))
    print("the test Batch for %d th iteration is " % index, len(test_batch_))

    cells = []
    with tf.variable_scope("RNN%d" % index):
        for i in range(num_layer):
            cell = tf.contrib.rnn.LSTMCell(num_units=sequence_length,
                                          reuse=tf.get_variable_scope().reuse)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        output, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


        ###Softmax Fully-connected
        X_for_softmax = tf.reshape(output, [-1, sequence_length])
        output = tf.contrib.layers.fully_connected(X_for_softmax, num_classes, activation_fn=None)
        output = tf.reshape(output, [-1, sequence_length, num_classes])

        sequence_loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y)
        loss = tf.reduce_mean(sequence_loss,name="loss")
        train = tf.train.AdamOptimizer(learning_rate=learning_rate,name="train").minimize(loss)

        # 결과값중 가장 큰 값을 1로 설정하는 함수가 argmax인 것
        prediction = tf.argmax(output, axis=2, name="prediction")
        correct_prediction = tf.equal(tf.argmax(output, axis=2), tf.argmax(Y, axis=2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

        tf.add_to_collection("vars",X)
        tf.add_to_collection("vars",Y)
        tf.add_to_collection("vars",dropout)
        #merged = tf.summary.merge([X,Y,dropout])
    return X, Y,loss, train, prediction, accuracy

def running_session(sess,Xt,Yt,Xte,Yte,index, accuracy, loss, train, prediction):

    avg_cost = 0
    ftrain = open("5-cross_train_label_result.txt", "w")
    t_emotion, tg_emotion = [], []
    trainingCount = 0
    activation = [loss,train]

    batch_, label_ = generate_batch(batch_size, Xt, Yt)
    test_batch_, test_label_ = generate_batch(batch_size, Xte, Yte)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord=coord,start=True)

    saver = tf.train.Saver()

    print("from the train set")

    for epoch in range(0, num_epoch):
        for i in range(len(batch_)):
            try:
                x_batch, y_batch = [batch_[i]], [label_[i]]
                l, _ = sess.run(activation, feed_dict={"X:0": x_batch, "Y:0": y_batch})
                avg_cost += l / batch_size

                ###############Prediction about the training data, which shows almost 98.6% correctness for prediction
                training_result = sess.run(prediction, feed_dict={"X:0": x_batch})
                for j in range(0, training_result.shape[1]):
                    t_emotion.append(emotionList[training_result[0][j]])
                for j in range(0, len(label_[i])):
                    tg_emotion.append(emotionList[np.ndarray.tolist(label_[i][j]).index(1)])
            except IndexError:
                pass
        # print("the length of the prediction label "+str(len(t_emotion)))
        ftrain.write("The is epoch %d" % epoch + "\n")
        ftrain.write("The predicted label with length %d\n" % len(t_emotion) + str(t_emotion) + "\n")
        ftrain.write("The true label with length %d\n" % len(tg_emotion) + str(tg_emotion) + "\n")

        for j in range(0, len(t_emotion)):
            if t_emotion[j] == tg_emotion[j]: trainingCount += 1
        ftrain.write("Total number of match is " + str(trainingCount) + '\n')

        print("epoch of " + str(epoch) + " :" + str(avg_cost))
        avg_cost = 0
        if i == len(batch_) - 1:
            t_emotion, tg_emotion = [], []
            trainingCount = 0
        tf.train.write_graph(sess.graph_def,"./tmp","test.pb",False)
        saver.export_meta_graph(ckpt_path + "/my_model.meta")
        saver.save(sess, ckpt_path + "/my_model.ckpt", global_step=index, meta_graph_suffix="meta", write_meta_graph=True)

    total_accuracy = 0
    fafterTraining = open("5-cross_test_label_result.txt", "w")
    afterCount = 0
    afterb_emotion, afterbg_emotion = [], []
    for i in range(0, len(test_batch_)):
        print("Accuracy of each in a row %d;" % i,
              sess.run(accuracy, feed_dict={"X:0": [test_batch_[i]], "Y:0": [test_label_[i]]}))
        total_accuracy += sess.run(accuracy, feed_dict={"X:0": [test_batch_[i]], "Y:0": [test_label_[i]]})
        result = sess.run(prediction, feed_dict={"X:0": [test_batch_[i]]})
        # fafterTraining.write("The length that will going to predict is %d"%len(test_batch_[i]))
        for j in range(0, result.shape[1]):
            afterb_emotion.append(emotionList[result[0][j]])
        for j in range(0, len(test_label_[i])):
            afterbg_emotion.append(emotionList[np.ndarray.tolist(test_label_[i][j]).index(1)])
        for j in range(0, len(afterb_emotion)):
            if afterb_emotion[j] == afterbg_emotion[j]: afterCount += 1
        # fafterTraining.write("Compare with the %d th number of the test batch" % i + "\n")
        print("The current match of the test data is %d" % afterCount)

    fafterTraining.write(
        "the predicted label is " + str(afterb_emotion) + "\n" + "true label is  " + str(afterbg_emotion) + "\n")
    fafterTraining.write("the matched emotion until now is %d" % afterCount + "\n")
    fafterTraining.write("The total number of label is %d" % len(afterb_emotion) + '\n')

    print("Accuracy of total %f" % (total_accuracy / (i + 1)))
    coord.request_stop()
    coord.join(threads)


def training(Xt, Yt, Xte, Yte, index):
    if index == 0:
        graph = tf.get_default_graph()
        variable_list = declaration_variables(Xt,Yt,Xte,Yte,index)
        X, Y, loss, train, prediction, accuracy = variable_list
        #tf.add_to_collection("loss",loss)
        #tf.add_to_collection("accuracy",accuracy)
        #tf.add_to_collection("train",train)
        #tf.add_to_collection("prediction",prediction)
    else:
        tf.reset_default_graph()
        new_graph = tf.Graph()
        with new_graph.as_default():
            variable_list = declaration_variables(Xt,Yt,Xte,Yte,index)
            X, Y, loss, train, prediction, accuracy = variable_list
            #tf.add_to_collection("loss", loss)
            #tf.add_to_collection("accuracy", accuracy)
            #tf.add_to_collection("train", train)
            #tf.add_to_collection("prediction", prediction)

    #X,Y,loss, train, batch_,label_,prediction,test_batch_,test_label_,accuracy = variable_list


    init_op = tf.global_variables_initializer()


    if index == 0:
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            running_session(sess,Xt,Yt,Xte,Yte,index,accuracy,loss,train,prediction)

    else:
        sess = tf.Session(graph=new_graph)
        #new_saver = tf.train.Saver()
        new_saver = tf.train.import_meta_graph(ckpt_path+"/my_model.meta")

        ckpt = tf.train.get_checkpoint_state(ckpt_path+"/my_model.ckpt")
        if ckpt and ckpt.model_checkpoint_path:
                new_saver.restore(sess,tf.train.latest_checkpoint("./tmp"))
        #activation = new_graph.get_operation_by_name("activation")
        loss = tf.get_collection("loss")
        train = tf.get_collection("train")
        #activation = [loss,train]
        accuracy = tf.get_collection("accuracy")
        prediction = tf.get_collection("prediction")
        with tf.Session(graph=new_graph) as sess:
            running_session(sess,Xt,Yt,Xte,Yte,index,accuracy,loss,train,prediction)





############Applying cross-validation################
Xtest = transcript[5]
Ytest = emotion[5]
transcriptLength = 0
emotionLength = 0
for i in range(0, cross_val - 2):
    transcriptLength += len(transcript[i])
    emotionLength += len(emotion[i])
forX = np.empty((50, transcriptLength), float)
forY = np.empty((7, emotionLength), int)
finfo = open("information.txt", "w")

for i in range(0, cross_val - 1):
    Xtraining = transcript[0:5]
    Ytraining = emotion[0:5]
    Xvalid = transcript[i]
    Yvalid = emotion[i]
    Xtraining.pop(i)
    Ytraining.pop(i)

    for j in range(0, len(Xtraining)):
        if j == 0:
            forX = Xtraining[j]
            forY = Ytraining[j]
        else:
            forX = np.append(forX, Xtraining[j], axis=0)
            forY = np.append(forY, Ytraining[j], axis=0)

    print("Cross validation with %d as Validation Set" % i)

    finfo.write("Cross validation with %d as Validation Set" % i + "\n")
    finfo.write("The length of Xtraining Data is %d" % transcriptLength + "\n")
    finfo.write("The length of Ytraining Data is %d" % emotionLength + "\n")
    finfo.write("The length of the validation data is " + str(Xvalid.shape[0]) + str(Xvalid.shape[1]) + '\n')

    training(forX, forY, Xvalid, Yvalid, i)

    forX = np.empty((50, transcriptLength), float)
    forY = np.empty((7, emotionLength), int)
    Xvalid = np.empty((Xvalid.shape[0], Xvalid.shape[1]))
    Yvalid = np.empty((Yvalid.shape[0], Yvalid.shape[1]))
    Xtraining, Ytraining, = [], []



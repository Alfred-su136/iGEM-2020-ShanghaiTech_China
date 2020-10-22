import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
class NN():

    def __init__(self,train_feature,train_label,validate_feature,validate_label):
        self.INPUT_NODE = 90       #输入层有90个神经元（蛋白质70+适体20）
        self.OUTPUT_NODE = 2        #输出层有2个神经元（判断能否结合，输入为one-hot）
        self.LAYER1_NODE = 50       #隐藏层有50个神经元
        self.LAYER2_NODE = 50
        self.BATCH_SIZE = 64
        self.TRAINING_STEPS = 10000
        self.LEARNING_RATE_BASE = 0.8
        self.LEARNING_RATE_DECAY = 0.99
        self.REGULARIZATION_RATE = 0.0001
        self.MODEL_SAVE_PATH = "model"
        self.MODEL_NAME = "model.ckpt"
        self.train_feature = train_feature
        self.train_label = train_label
        self.validate_feature = validate_feature
        self.validate_label = validate_label

    def inference(self,input_tensor):#双隐层神经网络的向前传播过程
        with tf.variable_scope('layer1',reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights",[self.INPUT_NODE,self.LAYER1_NODE],
                                      initializer = tf.truncated_normal_initializer(stddev = 0.1))
            tf.add_to_collection('losses',
                                 tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)(weights))
            biases = tf.get_variable("biases",[self.LAYER1_NODE],initializer = tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
            #引入激活函数使神经网络表示的函数为非线性

        with tf.variable_scope('layer2',reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", [self.LAYER1_NODE, self.LAYER2_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('losses',
                                 tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)(weights))
            biases = tf.get_variable("biases", [self.LAYER2_NODE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

        with tf.variable_scope('layer3',reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights",[self.LAYER2_NODE,self.OUTPUT_NODE],
                                      initializer = tf.truncated_normal_initializer(stddev = 0.1))
            tf.add_to_collection('losses',
                                 tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)(weights))
            biases = tf.get_variable("biases",[self.OUTPUT_NODE],initializer = tf.constant_initializer(0.0))
            layer3 = tf.matmul(layer2,weights)+biases
        return layer3


    def train(self):#训练过程
        x = tf.placeholder(tf.float32,[None,self.INPUT_NODE],name='X-input')
        y_ = tf.placeholder(tf.float32,[None,self.OUTPUT_NODE],name='y-input')

        y= self.inference(x)
        print(y)
        #加入正则项用以限制模型的复杂度避免过拟合
        regularization = tf.add_n(tf.get_collection('losses'))

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))

        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        global_step = tf.Variable(0,trainable=False)#训练轮数
        #learning_rate = self.LEARNING_RATE_BASE
        learning_rate = tf.train.exponential_decay(
            self.LEARNING_RATE_BASE,
            global_step,
            16,
            self.LEARNING_RATE_DECAY
        )#使用学习率衰减避免模型不收敛

        loss = cross_entropy_mean+regularization  #加入正则化避免模型过拟合
        #loss = cross_entropy_mean
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #使模型持久化，将模型存储在MODEL_SAVE_PATH/MODEL_NAME中
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(self.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            validate_feed = {x:self.validate_feature,y_:self.validate_label}#验证集 与训练集无交集，用来展示神经网络的结果

            for i in range(self.TRAINING_STEPS):
                if (i+1)*self.BATCH_SIZE%1024!=0:
                    xs,ys = self.train_feature[(i*self.BATCH_SIZE)%1024:((i+1)*self.BATCH_SIZE)%1024],self.train_label[(i*self.BATCH_SIZE)%1024:((i+1)*self.BATCH_SIZE)%1024]
                else:
                    xs,ys = self.train_feature[1024-self.BATCH_SIZE:1024],self.train_label[1024-self.BATCH_SIZE:1024]
                _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x: xs, y_: ys})


                if i%100 == 0:
                    validate_acc = sess.run(accuracy,feed_dict = validate_feed)
                    print("After %d training steps,loss on training batch is %g"%(step,loss_value))
                    #saver.save(sess,os.path.join(self.MODEL_SAVE_PATH,self.MODEL_NAME),global_step=global_step)
                    print("After %d training steps, validation accuracy is %g"%(i,validate_acc))
                    saver.save(sess, os.path.join(self.MODEL_SAVE_PATH, self.MODEL_NAME), global_step=global_step)

    def predict(self,test_feature):#输入数据，使用神经网络预测标记，输出0为预测不可结合，输出1为预测可以结合，可以将输出数据输出到一个csv文件中
        with tf.Graph().as_default() as tg:
            x= tf.placeholder(tf.float32,[None,self.INPUT_NODE])
            y=self.inference(x)
            preValue = tf.argmax(y,1)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                ckpt = tf.train.get_checkpoint_state(self.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    xs = test_feature[:]
                    preValue = sess.run(preValue, feed_dict={x: xs})
                    #print(type(preValue), len(preValue))
                    # return preValue

                    preValue = 1-preValue#如果返回0表示不能结合，返回1表示能结合

                    print(preValue)
                    data1 = pd.DataFrame(preValue)
                    data1.to_csv('predict_data.csv')
                    print("finish!!!")

                else:
                    print("No checkpoint file found")
                    return -1

def normalization(feature):
    fn = []
    for list in feature:
        fn.append([(i-min(list))/(max(list)-min(list)) for i in list])
    return fn



if __name__ == "__main__":
    label = np.load("label_balance.npy").tolist()
    feature = np.load("feature_balance.npy").tolist()
    feature_nm=normalization(feature)#特征归一化
    train_feature,validate_feature,train_label,validate_label = train_test_split(feature_nm,label,train_size=1024/1374,random_state=42)
    network = NN(train_feature,train_label,validate_feature,validate_label)
    tf.reset_default_graph()
    network.train()
    #network.predict([train_feature[3]])
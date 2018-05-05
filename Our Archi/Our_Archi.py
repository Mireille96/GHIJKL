import tensorflow as tf
import glob
import os
import cv2
import numpy as np

#from PIL import Image
#import matplotlib.pyplot as plt

RGBimage_list = []
Grayimage_list = []
#Temp_RGBimage_list = []
#Temp_Grayimage_list = []
RsizeOriginalW=400
RsizeOriginalH=400

x = tf.placeholder(tf.float32, [None, RsizeOriginalW*RsizeOriginalH])
y_ = tf.placeholder(tf.float32, [None,RsizeOriginalW,RsizeOriginalH,3])
keep_prob=tf.placeholder(tf.float32)


def read_images_in_folder():
    imagePath = glob.glob('/home/meraymedhat/DataSet 400x400/*.jpg')  
    RGBimage_stack = np.array(np.array([np.array(cv2.imread(imagePath[i])) for i in range(len(imagePath))]))
    grayimage_stack = np.array(np.array([np.array(cv2.imread(imagePath[i],cv2.IMREAD_GRAYSCALE)) for i in range(len(imagePath))]))
    return RGBimage_stack,grayimage_stack

def resize_image(image_stack, hResize, wResize): # Image resizing is performed here
    im_resized_stack = np.array( [np.array(cv2.resize(img, (hResize, wResize), interpolation=cv2.INTER_CUBIC)) for img in image_stack]) 
    return im_resized_stack

def Read_GreyTest_Images_in_folder():
    imagePath = glob.glob('/home/meraymedhat/Test/*.jpg')
    grayTestimage_stack = np.array( [np.array(cv2.imread(imagePath[i],cv2.IMREAD_GRAYSCALE)) for i in range(len(imagePath))] )
    print("Read Grey Test Images")
    return grayTestimage_stack
 
def Read_RGBLabelsTest_Images_in_folder():
    imagePath = glob.glob('/home/meraymedhat/Test/*.jpg')
    RGBTestimage_stack = np.array( [np.array(cv2.imread(imagePath[i])) for i in range(len(imagePath))] )
    return RGBTestimage_stack


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x,W):
    # output size= ((input size - no of filters ) / strides )+ 1 = ( (200-1)/1) )+1=200
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def MaxPool2d(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':weight_variable([5,5,1,32]),
                      'W_conv2':weight_variable([5,5,32,64]),
                      'W_conv3':weight_variable([5,5,64,128]),
                      'W_conv4':weight_variable([5,5,128,256]),
                      'W_conv5':weight_variable([5,5,256,3]) }
    biases = {'b_conv1':bias_variable([32]),
                      'b_conv2':bias_variable([64]),
                      'b_conv3':bias_variable([128]),
                      'b_conv4':bias_variable([256]),
                      'b_conv5':bias_variable([3])}
    x=(tf.reshape(x,shape=[-1,RsizeOriginalH,RsizeOriginalW,1])-128)/128
    conv1=tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
    conv2=tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv3=tf.nn.relu(conv2d(conv2,weights['W_conv3']) + biases['b_conv3'])
    conv3=MaxPool2d(conv3)
    conv4=tf.nn.relu(conv2d(conv3,weights['W_conv4']) + biases['b_conv4'])
    conv5=(conv2d(conv4,weights['W_conv5']) + biases['b_conv5'])
    output=(tf.nn.sigmoid(conv5))*255
    output = tf.image.resize_nearest_neighbor(output,[RsizeOriginalH,RsizeOriginalW])
    return output

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)    # lw el condition true hikml lw l22 hitl3 assertion error
    p = np.random.permutation(len(a))
    return a[p], b[p]

RGBimage_list, Grayimage_list = read_images_in_folder()
Grayimage_list =Grayimage_list.reshape([-1,RsizeOriginalW*RsizeOriginalH])
#Grayimage_list = resize_image(Grayimage_list, RsizeOriginalH, RsizeOriginalW).reshape([-1,RsizeOriginalW*RsizeOriginalH])
#RGBimage_list = resize_image(RGBimage_list, RsizeOriginalH, RsizeOriginalW)
unison_shuffled_copies(RGBimage_list,Grayimage_list)
prediction = convolutional_neural_network(x)
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.subtract(prediction,y_) ** 2) ** 0.5)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



def testNN():
    saver = tf.train.Saver()
    TempTestGray = Read_GreyTest_Images_in_folder().reshape([-1,RsizeOriginalW*RsizeOriginalH])
    #TempTestGray = resize_image(TempTestGray, RsizeOriginalH, RsizeOriginalW).reshape([-1,RsizeOriginalW*RsizeOriginalH])
    #c = TempTestGray[0]
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, "/home/meraymedhat/FiveWeightsModel/model/model.ckpt")   
        yy = sess.run(prediction,feed_dict = {x:TempTestGray,keep_prob: 0.5})
        i=0
        for y in yy:
          Image = TempTestGray[i]
          i=i+1
          res = np.floor(y)
          directory='/home/meraymedhat/FiveWeightsModel/Results'
          if not os.path.exists(directory):
            os.makedirs(directory)
          cv2.imwrite('/home/meraymedhat/FiveWeightsModel/Results/res.jpg',res);
          cv2.imwrite('/home/meraymedhat/FiveWeightsModel/Results/input.jpg',Image.reshape([-1,RsizeOriginalH,RsizeOriginalW])[0])
          img = cv2.imread("/home/meraymedhat/FiveWeightsModel/Results/res.jpg")
          cv2.startWindowThread()
          cv2.namedWindow("Colored Image")
          cv2.imshow("Colored Image", img)
          cv2.waitKey(0)

def trainNN():
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    directory='/home/meraymedhat/FiveWeightsModel/model'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "M:/Automatic Colorization/FiveWeightsModel/model/model.ckpt")
        for epoch in range(15000):
            epoch_loss = 0
            for i in range(int(6208/20)):
                print("Batch Num ",i + 1)
                a, c = sess.run([train_step,cross_entropy],feed_dict={x: Grayimage_list[i*20:(i+1)*20], y_: RGBimage_list[i*20:(i+1)*20], keep_prob: 0.5})
                epoch_loss +=c
                save_path = saver.save(sess, "/home/meraymedhat/FiveWeightsModel/model/model.ckpt")
            print("epoch: ",epoch + 1, ",Loss: ",epoch_loss)
            

trainNN()
#testNN()


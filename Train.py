import tensorflow as tf
import numpy as np
import os
import argparse
from NextBatch import Dataset
from vgg16 import *

parser = argparse.ArgumentParser(description='Tensorflow VGG16')
parser.add_argument('--epoch_size', type=int, default=1000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=9, help='Set batch size')
parser.add_argument('--log_dir', type=str, default='logs', help='log dir')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--optimization', type=str, default='adam', help='learning rate')

#https://www.cs.toronto.edu/~frossard/post/vgg16/

def main():
    # argment setting
    global args
    args = parser.parse_args()
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    log_dir = args.log_dir
    learning_rata = args.learning_rate
    optimization_type = args.optimization


    # data load
    train_file = './data/train_data.txt'
    test_file = './data/test_data.txt'

    Data = Dataset(train_file, test_file)
    n_batch = int(Data.train_size/batch_size)

    # input data & label
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    Y = tf.placeholder(tf.float32, [None, 2])


    # tensorflow define
    global_step = tf.Variable(0, trainable=False, name='global_step')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    with tf.Session(config=config) as sess:
        # ckpt = tf.train.get_checkpoint_state('./model')
        # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        #     sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir + os.path.sep, graph=sess.graph)


        # VGG model find-tunning
        vgg = vgg16(X, 'vgg16_weights.npz', sess)


        # loss define
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=vgg.probs, labels=Y))
        tf.summary.scalar("cross entropy loss", loss)


        # prediction accuracy
        prediction = tf.equal(tf.argmax(vgg.probs, 1), tf.argmax(Y, 1))
        with tf.name_scope("accuracy"):
            train_accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            valid_accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        tf.summary.scalar('train_accuracy', train_accuracy)
        tf.summary.scalar('valid_accuracy', valid_accuracy)


        # optimization
        vars = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS, scope='CNN')
        if optimization_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rata).minimize(loss)
        elif optimization_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rata).minimize(loss)

        # learning
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        for epoch in range(epoch_size):

            x_mini_batch, y_mini_batch = Data.next_batch(batch_size, 'train')
            _, avg_loss, logit, summary = sess.run([optimizer, loss, vgg.probs, merged], feed_dict={X:x_mini_batch, Y:y_mini_batch})
            train_writer.add_summary(summary, global_step=sess.run(global_step))

            if epoch % 100 == 0:
                print "Epoch:", epoch, "loss=", avg_loss
                # accuracy check
                ACC = 0
                for i in range(int(Data.test_size/batch_size)):
                    x_test_minibatch, y_test_minibatch = Data.next_batch(batch_size, 'train')
                    accuracy = sess.run(valid_accuracy, feed_dict={X:x_test_minibatch, Y:y_mini_batch})
                    ACC += accuracy
                ACC /= Data.test_size
                print "average accuracy=", ACC

        print "Optimization Finish"
        saver.save(sess, './model/model.ckpt')

        train_writer.close()


        # img1 = imread('/mnt/pose/tiny/21539.jpg', mode='RGB')
        # img1 = imresize(img1, (224, 224))
        #
        # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        # preds = (np.argsort(prob)[::-1])[0:5]
        # for p in preds:
        #     print class_names[p], prob[p]




if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
import random
import os
from pre_processing import visualize_images, create_prediction_image


# 3 conv layers , 1 dense layer in between and 1 output layer
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, img_size, img_size, 1])
    # stride = sliding, [1,?,?,1] dont change the ones
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, biases['bc1'])
    # relu can be used on convlayers as well
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # maxpooling reduces image to half resulition because 2*2

    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, biases['bc2'])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.nn.conv2d(conv2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, biases['bc3'])
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc1 = tf.reshape(conv3,
                     [-1, weights['wd1'].get_shape().as_list()[0]])  # flatten conv to dense -> 4D tensor to 2D tensor
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


def chunks(images, labels, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(images), n):
        # Create an index range for l of n items:
        yield (images[i:i + n], labels[i:i + n])


def split_training_and_testing(images, labels, training_percantage=0.8):
    set_length = len(images)
    training_length = int(set_length * training_percantage)
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []

    benign_images = []
    benign_labels = []
    malignant_images = []
    malignant_labels = []

    for i in range(len(images)):
        if np.array_equal([1, 0], labels[i]):
            benign_images.append(images[i])
            benign_labels.append(labels[i])

        if np.array_equal([0, 1], labels[i]):
            malignant_images.append(images[i])
            malignant_labels.append(labels[i])

    training_images = np.array(benign_images[:training_length // 2] + malignant_images[:training_length // 2])
    testing_images = np.array(benign_images[training_length // 2:] + malignant_images[training_length // 2:])
    training_labels = np.array(benign_labels[:training_length // 2] + malignant_labels[:training_length // 2])
    testing_labels = np.array(benign_labels[training_length // 2:] + malignant_labels[training_length // 2:])
    return training_images, training_labels, testing_images, testing_labels


def randomize_two_lists(images, labels):
    # shuffle lists but maintain order
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def display_wrong_images(images, labels, predictions, image_count=10):
    images_to_display = []
    labels_to_display = []

    counter = 0
    for idx, _ in enumerate(predictions):
        if np.array_equal(predictions[idx], labels[idx]):
            images_to_display.append(images[idx])
            labels_to_display.append(labels[idx])
            counter += 1
        if counter >= 10:
            break

    visualize_images(images_to_display, labels_to_display)


def check_positive_negative_count(labels):
    positive = 0
    negative = 0
    for elem in labels:
        if np.array_equal([1, 0], elem):
            negative += 1
        else:
            positive += 1
    print(positive, negative)


if __name__ == '__main__':

    learning_rate = 0.001
    batch_size = 64
    img_size = 128
    num_input = img_size * img_size  # input is 1D instead of 2D like an image
    num_classes = 2
    epochs = 25
    dropout = 0.9  # prob to keep =units (model tends to overfit, control this with this param)
    logs_path = 'logs/'

    # network architecture params
    output_1 = 128
    output_2 = 256
    output_3 = 512
    output_4 = 2048
    conv_size = 3  # dont pick to big, it slows the network down but does not provide big benefits

    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout

    #CNN settings
    weights = {
        # 5*5 conv with 1 input 32 outputs
        'wc1': tf.Variable(tf.random_normal([conv_size, conv_size, 1, output_1])),  # image res 80*80 just as example
        'wc2': tf.Variable(tf.random_normal([conv_size, conv_size, output_1, output_2])),  # image res 40*40
        'wc3': tf.Variable(tf.random_normal([conv_size, conv_size, output_2, output_3])),  # image res 20*20
        'wd1': tf.Variable(tf.random_normal([img_size // 8 * img_size // 8 * output_3, output_4])),  # image res 10*10
        'out': tf.Variable(tf.random_normal([output_4, num_classes]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([output_1])),
        'bc2': tf.Variable(tf.random_normal([output_2])),
        'bc3': tf.Variable(tf.random_normal([output_3])),
        'bd1': tf.Variable(tf.random_normal([output_4])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    tf.summary.scalar("loss", loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged_summary_op = tf.summary.merge_all()

    images = np.load('images.npy')
    images = np.divide(images, 255)  # normalize
    labels = np.load('labels.npy')

    images, labels = randomize_two_lists(images, labels)
    training_images, training_labels, testing_images, testing_labels = split_training_and_testing(images, labels,
                                                                                                  training_percantage=0.9)
    tmp_testing_images = testing_images.reshape(-1, img_size * img_size)

    model_name = 'model_{}_{}_{}_{}_{}_{}_{}_{}'.format(learning_rate, output_1, output_2, output_3, output_4,
                                                        conv_size, dropout,
                                                        img_size
                                                        )

    model_path = '{}/models/{}'.format(os.path.dirname(os.path.abspath(__file__)), model_name + '.ckpt')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    mode = 'TRAIN'
    with tf.Session() as sess:
        sess.run(init)

        try:
            saver.restore(sess, model_path)
            print('Loaded model')
        except ValueError as e:
            print('{} \nStarting from 0!'.format(e))

        if mode == 'PREDICT':
            image_to_predict = create_prediction_image('transfer_learning_model/test_images/test3.JPG')
            image_to_predict = image_to_predict.reshape(-1, img_size * img_size)
            logit, pred = sess.run([logits, prediction], feed_dict={X: image_to_predict, keep_prob: 1.0})
            print(logit, pred[0])

        if mode == 'MULTI_PREDICT':
            images_folder = 'transfer_learning_model/test_images/'
            images_to_predict = []
            counter = 100
            for image_to_predict in os.listdir(images_folder):
                if counter <= 0:
                    break
                if '.jpg' in image_to_predict or '.JPG' in image_to_predict:
                    images_to_predict.append(create_prediction_image(images_folder + image_to_predict))
                    # print(image_to_predict)
                    counter -= 1
            images_to_predict = np.array(images_to_predict).reshape(-1, img_size * img_size)
            pred = sess.run(prediction, feed_dict={X: images_to_predict, keep_prob: 1.0})
            print(pred)

        else:

            summary_writer = tf.summary.FileWriter(logs_path + model_name + '/', graph=tf.get_default_graph(),
                                                   flush_secs=60)
            for epoch in range(epochs):
                training_images, training_labels = randomize_two_lists(training_images, training_labels)
                for batch_x, batch_y in chunks(training_images, training_labels, batch_size):
                    batch_x = batch_x.reshape(-1, img_size * img_size)
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

                # get in sample accuracy
                tmp_training_images = training_images.reshape(-1, img_size * img_size)
                in_sample_loss, in_sample_acc = sess.run([loss_op, accuracy],
                                                         feed_dict={X: tmp_training_images[0:batch_size], Y: training_labels[0:batch_size],
                                                                    keep_prob: 1.0})

                # pred = sess.run(prediction,feed_dict={X: tmp_testing_images, keep_prob: 1.0})
                # get out of sample accuracy
                #dont let to big data for testing or training
                loss, acc, summary, pred = sess.run([loss_op, accuracy, merged_summary_op, prediction],
                                                    feed_dict={X: tmp_testing_images[0:batch_size], Y: testing_labels[0:batch_size],
                                                               keep_prob: 1.0})
                # display_wrong_images(testing_images,testing_labels,pred)
                # break
                summary_writer.add_summary(summary, epoch)
                print("Completed {}".format(epoch),
                      "Minibatch Loss= " + "{:.1f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(
                          acc) + ", In sample loss= {:.1f} , in sample acc= {:.3f}".format(in_sample_loss,
                                                                                           in_sample_acc))

                save_path = saver.save(sess, model_path)
                # print("Model saved in file: %s" % save_path)

            print("Optimization Finished!")

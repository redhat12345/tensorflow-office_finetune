import os
import sys
import math
import datetime
import logging

import numpy as np
import tensorflow as tf

from model import AlexNetModel
from preprocessor import BatchPreprocessor

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 500, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1',
                           'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string(
    'multi_scale', '256,257',
    'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network input size')
tf.app.flags.DEFINE_string('train_root_dir', '/home/wogong/Models/tf-office_finetune/',
                           'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')

NUM_CLASSES = 31
TRAINING_FILE = '/home/wogong/Datasets/office/amazon_list.txt'
VAL_FILE = '/home/wogong/Datasets/office/webcam_list.txt'
FLAGS = tf.app.flags.FLAGS
MAX_STEP = 10000
MODEL_NAME = 'amazon_to_webcam'


def decay(start_rate, epoch, num_epochs):
    return start_rate / pow(1 + 0.001 * epoch, 0.75)


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('ft_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir):
        os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir):
        os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir):
        os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('model name: {}\n'.format(MODEL_NAME))
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], 'x')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'y')
    decay_learning_rate = tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel(num_classes=NUM_CLASSES, dropout_keep_prob=dropout_keep_prob)
    loss = model.get_loss(x, y)
    train_op = model.optimize(decay_learning_rate, train_layers)

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the FileWriter
    train_writer = tf.summary.FileWriter(tensorboard_dir + '/train')
    test_writer = tf.summary.FileWriter(tensorboard_dir + '/val')

    # Summaries
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    train_preprocessor = BatchPreprocessor(
        dataset_file_path=TRAINING_FILE,
        num_classes=NUM_CLASSES,
        output_size=[227, 227],
        horizontal_flip=True,
        shuffle=True,
        multi_scale=multi_scale)
    val_preprocessor = BatchPreprocessor(
        dataset_file_path=VAL_FILE,
        num_classes=NUM_CLASSES,
        output_size=[227, 227],
        multi_scale=multi_scale,
        istraining=False)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        logger.info("Start training...")
        logger.info("tensorboard --logdir {}".format(tensorboard_dir))
        global_step = 0

        for epoch in range(FLAGS.num_epochs):
            # Reset the dataset pointers
            train_preprocessor.reset_pointer()

            step = 1

            while step < train_batches_per_epoch:
                global_step += 1
                rate = decay(FLAGS.learning_rate, global_step, MAX_STEP)

                batch_xs, batch_ys = train_preprocessor.next_batch(FLAGS.batch_size)
                summary, loss, _ = sess.run(
                    [merged, model.loss, train_op],
                    feed_dict={
                        x: batch_xs,
                        decay_learning_rate: rate,
                        y: batch_ys,
                        dropout_keep_prob: 0.5
                    })
                train_writer.add_summary(summary, global_step)

                step += 1

                if global_step % 10 == 0:
                    logger.info("epoch {}, step {}, loss {:.6f}".format(epoch, global_step, loss))
                    test_acc = 0.
                    test_count = 0

                    for _ in range((len(val_preprocessor.labels))):
                        batch_tx, batch_ty = val_preprocessor.next_batch(1)
                        acc = sess.run(correct, feed_dict={x: batch_tx, y: batch_ty, dropout_keep_prob: 1.})
                        test_acc += acc
                        test_count += 1
                    test_acc_ = test_acc / test_count
                    s = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=test_acc_)])
                    test_writer.add_summary(s, global_step)
                    logger.info("test accuracy: {:.4f}, {}/{}".format(test_acc_, test_acc, test_count))

                    # Reset the dataset pointers
                    val_preprocessor.reset_pointer()

                #save checkpoint of the model
                if global_step % 1000 == 0 and global_step > 0:
                    logger.info("saving checkpoint of model")
                    checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch' + str(global_step) + '.ckpt')
                    saver.save(sess, checkpoint_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s[%(module)s] %(levelname)s: %(message)s')
    logger = logging.getLogger("tf-finetune_office")

    tf.app.run()

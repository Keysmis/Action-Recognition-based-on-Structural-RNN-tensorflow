import tensorflow as tf
import numpy as np
import sys, random, math
import os, time
import cPickle as pickle
#from SRNN import SRNN_model
from SRNN_without_getgates import SRNN_model
from SRNN import LSTM_model
import read_data as rd

"""
import action_recognition_srnn as ma
tr,va,te = ma.synthetic_data(4,2,0,5,5)
b = ma.get_random_batch(tr,4,2)
"""

NUM_ACTIVITIES = 21

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_boolean("l2_reg", True, "Adds a L2 regularization term")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-6,
                          "Epsilon used for numerical stability in Adam optimizer.")
tf.app.flags.DEFINE_float("reg_factor", 0.6,
                          "Lambda for l2 regulariation.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_boolean("clip", False, "whether or not to clip gradients")
tf.app.flags.DEFINE_integer("batch_size", 12,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_units", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("attention_lstm_num_units", 128, "Number units in the attention LSTM.")
tf.app.flags.DEFINE_integer("attention_num_hidden_fc1", 128, "Number neurons in the attention hidden fc layer.")
tf.app.flags.DEFINE_integer("attention_placement", 0, "Where to put the attention")
tf.app.flags.DEFINE_integer("num_activities", NUM_ACTIVITIES, "Number of decoders, i.e. number of context chords")
tf.app.flags.DEFINE_integer("num_frames", 12, "Number of frames in each example.")
tf.app.flags.DEFINE_integer("num_joints", 15, "Number of frames in each example.")
tf.app.flags.DEFINE_integer("num_temp_features", 3, "Number of frames in each example.")
tf.app.flags.DEFINE_integer("num_st_features", 1, "Number of frames in each example.")
tf.app.flags.DEFINE_string("data", "JHMDB", "Data file name")
tf.app.flags.DEFINE_string("data_pickle", None, "optional pickle file containing the data ")
tf.app.flags.DEFINE_boolean("normalized", True, "Normalized raw joint positionsn")
tf.app.flags.DEFINE_boolean("srnn", True, "Whether to build a SRNN model or simple LSTM")
tf.app.flags.DEFINE_boolean("GD", False, "Uses Gradient Descent with adaptive learning rate")
tf.app.flags.DEFINE_string("train_dir", "models", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "logs", "Training directory.")
tf.app.flags.DEFINE_string("gpu", "/gpu:0", "GPU to run ")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_valid_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_epochs", 100,
                            "Maximium number of epochs for trainig.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 5,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("test_model", False,
                            "Evaluate an existing model on test data")

FLAGS = tf.app.flags.FLAGS





def extract_features(dir_name='/home/wpc/master-thesis-master/srnn-copy/data/JHMDB/joint_positions',
                     sub_activities=False, data='JHMDB', num_frames=FLAGS.num_frames, normalized=FLAGS.normalized,
                     cross_subject=False):

    # if not get_train_data ^ get_valid_data ^ get_test_data or get_train_data & get_valid_data & get_test_data:
    #     raise ValueError("Only one of training_data, valid_data and test_data must be True")

    if sub_activities:
        num_activities = 12
    else:
        num_activities = 21

    if data == 'MSR':
        dir_name = '/home/wpc/master-thesis-master/srnn-copy/MSRAction3DSkeleton(20joints)'
        train_data, train_num_video, valid_data, valid_num_video, max, min = rd.get_pos_imgsMRS_new(dir_name,
                                                                                                subset_joints=True)
        num_activities = 20
        num_joints = 20

    elif data == 'JHMDB':
        train_data, train_num_video, valid_data, valid_num_video = rd.get_pos_imgsJHMDB(dir_name, sub_activities=sub_activities, normalized=normalized,subset_joints=True)
        num_joints=15

    if data == 'JHMDB':
        # train_dataset = rd.extract_features(train_data, train_num_video, num_activities,
        #                                         num_considered_frames=num_frames, JHMDB=True)
        # valid_dataset = rd.extract_features(valid_data, valid_num_video, num_activities,
        #                                         num_considered_frames=num_frames, JHMDB=True)
        train_dataset = rd.extract_features(train_data, train_num_video, num_activities,
                                                num_considered_frames=num_frames,JHMDB=True)
        valid_dataset = rd.extract_features(valid_data, valid_num_video, num_activities,
                                                num_considered_frames=num_frames, JHMDB=True)
    else:
        train_dataset = rd.extract_features(train_data, train_num_video, num_activities,
                                                num_considered_frames=num_frames)
        valid_dataset = rd.extract_features(valid_data, valid_num_video, num_activities,
                                                num_considered_frames=num_frames)
    return train_dataset, valid_dataset




def split_data(data, validation_proportion):
    data_size = len(data[2])
    random.seed(14)
    valid_split_size = int(round(data_size * validation_proportion))
    valid_split = np.array({})
    train_split = np.array({})
    valid_idxs = random.sample(xrange(0, data_size), valid_split_size)
    train_idxs = range(data_size)
    for a in valid_idxs:
        train_idxs.remove(a)

    valid_dic = np.array([{}, {}, [], {}])
    train_dic = np.array([{}, {}, [], {}])

    for i in xrange(len(data)-1):
        dic = data[i]
        if i < 2:
            valid_sub_dic = {}
            train_sub_dic = {}
            for key in dic:
                valid_sub_dic[key] = dic[key][valid_idxs]
                train_sub_dic[key] = dic[key][train_idxs]
            valid_dic[i] = valid_sub_dic
            train_dic[i] = train_sub_dic
        else:
            valid_dic[i] = dic[valid_idxs]
            train_dic[i] = dic[train_idxs]

    valid_info = {}
    train_info = {}
    for key in valid_idxs:
        valid_info[key]=data[3][key]
    for key in train_idxs:
        train_info[key]=data[3][key]

    valid_dic[3] = valid_info
    train_dic[3] = train_info

    valid_split = np.append(valid_split, valid_dic)
    train_split = np.append(train_split, train_dic)
    valid_split = np.delete(valid_split, 0)
    train_split = np.delete(train_split, 0)
    return valid_split, train_split


def get_random_batch(data, batch_size):
    data_size = len(data[2])
    batch = np.array({})
    idxs = random.sample(xrange(0, data_size - 1), batch_size)
    info_idxs = np.array(data[3].keys())[idxs]
    batch_dic = np.array([{}, {}, []])
    for i in xrange(len(data) - 1):
        dic = data[i]
        if i < 2:
            batch_sub_dic = {}
            for key in dic:
                batch_sub_dic[key] = dic[key][idxs]
            batch_dic[i] = batch_sub_dic
        else:
            batch_dic[i] = dic[idxs]
    batch = np.append(batch, batch_dic)
    batch = np.delete(batch, 0)
    batch_info = {}
    for key in info_idxs:
        batch_info[key] = data[3][key]
    return batch, batch_info

def big_step(self, session, all_data, batch_size):
    size = len(all_data[2])

    missclassified = 0.0

    _, loss, outputs = self.step(session, all_data[0], all_data[1], all_data[2], True, batch_size)
    classes = np.argmax(outputs, 1)
    for d in range(size):
        if all_data[2][d][classes[d]] != 1:
            missclassified += 1
    error = missclassified / (size)
    return loss, error


def create_SRNN_model(session, forward_only, log_dir, result_file=None, same_param=False):
    model = SRNN_model(FLAGS.num_activities, FLAGS.num_frames, FLAGS.num_temp_features, FLAGS.num_st_features,
                       FLAGS.num_units, FLAGS.max_gradient_norm, FLAGS.learning_rate,
                       FLAGS.learning_rate_decay_factor, FLAGS.adam_epsilon, FLAGS.GD, forward_only=forward_only,
                       l2_regularization=FLAGS.l2_reg, weight_decay=FLAGS.reg_factor, log_dir=log_dir)

    if not same_param:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
            if result_file is not None:
                result_file.write("Continue training existing model! ")
            print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
            model.saver.restore(session, checkpoint.model_checkpoint_path)
            session.run(model.learning_rate.assign(float(FLAGS.learning_rate)))
            model.adam_epsilon = FLAGS.adam_epsilon
        else:
            print("Created model with fresh parameters.")
            if FLAGS.log_dir is not None:
                model.train_writer.add_graph(session.graph)
            session.run(tf.global_variables_initializer())
    return model


def main(_):

    tr, te = extract_features(num_frames=FLAGS.num_frames, data='JHMDB')
    dic = {'train': tr, 'test': te}

    test_data = dic['test']

    with tf.device(FLAGS.gpu):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.variable_scope("Model", reuse=None):  # , initializer=initializer):
                print("Creating SRNN model ")
                print("with %d units and %d bach-size." % (FLAGS.num_units, FLAGS.batch_size))


                if FLAGS.srnn:
                    model = create_SRNN_model(sess, False, FLAGS.log_dir)

                checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
                model.saver.restore(sess, checkpoint.model_checkpoint_path)

                test_loss, test_error = model.big_step(sess, test_data, len(test_data[2]))
                print("  total test  loss %.4f, total error  %.4f " % (test_loss, test_error))



if __name__ == '__main__':
    tf.app.run()

import tensorflow as tf

# from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import array_ops
# from tensorflow.python.util import nest
#
# from tensorflow.python.framework import ops
#
# from tensorflow.python.ops.math_ops import sigmoid
# from tensorflow.python.ops.math_ops import tanh
import numpy as np
from keras.layers import Dense, Dropout,Convolution2D
#from attention import attention
#from keras import regularizers
import gc

class SRNN_model(object):
    # SharedRNN(shared_layers, human_layers, object_layers, softmax_loss, trY_1, trY_2, 1e-3)

    def __init__(self, num_classes, num_frames, num_temp_features, num_st_features, num_units, max_gradient_norm,
                 learning_rate,
                 learning_rate_decay_factor, adam_epsilon, GD, attention_lstm_num_units,attention_num_hidden_fc1,forward_only=False, l2_regularization=False,
                 weight_decay=0, log_dir=None):
        """"
        Create S-RNN model
        edgeRNNs: dictionary with keys as RNN name and value is a list of layers
        nodeRNNs: dictionary with keys as RNN name and value is a list of layers
        nodeToEdgeConnections: dictionary with keys as nodeRNNs name and value is another
                dictionary whose keys are edgeRNNs the nodeRNN is connected to and value is a list
                of size-2 which indicate the features to choose from the unConcatenateLayer
        edgeListComplete:
        cost:
        nodeLabels:
        learning_rate:
        clipnorm:
        update_type:
        weight_decay:

        return:
        """
        self.save_summaries = log_dir is not None
        if self.save_summaries:
            print('Writing summaries for Tensorboard')
        num_layers = 1
        self.num_classes = num_classes
        self.num_temp_features = num_temp_features
        self.num_st_features = num_st_features
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        #self.learning_rate = float(learning_rate)
        #self.learning_rate_decay = tf.Variable(float(learning_rate), trainable=False)
        # self.learning_rate_decay_op = self.learning_rate.assign(
        #      self.learning_rate * 0.1)
        self.max_grad_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.adam_epsilon = adam_epsilon
        self.GD = GD
        self.weight_decay = weight_decay
        # self.previous_eval_loss = []
        # self.best_val_loss = np.inf
        # self.strikes = tf.Variable(0, trainable=False)
        self.temp_features_names =['face-face', 'neck-neck', 'belly-belly', 'rightShoulder-rightShoulder',
                                'leftShoulder-leftShoulder',
                                'rightElbow-rightElbow', 'leftElbow-leftElbow', 'rightArm-rightArm', 'leftArm-leftArm',
                                'rightHip-rightHip', 'leftHip-leftHip',
                                'rightKnee-rightKnee', 'leftKnee-leftKnee', 'rightLeg-rightLeg', 'leftLeg-leftLeg']

        self.st_features_names =['face-neck', 'face-belly', 'face-rightShoulder', 'face-leftShoulder', 'face-rightElbow',
                              'face-leftElbow', 'face-rightArm', 'face-leftArm', 'face-rightHip', 'face-leftHip',
                              'face-rightKnee', 'face-leftKnee', 'face-rightLeg', 'face-leftLeg',

                              'neck-belly', 'neck-rightShoulder', 'neck-leftShoulder', 'neck-rightElbow',
                              'neck-leftElbow',
                              'neck-rightArm', 'neck-leftArm', 'neck-rightHip', 'neck-leftHip', 'neck-rightKnee',
                              'neck-leftKnee',
                              'neck-rightLeg', 'neck-leftLeg',

                              'belly-rightShoulder', 'belly-leftShoulder', 'belly-rightElbow',
                              'belly-leftElbow', 'belly-rightArm', 'belly-leftArm',
                              'belly-rightHip', 'belly-leftHip',
                              'belly-rightKnee', 'belly-leftKnee',
                              'belly-rightLeg', 'belly-leftLeg',

                              'rightShoulder-leftShoulder', 'rightShoulder-rightElbow', 'rightShoulder-leftElbow',
                              'rightShoulder-rightArm', 'rightShoulder-leftArm', 'rightShoulder-rightHip',
                              'rightShoulder-leftHip',
                              'rightShoulder-rightKnee', 'rightShoulder-leftKnee', 'rightShoulder-rightLeg',
                              'rightShoulder-leftLeg',

                              'leftShoulder-rightElbow', 'leftShoulder-leftElbow',
                              'leftShoulder-rightArm', 'leftShoulder-leftArm', 'leftShoulder-rightHip',
                              'leftShoulder-leftHip',
                              'leftShoulder-rightKnee', 'leftShoulder-leftKnee', 'leftShoulder-rightLeg',
                              'leftShoulder-leftLeg',

                              'rightElbow-leftElbow', 'rightElbow-rightArm', 'rightElbow-leftArm',
                              'rightElbow-rightHip', 'rightElbow-leftHip',
                              'rightElbow-rightKnee', 'rightElbow-leftKnee', 'rightElbow-rightLeg',
                              'rightElbow-leftLeg',

                              'leftElbow-rightArm', 'leftElbow-leftArm', 'leftElbow-rightHip', 'leftElbow-leftHip',
                              'leftElbow-rightKnee',
                              'leftElbow-leftKnee', 'leftElbow-rightLeg', 'leftElbow-leftLeg',

                              'rightArm-leftArm', 'rightArm-rightHip', 'rightArm-leftHip', 'rightArm-rightKnee',
                              'rightArm-leftKnee',
                              'rightArm-rightLeg', 'rightArm-leftLeg',

                              'leftArm-rightHip', 'leftArm-leftHip', 'leftArm-rightKnee', 'leftArm-leftKnee',
                              'leftArm-rightLeg', 'leftArm-leftLeg',

                              'rightHip-leftHip', 'rightHip-rightKnee', 'rightHip-leftKnee', 'rightHip-rightLeg',
                              'rightHip-leftLeg',

                              'leftHip-rightKnee', 'leftHip-leftKnee', 'leftHip-rightLeg', 'leftHip-leftLeg',

                              'rightKnee-leftKnee', 'rightKnee-rightLeg', 'rightKnee-leftLeg',

                              'leftKnee-rightLeg', 'leftKnee-leftLeg',

                              'rightLeg-leftLeg']

        #nodes_names = {'face','neck','belly','right-shoulder','left-shoulder','right-elbow','left-elbow','right-arm','left-arm','right-hip','left-hip','right-knee','left-knee','right-leg', 'left-leg'}
        nodes_names = {'face',  'belly', 'right-elbow', 'left-elbow',
                       'right-arm', 'left-arm', 'right-knee', 'left-knee', 'right-leg',
                       'left-leg'}
        edgesRNN = {}
        nodesRNN = {}
        states = {}
        infos = {}
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        #self.batch_size = 36
        self.inputs = {}
        self.targets = tf.placeholder(tf.float32, shape=(None, num_classes), name='targets')
        for temp_feat in self.temp_features_names:
            infos[temp_feat] = {'input_gates': [], 'forget_gates': [], 'modulated_input_gates': [], 'output_gates': [],
                                'activations': [], 'state_c': [], 'state_m': []}
            self.inputs[temp_feat] = tf.placeholder(tf.float32, shape=(None, num_frames, self.num_temp_features),
                                                    name=temp_feat)
            if num_layers == 1:
                edgesRNN[temp_feat] = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                      activation=tf.nn.softsign)

            else:
                cells = []
                for _ in range(num_layers):
                    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                                      activation=tf.nn.softsign))
                    cells.append(cell)
                edgesRNN[temp_feat] = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            states[temp_feat] = edgesRNN[temp_feat].zero_state(self.batch_size, dtype=tf.float32)

        for st_feat in self.st_features_names:
            infos[st_feat] = {'input_gates': [], 'forget_gates': [], 'modulated_input_gates': [], 'output_gates': [],
                              'activations': [], 'state_c': [], 'state_m': []}
            self.inputs[st_feat] = tf.placeholder(tf.float32, shape=(None, num_frames, self.num_st_features),
                                                  name=st_feat)
            if num_layers == 1:
                 edgesRNN[st_feat] = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                      activation=tf.nn.softsign)
            else:
                cells = []
                for _ in range(num_layers):
                    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                                      activation=tf.nn.softsign))
                    cells.append(cell)
                edgesRNN[st_feat] = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            states[st_feat] = edgesRNN[st_feat].zero_state(self.batch_size, tf.float32)

        for node in nodes_names:
            infos[node] = {'input_gates': [], 'forget_gates': [], 'modulated_input_gates': [], 'output_gates': [],
                           'activations': [], 'state_c': [], 'state_m': []}
            self.inputs[node] = tf.placeholder(tf.float32, shape=(None, num_frames, None), name=node)
            if num_layers == 1:
                nodesRNN[node] = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                       activation=tf.nn.softsign)
            else:
                cells = []
                for _ in range(num_layers):
                    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                                      activation=tf.nn.softsign))
                    cells.append(cell)
                nodesRNN[node] = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            states[node] = nodesRNN[node].zero_state(self.batch_size, tf.float32)

        wholeRNN = tf.contrib.rnn.BasicLSTMCell(num_units*10, state_is_tuple=True,
                                                      activation=tf.nn.softsign)
        states_whole = wholeRNN.zero_state(self.batch_size, tf.float32)


        attention_out_size = 1
        attention_in_size = num_units

        sp_attention_LSTM = tf.contrib.rnn.BasicLSTMCell(attention_lstm_num_units, state_is_tuple=True)
        states['spatial_attention'] = sp_attention_LSTM.zero_state(self.batch_size,tf.float32)

        tp_attention_LSTM = tf.contrib.rnn.BasicLSTMCell(attention_lstm_num_units, state_is_tuple=True)
        states['tempral_attention'] = tp_attention_LSTM.zero_state(self.batch_size,tf.float32)

        weights = {
            'out': tf.Variable(tf.random_normal([num_units*num_frames,num_classes]),name='weights_out'),
            'sp_attention_FC1': tf.Variable(tf.random_normal([attention_lstm_num_units+attention_in_size,attention_num_hidden_fc1]),name='sp_weights_FC1'),
            'sp_attention_FC2': tf.Variable(tf.random_normal([attention_num_hidden_fc1,attention_out_size]),name='sp_weights_FC2'),
            'tp_attention_FC1': tf.Variable(tf.random_normal([1000 + attention_lstm_num_units, attention_out_size]),name='tp_weights_FC1'),
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]),name='biases_out'),
            'sp_attention_FC1': tf.Variable(tf.random_normal([attention_num_hidden_fc1]),name='sp_biases_FC1'),
            'sp_attention_FC2': tf.Variable(tf.random_normal([attention_out_size]),name='sp_biases_FC2'),
            'tp_attention_FC1': tf.Variable(tf.random_normal([attention_out_size]), name='tp_biases_FC1')
        }

        def spatial_attention(x_t, x_t1,scope):
            h_t1, states['spatial_attention'] = sp_attention_LSTM(x_t1, states['spatial_attention'],scope = scope)
            fc1 = tf.matmul(tf.concat([x_t, h_t1],1),weights['sp_attention_FC1']) + biases['sp_attention_FC1']
            fc2 = tf.matmul(tf.tanh(fc1), weights['sp_attention_FC2']) + biases['sp_attention_FC2']
            tmp_at = fc2
            #tmp_at = tf.nn.relu(fc2)
            # if attention_placement == 0:
            #     at =  tf.stack([tmp_at]*num_features_per_joints,2)
            #     shape_at = tf.shape(at)
            #     at = tf.reshape(at, [shape_at[0], shape_at[1]*shape_at[2]])
            # else:
            return tmp_at

        def tempral_attention(x_t,x_t1,scope):
            h_t1, states['tempral_attention'] = tp_attention_LSTM(x_t1, states['tempral_attention'],scope=scope)
            fc1 = tf.matmul(tf.concat([x_t, h_t1],1),weights['tp_attention_FC1']) + biases['tp_attention_FC1']
            tmp_at = tf.nn.softmax(fc1)
            #tmp_at = tf.nn.relu(fc2)
            # if attention_placement == 0:
            #     at =  tf.stack([tmp_at]*num_features_per_joints,2)
            #     shape_at = tf.shape(at)
            #     at = tf.reshape(at, [shape_at[0], shape_at[1]*shape_at[2]])
            # else:
            return tmp_at



        outputs = {}
        #final_outputs = []
        node_inputs = {}
        final_inputs_list = []
        #att_temp = []
        #attention_dense = {}
        #attention_dense_list = []
        #attention_fullbody_input_list = []
        def conv_2d(kernels,kernel_size):
            return Convolution2D(kernels,kernel_size,kernel_size,init="he_uniform",border_mode="same")

        def att_module(final_inputs_list,t_or_s,scope,time_steps):

            att_weight = []

            for time_step in range(len(final_inputs_list)):
                input_att = final_inputs_list[time_step]
                if time_step > 0:
                    input_att_t1 = final_inputs_list[time_step - 1]
                else:
                    input_att_t1 = tf.zeros_like(input_att)

                if t_or_s == 's':
                    if time_step > 0 : tf.get_variable_scope().reuse_variables()
                    at_shaped = spatial_attention(input_att, input_att_t1,scope)
                elif t_or_s == 't':
                    at_shaped = tempral_attention(input_att, input_att_t1,scope)
                att_weight.append(at_shaped)

            att_weight = tf.nn.softmax(att_weight)

            final_inputs_list = final_inputs_list*att_weight

            return final_inputs_list

        with tf.variable_scope("SRNN"):
            for time_step in range(num_frames):
                #final_inputs_list = []
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                #final_temp_inputs = []
                #final_inputs = []
                #attention_dense_list = []
                for temp_feat in self.temp_features_names:
                    inputs = self.inputs[temp_feat][:, time_step, :]
                    state = states[temp_feat]
                    scope = "lstm_" + temp_feat
                    outputs[temp_feat], states[temp_feat] = edgesRNN[temp_feat](inputs, state,scope=scope)
                    # attention_dense[temp_feat] = Dense(1, kernel_initializer=tf.random_normal_initializer(),
                    #       bias_initializer=tf.random_normal_initializer(),activation='sigmoid')(outputs[temp_feat])
                    # attention_dense_list.append(attention_dense[temp_feat])

                    #final_inputs.append(outputs[temp_feat])
                for st_feat in self.st_features_names:
                    inputs = self.inputs[st_feat][:, time_step, :]
                    state = states[st_feat]
                    scope = "lstm_" + st_feat
                    outputs[st_feat], states[st_feat] = edgesRNN[st_feat](inputs, state, scope=scope)

                    # attention_dense[st_feat] = Dense(1, kernel_initializer=tf.random_normal_initializer(),
                    #                                    bias_initializer=tf.random_normal_initializer(),activation='sigmoid')(outputs[st_feat])
                    # attention_dense_list.append(attention_dense[st_feat])

                    #final_inputs.append(outputs[st_feat])

                # attention_fullbody_input = tf.concat(attention_dense_list,1)
                # attention_fullbody_input = tf.nn.elu(attention_fullbody_input)
                # attention_fullbody_input_list.append(attention_fullbody_input)
                # fullbody_input = tf.concat(final_inputs, 1)
                # final_inputs_list.append(fullbody_input)
                # input_att = final_inputs_list[time_step]
                # if time_step > 0:
                #     input_att_t1 = final_inputs_list[time_step-1]
                # else:
                #     input_att_t1 = tf.zeros_like(input_att)
                #
                # at_shaped, at = attention(input_att, input_att_t1)
                #
                # final_inputs_list[time_step] = tf.multiply(final_inputs_list[time_step] ,at_shaped)

                #
                # fullbody_input = tf.concat(final_inputs, 1)
                # final_inputs_list.append(fullbody_input)
                #


                #attention_fullbody_input_list[time_step] = tf.multiply(at_shaped, attention_fullbody_input_list[time_step])




                node_inputs['face'] = [outputs['face-face'], outputs['face-belly'],
                                        outputs['face-rightElbow'],outputs['face-leftElbow'],outputs['face-rightArm'], outputs['face-leftArm'],
                                       outputs['face-rightKnee'],outputs['face-leftKnee'],outputs['face-rightLeg'],outputs['face-leftLeg']]

                with tf.variable_scope('attention_face'):
                #if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['face'] = att_module(node_inputs['face'],'s','attention_face',time_step)


                # node_inputs['neck'] = [outputs['face-neck'],outputs['neck-belly'],outputs['neck-rightShoulder'],outputs['neck-leftShoulder'],
                #                                  outputs['neck-rightElbow'],
                #                                  outputs['neck-leftElbow'],outputs['neck-rightArm'],outputs['neck-leftArm'],outputs['neck-rightHip'],
                #                                  outputs['neck-leftHip'],
                #                                  outputs['neck-rightKnee'],outputs['neck-leftKnee'],outputs['neck-rightLeg'],outputs['neck-leftLeg'],
                #                                  outputs['neck-neck']]
                #
                #
                # node_inputs['neck'] = att_module(node_inputs['neck'])

                #node_inputs['neck'] = tf.reshape(tf.transpose(conv_2d(1,1)(tf.nn.relu(node_inputs['neck'])),[0,3,2,1]),[self.batch_size,num_units])

                # node_inputs['elbow'] = tf.concat(
                #     [outputs['rightElbow-rightElbow'], outputs['leftElbow-leftElbow'],
                #      outputs['face-rightElbow'], outputs['face-leftElbow'],
                #      outputs['rightElbow-rightArm'],
                #      outputs['leftElbow-leftArm'], outputs['belly-rightElbow'],
                #      outputs['belly-leftElbow'], outputs['rightElbow-leftElbow']], 1)
                node_inputs['belly'] = [outputs['belly-belly'], outputs['face-belly'],outputs['belly-rightElbow'],
                                        outputs['belly-leftElbow'],outputs['belly-rightKnee'],
                                        outputs['belly-leftKnee'],outputs['belly-leftArm'], outputs['belly-rightArm'],outputs['belly-leftLeg'],
                                        outputs['belly-rightLeg']]

                with tf.variable_scope('attention_belly'):
                #if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['belly'] = att_module(node_inputs['belly'],'s','attention_belly',time_step)

                # node_inputs['right-shoulder'] = [outputs['face-rightShoulder'],outputs['neck-rightShoulder'],outputs['belly-rightShoulder'],outputs['rightShoulder-leftShoulder'],
                #                                            outputs['rightShoulder-rightElbow'],
                #                                            outputs['rightShoulder-leftElbow'],outputs['rightShoulder-rightArm'],outputs['rightShoulder-leftArm'],outputs['rightShoulder-rightHip'],
                #                                            outputs['rightShoulder-leftHip'],
                #                                            outputs['rightShoulder-rightKnee'],outputs['rightShoulder-leftKnee'],outputs['rightShoulder-rightLeg'],outputs['rightShoulder-leftLeg'],
                #                                            outputs['rightShoulder-rightShoulder']]

                #node_inputs['right-shoulder'] = att_module(node_inputs['right-shoulder'])


                # node_inputs['left-shoulder'] = [outputs['face-leftShoulder'],outputs['neck-leftShoulder'],outputs['belly-leftShoulder'],outputs['rightShoulder-leftShoulder'],
                #                                           outputs['leftShoulder-rightElbow'],
                #                                           outputs['leftShoulder-leftElbow'],outputs['leftShoulder-rightArm'],outputs['leftShoulder-leftArm'],outputs['leftShoulder-rightHip'],
                #                                           outputs['leftShoulder-leftHip'],
                #                                           outputs['leftShoulder-rightKnee'],outputs['leftShoulder-leftKnee'],outputs['leftShoulder-rightLeg'],outputs['leftShoulder-leftLeg'],
                #                                           outputs['leftShoulder-leftShoulder']]


                #node_inputs['left-shoulder'] = att_module(node_inputs['left-shoulder'])


                node_inputs['right-elbow'] = [outputs['face-rightElbow'],outputs['belly-rightElbow'],
                                                        outputs['rightElbow-leftElbow'],outputs['rightElbow-rightArm'],outputs['rightElbow-leftArm'],
                                                        outputs['rightElbow-rightKnee'],outputs['rightElbow-leftKnee'],outputs['rightElbow-rightLeg'],outputs['rightElbow-leftLeg'],
                                                        outputs['rightElbow-rightElbow']]
                with tf.variable_scope('attention_right-elbow'):
                #if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['right-elbow'] = att_module(node_inputs['right-elbow'],'s','attention_right-elbow',time_step)


                node_inputs['left-elbow'] = [outputs['face-leftElbow'],outputs['belly-leftElbow'],
                                                       outputs['rightElbow-leftElbow'],outputs['leftElbow-rightArm'],outputs['leftElbow-leftArm'],
                                                       outputs['leftElbow-rightKnee'],outputs['leftElbow-leftKnee'],outputs['leftElbow-rightLeg'],outputs['leftElbow-leftLeg'],
                                                       outputs['leftElbow-leftElbow']]

                with tf.variable_scope('attention_left-elbow'):
                #    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['left-elbow'] = att_module(node_inputs['left-elbow'],'s','attention_left-elbow',time_step)



                node_inputs['right-arm'] = [outputs['face-rightArm'],outputs['belly-rightArm'],
                                                      outputs['rightElbow-rightArm'],outputs['leftElbow-rightArm'],outputs['rightArm-leftArm'],
                                                      outputs['rightArm-rightKnee'],outputs['rightArm-leftKnee'],outputs['rightArm-rightLeg'],outputs['rightArm-leftLeg'],
                                                      outputs['rightArm-rightArm']]

                with tf.variable_scope('attention_right-arm'):
                #    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['right-arm'] = att_module(node_inputs['right-arm'],'s','attention_right-arm',time_step)

                node_inputs['left-arm'] = [outputs['face-leftArm'],outputs['belly-leftArm'],
                                                     outputs['rightElbow-leftArm'],outputs['leftElbow-leftArm'],outputs['rightArm-leftArm'],
                                                     outputs['leftArm-rightKnee'],outputs['leftArm-leftKnee'],outputs['leftArm-rightLeg'],outputs['leftArm-leftLeg'],
                                                     outputs['leftArm-leftArm']]

                with tf.variable_scope('attention_left-arm'):
                #    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['left-arm'] = att_module(node_inputs['left-arm'],'s','attention_left-arm',time_step)

                # node_inputs['right-hip'] = [outputs['face-rightHip'],outputs['neck-rightHip'],outputs['belly-rightHip'],outputs['rightShoulder-rightHip'],
                #                                       outputs['leftShoulder-rightHip'],
                #                                       outputs['rightElbow-rightHip'],outputs['leftElbow-rightHip'],outputs['rightArm-rightHip'],outputs['rightHip-leftHip'],
                #                                       outputs['leftArm-rightHip'],
                #                                       outputs['rightHip-rightKnee'],outputs['rightHip-leftKnee'],outputs['rightHip-rightLeg'],outputs['rightHip-leftLeg'],
                #                                       outputs['rightHip-rightHip']]


                #node_inputs['right-hip'] = att_module(node_inputs['right-hip'])


                # node_inputs['left-hip'] = [outputs['face-leftHip'],outputs['neck-leftHip'],outputs['belly-leftHip'],outputs['rightShoulder-leftHip'],
                #                                      outputs['leftShoulder-leftHip'],
                #                                      outputs['rightElbow-leftHip'],outputs['leftElbow-leftHip'],outputs['rightArm-leftHip'],outputs['leftArm-leftHip'],
                #                                      outputs['rightHip-leftHip'],
                #                                      outputs['leftHip-rightKnee'],outputs['leftHip-leftKnee'],outputs['leftHip-rightLeg'],outputs['leftHip-leftLeg'],
                #                                      outputs['leftHip-leftHip']]


                #node_inputs['left-hip'] = att_module(node_inputs['left-hip'])

                node_inputs['right-knee'] = [outputs['face-rightKnee'],outputs['belly-rightKnee'],
                                            outputs['rightElbow-rightKnee'],outputs['leftElbow-rightKnee'],outputs['rightArm-rightKnee'],outputs['leftArm-rightKnee'],
                                            outputs['rightKnee-leftKnee'],outputs['rightKnee-rightLeg'],outputs['rightKnee-leftLeg'],
                                            outputs['rightKnee-rightKnee']]

                with tf.variable_scope('attention_right-knee'):
                #    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['right-knee'] = att_module(node_inputs['right-knee'],'s','attention_right-knee',time_step)


                node_inputs['left-knee'] = [outputs['face-leftKnee'],outputs['belly-leftKnee'],
                                              outputs['rightElbow-leftKnee'],outputs['leftElbow-leftKnee'],outputs['rightArm-leftKnee'],outputs['leftArm-leftKnee'],
                                              outputs['rightKnee-leftKnee'],outputs['leftKnee-rightLeg'],outputs['leftKnee-leftLeg'],
                                              outputs['leftKnee-leftKnee']]

                with tf.variable_scope('attention_left-knee'):
                #    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['left-knee'] = att_module(node_inputs['left-knee'],'s','attention_left-knee',time_step)


                node_inputs['right-leg'] = [outputs['face-rightLeg'],outputs['belly-rightLeg'],
                                                      outputs['rightElbow-rightLeg'],outputs['leftElbow-rightLeg'],outputs['rightArm-rightLeg'],outputs['leftArm-rightLeg'],
                                                      outputs['rightKnee-rightLeg'],outputs['leftKnee-rightLeg'],outputs['rightLeg-leftLeg'],
                                                      outputs['rightLeg-rightLeg']]
                with tf.variable_scope('attention_right-leg'):
                #    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['right-leg'] = att_module(node_inputs['right-leg'],'s','attention_right-leg',time_step)

                node_inputs['left-leg'] = [outputs['face-leftLeg'],outputs['belly-leftLeg'],outputs['rightShoulder-leftLeg'],
                                                     outputs['leftElbow-leftLeg'],outputs['rightArm-leftLeg'],outputs['leftArm-leftLeg'],
                                                    outputs['rightKnee-leftLeg'],outputs['leftKnee-leftLeg'],outputs['rightLeg-leftLeg'],
                                                     outputs['leftLeg-leftLeg']]
                with tf.variable_scope('attention_left-leg'):
                #    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    node_inputs['left-leg'] = att_module(node_inputs['left-leg'],'s','attention_left-leg',time_step)

                #node_inputs['left-leg'] = tf.reshape(tf.transpose(conv_2d(1,1)(tf.nn.relu(node_inputs['left-leg'])),[0,3,2,1]),[self.batch_size,num_units])

                # node_inputs['arms'] = tf.concat(
                #     [outputs['rightArm-rightArm'], outputs['leftArm-leftArm'],outputs['face-rightArm'],outputs['rightElbow-rightArm'],
                #      outputs['leftElbow-leftArm'],outputs['face-leftArm'], outputs['belly-rightArm'], outputs['belly-leftArm'],
                #      outputs['rightArm-leftArm']], 1)
                #
                # node_inputs['knee'] = tf.concat([outputs['rightKnee-rightKnee'],outputs['leftKnee-leftKnee'],outputs['rightKnee-leftKnee'],
                #                                 outputs['belly-rightKnee'],outputs['belly-leftKnee'],outputs['rightKnee-rightLeg'],
                #                                  outputs['leftKnee-leftLeg']], 1)
                #
                #
                # node_inputs['legs'] = tf.concat(
                #     [outputs['rightLeg-rightLeg'], outputs['leftLeg-leftLeg'],outputs['rightKnee-rightLeg'],outputs['leftKnee-leftLeg'],
                #      outputs['belly-rightLeg'], outputs['belly-leftLeg'],outputs['rightLeg-leftLeg']], 1)





                node_output_list = []
                for node_name in nodes_names:
                    inputs = tf.concat(tf.unstack(node_inputs[node_name]),1)
                    inputs = tf.nn.elu(inputs)
                    state = states[node_name]
                    scope = "lstm_" + node_name
                    outputs[node_name], states[node_name] = nodesRNN[node_name](inputs, state, scope=scope)
                    node_output_list.append(outputs[node_name])
                #
                # fullbody_input = tf.concat(
                #     [node_inputs['face'],node_inputs['elbow'], node_inputs['belly'], node_inputs['knee'],node_inputs['arms'],
                #      node_inputs['legs']], 1)


                # state = states['wholeRNN']
                # scope = "lstm_" + 'wholeRNN'
                #node_output_list = att_module(node_output_list)

                fullbody_input = tf.concat(node_output_list, 1)
                final_inputs_list.append(fullbody_input)


        #with tf.variable_scope("temporal_attention",reuse=None):
             #final_inputs_list = tf.stack(att_module(final_inputs_list,'t'))

             #outputs, final_state = tf.nn.dynamic_rnn(wholeRNN, tf.stack(final_inputs_list), initial_state=states_whole, time_major=True)
             #outputs = tf.unstack(outputs)






                # input_att = final_inputs_list[time_step]
                # if time_step > 0:
                #     input_att_t1 = final_inputs_list[time_step - 1]
                # else:
                #     input_att_t1 = tf.zeros_like(input_att)
                #
                # at_shaped, at = attention(input_att, input_att_t1)
                #
                # final_inputs_list[time_step] = tf.multiply(final_inputs_list[time_step], at_shaped)


        # cells = []
        # for _ in range(1):
        #     cell = tf.contrib.rnn.BasicLSTMCell(num_units,activation=tf.nn.softsign )
        #     cells.append(cell)
        # cell_fw = tf.contrib.rnn.MultiRNNCell(cells)
        #
        # cells = []
        # for _ in range(1):
        #     cell = tf.contrib.rnn.BasicLSTMCell(num_units,activation=tf.nn.softsign)
        #     cells.append(cell)
        # cell_bw = tf.contrib.rnn.MultiRNNCell(cells)
        #
        # final_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw,final_inputs_list,dtype=tf.float32)





        # attention_inputs = tf.transpose(final_outputs, perm=[1, 0, 2])
        # #
        # alpha = attention(attention_inputs,100,return_alphas=True)
        # #
        # final_outputs = final_outputs * alpha
        #
        # split0,split1,split2,split3,split4,split5,split6,split7,split8,split9= tf.split(final_outputs, num_or_size_splits=10, axis=0)
        # split = [tf.squeeze(split0,axis=0),tf.squeeze(split1,axis=0),tf.squeeze(split2,axis=0),tf.squeeze(split3,axis=0),tf.squeeze(split4,axis=0),
        #          tf.squeeze(split5, axis=0),tf.squeeze(split6,axis=0),tf.squeeze(split7,axis=0),tf.squeeze(split8,axis=0),tf.squeeze(split9,axis=0)]



        self.infos = infos

        self.final_states = states
        #self.logits = tf.matmul(output, weights['out'], name="logits") + biases['out']
        # self.full_connect_layer = Dense(256,kernel_initializer=tf.random_normal_initializer(),bias_initializer=tf.random_normal_initializer())(final_outputs[-1])
        # self.dropout_layer =  tf.nn.dropout(Dense(256,kernel_initializer=tf.random_normal_initializer(),bias_initializer=tf.random_normal_initializer())(final_outputs[-1]),keep_prob=0.8)
        # self.logits = Dense(21,kernel_initializer=tf.random_normal_initializer(),bias_initializer=tf.random_normal_initializer())(final_outputs)
        self.logits = tf.layers.dense(tf.nn.elu(final_inputs_list[-1]),21,kernel_initializer=tf.random_normal_initializer(),bias_initializer=tf.random_normal_initializer(),name='dense_out')
        # self.logits = Dense(21, kernel_initializer=tf.random_normal_initializer(),
        #                     bias_initializer=tf.random_normal_initializer())(output)
        self.logits_drop = tf.nn.dropout(self.logits,keep_prob=0.5)
        self.predict = tf.nn.softmax(self.logits_drop)

        with tf.name_scope('cross_entropy'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_drop, labels=self.targets)
            self.cost = tf.reduce_mean(loss)
            self.cost_inference = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.targets))
        if self.save_summaries:
            tf.summary.scalar('cross_entropy', self.cost)

        tvars = tf.trainable_variables()

        ##learning rate!!!
        #
        # def get_learningrate():
        #     if self.strikes > 4:
        #         self.learning_rate_decay = tf.Variable(float(self.learning_rate_decay.eval() / 10), trainable=False)
        #         self.strikes = 0
        #     else:
        #         self.learning_rate_decay= self.learning_rate_decay
        #     return  self.learning_rate_decay

        # def exponential_decay_new(learning_rate, decay_rate):
        #     """Applies exponential decay to the learning rate.
        #
        #     When training a model, it is often recommended to lower the learning rate as
        #     the training progresses.  This function applies an exponential decay function
        #     to a provided initial learning rate.  It requires a `global_step` value to
        #     compute the decayed learning rate.  You can just pass a TensorFlow variable
        #     that you increment at each training step.
        #
        #     The function returns the decayed learning rate.  It is computed as:
        #
        #     ```python
        #     decayed_learning_rate = learning_rate *
        #                             decay_rate ^ (global_step / decay_steps)
        #     ```
        #
        #     If the argument `staircase` is `True`, then `global_step / decay_steps` is an
        #     integer division and the decayed learning rate follows a staircase function.
        #
        #     Example: decay every 100000 steps with a base of 0.96:
        #
        #     ```python
        #     ...
        #     global_step = tf.Variable(0, trainable=False)
        #     starter_learning_rate = 0.1
        #     learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        #                                                100000, 0.96, staircase=True)
        #     # Passing global_step to minimize() will increment it at each step.
        #     learning_step = (
        #         tf.train.GradientDescentOptimizer(learning_rate)
        #         .minimize(...my loss..., global_step=global_step)
        #     )
        #     ```
        #
        #     Args:
        #       learning_rate: A scalar `float32` or `float64` `Tensor` or a
        #         Python number.  The initial learning rate.
        #       global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
        #         Global step to use for the decay computation.  Must not be negative.
        #       decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        #         Must be positive.  See the decay computation above.
        #       decay_rate: A scalar `float32` or `float64` `Tensor` or a
        #         Python number.  The decay rate.
        #       staircase: Boolean.  If `True` decay the learning rate at discrete intervals
        #       name: String.  Optional name of the operation.  Defaults to
        #         'ExponentialDecay'.
        #
        #     Returns:
        #       A scalar `Tensor` of the same type as `learning_rate`.  The decayed
        #       learning rate.
        #
        #     Raises:
        #       ValueError: if `global_step` is not supplied.
        #     """
        #
        #     learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        #     dtype = learning_rate.dtype
        #     strikes = math_ops.cast(self.strikes, dtype)
        #     #decay_steps = math_ops.cast(decay_steps, dtype)
        #     decay_rate = math_ops.cast(decay_rate, dtype)
        #     if strikes.eval() > 4 :
        #         self.strikes = math_ops.multiply(self.strikes,0)
        #
        #         return math_ops.multiply(learning_rate, decay_rate)
        #     else:
        #         return learning_rate


        starter_learning_rate = self.learning_rate

        self.learning_rate_decay = tf.train.exponential_decay(
            starter_learning_rate,
            self.global_step,
            250,
            0.65,
            staircase=True
        )
        # self.learning_rate_decay = exponential_decay_new(
        #     starter_learning_rate,
        #     0.1
        # )

        if not forward_only:
            if self.GD:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_decay)

                clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
                self.gradients_norm = norm
                self.updates = optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=self.global_step)
            else:
                aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_decay, epsilon=self.adam_epsilon)
                gradients_and_params = optimizer.compute_gradients(self.cost, tvars,
                                                                   aggregation_method=aggregation_method)
                gradients, params = zip(*gradients_and_params)
                norm = tf.global_norm(gradients)
                self.gradients_norm = norm
                self.updates = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        self.merged = tf.summary.merge_all()
        # self.merged = tf.merge_all_summaries()
        if self.save_summaries:
            self.train_writer = tf.summary.FileWriter(log_dir + '/train')
            self.test_writer = tf.summary.FileWriter(log_dir + '/test')

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def step(self, session, temp_inputs, st_inputs, targets, forward_only, batch_size, get_info=False):
        input_feed = {}
        for temp_features_name in self.temp_features_names:
            input_feed[self.inputs[temp_features_name].name] = temp_inputs[temp_features_name]

        for st_features_name in self.st_features_names:
            input_feed[self.inputs[st_features_name].name] = st_inputs[st_features_name]

        input_feed[self.targets.name] = targets
        input_feed[self.batch_size] = batch_size

        if not forward_only:
            # output_feed = [self.updates, self.gradients_norm, self.cost]
            output_feed = [self.updates, self.gradients_norm, self.cost, self.predict]
        else:
            if get_info:
                output_feed = [self.cost, tf.nn.softmax(self.logits), self.infos]
            else:
                output_feed = [self.cost_inference, tf.nn.softmax(self.logits)]

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            # return outputs[1], outputs[2], None  # returs gradients norm and cost and no output
            return outputs[1], outputs[2], outputs[3]
        else:
            if get_info:
                return outputs[2], outputs[0], outputs[1]  # no gradients, cost and output
            else:
                return None, outputs[0], outputs[1]  # no gradients, cost and output

    def step_with_summary(self, session, temp_inputs, st_inputs, targets, forward_only, batch_size):
        input_feed = {}
        input_feed[self.batch_size.name] = batch_size
        #input_feed['batch_size'] = batch_size
        for temp_features_name in self.temp_features_names:
            input_feed[self.inputs[temp_features_name].name] = temp_inputs[temp_features_name]

        for st_features_name in self.st_features_names:
            input_feed[self.inputs[st_features_name].name] = st_inputs[st_features_name]

        input_feed[self.targets.name] = targets

        if not forward_only:
            output_feed = [self.updates, self.gradients_norm, self.cost]
        else:
            output_feed = [self.cost, tf.nn.softmax(self.logits)]

        outputs, summary = session.run([output_feed, self.merged], input_feed)

        if not forward_only:
            return outputs[1], outputs[2], None, summary  # returs gradients norm and cost and no output
        else:
            return None, outputs[0], outputs[1], summary  # no gradients, cost and output

    def steps(self, session, all_data, batch_size):
        size = len(all_data[2])
        num_batches = size / self.batch_size

        batch_losses = np.zeros(num_batches)
        batch_outputs = np.zeros((num_batches, self.batch_size, self.num_classes))
        missclassified = 0.0
        for i in range(num_batches):
            batch = self.get_batch(all_data, i, batch_size)
            _, batch_losses[i], batch_outputs[i] = self.step(session, batch[0], batch[1], batch[2], True, batch_size)
            classes = np.argmax(batch_outputs[i], 1)
            for d in range(self.batch_size):
                if batch[2][d][classes[d]] != 1:
                    missclassified += 1
        error = missclassified / (num_batches * self.batch_size)
        return np.mean(batch_losses), error

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

    def get_batch(self, data, batch_id, batch_size):

        batch = np.array({})
        idxs = range((batch_id * batch_size - batch_size), (batch_id * batch_size))

        batch_dic = np.array([{}, {}, []])

        for i in range(len(data) - 1):
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
        return batch

    def get_big_batch(self, data):

        batch = np.array({})
        idxs = range(len(data[2]))

        batch_dic = np.array([{}, {}, []])

        for i in range(len(data) - 1):
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
        return batch

    #
    #
    # def transport_validloss(self,valid_loss):
    #     self.valid_loss = valid_loss
    #     self.previous_eval_loss.append(valid_loss)
    #     improve_valid = self.previous_eval_loss[-1] < self.best_val_loss
    #     if improve_valid:
    #         self.best_val_loss = self.previous_eval_loss[-1]
    #     else:
    #         self.strikes = math_ops.add(self.strikes,1)
    #         print self.strikes

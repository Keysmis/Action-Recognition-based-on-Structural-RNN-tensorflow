import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
import numpy as np



class SRNN_model(object):
    # SharedRNN(shared_layers, human_layers, object_layers, softmax_loss, trY_1, trY_2, 1e-3)

    def __init__(self, num_classes, num_frames, num_temp_features, num_st_features, num_units, max_gradient_norm,
                 learning_rate,
                 learning_rate_decay_factor, adam_epsilon, GD, forward_only=False, l2_regularization=False,
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
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.max_grad_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.adam_epsilon = adam_epsilon
        self.GD = GD
        self.weight_decay = weight_decay

        self.temp_features_names = ['face-face', 'belly-belly','rightKnee-rightKnee', 'leftKnee-leftKnee','rightArm-rightArm', 'leftArm-leftArm',
                                    'rightShoulder-rightShoulder', 'leftShoulder-leftShoulder',
                                    'rightLeg-rightLeg', 'leftLeg-leftLeg','rightElbow-rightElbow','leftElbow-leftElbow']
        self.st_features_names =['face-leftArm', 'face-rightArm', 'face-belly','face-rightElbow','face-leftElbow',
                                 'face-rightShoulder','face-leftShoulder','rightShoulder-rightElbow',
                                 'leftShoulder-leftElbow', 'rightShoulder-rightArm', 'leftShoulder-leftArm',
                                 'rightElbow-rightArm','leftElbow-leftArm',
                                 'belly-rightElbow','belly-leftElbow',
                                  'belly-leftArm', 'belly-rightArm', 'belly-rightKnee','belly-leftKnee','rightKnee-rightLeg','leftKnee-leftLeg','belly-rightShoulder','belly-leftShoulder',
                                  'belly-rightLeg', 'belly-leftLeg', 'rightArm-leftArm', 'rightLeg-leftLeg','rightKnee-leftKnee','rightElbow-leftElbow','rightShoulder-leftShoulder']

        nodes_names = {'face','shoulder','elbow','arms', 'knee','legs', 'belly'}
        edgesRNN = {}
        nodesRNN = {}
        states = {}
        infos = {}
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.inputs = {}
        self.targets = tf.placeholder(tf.float32, shape=(None, num_classes), name='targets')


        for temp_feat in self.temp_features_names:
            infos[temp_feat] = {'input_gates': [], 'forget_gates': [], 'modulated_input_gates': [], 'output_gates': [],
                                'activations': [], 'state_c': [], 'state_m': []}
            self.inputs[temp_feat] = tf.placeholder(tf.float32, shape=(None, num_frames, self.num_temp_features),
                                                    name=temp_feat)
            if num_layers == 1:
                edgesRNN[temp_feat] = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                      reuse=tf.get_variable_scope().reuse,
                                                                      activation=tf.nn.softsign)

            else:
                cells = []
                for _ in range(num_layers):
                    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                                      reuse=tf.get_variable_scope().reuse,
                                                                                      activation=tf.nn.softsign))
                    cells.append(cell)
                edgesRNN[temp_feat] = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            states[temp_feat] = edgesRNN[temp_feat].zero_state(self.batch_size, tf.float32)

        for st_feat in self.st_features_names:
            infos[st_feat] = {'input_gates': [], 'forget_gates': [], 'modulated_input_gates': [], 'output_gates': [],
                              'activations': [], 'state_c': [], 'state_m': []}
            self.inputs[st_feat] = tf.placeholder(tf.float32, shape=(None, num_frames, self.num_st_features),
                                                  name=st_feat)
            if num_layers == 1:
                 edgesRNN[st_feat] = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                       reuse=tf.get_variable_scope().reuse,
                                                                      activation=tf.nn.softsign)
            else:
                cells = []
                for _ in range(num_layers):
                    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                                      reuse=tf.get_variable_scope().reuse,
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
                                                                      reuse=tf.get_variable_scope().reuse,
                                                                       activation=tf.nn.softsign)
            else:
                cells = []
                for _ in range(num_layers):
                    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True,
                                                                                      reuse=tf.get_variable_scope().reuse,
                                                                                      activation=tf.nn.softsign))
                    cells.append(cell)
                nodesRNN[node] = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            states[node] = nodesRNN[node].zero_state(self.batch_size, tf.float32)

        weights = {'out': tf.Variable(tf.random_normal([num_units * num_frames * 7, num_classes]))}
        biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

        outputs = {}
        final_outputs = []
        node_inputs = {}
        def linear(args, output_size, bias=True, bias_start=0.0, scope=None):
            """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
            Args:
            args: a 2D Tensor or a list of 2D, batch x n, Tensors.
            output_size: int, second dimension of W[i].
            bias: boolean, whether to add a bias term or not.
            bias_start: starting value to initialize the bias; 0 by default.
            scope: VariableScope for the created subgraph; defaults to "Linear".
            Returns:
            A 2D Tensor with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.
            """
            if args is None or (nest.is_sequence(args) and not args):
                raise ValueError("`args` must be specified")
            if not nest.is_sequence(args):
                args = [args]

            # Calculate the total size of arguments on dimension 1.
            total_arg_size = 0
            shapes = [a.get_shape().as_list() for a in args]
            for shape in shapes:
                if len(shape) != 2:
                    raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
                if not shape[1]:
                    raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
                else:
                    total_arg_size += shape[1]

            dtype = [a.dtype for a in args][0]

            # Now the computation.
            with vs.variable_scope(scope or "Linear"):
                matrix = vs.get_variable(
                    "Matrix", [total_arg_size, output_size], dtype=dtype)
                if len(args) == 1:
                    res = math_ops.matmul(args[0], matrix)
                else:
                    res = math_ops.matmul(array_ops.concat(args, 1), matrix)
                if not bias:
                    return res
                bias_term = vs.get_variable(
                    "Bias", [output_size],
                    dtype=dtype,
                    initializer=init_ops.constant_initializer(
                        bias_start, dtype=dtype))
                return res + bias_term

        # connect the edgesRNN to the corresponding nodeRNN
        with tf.variable_scope("SRNN"):
            for time_step in range(num_frames):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                for temp_feat in self.temp_features_names:
                    inputs = self.inputs[temp_feat][:, time_step, :]
                    state = states[temp_feat]
                    scope = "lstm_" + temp_feat
                    outputs[temp_feat], states[temp_feat] = edgesRNN[temp_feat](inputs, state,scope=scope)

                for st_feat in self.st_features_names:
                    inputs = self.inputs[st_feat][:, time_step, :]
                    state = states[st_feat]
                    scope = "lstm_" + st_feat
                    outputs[st_feat], states[st_feat] = edgesRNN[st_feat](inputs, state, scope=scope)
                node_inputs['face'] = tf.concat([outputs['face-face'], outputs['face-rightArm'], outputs['face-leftArm'],
                                                 outputs['face-belly'],outputs['face-rightElbow'],outputs['face-leftElbow'],
                                                 outputs['face-rightShoulder'],outputs['face-leftShoulder']], 1)

                node_inputs['shoulder'] = tf.concat(
                    [outputs['rightShoulder-rightShoulder'], outputs['leftShoulder-leftShoulder'],
                     outputs['face-rightShoulder'],outputs['face-leftShoulder'],
                     outputs['rightShoulder-rightElbow'],outputs['leftShoulder-leftElbow'],
                     outputs['rightShoulder-rightArm'],
                     outputs['leftShoulder-leftArm'], outputs['belly-rightShoulder'],
                     outputs['belly-leftShoulder'], outputs['rightShoulder-leftShoulder']], 1)


                node_inputs['elbow'] = tf.concat(
                    [outputs['rightElbow-rightElbow'], outputs['leftElbow-leftElbow'],
                     outputs['face-rightElbow'], outputs['face-leftElbow'],
                     outputs['rightShoulder-rightElbow'],outputs['leftShoulder-leftElbow'],
                     outputs['rightElbow-rightArm'],
                     outputs['leftElbow-leftArm'], outputs['belly-rightElbow'],
                     outputs['belly-leftElbow'], outputs['rightElbow-leftElbow']], 1)

                node_inputs['belly'] = tf.concat(
                    [outputs['belly-belly'], outputs['face-belly'],
                     outputs['belly-rightElbow'],outputs['belly-leftElbow'],
                     outputs['belly-rightKnee'],outputs['belly-leftKnee'],
                     outputs['belly-rightShoulder'],outputs['belly-leftShoulder'],
                     outputs['belly-leftArm'], outputs['belly-rightArm'],
                     outputs['belly-leftLeg'], outputs['belly-rightLeg']], 1)

                node_inputs['arms'] = tf.concat(
                    [outputs['rightArm-rightArm'], outputs['leftArm-leftArm'],outputs['face-rightArm'],outputs['rightElbow-rightArm'],outputs['leftElbow-leftArm'],
                     outputs['rightShoulder-rightArm'],outputs['leftShoulder-leftArm'],
                   outputs['face-leftArm'], outputs['belly-rightArm'], outputs['belly-leftArm'],outputs['rightArm-leftArm']], 1)

                node_inputs['knee'] = tf.concat([outputs['rightKnee-rightKnee'],outputs['leftKnee-leftKnee'],outputs['rightKnee-leftKnee'],
                                                outputs['belly-rightKnee'],outputs['belly-leftKnee'],outputs['rightKnee-rightLeg'],outputs['leftKnee-leftLeg']], 1)


                node_inputs['legs'] = tf.concat(
                    [outputs['rightLeg-rightLeg'], outputs['leftLeg-leftLeg'],outputs['rightKnee-rightLeg'],outputs['leftKnee-leftLeg'],
                     outputs['belly-rightLeg'], outputs['belly-leftLeg'],outputs['rightLeg-leftLeg']], 1)

                for node_name in nodes_names:
                    inputs = node_inputs[node_name]
                    state = states[node_name]
                    scope = "lstm_" + node_name
                    outputs[node_name], states[node_name] = nodesRNN[node_name](inputs, state, scope=scope)

                fullbody_input = tf.concat(
                    [outputs['face'],outputs['shoulder'],outputs['elbow'], outputs['belly'], outputs['knee'],outputs['arms'],
                     outputs['legs']], 1)
                final_outputs.append(fullbody_input)

        self.infos = infos
        output = tf.concat(final_outputs, 1, name="output_lastCells")
        self.final_states = states
        self.logits = tf.matmul(output, weights['out'], name="logits") + biases['out']
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
        starter_learning_rate = self.learning_rate
        self.learning_rate_decay = tf.train.exponential_decay(
            starter_learning_rate,
            self.global_step,
            60,
            0.80,
            staircase=True
        )


        if not forward_only:
            if self.GD:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

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
        # self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

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
        return batch

    def get_big_batch(self, data):

        batch = np.array({})
        idxs = range(len(data[2]))

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
        return batch


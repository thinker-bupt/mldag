import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse
import random
from sklearn.metrics import roc_auc_score
import multiprocessing
import queue
import threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def parse_args():
    parser = argparse.ArgumentParser(description="Run AITM.")
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='Batch size.')
    parser.add_argument('--embedding_dim', type=int, default=5,
                        help='Number of embedding dim.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.9,0.7,0.7]',
                        help='Keep probability. 1: no dropout.')
    parser.add_argument('--lamda', type=float, default=1e-6,
                        help='Regularizer weight.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Specify an optimizer type (adam, adagrad, gd, moment).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the results (0, 1 ... any positive integer)')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0, 1 ... any positive integer)')
    parser.add_argument('--prefix', type=str, required=True,
                        help='prefix for model_name path.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Which gpu to use.')
    parser.add_argument('--weight', type=float, default=0.6,
                        help='label constraint weight.')
    return parser.parse_args()


args = parse_args()
all_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301']
vocabulary_size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4}
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def print_info(prefix, result, time):
    print(prefix + '[%.1fs]: \n'
                   'click:     AUC:%.6f\n'
                   'conv:      AUC:%.6f\n'    
                   'ctcv:      AUC:%.6f\n'
                   'ctr_loss:      %.6f\n'
                   'cvr_loss:      %.6f\n'
                   'ctcvr_loss:    %.6f'
          % tuple([time] + result))


class GeneratorEnqueuer(object):
    """From keras source code training.py
    Builds a queue out of a data generator.

    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading
    """

    def __init__(self, generator, pickle_safe=False):
        self._generator = generator
        self._pickle_safe = pickle_safe
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.finish = False

    def start(self, workers=1, max_q_size=10, wait_time=0.05):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._pickle_safe or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except StopIteration:
                    self.finish = True
                    break
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._pickle_safe:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._pickle_safe:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed()
                    thread = multiprocessing.Process(
                        target=data_generator_task)
                    thread.daemon = True
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except BaseException:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._pickle_safe:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._pickle_safe:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None
        
"""
    Input shape
    - [batch_size, feature_dims]
    Output shape
    - task_nums * [batch_size, hidden_units]

"""
class PLE(tf.keras.layers.Layer):
    def __init__(self, hidden_units=64, expert_nums=2, task_nums=2, **kwargs):
        self.hidden_units = hidden_units
        self.expert_nums = expert_nums
        self.task_nums = task_nums
        super(PLE, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.domain_layers = [tf.keras.layers.Dense(self.hidden_units, activation='relu') for _ in range(self.task_nums)]
        self.expert_layers = [tf.keras.layers.Dense(self.hidden_units, activation='relu') for _ in range(self.expert_nums)]
        self.gate_layers = [tf.keras.layers.Dense(self.expert_nums + 1, activation='softmax') for _ in range(self.task_nums)]
        
    def call(self, inputs):
        domain_outputs, expert_outputs, gate_outputs, final_outputs = [], [], [], []
        
        # task_nums * (None, hidden_units, 1)
        for domain_layer in self.domain_layers:
            domain_output = tf.keras.layers.Dropout(1 - eval(args.keep_prob)[0])(inputs)
            domain_output = tf.expand_dims(domain_layer(domain_output), axis=2)
            domain_outputs.append(domain_output)
        
        # expert_nums * (None, hidden_units, 1)
        for expert_layer in self.expert_layers:
            expert_output = tf.keras.layers.Dropout(1 - eval(args.keep_prob)[0])(inputs)
            expert_output = tf.expand_dims(expert_layer(expert_output), axis=2)
            expert_outputs.append(expert_output)

        # task_nums * (None, 1, expert_nums + 1)
        for gate_layer in self.gate_layers:
            gate_output = gate_layer(inputs)
            gate_outputs.append(tf.expand_dims(gate_output, axis=1))

        for i, gate_output in enumerate(gate_outputs):
            # (None, hidden_units, expert_nums + 1)
            task_output = tf.concat([domain_outputs[i]] + expert_outputs, axis=2)
            
            # (None, hidden_units, expert_nums + 1)
            weighted_expert_output = task_output * gate_output
            task_output = tf.reduce_sum(weighted_expert_output, axis=2)
            final_outputs.append(task_output)
        return final_outputs


class AITM(object):
    def __init__(self, vocabulary_size, embedding_dim, epoch, batch_size, learning_rate, lamda,
                 keep_prob, optimizer_type, verbose, early_stop,
                 prefix, random_seed=2020):
        # bind params to class
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.lamda = lamda
        self.epoch = 0
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        print('dropout:{}'.format(self.keep_prob))
        self.no_dropout = np.array([1 for _ in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.early_stop = early_stop
        self.prefix = prefix
        # init all variables in a tensorflow graph
        self._init_graph_AITM()

    def _init_graph_AITM(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        print('Init raw AITM graph')
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Variables init.
            self.weights = self._initialize_weights()
            self.train_labels_click = tf.placeholder(
                tf.float32, shape=[None, 1], name='click')
            self.train_labels_purchase = tf.placeholder(
                tf.float32, shape=[None, 1], name='purchase')

            self.inputs_placeholder = []
            for column in all_columns:
                self.inputs_placeholder.append(tf.placeholder(
                    tf.int64, shape=[None, 1], name=column))

            feature_embedding = []
            for column, feature in zip(all_columns, self.inputs_placeholder):
                embedded = tf.nn.embedding_lookup(self.weights['feature_embeddings_{}'.format(
                    column)], feature)  # [None , 1, K]*num_features
                feature_embedding.append(embedded)
            feature_embedding = tf.keras.layers.concatenate(feature_embedding)
            feature_embedding = tf.squeeze(feature_embedding, axis=1)

            ple_result = PLE(hidden_units=64, expert_nums=2, task_nums=2)(feature_embedding)
            # ctr
            self.tower_click = ple_result[0]
            self.tower_click = tf.keras.layers.Dropout(
                1 - self.keep_prob[1])(self.tower_click)
            self.tower_click = tf.keras.layers.Dense(
                32, activation='relu')(self.tower_click)
            self.tower_click = tf.keras.layers.Dropout(
                1 - self.keep_prob[2])(self.tower_click)

            # cvr
            self.tower_purchase = ple_result[1]
            self.tower_purchase = tf.keras.layers.Dropout(
                1 - self.keep_prob[1])(self.tower_purchase)
            self.tower_purchase = tf.keras.layers.Dense(
                32, activation='relu')(self.tower_purchase)
            self.tower_purchase = tf.keras.layers.Dropout(
                1 - self.keep_prob[2])(self.tower_purchase)

            self.click = tf.keras.layers.Dense(1)(self.tower_click)
            self.purchase = tf.keras.layers.Dense(1)(self.tower_purchase)
            
            self.click = tf.sigmoid(self.click, name="click_pred")
            self.purchase = tf.sigmoid(self.purchase, name="purchase_pred")

            # Compute the loss.
            # L2
            reg_variables = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.lamda > 0:
                reg_loss = tf.add_n(reg_variables)
            else:
                reg_loss = 0
                
            self.ctr_loss = tf.losses.log_loss(self.train_labels_click, self.click)
            self.ctcvr_loss = tf.losses.log_loss(self.train_labels_purchase, self.purchase)

            self.loss = self.ctr_loss + self.ctcvr_loss + reg_loss
            
            # -------watch cvr--------
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Optimizer.
                if self.optimizer_type == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                                            beta2=0.999, epsilon=1e-8).minimize(self.loss)
                elif self.optimizer_type == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                               initial_accumulator_value=1e-8).minimize(self.loss)
                elif self.optimizer_type == 'gd':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                        self.loss)
                elif self.optimizer_type == 'moment':
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver(var_list=tf.global_variables())
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.InteractiveSession(
                config=tf.ConfigProto(
                    gpu_options=gpu_options))
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)
                
    def _initialize_weights(self):
        '''
        initialize parameters.
        '''
        all_weights = dict()
        l2_reg = tf.contrib.layers.l2_regularizer(self.lamda)
        # embedding
        for column in all_columns:
            all_weights['feature_embeddings_{}'.format(column)] = tf.get_variable(
                initializer=tf.random_normal(
                    shape=[
                        vocabulary_size[column],
                        self.embedding_dim],
                    mean=0.0,
                    stddev=0.01),
                regularizer=l2_reg, name='feature_embeddings_{}'.format(column))  # vocabulary_size * K
        return all_weights

    def fit_on_batch(self, data):
        '''
        Fit on a batch data.
        :param data: a batch data.
        :return: The LogLoss.
        '''
        train_ids = {}
        for column_name, column_placeholder in zip(
                all_columns, self.inputs_placeholder):
            train_ids[column_placeholder] = data['ids_{}'.format(column_name)]
        feed_dict = {
            self.train_labels_click: data['y_click'],
            self.train_labels_purchase: data['y_purchase']}
        feed_dict.update(train_ids)

        loss, ctr_loss, ctcvr_loss, _ = self.sess.run(
            (self.loss, self.ctr_loss, self.ctcvr_loss, self.optimizer), feed_dict=feed_dict)
        return loss, ctr_loss, ctcvr_loss

    def fit(self, train_path, dev_path,
            pickle_safe=False, max_q_size=40, workers=1):
        '''
        Fit the train data.
        :param train_path: train path.
        :param dev_path:  validation path.
        :param pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
        :param max_q_size: maximum size for the generator queue
        :param workers: maximum number of processes to spin up
                when using process based threading
        :return: None
        '''
        max_acc = -np.inf
        best_epoch = 0
        earlystop_count = 0
        enqueuer = None
        wait_time = 0.001  # in seconds
        tf.keras.backend.set_learning_phase(1)
        try:
            train_gen = self.iterator(train_path, shuffle=True)
            enqueuer = GeneratorEnqueuer(
                train_gen, pickle_safe=pickle_safe)
            enqueuer.start(max_q_size=max_q_size, workers=workers)
            t1 = time.time()
            train_loss = 0.
            ctr_loss = 0.
            ctcvr_loss = 0. 
            nb_sample = 0
            i = 0
            while True:
                # get a batch
                generator_output = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_output = enqueuer.queue.get()
                        break
                    elif enqueuer.finish:
                        break
                    else:
                        time.sleep(wait_time)
                # Fit training, return loss...
                if generator_output is None:  # epoch end
                    break
                nb_sample += len(generator_output['y_click'])
                result = self.fit_on_batch(generator_output)
                train_loss += result[0]
                ctr_loss += result[1]
                ctcvr_loss += result[2]
                if self.verbose > 0:
                    if (i + 1) % 200 == 0:
                        print('[%d] step %d, loss: %.6f, ctr_loss: %.6f, ctcvr_loss: %.6f' %
                              (nb_sample, (i + 1), train_loss / (i + 1), ctr_loss / (i + 1), ctcvr_loss / (i + 1)))
                i += 1
            # validation
            tf.keras.backend.set_learning_phase(0)
            t2 = time.time()
            dev_gen = self.iterator(dev_path)
            true_pred = self.evaluate_generator(
                dev_gen, max_q_size=max_q_size, workers=workers, pickle_safe=pickle_safe)
            valid_result = self.evaluate(true_pred)

            if self.verbose > 0:
                print_info(
                    "Epoch %d [%.1f s]\t Dev" %
                    (self.epoch + 1, t2 - t1), valid_result, time.time() - t2)
            if self.early_stop > 0:
                self.save_path = self.saver.save(self.sess,
                                                 save_path='./checkpoint/best_model_{}_epoch_{}.model'.format(
                                                     self.prefix, self.epoch + 1),
                                                 latest_filename='check_point_{}_epoch_{}'.format(self.prefix, self.epoch + 1))
                best_epoch = self.epoch + 1
        

        finally:
            if enqueuer is not None:
                enqueuer.stop()
        self.epoch += 1

    def evaluate_generator(self, generator, max_q_size=40,
                           workers=1, pickle_safe=False):
        '''
        See GeneratorEnqueuer Class about the following params.
        :param generator: the generator which return the data.
        :param max_q_size: maximum size for the generator queue
        :param workers: maximum number of processes to spin up
                when using process based threading
        :param pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
        :return: true labels, prediction probabilities.
        '''
        wait_time = 0.001
        enqueuer = None
        dev_y_true_click = []
        dev_y_true_purchase = []
        dev_y_pred_click = []
        dev_y_pred_purchase = []
        try:
            enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
            enqueuer.start(workers=workers, max_q_size=max_q_size)
            nb_dev = 0
            while True:
                dev_batch = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        dev_batch = enqueuer.queue.get()
                        break
                    elif enqueuer.finish:
                        break
                    else:
                        time.sleep(wait_time)
                # Fit training, return loss...
                if dev_batch is None:
                    break
                nb_dev += len(dev_batch['y_click'])
                train_ids = {}
                for column_name, column_placeholder in zip(
                        all_columns, self.inputs_placeholder):
                    train_ids[column_placeholder] = dev_batch['ids_{}'.format(
                        column_name)]
                feed_dict = {
                    self.train_labels_click: dev_batch['y_click'],
                    self.train_labels_purchase: dev_batch['y_purchase']}
                feed_dict.update(train_ids)
                
                predictions = self.sess.run(
                    [self.click, self.purchase], feed_dict=feed_dict)
                dev_y_true_click += list(dev_batch['y_click'])
                dev_y_true_purchase += list(dev_batch['y_purchase'])
                dev_y_pred_click += list(predictions[0])
                dev_y_pred_purchase += list(predictions[1])
            
            # to row vectors
            dev_y_true_click = np.reshape(dev_y_true_click, (-1,))
            dev_y_true_purchase = np.reshape(dev_y_true_purchase, (-1,))
            dev_y_pred_click = np.reshape(dev_y_pred_click, (-1,))
            dev_y_pred_purchase = np.reshape(dev_y_pred_purchase, (-1,))
            print('Evaluate on %d samples.' % nb_dev)
        finally:
            if enqueuer is not None:
                enqueuer.stop()

        dev_cvr = np.array([dev_y_pred_purchase[i] / dev_y_pred_click[i] for i in range(len(dev_y_pred_purchase))])
        return {'click_true': dev_y_true_click, 'ctcv_true': dev_y_true_purchase,
                'pctr': dev_y_pred_click, 'pctcvr': dev_y_pred_purchase,
                'ppcvr': dev_cvr
                 }

    def iterator(self, path, shuffle=False):
        '''
        Generator of data.
        :param path: data path.
        :param shuffle: whether to shuffle the data. It should be True for training set.
        :return: a batch data.
        '''
        prefetch = 50  # prefetch number of batches.
        batch_lines = []
        with open(path, 'r') as fr:
            lines = []
            # remove csv header
            fr.readline()
            for prefetch_line in fr:
                lines.append(prefetch_line)
                if len(lines) >= self.batch_size * prefetch:
                    if shuffle:
                        random.shuffle(lines)
                    for line in lines:
                        batch_lines.append(line.split(','))
                        if len(batch_lines) >= self.batch_size:
                            batch_array = np.array(batch_lines)
                            batch_lines = []
                            batch_data = {}
                            batch_data['y_click'] = batch_array[:,
                                                                0:1].astype(np.float64)
                            batch_data['y_purchase'] = batch_array[:,
                                                                   1:2].astype(np.float64)
                            for i, column in enumerate(all_columns):
                                batch_data['ids_{}'.format(
                                    column)] = batch_array[:, i + 2:i + 3].astype(np.int64)
                            yield batch_data
                    lines = []
            if 0 < len(lines) < self.batch_size * prefetch:
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    batch_lines.append(line.split(','))
                    if len(batch_lines) >= self.batch_size:
                        batch_array = np.array(batch_lines)
                        batch_lines = []
                        batch_data = {}
                        batch_data['y_click'] = batch_array[:,
                                                            0:1].astype(np.float64)
                        batch_data['y_purchase'] = batch_array[:,
                                                               1:2].astype(np.float64)
                        for i, column in enumerate(all_columns):
                            batch_data['ids_{}'.format(
                                column)] = batch_array[:, i + 2:i + 3].astype(np.int64)
                        yield batch_data
                if 0 < len(batch_lines) < self.batch_size:
                    batch_array = np.array(batch_lines)
                    batch_data = {}
                    batch_data['y_click'] = batch_array[:,
                                                        0:1].astype(np.float64)
                    batch_data['y_purchase'] = batch_array[:,
                                                           1:2].astype(np.float64)
                    for i, column in enumerate(all_columns):
                        batch_data['ids_{}'.format(
                            column)] = batch_array[:, i + 2:i + 3].astype(np.int64)
                    yield batch_data

    def evaluate(self, true_pred):
        '''
        Evaluation Metrics.
        :param true_pred: dict that contains the label and prediction.
        :return: click_auc, purchase_auc
        '''
        conv_true = true_pred['ctcv_true'][true_pred['click_true'] == 1]
        pcvr = true_pred['ppcvr'][true_pred['click_true'] == 1]
        
        auc_click = roc_auc_score(
            y_true=true_pred['click_true'],
            y_score=true_pred['pctr'])
        auc_conv = roc_auc_score(
            y_true=conv_true,
            y_score=pcvr)
        auc_ctcv = roc_auc_score(
            y_true=true_pred['ctcv_true'],
            y_score=true_pred['pctcvr'])
        
        click_logloss = -np.mean(true_pred['click_true'] * np.log(true_pred['pctr']) + (1 - true_pred['click_true']) * np.log((1 - true_pred['pctr'])))
        conv_logloss = -np.mean(conv_true * pcvr + (1 - conv_true) * np.log((1 - pcvr)))
        ctcv_logloss = -np.mean(true_pred['ctcv_true'] * np.log(true_pred['pctcvr']) + (1 - true_pred['ctcv_true']) * np.log((1 - true_pred['pctcvr'])))
        
        return [auc_click, auc_conv, auc_ctcv, click_logloss, conv_logloss, ctcv_logloss]
    

def save_df(true_pred, path):
    df = pd.DataFrame()
    df['click'] = true_pred['click_true']
    df['ctcv'] = true_pred['ctcv_true']
    df['pctr'] = true_pred['pctr']
    df['ppcvr'] = true_pred['ppcvr']
    df['pctcvr'] = true_pred['pctcvr']
    df.to_csv(path, index=False)


if __name__ == '__main__':
    data_path = 'data/'
    train_path, dev_path, test_path = os.path.join(data_path, 'ctr_cvr.train'), \
        os.path.join(
        data_path, 'ctr_cvr.dev'), os.path.join(
        data_path, 'ctr_cvr.test')
    max_q_size = 100
    workers = 1
    pickle_safe = False

    args.prefix = args.prefix.replace('"', '')
    print(eval(args.keep_prob))
    # Training
    t1 = time.time()
    model = AITM(vocabulary_size=vocabulary_size, embedding_dim=args.embedding_dim,
                 epoch=args.epoch,
                 batch_size=args.batch_size, learning_rate=args.lr, lamda=args.lamda,
                 keep_prob=eval(args.keep_prob), optimizer_type=args.optimizer, verbose=args.verbose, early_stop=args.early_stop,
                 prefix=args.prefix)
    
    file_name = 'PLE.csv'
    with open(file_name, 'a+') as f:
        f.writelines('model,pctr_auc,pcvr_auc,pctcvr_auc,ctr_loss,cvr_loss,ctcvr_loss\n')
    
    for i in range(6):
        model.fit(train_path, dev_path, pickle_safe=pickle_safe, max_q_size=max_q_size, workers=workers)
        # restore the best model
        # model.save_path = './checkpoint/best_ model_{}_epoch_{}.model'.format(args.prefix, 3)
        # model.saver.restore(model.sess, save_path=model.save_path)
        tf.keras.backend.set_learning_phase(0)

        # Test
        t = time.time()
        test_gen = model.iterator(test_path)
        true_pred = model.evaluate_generator(test_gen, max_q_size=max_q_size,
                                             workers=workers,
                                             pickle_safe=pickle_safe)
        test_result = model.evaluate(true_pred)
        print_info('Test', test_result, time.time() - t)
        
        with open(file_name, 'a+') as f:
            f.writelines('PLE_epoch%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % tuple([i] + test_result))

        save_df(true_pred, './{}_epoch_{}_result.csv'.format(args.prefix, i + 1))
        
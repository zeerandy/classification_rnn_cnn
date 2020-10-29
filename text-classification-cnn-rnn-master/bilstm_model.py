# coding: utf-8

import tensorflow as tf


class BiLSTMConfig(object):
    """CNN配置参数"""

    # 模型参数
    embedding_dim = 300      # 词向量维度
    seq_length = 128        # 序列长度
    num_classes = 2        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 3           # 隐藏层层数
    hidden_dim = 256        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru
    hiddenSize = 256

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 8         # 每批训练大小
    num_epochs = 15          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

# 构建模型
class BiLSTM(object):
    """
    Bi-LSTM 用于文本分类
    """

    def __init__(self, config, embedding):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.embeddings = embedding

        self.bilstm()

    def bilstm(self):
        """rnn模型"""

        def lstm_cell():  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("bilstm"):
            # # 多层rnn网络
            # cells = [dropout() for _ in range(self.config.num_layers)]
            # rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            # _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            # last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

            with tf.name_scope("Bi-LSTM"):
                # 定义前向LSTM结构
                lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=self.config.hiddenSize, state_is_tuple=True),
                    output_keep_prob=self.config.dropout_keep_prob)
                # 定义反向LSTM结构
                lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=self.config.hiddenSize, state_is_tuple=True),
                    output_keep_prob=self.config.dropout_keep_prob)

                # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                              self.embedding_inputs,
                                                                              dtype=tf.float32,
                                                                              scope="bi-lstm")

                # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                self.embedding_inputs = tf.concat(outputs, 2)

                # 去除最后时间步的输出作为全连接的输入
            finalOutput = self.embedding_inputs[:, -1, :]

            outputSize = self.config.hiddenSize * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
            last = tf.reshape(finalOutput, [-1, outputSize])

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))





        # # 词嵌入层
        # with tf.name_scope("embedding"):
        #     # 利用预训练的词向量初始化词嵌入矩阵
        #     self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
        #     # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        #     self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
        #
        # # 定义两层双向LSTM的模型结构
        # with tf.name_scope("Bi-LSTM"):
        #     for idx, hiddenSize in enumerate(self.config.hiddenSizes):
        #         with tf.name_scope("Bi-LSTM" + str(idx)):
        #             # 定义前向LSTM结构
        #             lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
        #                 tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
        #                 output_keep_prob=self.config.dropout_keep_prob)
        #             # 定义反向LSTM结构
        #             lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
        #                 tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
        #                 output_keep_prob=self.config.dropout_keep_prob)
        #
        #             # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
        #             # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
        #             # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
        #             outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
        #                                                                           self.embeddedWords,
        #                                                                           dtype=tf.float32,
        #                                                                           scope="bi-lstm" + str(idx))
        #
        #             # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
        #             self.embeddedWords = tf.concat(outputs, 2)
        #
        # # 去除最后时间步的输出作为全连接的输入
        # finalOutput = self.embeddedWords[:, -1, :]
        #
        # outputSize = config.model.hiddenSizes[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        # output = tf.reshape(finalOutput, [-1, outputSize])  # reshape成全连接层的输入维度
        #
        # # 全连接层的输出
        # with tf.name_scope("output"):
        #     outputW = tf.get_variable(
        #         "outputW",
        #         shape=[outputSize, 1],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #
        #     outputB = tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
        #     l2Loss += tf.nn.l2_loss(outputW)
        #     l2Loss += tf.nn.l2_loss(outputB)
        #     self.predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
        #     self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binaryPreds")
        #
        # # 计算二元交叉熵损失
        # with tf.name_scope("loss"):
        #     losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
        #     self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss
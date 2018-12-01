import tensorflow as tf

# reference to transH
# 1. multi-view attention, cal A, S
# 2. semantics space reconstruct: A+S=W'
#    a. two space, semantic space and word space, word==>semantics
#    b. three space, aspect,sentiment and word space,
# 3. copy mechanism for background word: A+S+W'=W'
# 4. min entropy for sentence and word consistency
# 5. regular term and prior knowledge


def attention_layer(query, keys, weights):
    '''
    :param query: words = [batch_size*seq_length, word_emb_size]
    :param keys: aspect/sentiment embeddings = [aspect/sentiment_num, aspect/sentiment_emb_size]
    :param weights: [q_emb_size, k_emb_size]
    :return: attention_weights = [batch_size*seq_length, v_emb_size]
    '''

    # [batch_size*seq_length, k_emb_size]
    query_ = tf.matmul(query, weights)
    # [batch_size*seq_length, k_num]
    attention_weights = tf.matmul(query_, tf.transpose(keys))
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)

    return attention_weights


def weighted_sum(weight, values):
    return tf.matmul(weight, values)


def cal_diversity(embeddings, rows):
    embeddings_ = tf.nn.l2_normalize(embeddings, axis=-1)
    unit_matrix = tf.eye(num_rows=rows)
    return tf.reduce_mean(tf.sqrt(unit_matrix - tf.matmul(embeddings_, tf.transpose(embeddings_))))


def hinge_loss(semantics_emb, semantics_emb_neg, semantics_emb_re, margin_value):
    # [batch_size*seq_length, semantic_emb_size]
    pos_distance = tf.reduce_sum(tf.square(semantics_emb-semantics_emb_re), axis=-1, keepdims=True)
    neg_distance = tf.reduce_sum(tf.square(semantics_emb_neg-semantics_emb_re), axis=-1, keepdims=True)
    margin = tf.ones_like(semantics_emb) * margin_value
    zeros = tf.zeros_like(semantics_emb)
    max_margin_loss = tf.concat([pos_distance+margin-neg_distance, zeros], axis=-1)
    max_margin_loss = tf.reduce_sum(tf.reduce_max(max_margin_loss, axis=-1))

    return max_margin_loss


def norm_regular():
    pass


def cal_entropy(distribution, smoothing_factor):
    return -tf.reduce_sum(tf.multiply(distribution, tf.log(distribution+smoothing_factor)), axis=-1, keepdims=True)


# two space: semantic space and word space
def asess_model(pos_data, neg_data, vocab_size, aspect_num, sentiment_num,
                word_emb_size, semantics_emb_size, margin, learning_rate,
                batch_size, seq_length, w2v_embeddings, mode, smoothing_factor):

    with tf.name_scope('embedding'):
        word_embeddings = tf.get_variable(name='word_embeddings',
                                          shape=[vocab_size, word_emb_size],
                                          initializer=tf.truncated_normal_initializer(),
                                          trainable=False)
        aspect_embeddings = tf.get_variable(name='aspect_embeddings',
                                            shape=[aspect_num, semantics_emb_size],
                                            initializer=tf.truncated_normal_initializer())
        sentiment_embeddings = tf.get_variable(name='sentiment_embeddings',
                                               shape=[sentiment_num, semantics_emb_size],
                                               initializer=tf.truncated_normal_initializer())

    tf.assign(word_embeddings, w2v_embeddings)
    # ------------- first step: multi-view attention rebuild -------------

    # [batch_size, seq_length, word_emb_size]
    pos_words_emb = tf.nn.embedding_lookup(word_embeddings, ids=pos_data)

    with tf.name_scope('att_weight'):
        aspect_att = tf.get_variable(name='aspect_att',
                                     shape=[word_emb_size, semantics_emb_size],
                                     initializer=tf.truncated_normal_initializer())
        sentiment_att = tf.get_variable(name='sentiment_att',
                                        shape=[word_emb_size, semantics_emb_size],
                                        initializer=tf.truncated_normal_initializer())

    aspect_weight = attention_layer(query=tf.reshape(pos_words_emb, shape=[-1, word_emb_size]),
                                    keys=aspect_embeddings,
                                    weights=aspect_att)
    sentiment_weight = attention_layer(query=tf.reshape(pos_words_emb, shape=[-1, word_emb_size]),
                                       keys=sentiment_embeddings,
                                       weights=sentiment_att)
    copy_weight = tf.get_variable(name='background_rec',
                                  shape=[semantics_emb_size, semantics_emb_size],
                                  initializer=tf.truncated_normal_initializer())

    # [batch_size*seq_length, semantics_emb_size]
    aspect_rep = weighted_sum(aspect_weight, aspect_embeddings)
    sentiment_rep = weighted_sum(sentiment_weight, sentiment_embeddings)

    # --------------  second step: embedding reconstruct in semantics space
    project_matrix = tf.get_variable(name='project_matrix',
                                     shape=[word_emb_size, semantics_emb_size],
                                     initializer=tf.truncated_normal_initializer())

    # [batch_size*seq_length, semantics_emb_size]
    semantics_emb = tf.matmul(tf.reshape(pos_words_emb, shape=[-1, word_emb_size]), project_matrix)

    # scaled dot product, [batch_size*seq_length, 1]
    scale_factor = tf.sqrt(tf.to_float(semantics_emb_size))
    aspect_prob = tf.reduce_sum(tf.multiply(aspect_rep, semantics_emb), axis=-1, keepdims=True) / scale_factor
    sentiment_prob = tf.reduce_sum(tf.multiply(sentiment_rep, semantics_emb), axis=-1, keepdims=True) / scale_factor

    # [batch_size*seq_length, 2, semantics_emb_size]
    semantics_emb_ = tf.reshape(tf.concat([aspect_rep, sentiment_rep], axis=-1), shape=[-1, 2, semantics_emb_size])
    semantics_prob = tf.reshape(tf.concat([aspect_prob, sentiment_prob], axis=-1), shape=[-1, 2, 1])
    # a*A+b*S=W', W' = [batch_size*seq_length, semantics_emb_size]
    semantics_emb_ = tf.reduce_sum(tf.multiply(semantics_emb_, semantics_prob), axis=1)

    # copy_gate
    gamma = tf.matmul(semantics_emb, copy_weight)
    gamma = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(gamma, semantics_emb), axis=-1, keepdims=True))

    semantics_emb_r = (1-gamma)*semantics_emb_ + gamma*semantics_emb

    # word semantics distribution = [batch_size*seq_length, aspect/sentiment num]
    word_aspect_weight = (1-gamma) * aspect_prob * aspect_weight
    word_sentiment_weight = (1-gamma) * sentiment_prob * sentiment_weight

    end_points = {}

    if neg_data is None:
        assert mode in ['word', 'sentence']
        if mode == 'word':
            end_points['background_prob'] = gamma
            end_points['word_aspect_weight'] = word_aspect_weight
            end_points['word_sentiment_weight'] = word_sentiment_weight
        else:
            # sentence semantics distribution = [batch_size*seq_length, aspect/sentiment num]
            sentence_aspect = tf.reduce_sum(tf.reshape(aspect_weight, shape=[-1, seq_length, aspect_num]), axis=1)
            sentence_aspect = tf.nn.softmax(sentence_aspect)
            sentence_sentiment = tf.reduce_sum(tf.reshape(sentiment_weight, shape=[-1, seq_length, sentiment_num]),
                                               axis=1)
            sentence_sentiment = tf.nn.softmax(sentence_sentiment)
            end_points['sentence_aspect_weight'] = sentence_aspect
            end_points['sentence_sentiment_weight'] = sentence_sentiment
    else:
        neg_words_emb = tf.nn.embedding_lookup(word_embeddings, ids=neg_data)
        semantics_emb_neg = tf.matmul(tf.reshape(neg_words_emb, shape=[-1, word_emb_size]), project_matrix)
        # max_margin loss
        reconstruct_loss = hinge_loss(semantics_emb=semantics_emb,
                                      semantics_emb_neg=semantics_emb_neg,
                                      semantics_emb_re=semantics_emb_r,
                                      margin_value=margin)
        # regular
        # consistency regular
        word_background_entropy = tf.reduce_mean(cal_entropy(tf.concat([gamma, 1-gamma], axis=-1), smoothing_factor))
        word_aspect_entropy = tf.reduce_mean((1-gamma)*aspect_prob*cal_entropy(aspect_weight, smoothing_factor))
        word_sentiment_entropy = tf.reduce_mean((1-gamma)*sentiment_prob*cal_entropy(sentiment_weight, smoothing_factor))

        # sentence semantics distribution = [batch_size*seq_length, aspect/sentiment num]
        sentence_aspect_weight = tf.reduce_sum(tf.reshape(aspect_weight, shape=[-1, seq_length, aspect_num]), axis=1)
        sentence_aspect_distribution = tf.nn.softmax(sentence_aspect_weight)
        sentence_sentiment_weight = tf.reduce_sum(tf.reshape(sentiment_weight, shape=[-1, seq_length, sentiment_num]),
                                           axis=1)
        sentence_sentiment_distribution = tf.nn.softmax(sentence_sentiment_weight)

        sentence_aspect_entropy = tf.reduce_mean(cal_entropy(sentence_aspect_distribution, smoothing_factor))
        sentence_sentiment_entropy = tf.reduce_mean(cal_entropy(sentence_sentiment_distribution, smoothing_factor))

        consistency_regular = (word_background_entropy + word_aspect_entropy + word_sentiment_entropy +
                               sentence_aspect_entropy + sentence_sentiment_entropy)

        # copy gate regular
        copy_regular = tf.reduce_mean(gamma)

        # diversity regular
        aspect_diversity = cal_diversity(aspect_embeddings, rows=aspect_num)
        sentiment_diversity = cal_diversity(sentiment_embeddings, rows=sentiment_num)
        diversity_regular = aspect_diversity + sentiment_diversity

        # norm regular
        pass

        total_loss = reconstruct_loss
        # total_loss += consistency_regular
        # total_loss += diversity_regular
        total_loss += copy_regular

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss)

        end_points = {'train_op': train_op,
                      'loss': total_loss,
                      'aspect_embeddings': aspect_embeddings,
                      'sentiment_embeddings': sentiment_embeddings}

    return end_points






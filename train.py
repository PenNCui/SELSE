from DataManager.dataProcesser import *
from DataManager.dataReader import *
from inference import *
from w2v import *


def train_model(args, domain, prefix):

    assert domain in ['restaurant', 'beer', 'game']

    # load args
    batch_size = int(args.get('exp-set', 'batch_size'))
    trunc_ratio = float(args.get('exp-set', 'trunc_ratio'))
    epoch_num = int(args.get('exp-set', 'epoch_num'))
    vocab_capacity = int(args.get('exp-set', 'vocab_capacity'))
    min_count = int(args.get('exp-set', 'min_count'))
    negative_num = int(args.get('exp-set', 'negative_num'))
    aspect_num = int(args.get('exp-set', 'aspect_num'))
    sentiment_num = int(args.get('exp-set', 'sentiment_num'))
    margin = float(args.get('exp-set', 'margin'))
    semantics_emb_size = int(args.get('exp-set', 'semantics_emb_size'))
    word_emb_size = int(args.get('exp-set', 'word_emb_size'))
    learning_rate = float(args.get('exp-set', 'learning_rate'))
    smoothing_factor = float(args.get('exp-set', 'smoothing_factor'))

    domain_data = os.path.join(args.get('paths', 'data_dir'), domain)
    train_data = os.path.join(domain_data, 'train_clean.txt')
    vocab_save_path = os.path.join(domain_data, 'vocab')
    data_save_path = os.path.join(domain_data, 'train.json')
    model_dir = os.path.join(args.get('paths', 'model_dir'))

    word2id, id2word, word2freq, word2sentence, max_length = build_vocab(file_path=train_data,
                                                                         vocab_capacity=vocab_capacity,
                                                                         min_count=min_count,
                                                                         save_path=vocab_save_path)

    w2v_model = load_model(args.get('paths', 'w2v'))
    w2v_embedding = load_embedding_matrix(model=w2v_model, vocab=word2id, embedding_size=word_emb_size)

    paddle_length = int(max_length*trunc_ratio)
    train_data_pos, train_data_neg = generate_data(file_path=train_data,
                                                   vocab=word2id,
                                                   word2freq=word2freq,
                                                   max_length=paddle_length,
                                                   negative_num=negative_num,
                                                   save_data=data_save_path)

    pos_placeholder = tf.placeholder(dtype=tf.int32, shape=[batch_size, paddle_length])
    neg_placeholder = tf.placeholder(dtype=tf.int32, shape=[batch_size, paddle_length])
    word_emb_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(word2id), word_emb_size])

    end_points = asess_model(pos_data=pos_placeholder,
                             neg_data=neg_placeholder,
                             vocab_size=len(word2id),
                             aspect_num=aspect_num,
                             sentiment_num=sentiment_num,
                             margin=margin,
                             semantics_emb_size=semantics_emb_size,
                             seq_length=paddle_length,
                             word_emb_size=word_emb_size,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             w2v_embeddings=word_emb_placeholder,
                             mode=None,
                             smoothing_factor=smoothing_factor)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        sess.run(init_op)
        batch_num = len(train_data_pos) // batch_size
        for epoch in range(epoch_num):
            for batch in range(batch_num):
                # print(train_data_pos[batch*batch_size:(batch+1)*batch_size])
                # print(train_data_neg[batch*batch_size:(batch+1)*batch_size])
                loss, _ = sess.run(fetches=[end_points['loss'], end_points['train_op']],
                                   feed_dict={pos_placeholder: train_data_pos[batch*batch_size:(batch+1)*batch_size],
                                              neg_placeholder: train_data_neg[batch*batch_size:(batch+1)*batch_size],
                                              word_emb_placeholder: w2v_embedding})
                print('In %d epoch, %d batch, loss is %f' % (epoch, batch, loss))
            saver.save(sess, save_path=os.path.join(model_dir, prefix))

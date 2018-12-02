from inference import *
from w2v import *
from DataManager.dataReader import *
import math


def get_word_semantics_weight(args, domain):
    assert domain in ['restaurant', 'game']
    # load args
    aspect_num = int(args.get('exp-set', 'aspect_num'))
    sentiment_num = int(args.get('exp-set', 'sentiment_num'))
    margin = float(args.get('exp-set', 'margin'))
    semantics_emb_size = int(args.get('exp-set', 'semantics_emb_size'))
    word_emb_size = int(args.get('exp-set', 'word_emb_size'))
    learning_rate = float(args.get('exp-set', 'learning_rate'))
    smoothing_factor = float(args.get('exp-set', 'smoothing_factor'))
    reconstruct_tolerance = float(args.get('exp-set', 'reconstruct_tolerance'))

    domain_data = os.path.join(args.get('paths', 'data_dir'), domain)
    model_dir = os.path.join(args.get('paths', 'model_dir'))
    vocab_save_path = os.path.join(domain_data, 'vocab')

    words_semantics = {}

    vocab = read_vocab(vocab_save_path)

    w2v_model = load_model(args.get('paths', 'w2v'))
    w2v_embedding = load_embedding_matrix(model=w2v_model, vocab=vocab, embedding_size=word_emb_size)

    pos_placeholder = tf.placeholder(dtype=tf.int32, shape=[1, ])
    word_emb_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(vocab), word_emb_size])

    end_points = asess_model(pos_data=pos_placeholder,
                             neg_data=None,
                             aspect_num=aspect_num,
                             sentiment_num=sentiment_num,
                             learning_rate=learning_rate,
                             mode='word',
                             semantics_emb_size=semantics_emb_size,
                             seq_length=1,
                             vocab_size=len(vocab),
                             w2v_embeddings=word_emb_placeholder,
                             word_emb_size=word_emb_size,
                             margin=margin,
                             batch_size=1,
                             smoothing_factor=smoothing_factor)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckp = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, ckp)
        for word in vocab:
            words_semantics[word] = sess.run(fetches=[end_points['background_prob'],
                                                      end_points['word_aspect_weight'],
                                                      end_points['word_sentiment_weight']],
                                             feed_dict={pos_placeholder: np.array([vocab[word]]),
                                                        word_emb_placeholder: w2v_embedding})
            # print(word, words_semantics[word])
    return words_semantics


def get_sentence_semantic_weight(args, domain):

    # load args
    aspect_num = int(args.get('exp-set', 'aspect_num'))
    sentiment_num = int(args.get('exp-set', 'sentiment_num'))
    margin = float(args.get('exp-set', 'margin'))
    semantics_emb_size = int(args.get('exp-set', 'semantics_emb_size'))
    word_emb_size = int(args.get('exp-set', 'word_emb_size'))
    learning_rate = float(args.get('exp-set', 'learning_rate'))
    smoothing_factor = float(args.get('exp-set', 'smoothing_factor'))
    reconstruct_tolerance = float(args.get('exp-set', 'reconstruct_tolerance'))

    model_dir = os.path.join(args.get('paths', 'model_dir'))
    domain_data_dir = os.path.join(args.get('paths', 'data_dir'), domain)
    file_path = os.path.join(domain_data_dir, 'test.txt')
    vocab_save_path = os.path.join(domain_data_dir, 'vocab')

    vocab = read_vocab(vocab_save_path)

    words_semantics = {}

    w2v_model = load_model(args.get('paths', 'w2v'))
    w2v_embedding = load_embedding_matrix(model=w2v_model, vocab=vocab, embedding_size=word_emb_size)

    pos_placeholder = tf.placeholder(dtype=tf.int32, shape=[1, ])
    word_emb_placeholder = tf.placeholder(dtype=tf.int32, shape=[len(vocab), word_emb_size])

    sentences = generate_data(file_path=file_path, max_length=63, negative_num=0,
                              save_data=None, vocab=vocab, word2freq=None)

    end_points = asess_model(pos_data=pos_placeholder,
                             neg_data=None,
                             aspect_num=aspect_num,
                             sentiment_num=sentiment_num,
                             learning_rate=learning_rate,
                             mode='sentence',
                             semantics_emb_size=semantics_emb_size,
                             seq_length=1,
                             vocab_size=len(vocab),
                             w2v_embeddings=w2v_embedding,
                             word_emb_size=word_emb_placeholder,
                             margin=margin,
                             batch_size=1,
                             smoothing_factor=smoothing_factor)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckp = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, ckp)
        for sentence in sentences:
            sentence_aspect_weight, sentence_sentiment_weight = sess.run(fetches=[end_points['sentence_aspect_weight'],
                                                                                  end_points['sentence_sentiment_weight'],],
                                                                         feed_dict={pos_placeholder: sentence,
                                                                                    word_emb_placeholder: w2v_embedding})


def rank_by_weight(word_semantics, aspect_num, sentiment_num, k):
    words_info = list(word_semantics.items())
    # print(words_info[0])
    # print(words_info[0][1][2])
    background_words = []
    aspect_words = [[] for i in range(aspect_num)]
    sentiment_words = [[] for j in range(sentiment_num)]
    words_info.sort(key=lambda info: info[1][0][0][0], reverse=True)
    for i in range(k):
        background_words.append([words_info[i][0], words_info[i][1][0][0][0]])
    for i in range(aspect_num):
        words_info.sort(key=lambda info: info[1][1][0][i], reverse=True)
        for j in range(k):
            aspect_words[i].append([words_info[j][0], words_info[j][1][1][0][i]])
    for i in range(sentiment_num):
        words_info.sort(key=lambda info: info[1][2][0][i], reverse=True)
        for j in range(k):
            sentiment_words[i].append([words_info[j][0], words_info[j][1][2][0][i]])

    return background_words, aspect_words, sentiment_words


def calculate_coherence_score(words, invert_index):
    '''
    :param words: aspect words or sentiment words
    :param invert_index: {word: document_id}
    :return: coherence_score
    '''
    coherence_score = 0
    word_num = len(words)
    for i in range(1, word_num):
        for j in range(0, i):
            co_occurrence = len(invert_index[words[i]] & invert_index[words[j]])
            document_frequency = len(invert_index[words[j]])
            coherence_score += math.log((co_occurrence+1)/document_frequency)
    return coherence_score





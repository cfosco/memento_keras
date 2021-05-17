import numpy as np
import os
import json
import i3d_config as cfg
from generator import load_vids_opencv, load_hmdb_npy_rgb
import keras
import keras.backend as K
import io

def prepare_caption_data(tokenized_captions_json_path, word_embeddings=None,
                        return_backward=False, caption_format='index_list', names_with_slash=True):

    '''
    Prepares tokenized captions from a given json to be fed to a captioning model.

    Inputs
    ------
        tokenized_captions_json_path: string, path to the tokenized captions json.
        word_embeddings: path to the word embeddings or pre-loaded embedding dictionary
        caption_format: string that determines the format and type of the
         return variables. Can take the following values:
            'index_triangular_matrix'
            'embedding_triangular_matrix'
            'index_list'
            'embedding_list'

    Returns
    -------
        input_captions: dictionary mapping a video name to the caption input of
        the model. Multiple different arrays can be returned depending on the
        value of caption_format.

        target_captions: dictionary mapping a video name to the desired target
        of the captioning task for that video. Changes depending on the value of
        caption_format.
    '''

    if type(word_embeddings) is str:
        word_embeddings = json.load(open(word_embeddings, 'r'))
        
    if isinstance(tokenized_captions_json_path, str):
        tokenized_captions = json.load(open(tokenized_captions_json_path, 'r'))
        ##DEBUG
        print("LOADING FROM JSON")
        print(list(tokenized_captions.keys())[:5])
        print(list(tokenized_captions.values())[:5])
    else:
        tokenized_captions = tokenized_captions_json_path
        ##DEBUG
        print("CAPTIONS ALREADY TOKENIZED")
        print(list(tokenized_captions.keys())[:5])
        print(list(tokenized_captions.values())[:5])
        
        
    input_captions = {}
    target_captions = {}
    
    for vid_name, cap_dict in tokenized_captions.items():
        if not names_with_slash:
            vid_name = vid_name.split("/")[0].replace("+","-")+'_'+"".join(vid_name.split("/")[1:])
            vid_name = vid_name[:-4]+".npy"

        input_captions[vid_name] = []
        target_captions[vid_name] = []
        for i,cap in enumerate(cap_dict['indexed_captions']):
            if caption_format == 'index_triangular_matrix':
                input_captions[vid_name].append(prepare_as_triangular(cap))
                target_captions[vid_name].append(cap[1:]+[0])
            elif caption_format == 'embedding_triangular_matrix':
                input_captions[vid_name].append(prepare_as_triangular(cap, embedding = word_embeddings))
                target_captions[vid_name].append(transform_into_embedding(cap_dict['tokenized_captions'][i], word_embeddings))
            elif caption_format == 'embedding_list':
                input_captions[vid_name].append(transform_into_embedding(cap_dict['tokenized_captions'][i], word_embeddings))
                target_captions[vid_name].append(transform_into_embedding(cap_dict['tokenized_captions'][i], word_embeddings))
            elif caption_format == 'index_list':
                input_captions[vid_name].append(cap)
                target_captions[vid_name].append(cap[1:]+[0])
            else:
                raise ValueError("Unknown caption format %s" % caption_format)



    return input_captions, target_captions


def load_videomem_captions(filepath):    
    d = {}
    for l in open(filepath, 'r'):
        name = l.split()[0]
        caption = l.split()[1].replace('-', ' ')
        d[name] = caption
#         input_captions[name] = tokenize(caption)
#         target_captions[name] = tokenize(caption)[1:]+[0]
    
    return d

def transform_into_embedding(list_of_words, embedding, offset_by_one=True, max_cap_len=cfg.MAX_CAP_LEN):

    emb_list = []

    emb_len = len(embedding[list_of_words[0]])

    c = 1 if offset_by_one else 0

    for l in list_of_words[c:]:
        emb_list.append(embedding[l])

    for i in range(max_cap_len-len(list_of_words) + c ):
        emb_list.append([0]*emb_len)

    # print("LEN OF LIST OF WORDS AND EMBEDDING LIST:", len(list_of_words), len(emb_list))
    # print(emb_list)

    return emb_list


def prepare_as_triangular(cap, return_backward=False, embedding=None):
    cap_len = next((i for i, x in enumerate(cap) if x==0), cfg.MAX_CAP_LEN)

    if embedding is not None:
        cap = transform_into_embedding(cap, embedding, offset_by_one=False)
    try:
        cap_tiled = np.tile(cap, (cap_len-1,1))
    except:
        print(len(cap))
        print(cap_len)
        print(cap)

    # Diagonalizing for forward direction
    cap_matrix_forw = np.tril(cap_tiled)

    if return_backward:
        cap_tiled_bw = np.tile(cap[:cap_len], (cap_len,1))
        # Diagonalizing for backward direction
        cap_matrix_backw = np.triu(cap_tiled_bw).fliplr()[::-1]


    return cap_matrix_forw

def prepare_one_caption_as_embedding_triangular_matrix():
    cap_len = next((i for i, x in enumerate(cap) if x==0), None)
    cap_tiled = np.tile(cap, (cap_len,1))

    # Diagonalizing for forward direction
    cap_matrix_forw = np.tril(cap_tiled)

    if return_backward:
        cap_tiled_bw = np.tile(cap, )
        # Diagonalizing for backward direction
        cap_matrix_backw = np.triu()

    return cap_matrix_forw


def load_videos_and_partial_caption(filenames, path, is_train=False, **kwargs):
    idx_cap = kwargs['idx_cap']
    idx_seq = kwargs['idx_seq']
    input_captions = kwargs['input_captions']

    vids = load_vids_opencv(filenames, path, is_train=is_train)
    caps = []
    for n in filenames:
        i = idx_seq[n]
        caps.append(input_captions[n][idx_cap][i])
    return [vids, np.array(caps)]

def load_npy_and_partial_caption(filenames, path, is_train=False, **kwargs):
    idx_cap = kwargs['idx_cap']
    idx_seq = kwargs['idx_seq']
    input_captions = kwargs['input_captions']

    vids = load_hmdb_npy_rgb(filenames, path, is_train=is_train)
    caps = []
    for n in filenames:
        i = idx_seq[n]
        caps.append(input_captions[n][idx_cap][i])
    return [vids, np.array(caps)]

def load_npy_and_full_caption(filenames, path, is_train=False, **kwargs):
    idx_cap = kwargs['idx_cap']
    input_captions = kwargs['input_captions']

    vids = load_hmdb_npy_rgb(filenames, path, is_train=is_train)
    caps = []
    for n in filenames:
        caps.append(input_captions[n][idx_cap])
    return [vids, np.array(caps)]

def load_labels_mem_alpha_words(filenames, str2label_dict, label_array=None, **kwargs):
    idx_cap = kwargs['idx_cap']
    idx_seq = kwargs['idx_seq']
    len_vocab = kwargs['len_vocab']

    # print("LABELS_LOADING: idx_cap %d, idx_seq %s" % (idx_cap, idx_seq))
    mem_alpha = []
    words = []
    for i, file in enumerate(filenames):
        mem = str2label_dict[file][0]
        alpha = str2label_dict[file][1]
        word = str2label_dict[file][2][idx_cap][idx_seq[file]]
        mem_alpha.append( [mem,alpha])
        onehot_word = np.zeros((len_vocab,))
        onehot_word[word] = 1
        words.append(onehot_word)

    return [mem_alpha, np.array(words)]

def load_labels_mem_alpha_caption(filenames, str2label_dict, label_array=None, **kwargs):
    idx_cap = kwargs['idx_cap']
    len_vocab = kwargs['len_vocab']

    # print("LABELS_LOADING: idx_cap %d, idx_seq %s" % (idx_cap, idx_seq))
    mem_alpha = []
    sentences = []
    for i, file in enumerate(filenames):
        mem = str2label_dict[file][0]
        alpha = str2label_dict[file][1]
        sentence = str2label_dict[file][2][idx_cap]
        mem_alpha.append([mem,alpha])
        sentences.append(keras.utils.to_categorical(sentence, num_classes=len_vocab))

    return [mem_alpha, sentences]

def load_labels_mem_caption(filenames, str2label_dict, label_array=None, **kwargs):
    idx_cap = kwargs['idx_cap']
    len_vocab = kwargs['len_vocab']

    mem = []
    sentences = []
    for i, file in enumerate(filenames):
#         print("str2label_dict[file]",str2label_dict[file])
        m = str2label_dict[file][0]
        sentence = str2label_dict[file][1][idx_cap]
        mem.append(m)
        sentences.append(keras.utils.to_categorical(sentence, num_classes=len_vocab))
#         print("sentences SHOULD BE CATEGORICAL",sentences)
    return [mem, sentences]


def load_labels_mot_caption(filenames, str2label_dict, label_array=None, **kwargs):
    idx_cap = kwargs['idx_cap']
    len_vocab = kwargs['len_vocab']

    mot_list = []
    sentences = []
    for i, file in enumerate(filenames):
        mot = str2label_dict[file][0] # Memory over Time
        sentence = str2label_dict[file][2][idx_cap]
        mot_list.append(mot)
        sentences.append(keras.utils.to_categorical(sentence, num_classes=len_vocab))

    return [mot_list, sentences]

def load_labels_mem_alpha_embedding(filenames, str2label_dict, label_array=None, **kwargs):

    idx_cap = np.random.randint(len(str2label_dict[filenames[0]][2]))
    mem_alpha = []
    sentence_embeddings = []
    k=-1
    for i, file in enumerate(filenames):
        mem = str2label_dict[file][0]
        alpha = str2label_dict[file][1]
        positive_emb = str2label_dict[file][2][idx_cap]
        negative_emb = str2label_dict[filenames[k]][2][idx_cap]
        mem_alpha.append([mem,alpha])
        sentence_embeddings.append([positive_emb, negative_emb])
        k+=1

    return [mem_alpha, sentence_embeddings]


def create_synched_loading_functions(video_loading_func, input_captions):
    idx_cap = 0
    idx_seq = {}

    def load_func(filenames, path, is_train=False):
        nonlocal idx_cap
        nonlocal idx_seq

        idx_cap = np.random.randint(len(input_captions[filenames[0]]))
        idx_seq = {n:np.random.randint(len(input_captions[n][idx_cap])) for n in filenames}

        # print("INPUT_LOADING: idx_cap %d, idx_seq %s" % (idx_cap, idx_seq))
        vids = video_loading_func(filenames, path, is_train=is_train)
        caps = []
        for n in filenames:
            i = idx_seq[n]
            caps.append(input_captions[n][idx_cap][i])
        return [vids, np.array(caps)]

    def load_labels_func(filenames, str2label_dict, label_array=None, reset=False):
        nonlocal idx_cap
        nonlocal idx_seq

        # print("LABELS_LOADING: idx_cap %d, idx_seq %s" % (idx_cap, idx_seq))
        mem_alpha = []
        words = []
        for i, file in enumerate(filenames):
            mem = str2label_dict[file][0]
            alpha = str2label_dict[file][1]
            word = str2label_dict[file][2][idx_cap][idx_seq[file]]
            mem_alpha.append( [mem,alpha])
            words.append(word)

        return [mem_alpha, words]

    return load_func, load_labels_func

def create_video_and_caption_loading_function(video_loading_func, caption_inputs):
    '''DEPRECATED'''

    counter_cap = 0
    counter_seq = 0

    def load_func(filenames, path, is_train=False, reset=False):
        nonlocal counter_cap
        nonlocal counter_seq
        if reset:
            counter_cap = 0
            counter_seq = 0
            return
        counter_cap += 1
        counter_seq += 1
        if not counter_cap % len(caption_inputs[filenames[0]]):
            counter_cap = 0
        if not counter_seq % len(caption_inputs[filenames[0]][0]):
            counter_seq = 0

        vids = video_loading_func(filenames, path, is_train=is_train)
        caps = np.array([caption_inputs[n][counter_cap][counter_seq] for n in filenames])
        return [vids, caps]

    return load_func

def create_video_and_word_label_function(caption_inputs):
    '''DEPRECATED'''

    counter_cap = 0
    counter_seq = 0

    def load_labels_func(filenames, str2label_dict, label_array=None, reset=False):

        nonlocal counter_cap
        nonlocal counter_seq
        if reset:
            counter_cap = 0
            counter_seq = 0
            return
        counter_cap += 1
        counter_seq += 1
        if not counter_cap % len(str2label_dict[filenames[0]][-1]):
            counter_cap = 0
        if not counter_seq % len(caption_inputs[filenames[0]][0]):
            counter_seq = 0

        mem_alpha = []
        words = []
        for i, file in enumerate(filenames):
            mem = str2label_dict[file][0]
            alpha = str2label_dict[file][1]
            word = str2label_dict[file][2][counter_cap][counter_seq]
            mem_alpha.append( [mem,alpha])
            words.append(word)

        return [mem_alpha, words]

    return load_labels_func

def add_caption_to_str2label(str2label, caption_targets):

    str2label_with_caps={}
    for k,v in caption_targets.items():
        if k not in str2label:
            print(k, "not in str2label")
            continue
        str2label_with_caps[k] = [str2label[k][0], str2label[k][1], v]

    return str2label_with_caps

def add_sentence_embeddings_to_str2label(str2label, embeddings_dict, average=True):

    str2label_with_emb={}
    for k,v in embeddings_dict.items():
        if average:
            emb = [np.mean(v['embedded'], axis=0)]
        else:
            emb = v['embedded']
        str2label_with_emb[k] = [str2label[k][0], str2label[k][1], emb]

    return str2label_with_emb

def generate_train_val_test_split(original_pickle, missing_vids, clean_vids):

    # original pickle
    pickle_ca = pickle.load(open(os.path.join(labels_path, 'old/train_test_split_memento.pkl'), 'rb'))

    # load missing vids
    removed_vids = json.load(open(os.path.join(labels_path, 'duplicates.json'))) + [['singing/5-5-2-5-8-3-5-2-16655258352_29.mp4']]

    # load clean vids
    clean_set = json.load(open("../../memento_data/clean_10k.json"))

    removed_vids2 = []
    for pair in removed_vids:
        for c in pair:
            spl = c.split('/')[0]
            spl = spl.replace('+','-')
            new_c = spl + '_' + "".join(c.split('/')[1:])
            removed_vids2.append(new_c)

    train_data=[]
    removed_from_train = 0
    for p in pickle_ca[0]:
        if p in removed_vids2:
            removed_from_train+=1
        else:
            train_data.append(p)

    val_data=[]
    removed_from_val = 0
    for p in pickle_ca[1]:
        if p in removed_vids2:
            removed_from_val+=1
        else:
            val_data.append(p)

    test_data=[]
    for c in clean_set:
        spl = c.split('/')[0]
        spl = spl.replace('+','-')
        new_c = spl + '_' + "".join(c.split('/')[1:])

        if new_c not in removed_vids2 and new_c not in train_data and new_c not in val_data:
            test_data.append(new_c)

    print("len(train_data), len(val_data), len(test_data)",len(train_data), len(val_data), len(test_data))

    final_splits = {"train":train_data, "val":val_data, "test":test_data}
    with open('../../memento_data/memento_train_val_test.json', 'w+') as f:
        json.dump(final_splits, f)





def to_words(caption_index, index_to_token_dict=None, vocab_path=cfg._VOCAB_PATH):

    if index_to_token_dict is None:
        vocab = json.load(open(vocab_path))
        idx2word = {i+1:elt for i, elt in enumerate(vocab)}
        idx2word[0]='0'

    if isinstance(caption_index, (int,float,np.int64)):
        return idx2word[caption_index]

    if isinstance(caption_index, (list,np.ndarray)):
        if isinstance(caption_index[0], (list,np.ndarray)):
            return [[idx2word[c] for c in cap] for cap in caption_index ]
        return [idx2word[c] for c in caption_index]

    return [idx2word[c] for c in caption_index]


def get_embedding_matrix(word_embeddings_path, vocab_path):

    vocab = json.load(open(vocab_path))
    word_embeddings = json.load(open(word_embeddings_path))

    assert len(vocab)==len(word_embeddings)

    word2idx = {elt:i+1 for i, elt in enumerate(vocab)}
    vocab_dim = len(vocab)
    emb_dim = len(list(word_embeddings.values())[0])

    embedding_matrix = np.zeros((vocab_dim+1, emb_dim))
    for word,embedding in word_embeddings.items():
        index = word2idx[word]
        embedding_matrix[index] = embedding

    return embedding_matrix

def generate_embedding_json_from_fasttext(tokenizer, fasttext_path, out_path, dummy_magn = 0.001, emb_size = 300):
    """Generates a json with a dictionary mapping word to embeddings. The embeddings are extracted from fasttext.
    If a word is not found in the embedding, a dummy embedding 
    """
    
    data = load_vec_file(fasttext_path)
    vocab_embedding = {}
    c=0

    for w in tokenizer.word_counts.keys():
        try:
            emb = np.array([float(t) for t in data[w]])
            emb_size = len(emb)
        except:
            c+=1
            print("Total not in embedding so far:", c, '. Appending dummy embedding vector - np.ones(EMB_SIZE)*(%s)' % dummy_magn)
            emb = np.ones(emb_size)*dummy_magn

        vocab_embedding[w] = list(emb)

    with open(out_path, 'w+') as f:
        json.dump( vocab_embedding, f )

    print("saved vocab embedding")
    
def load_vec_file(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(t) for t in tokens[1:]] 
    return data


def spellcheck_captions(captions_input, captions_output=None, save_json=True):
    
    videomem_captions = load_videomem_captions(captions_input)
    
    from autocorrect import Speller
    spell = Speller(lang='en')

    videomem_captions_spellchecked = {}

    # Spellcheck
    for name, cap in videomem_captions.items():
        videomem_captions_spellchecked[name] = spell(cap)
        
    # Save spellchecked captions
    if save_json:
        json.dump(videomem_captions_spellchecked, open(captions_output, "w+"))
    return videomem_captions
    
    

def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.sum(K.square(y_pred - y_true[:,0]), axis=-1) - 0.5*K.sum(K.square(y_pred - y_true[:,1]), axis=-1) + margin))


def reset_generator_state(gens):
    for gen in gens:
        gen.load_func(reset=True)
        gen.load_label_func(reset=True)

def show_results():
    pass

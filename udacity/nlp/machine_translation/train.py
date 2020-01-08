# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import multiprocessing
multiprocessing.set_start_method('spawn', True)

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import collections
import helper
import numpy as np
import project_tests as tests

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
# Load English data
english_sentences = helper.load_data('/home/houbowei/Artificial_intellectual_disabilities/udacity/nlp/machine_translation/data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('/home/houbowei/Artificial_intellectual_disabilities/udacity/nlp/machine_translation/data/small_vocab_fr')
# %%
for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))

# %%
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


# %%
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    
    tokenizer = Tokenizer(num_words=None,char_level=False)
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)
    # TODO: Implement
    return sequences, tokenizer
tests.test_tokenize(tokenize)

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)

for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))

# %%
def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    pad = pad_sequences(x, maxlen=length, dtype='int32', padding='post', truncating='post')
    return pad
tests.test_pad(pad)

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))

# %%
def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

# %%
def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')


# %%
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy


# %%
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import InputLayer

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the layers
    learning_rate = 0.01
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape[1:]))
    model.add(GRU(128,return_sequences = True))
    model.add(TimeDistributed(Dense(512,activation='tanh')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate))
    return model


tests.test_simple_model(simple_model)
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)
simple_rnn_model.fit(tmp_x, preproc_french_sentences,
                     batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
# %%
from keras.layers.embeddings import Embedding

def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    learning_rate = 0.01
    inputs = Input(shape=input_shape[1:])
    embedding_layer =Embedding(input_dim=english_vocab_size, output_dim=output_sequence_length,mask_zero=False)(inputs)
    hidden_layer = GRU(output_sequence_length,return_sequences=True)(embedding_layer)
    outputs=TimeDistributed(Dense(french_vocab_size,activation='softmax'))(hidden_layer)
    
    model=Model(inputs=inputs,outputs=outputs)
    model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])
    return model
tests.test_embed_model(embed_model)


# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

# TODO: Train the neural network
embed_rnn_model = embed_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    english_vocab_size,
    french_vocab_size)
embed_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# TODO: Print prediction(s)
print(logits_to_text(embed_rnn_model.predict(tmp_x[:1])[0],french_tokenizer))
      
# %%
from keras.layers import Bidirectional
def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    learning_rate=0.01

    inputs = Input(shape=input_shape[1:])
    hidden_layer = Bidirectional(GRU(output_sequence_length,return_sequences=True))(inputs)
    outputs = TimeDistributed(Dense(french_vocab_size,activation='softmax'))(hidden_layer)
    
    model = Model(input=inputs,outputs=outputs)
    model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])
    
    return model
tests.test_bd_model(bd_model)


# TODO: Train and Print prediction(s)

# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2],1))

# TODO: Train the neural network
bd_rnn_model = bd_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    english_vocab_size,
    french_vocab_size)
bd_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# TODO: Print prediction(s)
print(logits_to_text(bd_rnn_model.predict(tmp_x[:1])[0],french_tokenizer))
      

# %%
from keras.layers import Reshape
# reference link:https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/
def student_encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # OPTIONAL: Implement
    # Define an input sequence and process it.
    learning_rate = 1e-3
    model = Sequential()
    model.add(Reshape((3, 4), input_shape=(12,)))
    model.add(Embedding(english_vocab_size,
                        32,
                        input_length=input_shape[1],
                        mask_zero=True,
                        input_shape=input_shape[1:]))
    model.add(GRU(64))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(64, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(optimizer=Adam(learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tests.test_encdec_model(encdec_model)
encdec_rnn_model = student_encdec_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index),
    len(french_tokenizer.word_index))
encdec_rnn_model.summary()
tmp_x = np.squeeze(tmp_x)
encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(encdec_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

# %%
def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # OPTIONAL: Implement
    learning_rate = 0.01
    latent_dim = 128
    
    #Config Encoder
    encoder_inputs = Input(shape=input_shape[1:])
    encoder_gru = GRU(output_sequence_length)(encoder_inputs)
    encoder_outputs = Dense(latent_dim, activation='relu')(encoder_gru)
    
    #COnfig Decoder
    decoder_inputs = RepeatVector(output_sequence_length)(encoder_outputs)
    decoder_gru = GRU(latent_dim, return_sequences=True)(decoder_inputs)
    output_layer = TimeDistributed(Dense(french_vocab_size, activation='softmax'))
    outputs = output_layer(decoder_gru)

    #Create Model from parameters defined above
    model = Model(inputs=encoder_inputs, outputs=outputs)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_encdec_model(encdec_model)


# OPTIONAL: Train and Print prediction(s

# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
encdec_rnn_model = encdec_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index) + 1,
    len(french_tokenizer.word_index) + 1)
encdec_rnn_model.summary()
encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(encdec_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

### 模型 5：自定义（实现）
# 请使用从之前的模型中学到的所有知识创建一个包含嵌入和双向 RNN 的模型。


# %%
def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    
    learning_rate = 0.01
    
    
    inputs = Input(shape=input_shape[1:])
    embedding_layer = Embedding(input_dim=english_vocab_size,
                                output_dim=output_sequence_length,
                                mask_zero=False)(inputs)
    bd_layer = Bidirectional(GRU(output_sequence_length))(embedding_layer)
    encoding_layer = Dense(128, activation='relu')(bd_layer)
    decoding_layer = RepeatVector(output_sequence_length)(encoding_layer)
    output_layer = Bidirectional(GRU(128, return_sequences=True))(decoding_layer)
    outputs = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(output_layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model
tests.test_model_final(model_final)


print('Final Model Loaded')
# TODO: Train the final model

# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

# TODO: Train the neural network
final_rnn_model = model_final(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    english_vocab_size,
    french_vocab_size)
final_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# TODO: Print prediction(s)
print(logits_to_text(final_rnn_model.predict(tmp_x[:1])[0],french_tokenizer))
      

# %% [markdown]
# ## 预测（实现）

# %%
def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    # TODO: Train neural network using model_final
    
    model = model_final(x.shape,
                        y.shape[1],
                       len(x_tk.word_index) + 1,
                       len(y_tk.word_index) + 1)
    model.summary()
    model.fit(x, y, batch_size=1024, epochs=10, validation_split=0.2)

    
    ## DON'T EDIT ANYTHING BELOW THIS LINE
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)

# %% [markdown]
# ## 提交
# 准备好提交后，请完成以下步骤：
# 1. 阅读[审阅标准](https://review.udacity.com/#!/rubrics/1004/view)，确保提交内容满足所有要求，以便通过审阅
# 2. 生成此 notebook 的 HTML 版本
# 
#   - 运行下个单元格，以便尝试自动生成（这是在 Workspace 中的推荐方法）
#   - 依次转到**文件 -> 下载为 -> HTML (.html)**
#   - 从 shell 终端中使用 `nbconvert` 手动生成副本
# %% [markdown]
# $ pip install nbconvert
# $ python -m nbconvert machine_translation.ipynb
# %% [markdown]
# 3. 提交项目
# 
#   - 如果是在 Workspace 中，直接点击“提交项目”按钮（位于右下角）
#   
#   - 否则，压缩以下文件并提交
#   - `helper.py`
#   - `machine_translation.ipynb`
#   - `machine_translation.html`
#     - 可以通过依次转到**文件 -> 下载为 -> HTML (.html)** 导出 notebook。
# 
# ### 生成 html
# 
# **在运行下个单元格前先保存 notebook，以导出 HTML。** 然后提交项目。

# %%
# Save before you run this cell!
get_ipython().getoutput('jupyter nbconvert *.ipynb')


# %%
['[NbConvertApp] Converting notebook machine_translation.ipynb to html',
 '[NbConvertApp] Writing 305996 bytes to machine_translation.html']

# %% [markdown]
# ## 可选增强方式
# 
# 此项目侧重于学习各种机器翻译网络架构，但是没有根据最佳做法（将数据拆分为测试集和训练集）评估模型，因此模型的准确率夸大了。请使用 [`sklearn.model_selection.train_test_split()`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 函数创建单独的训练集和测试集，然后仅使用训练集重新训练每个模型，并使用预留的测试集评估预测准确率。“最佳”模型变了吗？


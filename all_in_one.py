from google.colab import drive
drive.mount ("/content/gdrive")

!pip install numpy==1.12.1
!pip install tensorflow==1.13.1

import os
import re
import copy
import codecs
import librosa
import matplotlib
import numpy as np
import unicodedata
from tqdm import tqdm
matplotlib.use ('pdf')
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from __future__ import print_function, division

class Hyperparams:
  prepro = False
  vocab = "PE абвгдеёжзийклмнопрстуфхцчшщъыьэюяәіңғүұқөһ'.?"
  data = "/content/gdrive/MyDrive/ISSAI_KazakhTTS/"
  test_data = "/content/gdrive/MyDrive/ISSAI_KazakhTTS/transcript.txt"
  max_duration = 56.0

  sr = 22050
  n_fft = 2048
  frame_shift = 0.0125
  frame_length = 0.05
  hop_length = int (sr * frame_shift)
  win_length = int (sr * frame_length)
  n_mels = 80
  power = 1.2
  n_iter = 50
  preemphasis = 0.97
  max_db = 100
  ref_db = 20

  embed_size = 256
  encoder_num_banks = 16
  decoder_num_banks = 8
  num_highwaynet_blocks = 4
  r = 5
  dropout_rate = 0.5

  lr = 0.001
  logdir = "/content/gdrive/MyDrive/logdir/01"
  sampledir = "/content/gdrive/MyDrive/samples"
  batch_size = 32
# batch_size = 16
# batch_size = 8
# batch_size = 1
##### FIFTH
# from hyperparams import Hyperparams as hp
hp = Hyperparams ()

def get_spectrograms (fpath):
  # num = np.random.randn ()
  # if num < 0.2:
  #   y, sr = librosa.load (fpath, sr = hp.sr)
  # else:
  #   if num < 0.4:
  #     tempo = 1.1
  #   elif num < 0.6:
  #     tempo = 1.2
  #   elif num < 0.8:
  #     tempo = 0.9
  #   else:
  #     tempo = 0.8
  # cmd = "ffmpeg -i {} -y ar {} -hide_banner -loglevel panic -ac 1 -filter:a atempo={} -vn temp.wav".format (fpath, hp.sr, tempo)
  # os.system (cmd)
  # y, sr = librosa.load ('temp.wav', sr = hp.sr)

  y, sr = librosa.load (fpath, sr = hp.sr)
  y, _ = librosa.effects.trim (y)
  y = np.append (y[0], y[1:] - hp.preemphasis * y[:-1])
  linear = librosa.stft (y = y, n_fft = hp.n_fft, hop_length = hp.hop_length, win_length = hp.win_length)
  mag = np.abs (linear) # (1 + n_fft//2, T)
  mel_basis = librosa.filters.mel (hp.sr, hp.n_fft, hp.n_mels) # (n_mels, 1 + n_fft//2)
  mel = np.dot (mel_basis, mag) # (n_mels, t)
  mel = 20 * np.log10 (np.maximum (1e-5, mel))
  mag = 20 * np.log10 (np.maximum (1e-5, mag))

  mel = np.clip ((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
  mag = np.clip ((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

  mel = mel.T.astype (np.float32) # (T, n_mels)
  mag = mag.T.astype (np.float32) # (T, 1 + n_fft//2)

  return mel, mag

def spectrogram2wav (mag):
  mag = mag.T
  mag = (np.clip (mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
  mag = np.power (10.0, mag * 0.05)
  wav = griffin_lim (mag)
  wav = signal.lfilter ([1], [1, -hp.preemphasis], wav)
  wav, _ = librosa.effects.trim (wav)
  return wav.astype (np.float32)

def griffin_lim (spectrogram):
  X_best = copy.deepcopy (spectrogram)
  for i in range (hp.n_iter):
    X_t = invert_spectrogram (X_best)
    est = librosa.stft (X_t, hp.n_fft, hp.hop_length, win_length = hp.win_length)
    phase = est / np.maximum (1e-8, np.abs (est))
    X_best = spectrogram * phase
  X_t = invert_spectrogram (X_best)
  y = np.real (X_t)
  return y

def invert_spectrogram (spectrogram):
  return librosa.istft (spectrogram, hp.hop_length, win_length = hp.win_length, window = "hann")

def plot_alignment (alignment, gs):
  fig, ax = plt.subplots ()
  im = ax.imshow (alignment)
  fig.colorbar (im)
  plt.title ('{} Steps'.format (gs))
  plt.savefig ('{}/alignment_{}k.png'.format (hp.logdir, gs // 1000), format = 'png')

def learning_rate_decay (init_lr, global_step, warmup_steps = 4000.0):
  step = tf.cast (global_step + 1, dtype = tf.float32)
  return init_lr * warmup_steps ** 0.5 * tf.minimum (step * warmup_steps ** -1.5, step ** -0.5)

def load_spectrograms (fpath):
  fname = os.path.basename (fpath)
  mel, mag = get_spectrograms (fpath)
  t = mel.shape[0]
  num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
  mel = np.pad (mel, [[0, num_paddings], [0, 0]], mode = "constant")
  mag = np.pad (mag, [[0, num_paddings], [0, 0]], mode = "constant")
  return fname, mel.reshape ((-1, hp.n_mels * hp.r)), mag

def load_vocab ():
  char2idx = {char: idx for idx, char in enumerate (hp.vocab)}
  idx2char = {idx: char for idx, char in enumerate (hp.vocab)}
  return char2idx, idx2char

def text_normalize(text):
  text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn') # Strip accents

  text = text.lower()
  text = re.sub("[^{}]".format(hp.vocab), " ", text)
  text = re.sub("[ ]+", " ", text)
  return text

def load_data(mode="train"):
  # Load vocabulary
  char2idx, idx2char = load_vocab()

  if mode in ("train", "eval"):
    fpaths, text_lengths, texts = [], [], []
    transcript = os.path.join(hp.data, 'transcript.txt')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    total_hours = 0
    if mode=="train":
      lines = lines[1:]
    else:
      lines = lines[:1]

    for line in lines:
    # fname, _, text = line.strip().split("|")
      fname, _, text, duration_time_spent = line.strip().split("|")
    # print(f"fname = {fname}")

    # fpath = os.path.join(hp.data, "wavs", fname + ".wav")
      fpath = os.path.join(hp.data, fname)
    # print(f"fpath = {fpath}")
      fpaths.append(fpath)

      text = text_normalize(text) + "E"  # E: EOS
      text = [char2idx[char] for char in text]
    # print(f"text = {text}")
      text_lengths.append(len(text))
      texts.append(np.array(text, np.int32).tostring())

    return fpaths, text_lengths, texts
  else:
    lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
    sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
    lengths = [len(sent) for sent in sents]
    maxlen = sorted(lengths, reverse=True)[0]
    texts = np.zeros((len(sents), maxlen), np.int32)
    for i, sent in enumerate(sents):
      texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts

def get_batch():
  with tf.device('/cpu:0'):
    # Load data
    fpaths, text_lengths, texts = load_data() # list
    maxlen, minlen = max(text_lengths), min(text_lengths)

    # Calc total batch count
    num_batch = len(fpaths) // hp.batch_size

    fpaths = tf.convert_to_tensor(fpaths)
    text_lengths = tf.convert_to_tensor(text_lengths)
    texts = tf.convert_to_tensor(texts)

    # Create Queues
    fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)
  # fpath, text_length, text = tf.data.Dataset.from_tensor_slices([fpaths, text_lengths, texts])

    text = tf.decode_raw(text, tf.int32)  # (None,)

    if hp.prepro:
      def _load_spectrograms(fpath):
        fname = os.path.basename(fpath)
        mel = "mels/{}".format(fname.replace("wav", "npy"))
        mag = "mags/{}".format(fname.replace("wav", "npy"))
        return fname, np.load(mel), np.load(mag)

      fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
    else:
      fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
    fname.set_shape(())
    text.set_shape((None,))
    mel.set_shape((None, hp.n_mels*hp.r))
    mag.set_shape((None, hp.n_fft//2+1))

    # Batching
    _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(input_length=text_length, tensors=[text, mel, mag, fname], batch_size=hp.batch_size, bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)], num_threads=16, capacity=hp.batch_size * 4, dynamic_pad=True)

  return texts, mels, mags, fnames, num_batch

def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, num_units], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    if zero_pad:
      lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
  return tf.nn.embedding_lookup(lookup_table, inputs)

def bn(inputs, is_training=True, activation_fn=None, scope="bn", reuse=None):
  inputs_shape = inputs.get_shape()
  inputs_rank = inputs_shape.ndims

    if inputs_rank in [2, 3, 4]:
    if inputs_rank == 2:
      inputs = tf.expand_dims(inputs, axis=1)
      inputs = tf.expand_dims(inputs, axis=2)
    elif inputs_rank == 3:
      inputs = tf.expand_dims(inputs, axis=1)

    outputs = tf.contrib.layers.batch_norm(inputs=inputs, center=True, scale=True, updates_collections=None, is_training=is_training, scope=scope, fused=True, reuse=reuse)
    # restore original shape
    if inputs_rank == 2:
      outputs = tf.squeeze(outputs, axis=[1, 2])
    elif inputs_rank == 3:
      outputs = tf.squeeze(outputs, axis=1)
  else:  # fallback to naive batch norm
    outputs = tf.contrib.layers.batch_norm(inputs=inputs, center=True, scale=True, updates_collections=None, is_training=is_training, scope=scope, reuse=reuse, fused=False)
  if activation_fn is not None:
    outputs = activation_fn(outputs)
  return outputs

def conv1d(inputs, filters=None, size=1, rate=1, padding="SAME", use_bias=False, activation_fn=None, scope="conv1d", reuse=None):
  with tf.variable_scope(scope):
    if padding.lower()=="causal":
        # pre-padding for causality
        pad_len = (size - 1) * rate  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list[-1]

    params = {"inputs":inputs, "filters":filters, "kernel_size":size,
            "dilation_rate":rate, "padding":padding, "activation":activation_fn,
            "use_bias":use_bias, "reuse":reuse}

    outputs = tf.layers.conv1d(**params)
  return outputs

def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    outputs = conv1d(inputs, hp.embed_size//2, 1) # k=1
    for k in range(2, K+1): # k = 2...K
        with tf.variable_scope("num_{}".format(k)):
            output = conv1d(inputs, hp.embed_size // 2, k)
            outputs = tf.concat((outputs, output), -1)
    outputs = bn(outputs, is_training=is_training, activation_fn=tf.nn.relu)
  return outputs

def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    if num_units is None:
      num_units = inputs.get_shape().as_list[-1]

    cell = tf.contrib.rnn.GRUCell(num_units)
    if bidirection:
      cell_bw = tf.contrib.rnn.GRUCell(num_units)
      outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
      return tf.concat(outputs, 2)
    else:
      outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
      return outputs

def attention_decoder(inputs, memory, num_units=None, scope="attention_decoder", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    if num_units is None:
      num_units = inputs.get_shape().as_list[-1]

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory)
    decoder_cell = tf.contrib.rnn.GRUCell(num_units)
    cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, num_units, alignment_history=True)
    outputs, state = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32) #( N, T', 16)

  return outputs, state

def prenet(inputs, num_units=None, is_training=True, scope="prenet", reuse=None):
  if num_units is None:
    num_units = [hp.embed_size, hp.embed_size//2]

  with tf.variable_scope(scope, reuse=reuse):
    outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
    outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout1")
    outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
    outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout2")
  return outputs # (N, ..., num_units[1])

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
  if not num_units:
    num_units = inputs.get_shape()[-1]

  with tf.variable_scope(scope, reuse=reuse):
    H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
    T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1.0), name="dense2")
    outputs = H*T + inputs*(1.-T)
  return outputs

def encoder(inputs, is_training=True, scope="encoder", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    # Encoder pre-net
    prenet_out = prenet(inputs, is_training=is_training) # (N, T_x, E/2)

    # Encoder CBHG 
    ## Conv1D banks
    enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, T_x, K*E/2)

    ## Max pooling
    enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  # (N, T_x, K*E/2)

    ## Conv1D projections
    enc = conv1d(enc, filters=hp.embed_size//2, size=3, scope="conv1d_1") # (N, T_x, E/2)
    enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

    enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2")  # (N, T_x, E/2)
    enc = bn(enc, is_training=is_training, scope="conv1d_2")

    enc += prenet_out # (N, T_x, E/2) # residual connections

    ## Highway Nets
    for i in range(hp.num_highwaynet_blocks):
      enc = highwaynet(enc, num_units=hp.embed_size//2, scope='highwaynet_{}'.format(i))
    ## Bidirectional GRU
    memory = gru(enc, num_units=hp.embed_size//2, bidirection=True) # (N, T_x, E)

  return memory

def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    # Decoder pre-net
    inputs = prenet(inputs, is_training=is_training)  # (N, T_y/r, E/2)

    # Attention RNN
    dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size) # (N, T_y/r, E)

    ## for attention monitoring
    alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

    # Decoder RNNs
    dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1") # (N, T_y/r, E)
    dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2") # (N, T_y/r, E)

    # Outputs => (N, T_y/r, n_mels*r)
    mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)

  return mel_hats, alignments

def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    # Restore shape -> (N, Ty, n_mels)
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

    # Conv1D bank
    dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training) # (N, T_y, E*K/2)

    # Max pooling
    dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same") # (N, T_y, E*K/2)

    ## Conv1D projections
    dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, T_x, E/2)
    dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

    dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  # (N, T_x, E/2)
    dec = bn(dec, is_training=is_training, scope="conv1d_2")

    # Extra affine transformation for dimensionality sync
    dec = tf.layers.dense(dec, hp.embed_size//2) # (N, T_y, E/2)

    # Highway Nets
    for i in range(4):
      dec = highwaynet(dec, num_units=hp.embed_size//2, scope='highwaynet_{}'.format(i)) # (N, T_y, E/2)

    # Bidirectional GRU    
    dec = gru(dec, hp.embed_size//2, bidirection=True) # (N, T_y, E)

    # Outputs => (N, T_y, 1+n_fft//2)
    outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

  return outputs

class Graph:
  def __init__(self, mode="train"):
    self.char2idx, self.idx2char = load_vocab()
    is_training=True if mode=="train" else False
    if mode=="train":
      self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
    elif mode=="eval":
    # self.x = tf.placeholder(tf.int32, shape=(None, None), name="useless_x")
      self.x = tf.placeholder(tf.int32, shape=(None, None))
      self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
      self.z = tf.placeholder(tf.float32, shape=(None, None, 1+hp.n_fft//2))
      self.fnames = tf.placeholder(tf.string, shape=(None,))
    else: # Synthesize
      self.x = tf.placeholder(tf.int32, shape=(None, None))
      self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels * hp.r))
    self.encoder_inputs = embed(self.x, len(hp.vocab), hp.embed_size) # (N, T_x, E)
    self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)
    self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)
    with tf.variable_scope("net"):
      self.memory = encoder(self.encoder_inputs, is_training=is_training) # (N, T_x, E)
      self.y_hat, self.alignments = decoder1(self.decoder_inputs, self.memory, is_training=is_training) # (N, T_y//r, n_mels*r)
      self.z_hat = decoder2(self.y_hat, is_training=is_training) # (N, T_y//r, (1+n_fft//2)*r)
    self.audio = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)
    if mode in ("train", "eval"):
      self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y))
      self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z))
      self.loss = self.loss1 + self.loss2
    # self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.global_step = tf.Variable(1, name='global_step', trainable=False)
      self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
      self.gvs = self.optimizer.compute_gradients(self.loss)
      self.clipped = []
      for grad, var in self.gvs:
        grad = tf.clip_by_norm(grad, 5.)
        self.clipped.append((grad, var))
      self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
      tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
      tf.summary.scalar('{}/loss'.format(mode), self.loss)
      tf.summary.scalar('{}/lr'.format(mode), self.lr)
      tf.summary.image("{}/mel_gt".format(mode), tf.expand_dims(self.y, -1), max_outputs=1)
      tf.summary.image("{}/mel_hat".format(mode), tf.expand_dims(self.y_hat, -1), max_outputs=1)
      tf.summary.image("{}/mag_gt".format(mode), tf.expand_dims(self.z, -1), max_outputs=1)
      tf.summary.image("{}/mag_hat".format(mode), tf.expand_dims(self.z_hat, -1), max_outputs=1)
      tf.summary.audio("{}/sample".format(mode), tf.expand_dims(self.audio, 0), hp.sr)
      self.merged = tf.summary.merge_all()
!pip show numpy tensorflow

if __name__ == '__main__':
  g = Graph(); print("Training Graph loaded")
  # with g.graph.as_default():
# iteration_counter = 0
  sv = tf.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)
# print("Before \"with\"")
  with sv.managed_session() as sess:
  # while iteration_counter < 1:
      for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
        print("\n\t\tIn for _")
      # print("iteration_counter < 1")
        _, gs = sess.run([g.train_op, g.global_step])
      # print("\tStill in while 1, after sess.run")
        print("\n\t\tAfter sess.run")
        # Write checkpoint files
        if gs % 1000 == 0:
        # print("\n\t\t\tIn gs % 1000")
          sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs//1000))
        # print("\n\t\t\tAfter sv.saver.save")

            # plot the first alignment for logging
          al = sess.run(g.alignments)
        # print("\n\t\t\tAfter sess.run")
          plot_alignment(al[0], gs)
        # print("\n\t\t\tAfter plot_alignment")
  print("Training done")

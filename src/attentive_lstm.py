from keras import backend as K
from keras import regularizers, constraints, initializers, activations
# from keras.layers.recurrent import RNN, Layer, _generate_dropout_mask, _generate_dropout_ones
from keras.layers import RNN, LSTM, Layer
from keras.engine import InputSpec
from keras.legacy import interfaces
import warnings


# Copied from original keras source
def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):

    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class AttentiveLSTMCell(Layer):
    """Cell class for the AttentiveLSTM layer.
    """

    def __init__(self, units,
                 annotations,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(AttentiveLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.annotations = annotations

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):

        # annotation dimmensions
        self.batch_size, self.annotation_timesteps, self.annotation_units = K.int_shape(self.annotations)

        input_dim = input_shape[-1]  # size of a feature. i.e, the size of a word embedding
        input_dim += self.annotation_units  # give space for context vector (will be appended at each timestep)

        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 4,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 6,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3: self.units * 4]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3: self.units * 4]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

        # Built attention mechanism.

        # energy is calculated by
        # j being for the index of what to attent to, t being the current timestep of this deocder
        # e_{j,t} = V_a * tanh(W_a*h_t + U_a*h_j)

        self.kernel_u = self.add_weight(shape=(self.annotation_units, self.units),
                                        name='attentive_kernel_u',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.kernel_w = self.add_weight(shape=(self.units, self.units),
                                        name='attentive_kernel_w',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.kernel_v = self.add_weight(shape=(self.units,),
                                        name='attentive_kernel_v',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_u = self.bias[self.units * 4: self.units * 5]
            self.bias_v = self.bias[self.units * 5:]
        else:
            self.bias_u = None
            self.bias_v = None

        # U_a*h_j \forall j
        self._uh = _time_distributed_dense(self.annotations, self.kernel_u, b=self.bias_u,
                                           input_dim=self.annotation_units,
                                           timesteps=self.annotation_timesteps,
                                           output_dim=self.units)

        self.built = True

    def call(self, inputs, states, training=None):

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1] + self.annotation_units),
                self.dropout,
                training=training,
                count=4)

        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # attention mechanism

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(h_tm1, self.annotation_timesteps)

        # multiplty the weight matrix with the repeated (current) hidden state
        _Wxstm = K.dot(_stm, self.kernel_w)

        # calculate the attention probabilities
        et = K.dot(activations.tanh(_Wxstm + self._uh), K.expand_dims(self.kernel_v))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.annotation_timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.annotations, axes=1), axis=1)

        # append the context vector to the inputs
        inputs = K.concatenate([inputs, context])

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)

            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1

            i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.recurrent_kernel_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o))

        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}

        base_config = super(AttentiveLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentiveLSTM(RNN):
    """Long-Short Term Memory layer used for decoding where a context vector
    from the docoding sequence is appended to the input of the encoder at every
    timestep
    # References:
        Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
        "Neural machine translation by jointly learning to align and translate."
        arXiv preprint arXiv:1409.0473 (2014).
    """

    def __init__(self, units,
                 annotations,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):

        if K.backend() == 'theano':
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        self.annotations = annotations

        cell = AttentiveLSTMCell(units,
                                 self.annotations,
                                 activation=activation,
                                 recurrent_activation=recurrent_activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 recurrent_initializer=recurrent_initializer,
                                 unit_forget_bias=unit_forget_bias,
                                 bias_initializer=bias_initializer,
                                 kernel_regularizer=kernel_regularizer,
                                 recurrent_regularizer=recurrent_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 kernel_constraint=kernel_constraint,
                                 recurrent_constraint=recurrent_constraint,
                                 bias_constraint=bias_constraint,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 implementation=implementation)

        super(AttentiveLSTM, self).__init__(cell,
                                            return_sequences=return_sequences,
                                            return_state=return_state,
                                            go_backwards=go_backwards,
                                            stateful=stateful,
                                            unroll=unroll,
                                            **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None

        if initial_state is not None:
            raise ValueError("you can not send a hidden state into an attention"
                             "lstm. This is becasue the attention mechanism describes"
                             "the initial states already.")

        c0 = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        c0 = K.sum(c0, axis=(1, 2))  # (samples,)
        c0 = K.expand_dims(c0)  # (samples, 1)
        c0 = K.tile(c0, [1, self.cell.state_size[1]])

        if self.cell.units == self.cell.annotation_units:
            h0 = activations.tanh(K.dot(self.annotations[:, 0], self.cell.kernel_w))
            initial_state = [h0, c0]
        elif 2*self.cell.units == self.cell.annotation_units:
            # bidireciton is used. Take the backwards direction (as described in paper)
            h0 = activations.tanh(K.dot(self.annotations[:, 0, self.cell.units:], self.cell.kernel_w))
            initial_state = [h0, c0]
        else:
            warnings.warn("annotation (attention) shapes do not allow for initial state setting")
            initial_state = None

        return super(AttentiveLSTM, self).call(inputs,
                                               mask=mask,
                                               training=training,
                                               initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    
    
class AttentionLSTM(LSTM):
    """LSTM with attention mechanism
    This is an LSTM incorporating an attention mechanism into its hidden states.
    Currently, the context vector calculated from the attended vector that is fed
    into the model's internal states, closely following the model by Xu et al.
    (2016, Sec. 3.1.2), using a soft attention model following
    Bahdanau et al. (2014).
    The layer expects two inputs instead of the usual one:
        1. the "normal" layer input; and
        2. a 3D vector to attend.
    Args:
        attn_activation: Activation function for attentional components
        attn_init: Initialization function for attention weights
        output_alpha (boolean): If true, outputs the alpha values, i.e.,
            what parts of the attention vector the layer attends to at each
            timestep.
    References:
        * Bahdanau, Cho & Bengio (2014), "Neural Machine Translation by Jointly
          Learning to Align and Translate", <https://arxiv.org/pdf/1409.0473.pdf>
        * Xu, Ba, Kiros, Cho, Courville, Salakhutdinov, Zemel & Bengio (2016),
          "Show, Attend and Tell: Neural Image Caption Generation with Visual
          Attention", <http://arxiv.org/pdf/1502.03044.pdf>
    See Also:
        `LSTM`_ in the Keras documentation.
        .. _LSTM: http://keras.io/layers/recurrent/#lstm
    """
    def __init__(self, *args, attn_activation='tanh', attn_init='orthogonal',
                 output_alpha=False, **kwargs):
        self.attn_activation = activations.get(attn_activation)
        self.attn_init = initializers.get(attn_init)
        self.output_alpha = output_alpha
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise Exception('Input to AttentionLSTM must be a list of '
                            'two tensors [lstm_input, attn_input].')
        
        ## DEBUG
        print("input_shape",input_shape)
        input_shape, attn_input_shape = input_shape
        print("input_shape",input_shape)
        super().build(input_shape)
        self.input_spec.append(InputSpec(shape=attn_input_shape))

        # weights for attention model
        self.U_att = self.inner_init((self.output_dim, self.output_dim),
                                     name='{}_U_att'.format(self.name))
        self.W_att = self.attn_init((attn_input_shape[-1], self.output_dim),
                                    name='{}_W_att'.format(self.name))
        self.v_att = self.init((self.output_dim, 1),
                               name='{}_v_att'.format(self.name))
        self.b_att = K.zeros((self.output_dim,), name='{}_b_att'.format(self.name))
        self.trainable_weights += [self.U_att, self.W_att, self.v_att, self.b_att]

        # weights for incorporating attention into hidden states
        if self.consume_less == 'gpu':
            self.Z = self.init((attn_input_shape[-1], 4 * self.output_dim),
                               name='{}_Z'.format(self.name))
            self.trainable_weights += [self.Z]
        else:
            self.Z_i = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_i'.format(self.name))
            self.Z_f = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_f'.format(self.name))
            self.Z_c = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_c'.format(self.name))
            self.Z_o = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_o'.format(self.name))
            self.trainable_weights += [self.Z_i, self.Z_f, self.Z_c, self.Z_o]
            self.Z = K.concatenate([self.Z_i, self.Z_f, self.Z_c, self.Z_o])

        # weights for initializing states based on attention vector
        if not self.stateful:
            self.W_init_c = self.attn_init((attn_input_shape[-1], self.output_dim),
                                           name='{}_W_init_c'.format(self.name))
            self.W_init_h = self.attn_init((attn_input_shape[-1], self.output_dim),
                                           name='{}_W_init_h'.format(self.name))
            self.b_init_c = K.zeros((self.output_dim,),
                                    name='{}_b_init_c'.format(self.name))
            self.b_init_h = K.zeros((self.output_dim,),
                                    name='{}_b_init_h'.format(self.name))
            self.trainable_weights += [self.W_init_c, self.b_init_c,
                                       self.W_init_h, self.b_init_h]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return super().get_output_shape_for(input_shape[0])

    def compute_mask(self, input, input_mask=None):
        if input_mask is not None:
            input_mask = input_mask[0]
        return super().compute_mask(input, input_mask=input_mask)

    def get_initial_states(self, x_input, x_attn, mask_attn):
        # set initial states from mean attention vector fed through a dense
        # activation
        mean_attn = K.mean(x_attn * K.expand_dims(mask_attn), axis=1)
        h0 = K.dot(mean_attn, self.W_init_h) + self.b_init_h
        c0 = K.dot(mean_attn, self.W_init_c) + self.b_init_c
        return [self.attn_activation(h0), self.attn_activation(c0)]

    def call(self, x, mask=None):
        assert isinstance(x, list) and len(x) == 2
        x_input, x_attn = x
        if mask is not None:
            mask_input, mask_attn = mask
        else:
            mask_input, mask_attn = None, None
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x_input, x_attn, mask_attn)
        constants = self.get_constants(x_input, x_attn, mask_attn)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask_input,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        x_attn = states[4]
        mask_attn = states[5]
        attn_shape = self.input_spec[1].shape

        #### attentional component
        # alignment model
        # -- keeping weight matrices for x_attn and h_s separate has the advantage
        # that the feature dimensions of the vectors can be different
        h_att = K.repeat(h_tm1, attn_shape[1])
        att = time_distributed_dense(x_attn, self.W_att, self.b_att)
        energy = self.attn_activation(K.dot(h_att, self.U_att) + att)
        energy = K.squeeze(K.dot(energy, self.v_att), 2)
        # make probability tensor
        alpha = K.exp(energy)
        if mask_attn is not None:
            alpha *= mask_attn
        alpha /= K.sum(alpha, axis=1, keepdims=True)
        alpha_r = K.repeat(alpha, attn_shape[2])
        alpha_r = K.permute_dimensions(alpha_r, (0, 2, 1))
        # make context vector -- soft attention after Bahdanau et al.
        z_hat = x_attn * alpha_r
        z_hat = K.sum(z_hat, axis=1)

        if self.consume_less == 'gpu':
            z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) \
                + K.dot(z_hat, self.Z) + self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]
        else:
            if self.consume_less == 'cpu':
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')

            z0 = x_i + K.dot(h_tm1 * B_U[0], self.U_i) + K.dot(z_hat, self.Z_i)
            z1 = x_f + K.dot(h_tm1 * B_U[1], self.U_f) + K.dot(z_hat, self.Z_f)
            z2 = x_c + K.dot(h_tm1 * B_U[2], self.U_c) + K.dot(z_hat, self.Z_c)
            z3 = x_o + K.dot(h_tm1 * B_U[3], self.U_o) + K.dot(z_hat, self.Z_o)

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3)

        h = o * self.activation(c)
        if self.output_alpha:
            return alpha, [h, c]
        else:
            return h, [h, c]

    def get_constants(self, x_input, x_attn, mask_attn):
        constants = super().get_constants(x_input)
        attn_shape = self.input_spec[1].shape
        if mask_attn is not None:
            if K.ndim(mask_attn) == 3:
                mask_attn = K.all(mask_attn, axis=-1)
        constants.append(x_attn)
        constants.append(mask_attn)
        return constants

    def get_config(self):
        cfg = super().get_config()
        cfg['output_alpha'] = self.output_alpha
        cfg['attn_activation'] = self.attn_activation.__name__
        return cfg

    @classmethod
    def from_config(cls, config):
        instance = super(AttentionLSTM, cls).from_config(config)
        if 'output_alpha' in config:
            instance.output_alpha = config['output_alpha']
        if 'attn_activation' in config:
            instance.attn_activation = activations.get(config['attn_activation'])
        return instance
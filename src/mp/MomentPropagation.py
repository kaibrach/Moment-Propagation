"""Moment Propagation Class

The following code contains the construct for doing MomentPropagation(MP) in DNN. Every function below
(e.g. Dense, Conv2D, etc) is designed to propagate the first two moments (Expectation E; Variance V)  based on a given input model
for the current iterated layer. It is also possible to only propagate the variance and keep the expecation from the base model.

For information regarding Propagation of uncertainty see https://en.wikipedia.org/wiki/Propagation_of_uncertainty

If your model contains a layer that is not already part of the MP class,
just add a new function with the same "Class-Name" as the layer.

For example if you have a layer called `MyCustomLayer` in your model, your function should look like this

```python
def MyCustomLayer(self,E,V,L):
    # E = Expectation
    # V = Variance
    # L = Layer

    logging.info(f'MyCustomLayer {E.shape}')
    ''' Your Moment Propagation implementation '''

    return E, V
```

I implemented a `debug` function that can be used to debug through the code with real values.
That helps a lot if you want to check the output of your function.

===== Example Usage =====
model = tensorflow functional or sequencial model with dropout

# Creatint Moment Progagation model
model_mp = mp.MP()
model_mp.create_MP_Model(model=model, use_mp=True, verbose=False)

pred_mp,var_mp=model_mp.model.predict(x=x_test)


===== Example Usage (DEBUG) =====
model_mp = mp.MP(model=model)

# Debug entire model
pred_mp, var_mp = model_mp.debug_model(batch_size=1, debug_input_tensor=x_test, use_mp=True)

# Example for debugging a layer
E,V,E_dbg,V_dbg = model_mp.debug_layer(6)

# Example for creting a layer evaluation generator
for l in model_mp.get_layer_evaluation_generator(X_random):
    name = l['layer_name']
    layer = l['layer']
    inp = l['inputs']
    out_b_a = l['output_b_a']
    out = l['outputs']
    plt.figure(figsize=(15,5))
    plt.hist(inp)
    plt.hist(out[0])
    plt.show()
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging
import sys
from packaging import version
from enum import Enum

'''
    Need to import private functions from tensorflow.python.keras instead of tf.keras.backend 
    to get symbolic_learning_phase() to use with K.function an eager execution enabled. 
    Otherwise we will run into an variance
    see my entry on https://github.com/tensorflow/tensorflow/issues/34201
    
    (this was an issue in older TF versions, with newer version this should not be an issue anymore but I did not updated it)
'''
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.conv_utils import convert_data_format
from tensorflow.python.keras.backend import eager_learning_phase_scope


''' Some variables '''
tfk = tf.keras
tfd = tfp.distributions


class MP:

    def __init__(self, model=None):
        self._use_mp = True
        self._nn_model = model
        self._mp_model = None
        self._weight_scale = 1.0 # Weight inference scaling factor

        self.DEBUG = Enum('DEBUG','model layer plot tensor')
        self._debug = dict({self.DEBUG.model: False,
                            self.DEBUG.layer: False,
                            self.DEBUG.plot: False,
                            self.DEBUG.tensor: None
                           })

        self.init_logger()

    def init_logger(self, log_level=logging.INFO):
        """ Logger
        :param log_level:
        """
        log_format = (
            # '%(asctime)s - '
            # '%(name)s - '
                self.__class__.__name__ +
                '(%(funcName)s) - '
                '%(levelname)s - '
                '%(message)s'
        )

        bold_seq = '\033[1m'
        mp_format = (
            # f'{bold_seq} '
            # '%(log_color)s '
            f'{log_format}'
        )

        ''' If one of the debugs is enabled reset the loglevel'''
        if not all((not value or value is None) for value in self._debug.values()):
            ll = logging.DEBUG
        else:
            ll = log_level

        # Reset the logging state
        logging.getLogger(__name__).setLevel(ll)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(stream=sys.stdout, format=mp_format)

    @staticmethod
    #@tf.function
    def norm_pdf(x, loc=0., scale=1.):
        '''
            PDF of the normal distribution
        '''
        dist = tfd.Normal(loc=loc, scale=scale)
        return dist.prob(x)

    @staticmethod
    #@tf.function
    def norm_cdf(x, loc=0., scale=1.):
        '''
            CDF of the normal distribution
        '''
        dist = tfd.Normal(loc=loc, scale=scale)
        return dist.cdf(x)

    @staticmethod
    def Gaussian_Sigmoid(E, V):
        """
        See Kevin Murphys Book: Machine Learning - A Probabilistic Perspective (Formula 8.68)
        :param E: Expectation from previous layer
        :param V: Variance from previous layer
        :return: Expectation (E) and Variance (V)
        """

        V = tf.cast(tf.maximum(V, 1e-14), dtype=tf.float32)
        lamb = tf.constant(np.pi / 8., dtype=tf.float32)
        denom = tf.sqrt(1. / lamb + V)
        denom2 = tf.sqrt(1. + lamb * V)

        E = MP.norm_cdf((E / denom))

        #todo: Seems to be correct, but need to be checked (was not necessary in our case)
        V = E*(1.-E)*(1.- 1./ denom2)

        ''' Return Sigmoid Expectation and Variance (Variance not necessary for Softmax)'''
        return E, V

    @staticmethod
    def Gaussian_ReLU(E, V):
        """ See formula Gaussian ReLU (Equation 44) and notebook Eval_Gaussian_ReLU.ipynb.
        The implementation below is based on the variable names in the paper.

        :param E: Expectation from previous layer
        :param V: Variance from previous layer
        :return: Expectation (E) and Variance (V)
        """

        V = tf.cast(tf.maximum(V, 1e-14), dtype=tf.float32)
        S = tf.sqrt(V) # Sigma (S) = standard deviation
        alpha = E / S

        E_R = (E * MP.norm_cdf(alpha)) + (S * MP.norm_pdf(alpha))
        V_R = ((E ** 2 + V) * MP.norm_cdf(alpha)) + ((E * S) * MP.norm_pdf(alpha)) - E_R ** 2

        ''' Expecation and Variance of a Gaussian going through ReLU'''
        return E_R, V_R


    @staticmethod
    #@tf.function
    def Gaussian_MaxPooling(E, V, pool_size=(2, 2), strides=(2, 2), padding='SAME', reverse=True):

        #@tf.function
        def max_gaussian(X1, X2):
            ''' X1 is the first stacked vector of random variables containing 𝜇1,𝜎2_1
                X2 is the second stacked vector of random variables containing 𝜇2,𝜎2_1
                𝜇1 is the expectation (mean) of the first normally distributed random variable.
                𝜎2_1 is the variance of the first normally distributed random variable.
                𝜇2 is the expectation(mean) of the second normally distributed random variable.
                𝜎2_2 is the variance of the second normally distributed random variable.
                Φ is the MP.cdf of the standard normal.
                𝜙 is the MP.pdf of the standard normal.
                Θ((⎯⎯√𝜎2_1+𝜎2_2)) is the normalization
            '''

            m1 = X1[..., 0]
            v1 = X1[..., 1]
            m2 = X2[..., 0]
            v2 = X2[..., 1]

            ''' Also possible to use slice but makes no difference in time'''
            # m1 = tf.reshape(tf.slice(X1,[0,0],[-1,1]),[-1])
            # v1 = tf.reshape(tf.slice(X1,[0,1],[-1,1]),[-1])
            # m2 = tf.reshape(tf.slice(X2,[0,0],[-1,1]),[-1])
            # v2 = tf.reshape(tf.slice(X2,[0,1],[-1,1]),[-1])


            ''' Important: use tf.maximum otherwise we could run into trouble'''
            theta = tf.math.sqrt(tf.maximum(v1 + v2, 1e-14))
            alpha = tf.math.divide((m1 - m2), theta)

            '''Create the 3 terms for calculation of Expected Value (E(Y)) '''
            EY_term1 = tf.math.multiply(m1, MP.norm_cdf(alpha))
            EY_term2 = tf.math.multiply(m2, MP.norm_cdf(-alpha))
            EY_term3 = tf.math.multiply(theta, MP.norm_pdf(alpha))

            ''' Calulate E(Y)'''
            EY = EY_term1 + EY_term2 + EY_term3

            '''Create the 3 terms for calculation of Variance (E(Y^2))'''
            EY2_term1 = tf.math.multiply((v1 + tf.math.square(m1)), MP.norm_cdf(alpha))
            EY2_term2 = tf.math.multiply((v2 + tf.math.square(m2)), MP.norm_cdf(-alpha))
            EY2_term3 = tf.math.multiply(tf.math.multiply((m1 + m2), theta), MP.norm_pdf(alpha))

            ''' Calulate E(Y^2)'''
            EY2 = EY2_term1 + EY2_term2 + EY2_term3

            ''' Calulation of mean and variance'''
            E_max = EY
            V_max = EY2 - tf.math.square(EY)

            ''' Necessary to return as stacked values'''
            return tf.stack(values=[E_max, V_max], axis=-1)

        ''' Extract the number of channels from the input'''
        n_channels = E.get_shape().as_list()[-1]
        ''' n_pool is the product of the pooling size (e.g. (2,2) = 4 or (3,3) = 9)'''
        n_pool = tf.math.reduce_prod(pool_size)

        ''' 
            All extracted patches are stacked in the depth (last) dimension of the output.
            Patches=n_pool*n_channels
        '''
        patch_E = tf.image.extract_patches(
            E,
            sizes=(1,) + pool_size + (1,),
            strides=(1,) + strides + (1,),
            padding=padding.upper(),
            rates=[1, 1, 1, 1]
        )
        patch_V = tf.image.extract_patches(
            V,
            sizes=(1,) + pool_size + (1,),
            strides=(1,) + strides + (1,),
            padding=padding.upper(),
            rates=[1, 1, 1, 1]
        )

        ''' Get the shape of the patch for further reshaping'''
        patch_shape = patch_E.get_shape().as_list()
        if patch_shape[0] is None:
            patch_shape[0] = -1

        ''' 
            1. Reshape the extracted patches from N,H,W,P*C to N*H*W,P,C
            We can do the following sequence also step by step, but this is faster.
        '''
        patch_E = tf.reshape(patch_E, (-1, n_pool, n_channels))
        patch_V = tf.reshape(patch_V, (-1, n_pool, n_channels))

        ''' 2. Transpose the patch from N*H*W,P,C to P,N*H*W,C'''
        patch_E = tf.transpose(patch_E, (1, 0, 2))
        patch_V = tf.transpose(patch_V, (1, 0, 2))

        '''     
            3. Now we can reshape the elements from P,N*H*W,C to P,N*H*W*C and have everything in a single batch
            dimensioned by the number pooling batches. This is necessary for further processing, thus tf.scan unpacks
            all elements on dimension 0. The dimension 0 is now the pooling_batch dimension and dimension 1 corresponds to
            the N,H,W,C image
        '''
        patch_E = tf.reshape(patch_E, (n_pool, -1))
        patch_V = tf.reshape(patch_V, (n_pool, -1))

        #todo: Sort patch_mean and patch_max by the patach_mean values(Check for ascending or descending order)
        # https://arxiv.org/pdf/1511.06306.pdf and http://users.eecs.northwestern.edu/~haizhou/publications/isqed06sinha.pdf
        # Seems that the approximation error is small if we sort the values beforehand. Did not implemented this here, but
        # did some pre-evaluation and did not found any significant improvement.
        # But if someone interested, feel free to implement :-)

        '''
            4. Now we Stack the patches and calculate the maximum of gaussian random variables.
            This means we stack mean and variance by pooling batch which leads to the shape
            n_pool,N*H*W*C,2
        '''
        stack_E_V_by_pool_batch = tf.stack([patch_E, patch_V], -1)

        # stack_by_pool_batch = tf.sort(
        #     stack_by_pool_batch, axis=0, direction='ASCENDING'
        # )

        '''
            Scan (https://www.tensorflow.org/api_docs/python/tf/scan) is very helpful here.
            The elements are made of the tensors unpacked from elems parameter on dimension 0.
            The callable fn takes two tensors as arguments. The first argument is the accumulated value computed from the
            preceding invocation of fn, and the second is the value at the current position of elems.
            In short this explains our gaussian maxpooling algorithm problem: :-)
            We calculate the maximum of the random variables per pooling depth by using the tf.scan function.
            tf.scan repeatedly applies the callable fn to a sequence of elements from first to last, thus we can
            calculate the following max(max(X1,X2),X3) where X1-X3 are vectors within stack_by_pool_batch containing
            |E|V| in the first and second column.
            See Eval_Gaussian_MaxPooling.ipynb for scan examples 
        '''
        stack_E_V_by_pool_batch = tf.scan(max_gaussian, stack_E_V_by_pool_batch, reverse=reverse)

        ''' Extract expectation and variance from the stacked batches. We have to take care to take the correct one'''
        if reverse:
            e = stack_E_V_by_pool_batch[0, :, 0]
            v = stack_E_V_by_pool_batch[0, :, 1]
        else:
            e = stack_E_V_by_pool_batch[-1, :, 0]
            v = stack_E_V_by_pool_batch[-1, :, 1]

        ''' Reshape to the downsampled Pooling size '''
        E_P = tf.reshape(e, (patch_shape[0:3] + [n_channels, ]))
        V_P = tf.reshape(v, (patch_shape[0:3] + [n_channels, ]))

        return E_P, V_P

    @staticmethod
    def Gaussian_Softmax(E, V):

        '''
            Implementation of the Formula (35) from https://arxiv.org/ftp/arxiv/papers/1703/1703.00091.pdf to calculate the
            first moment (E(x)) of the softmax.
            We assume that the input-vectors contain the mean values and the variances of the normal distribution random variables
            we propagated through the network.

            Important:
            * Use tf.shape(x) for the Batch-Dimension (Runtime Shape of the tensor)
            * Use x.shape[1] for the static Dimension
            Otherwise this will not work properly
            for reference see Reference Eval_Gaussian_Softmax.ipynb
      '''

        B = tf.shape(E)[0]  # (B)atch Dimension
        C = E.shape[1]      # Number of (C)lasses

        # # We expand along a new axis index by l=k'
        E_bkl = tf.tile(tf.reshape(E, (B, C, 1)), (1, 1, C)) - tf.tile(tf.reshape(E, (B, 1, C)), (1, C, 1))
        V_bkl = tf.tile(tf.reshape(V, (B, C, 1)), (1, 1, C)) + tf.tile(tf.reshape(V, (B, 1, C)), (1, C, 1))
        GSIG = MP.Gaussian_Sigmoid(E_bkl, V_bkl)
        ESIG = GSIG[0]
        VSIG = GSIG[1]
        sig_sum = tf.math.reduce_sum(tf.math.divide(1.0 , ESIG), axis=2)
        E_SM = tf.math.divide(1.0, (tf.cast(-C, dtype=tf.float32) + sig_sum))

        # Todo: need to be investigated if this is correct for the variance (but is not necessary)
        denom_V_SM = tf.sqrt((1. + tf.sqrt(np.pi / 8.) * V))
        #denom_vsm2 = tf.sqrt(1. + tf.sqrt((3 / np.pi ** 2)) * variance)

        V_SM =  E_SM*(1. - E_SM)*(1. - (1./denom_V_SM))

        # Return Softmax Expectation and Variance
        return E_SM, V_SM

    @staticmethod
    def Activation_Softmax(E, V):

        """ Because we do not want to change the behavior w.r.t. the other activation derivatives,
        we need to calculate the softmax first, to estimate the derivative.
        We could also take the output of the last affine layer (if activation softmax is used), but then we have
        to change the behavior of the layer propagation in between.
        Long story short, we want to have the same behaviour as the others, thus we calculate the softmax here
        (see: Eval_Jacobi_Matrix.ipynb for calculation)
        # Todo: Need to write the the formula from my Bleistift-Gekritzel to latex-formula :)

        :param E: Expectation as logits from Dense Layer
        :param V: Variance
        :return: Expectation (E) and Variance (V)
        """

        ''' 1. Calculate the Derivative of the softmax to propagate the variance.
            We can do that because the variance does not depend on the network output
        '''

        # m, n = input layer shape
        sm = tf.nn.softmax(E)

        ''' First we create for each example feature vector, it's outer product with itself
            ( p1^2  p1*p2  p1*p3 .... )
            ( p2*p1 p2^2   p2*p3 .... )
            ( ...                     )
        '''
        tensor1 = tf.einsum('ij,ik->ijk', sm, sm)  # (m, n, n)
        ''' Second we need to create an (n,n) identity of the feature vector
            ( p1  0  0  ...  )
            ( 0   p2 0  ...  )
            ( ...            )
        '''
        tensor2 = tf.einsum('ij,jk->ijk', sm, tf.eye(sm.shape[1]))  # (m, n, n)
        ''' Then we need to subtract the first tensor from the second
            ( p1 - p1^2   -p1*p2   -p1*p3  ... )
            ( -p1*p2     p2 - p2^2   -p2*p3 ...)
            ( ...                              )
        '''
        df_dx = tensor2 - tensor1
        df_dx = tf.dtypes.cast(df_dx, tf.float32)

        ''' Finally, we multiply the df_dx with the variance (variance) to get the gradient w.r.t. x
            In our case we are interested in calculating the variance.
        '''
        V = tf.einsum('ijk,ik->ij', df_dx ** 2, V)  # (m, n)

        return sm, V

    def create_MP_Model(self, model=None, use_mp=True, verbose=True):
        """

        Args:
            model: Trained normal network model with dropout
            use_mp: If True: Use Moment Propagation
                    If False: Use Error Propagation
            verbose: If True: Init logger for Info
                     If False: Init logger for Warnings only

        Returns: Tensorflow Model

        """

        if self._nn_model is None and model is None:
            raise ValueError("You have not set a nn_model for creating the MP_Model. "
                             "Please set a trained \"nn_model\" beforehand")
        elif model is not None:
            self._nn_model = model

        V = None
        E = None
        self._use_mp = use_mp

        # check if one of the debug options is enabled
        if all((not value or value is None) for value in self._debug.values()):
            if not verbose:
                self.init_logger(logging.WARNING)
            else:
                self.init_logger(logging.INFO)

        logging.info(f'--------- Start Moment Propagation for {len(self._nn_model.layers) - 1} Layers ---------\n')

        for i, l in enumerate(self._nn_model.layers):
            ''' 
                This is the main loop for creating the MP model based on the dropout input model.
                We iterate through all layers and propagate the moments through the network
            '''
            E, V = getattr(self, str(l.__class__.__name__))(E, V, l)

        ''' Check if variance is still None'''
        if V is None:
            # raise Runtimevariance('Mhhh something went wrong... Check if there is a dropout layer in model!')
            logging.warning('Mhhh something went wrong... Check if there is a dropout layer in model!')
            # set variance to zero in this case
            V = tf.zeros_like(self._nn_model.outputs)[0, 0:]

        # if isinstance(variance, (np.ndarray)):
        #     variance = tf.convert_to_tensor(variance, dtype=tf.float32)

        ''' Concatenate Expectation and Variance
            Either uses tfk.layers.concatenate() to have one output (e.g. None,2) or add the variance
            as a list item [variance] and you will get two separate outputs [e.g. (None,1),(None,1)].
            I used the second one because it is more intuitive
        '''
        mp_model_outputs = [E] + [V]

        if not self._debug[self.DEBUG.model]:
            ''' Create a new MP model with E + V as output'''
            self._mp_model = tfk.Model(inputs=self._nn_model.inputs, outputs=mp_model_outputs)
            return self._mp_model
        else:
            logging.warning('Do not create a model in Debug-Mode')

            ''' If we doing pure Error-Propagation we need to create a debug output, thus this line can handle both'''
            E = self._debug_output(mp_model_outputs[0])

            self._mp_model = None
            return [np.array(E.numpy()), np.array(V.numpy())]

    def debug_layer(self, layer_id=None, E_debug_tensor=None, V_debug_tensor=None, batch_size=None):
        """
        Function for debugging a layer using either an own tensor or it is created a random tensor automatically

        :param layer_id: Id of the model layer which shall be called by the function
        :param E_debug_tensor: Own Tensor for debugging Expectation
        :param V_debug_tensor: Own Tensor for debugging Variance
        :param batch_size: Batch_Size for auto generated tensor. If not specified the batch_size is 1 for higher
        dimensional layer inputs and 10 for one-dimensional inputs
        :return: No return
        """
        assert layer_id is not None, "Please specify a valid layer_id!!"
        try:
            layer = self._nn_model.layers[layer_id]
        except RuntimeError as e:
            raise e
        finally:
            print('---------------------------------------------------')
            print('-- Network contains the following IDs and shapes --')
            print('---------------------------------------------------')
            for i in range(len(self._nn_model.layers)):
                l = self._nn_model.layers[i]
                print(i, l.__class__.__name__,
                      l.input.shape,
                      '-->',
                      l.output.shape)

        ''' Check the debug Tensor, either create a random one or use the debug_input_tensor'''
        E_debug_tensor = self._debug_create_tensor(layer.input.shape, E_debug_tensor, batch_size)
        V_debug_tensor = self._debug_create_tensor(layer.input.shape, V_debug_tensor, batch_size)

        self._debug[self.DEBUG.layer] = True
        #self._debug[self.DEBUG.tensor] = E_debug_tensor

        # call function for debugging
        E,V = getattr(self, str(layer.__class__.__name__))(E_debug_tensor, V_debug_tensor, layer)
        return (E,V,E_debug_tensor, V_debug_tensor)

    def debug_model(self, debug_input_tensor=None, batch_size=None, debug_plot=False, use_mp=False):
        """
        Function for debugging the entire model using either an own tensor or it is created a random tensor automatically.
        :param debug_input_tensor: Own Tensor for debugging
        :param batch_size: Batch_Size for auto generated tensor. If not specified the batch_size is 1 for higher
        dimensional layer inputs and 10 for one-dimensional inputs
        :param debug_plot: Shows some intermediate plots for the layer actually iterated.
        :param use_mp: Enable/disable Moment Propagation
        :return: No return
        """
        debug_input_tensor = self._debug_create_tensor(self._nn_model.input.shape, debug_input_tensor, batch_size)

        ''' Check the debug Tensor, either create a random one or use the debug_input_tensor'''
        self._debug[self.DEBUG.model] = True
        self._debug[self.DEBUG.tensor] = debug_input_tensor
        self._debug[self.DEBUG.plot] = debug_plot

        self.init_logger()
        return self.create_MP_Model(use_mp=use_mp)

    def _debug_create_tensor(self, input_shape, debug_input_tensor=None, batch_size=None):
        """
        Creates an automatic random tensor of correct shape if debug_input_tensor is None. Otherwise debug_input_tensor
        will be proofed if it has the correct shape.
        :param input_shape: Required input_shape of the layer processing
        :param debug_input_tensor: Own Tensor for debugging
        :param batch_size: Batch_Size for auto generated tensor. If not specified the batch_size is 1 for higher
        dimensional layer inputs and 10 for one-dimensional inputs
        :return: Debug Tensor
        """
        if debug_input_tensor is not None:
            if input_shape[1:] != debug_input_tensor.shape[1:]:
                raise Exception(f'Input_Tensor has incorrect shape {debug_input_tensor.shape}\n'
                                f'Please specify an input tensor with correct shape {input_shape}!')

        if debug_input_tensor is None:
            if batch_size is None:
                batch_size = 1
                if len(input_shape) == 2 and input_shape[1] == 1:
                    batch_size = 10

            input_shape = list(input_shape)
            input_shape[0] = batch_size
            debug_tensor = tf.convert_to_tensor(np.random.choice(np.arange(-10., 10.), input_shape),
                                                dtype=tf.float32)
        else:
            debug_tensor = tf.convert_to_tensor(debug_input_tensor, dtype=tf.float32)
            if batch_size is not None:
                debug_tensor = debug_tensor[0:batch_size]
        return debug_tensor

    def _debug_output(self, output):

        ''' If you want to debug the values Eager Execution needs to be enabled'''
        if not tf.executing_eagerly():
            logging.warning('If eager execution is disabled not all debug values can be visualized!'
                            ' Enable Eager Execution for proper debugging!')

        # For debugging the layer we want a tensor only
        if self._debug[self.DEBUG.layer]:
            out = self._debug_create_tensor(output.shape)

        elif self._debug[self.DEBUG.model]:

            if version.parse(tf.__version__) <= version.parse("2.2.0"):
                # Only possible for Tensorflow Version < 2.3!!! For Tensorflow version > 2.3 see here:
                # see also https://github.com/tensorflow/tensorflow/issues/34201

                try:
                    functor = K.function([self._nn_model.input,
                                          K.symbolic_learning_phase()],
                                         output)  # evaluation function
                    out = functor([self._debug[self.DEBUG.tensor], False])
                    out = tf.convert_to_tensor(out, dtype=tf.float32)
                except:
                    out = tf.convert_to_tensor(output, dtype=tf.float32)

            else:
                with eager_learning_phase_scope(value=0):  # 0=test, 1=train
                    try:
                        functor = K.function([self._nn_model.input],
                                             output)  # evaluation function
                        out = functor([self._debug[self.DEBUG.tensor]])
                        out = tf.convert_to_tensor(out, dtype=tf.float32)
                    except:
                        out = tf.convert_to_tensor(output, dtype=tf.float32)

        return out

    def get_layer_evaluation_generator(self, debug_input_tensor=None, dropout_fw=False):

        """
        :param debug_input_tensor:
        :return:
        """
        ''' All outputs we wanna have during the evaluation
            1. Input to the layers
            2. Layer output without activation
            3. Layer output with activation
        '''
        outputs = [[layer.input,  # Input to the layers
                    layer.output  # Output with activation
                    ] for layer in self._nn_model.layers]# if layer.__class__.__name__ != 'InputLayer']

        if version.parse(tf.__version__) <= version.parse("2.2.0"):
            # Only possible for Tensorflow Version < 2.3!!! For Tensorflow version > 2.3 see here:
            # see also https://github.com/tensorflow/tensorflow/issues/34201

            ''' Keras function for getting the layer ouputs'''
            functor = K.function([self._nn_model.input,
                                  K.symbolic_learning_phase()],
                                  outputs)  # evaluation function
            # Create all the intemediate outputs with dropout disables. If Dropout is enables, we have MC-Dropout
            layer_outs = functor([debug_input_tensor, dropout_fw])

        else:
            with eager_learning_phase_scope(value=dropout_fw):  # 0=test, 1=train

                ''' Keras function for getting the layer outputs'''
                functor = K.function([self._nn_model.input],
                                      outputs)  # evaluation function
                # Create all the intemediate outputs with dropout disables. If Dropout is enables, we have MC-Dropout
                layer_outs = functor([debug_input_tensor])

        ''' Need to set the following values to handle debugging internally'''
        debug_input_tensor = self._debug_create_tensor(self._nn_model.input.shape,
                                                       debug_input_tensor)
        self._debug[self.DEBUG.model] = True
        self._debug[self.DEBUG.tensor] = debug_input_tensor

        for i, o in enumerate(zip(layer_outs, self._nn_model.layers)):
            i += 1
            l = o[1]
            inputs = o[0][0]
            output = o[0][1]  # Output after activation
            weights = []
            if l.weights:
                weights = l.weights[0].numpy()

            # We try to extract the output before activation function to see the distribution
            try:
                ''' Keras function for getting the layer outputs'''
                functor = K.function([self._nn_model.input],
                                      l.output.op.inputs[0].op.inputs[0])  # evaluation function
                # Create all the intemediate outputs with dropout disables. If Dropout is enables, we have MC-Dropout
                output_b_a = functor([debug_input_tensor]) # Output before activation
            except:
                output_b_a = []
            model_dict = {'layer_name': l.__class__.__name__,
                          'layer': l,
                          'inputs': inputs,
                          'weights':weights,
                          'output_b_a': output_b_a,
                          'outputs': output
                          }
            yield model_dict



    @property
    def model(self) -> tfk.Model:
        return self._mp_model

    @property
    def use_moment_propagation(self):
        return self._use_mp

    @use_moment_propagation.setter
    def use_moment_propagation(self, value):
        self._use_mp = value

    def MaxPooling2D(self, E, V, L):
        """
            Hint: If you want to debug and watch the output clearly transpose variables
            e.g.
                inputs_np = inputs.numpy()[0, :, :, ].transpose(2, 0, 1)
                mu_np = mu.numpy()[0, :, :, ].transpose(2, 0, 1)
                gradient_np = gradient.numpy()[0, :, :, ].transpose(2, 0, 1)

            ############## MAX ##############

            inputs_np

               / 14.0  11.0  16.0  13.0 /
              / 16.0  19.0  16.0  2.0  /
             / 2.0   6.0   11.0  17.0 /
            / 16.0  7.0   19.0  9.0  /

            max_pool

             / 19.0  16.0 /
            / 16.0  19.0 /

            gradient_np

               / 0.0   0.0   1.0   0.0  /
              / 0.0   1.0   0.0   0.0  /
             / 0.0   0.0   0.0   0.0  /
            / 1.0   0.0   1.0   0.0  /

            The formula for propagating the variance through max-pooling can be described as follows:

            1. Calculate the gradient w.r.t the inputs
            2. The gradient contain 1. for the max values of the input_tensor pooling-window
            3. Calculate the indices of the max values by
                3.1 Flatten the gradients
                3.2 Find all indices of the flatten gradient where the element equals 1.
                    e.g. flatten_gradient = 1 0 0 1 1 0 1 --> idx = 0 3 4 6
            4. Gather all elements from the variance based on the indices
            5. Reshape the variance


            :param E: Expectation from previous layer
            :param V: Variance from previous layer
            :param L: Current Layer for doing Moment Propagation
            :return: Expectation (E) and Variance (V)
        """

        if V is not None:
            if self._use_mp:
                # see(Eval_Gaussian_MaxPooling.ipynb)
                E, V = self.Gaussian_MaxPooling(E=E,
                                                V=V,
                                                pool_size=L.pool_size,
                                                strides=L.strides,
                                                padding=L.padding.upper())
            else:
                with tf.GradientTape() as gt:
                    ''' By default, GradientTape doesn’t track constants,
                     so we must instruct it to with: gt.watch(variable)
                     '''

                    # check if one of the debug options is enabled
                    if not all((not value or value is None) for value in self._debug.values()):
                        E = self._debug_output(E)

                    gt.watch(E)
                    E_pool = tf.nn.max_pool2d(input=E, ksize=L.pool_size, strides=L.strides,
                                         padding=L.padding.upper())

                    # calculate the gradient of the pooling output w.r.t the inputs
                    gradient = gt.gradient(E_pool, E)

                gradient_flatten = tf.reshape(gradient, [-1])
                # use tf.where(tf.equal(gradient_flatten, 1)) instead of tf.where(gradient_flatten == 1)
                # otherwise tf.where will return 0 when eager execution is disabled --> :-(
                idx = tf.reshape(tf.where(tf.equal(gradient_flatten, 1)), [-1])
                variance_flatten = tf.reshape(V, [-1])
                V = tf.gather(variance_flatten, idx)

                # Reshape to Down-sampled shape
                E = E_pool
                V = tf.reshape(V, tf.shape(E_pool))

                #logging.info(f'Propagate variance {V.shape}')


        else:
            E = tf.nn.max_pool2d(input=E, ksize=L.pool_size, strides=L.strides,
                                 padding=L.padding.upper())
        return E, V

    # Todo: Code need to be updated.
    def AveragePooling2D(self, E, V, L):
        """
            Hint: If you want to debug and watch the output clearly transpose variables
            e.g.
            V_np = variance.numpy()[0,:,:,].transpose(2,0,1)
            E_np = mu.numpy()[0,:,:,].transpose(2,0,1)

            ############## AVG ##############

            V_np
            / 14.0  11.0  16.0  13.0 /
            / 16.0  19.0  16.0  2.0  /
            / 2.0   6.0   11.0  17.0 /
            / 16.0  7.0   19.0  9.0  /

            avg_pool
            / 15.0  11.7 /
            / 7.75  14.0 /

            gradient = 1/m (m = product of pooling size)
            / 0.25  0.25  0.25  0.25 /
            / 0.25  0.25  0.25  0.25 /
            / 0.25  0.25  0.25  0.25 /
            / 0.25  0.25  0.25  0.25 /

            The formula for propagating the variance through average-pooling can be described as follows:

            1. Because the gradient is constant (see above) we can pool (down-sample) the variance directly
            2. Calculate the constant gradient w.r.t. the pooling sizes = 1/m (m=product of pooling sizes)
            3. Multiply the squared gradient with the pooled_variance

            variance_pooled = down-sampling of variance
            gradient = 1/m (m=product of pooling sizes)
            variance = gradient^2 x variance_pooled

        :param E: Expectation from previous layer
        :param V: Variance from previous layer
        :param L: Current Layer for doing Moment Propagation
        :return: Expectation (E) and Variance (V)
        """

        # Todo: Check code here with eval_pooling.ipynb

        ''' Down-sample the mean '''
        E = tf.nn.avg_pool2d(input=E, ksize=L.pool_size, strides=L.strides,
                             padding=L.padding.upper())

        if V is not None:
            ''' Down-sample the variance '''
            variance_pooled = tf.nn.avg_pool2d(input=V, ksize=L.pool_size, strides=L.strides,
                                               padding=L.padding.upper())

            ''' Calculate the gradient '''
            gradient = tf.math.divide(1., (L.pool_size[0] * L.pool_size[1]))

            ''' Propagate the variance by multiply the square of the gradient with the pooling output'''
            V = tf.math.multiply((gradient ** 2), variance_pooled)

            #logging.info(f'Propagate variance {V.shape}')

        return E, V

    def Flatten(self, E, V, L):
        """
        Flatten the variance based on the layer output shape
        :param E: Expectation from previous layer
        :param V: Variance from previous layer
        :param L: Current Layer for doing Moment Propagation
        :return: Flattened Expectation (E) and Variance (V)
        """

        #logging.info(f'Propagate variance {V.shape}')

        variance_flatten_dim = L.output_shape[1]

        '''Propagate variance through Flatten layer'''
        E = tf.reshape(tensor=E, shape=(-1, variance_flatten_dim))
        if V is not None:
            V = tf.reshape(tensor=V, shape=(-1, variance_flatten_dim))
        return E, V

    def Activation_Eagerly(self, E, V, AF):
        """
        :param E: Expectation
        :param V: Variance
        :param AF: Activation Function
        :return: Expectation (E) and Variance (V)
        """
        if V is not None:
            with tf.GradientTape() as gt:
                '''By default, GradientTape doesn’t track constants,
                 so we must instruct it to with: gt.watch(variable)
                 Todo: Explain how it works
                 '''
                gt.watch(E)
                y = AF(E)

            gradient = gt.gradient(y, E)

            if AF.__name__ != 'softmax':
                E, V = AF(E), tf.math.multiply(gradient, V)
                return E, V
            else:
                ''' Calculate the Jacobian of the Softmax activation function.
                    Unfortunately it is not possible to use gradienttape to calulate the jacobian
                    if eager execution is enabled (see Softmax and Gaussian_Softmax notebooks)
                '''
                if self._use_mp:
                    E, V = self.Gaussian_Softmax(E=E, V=V)
                else:
                    E, V = self.Activation_Softmax(E=E, V=V)
                return E, V
        else:
            E = AF(E)
            return E, V

    @tf.function
    def Activation_not_Eagerly(self, E, V, AF):
        """
        :param E: Expectation
        :param V: Variance
        :param AF: Activation Function
        :return: Expectation (E) and Variance (V)
        """
        if V is not None:
            with tf.GradientTape() as gt:
                '''By default, GradientTape doesn’t track constants,
                 so we must instruct it to with: gt.watch(variable)
                 '''
                gt.watch(E)
                y = AF(E)

            if AF.__name__ != 'softmax':
                # We need the gradient of the function
                # If we sum by the colums( axis 1) we get the gradient
                # gradient = tf.reduce_sum(jacobian, axis=1)
                gradient = gt.gradient(y, E)

                E, V = AF(E), tf.math.multiply(gradient, V)
                return E, V
            else:

                # Check if we propagat the distribution
                if self._use_mp:
                    E, V = self.Gaussian_Softmax(E=E, V=V)
                else:
                    # We need the jacobian of the function
                    jacobian = gt.batch_jacobian(y, E)
                    E = AF(E)
                    V = tf.einsum('ijk,ik->ij', jacobian ** 2, V)

                return E, V
        else:
            E = AF(E)
            return E, V

    def Activation(self, E, V, L):
        """ Activation could be an own layer or part of another layer, thus
            we have to distinguish which input shall be taken.
            For propagating the variance we always need the input from affine layers without activation.

            :param E: Expectation from previous layer
            :param V: Variance from previous layer
            :param L: Current Layer for doing Moment Propagation
            :return: Expectation (E) and Variance (V)
        """

        # check if one of the debug options is enabled
        if not all((not value or value is None) for value in self._debug.values()):
            E = self._debug_output(E)
            if V is not None:
                V = self._debug_output(V)

        ''' Either do approximation or calculate analytically
            But this is only possible if activation == ReLU
        '''
        if self._use_mp and L.activation.__name__ == 'relu':
            if V is not None:
                E, V = self.Gaussian_ReLU(E, V)
            else:
                E = L.activation(E)

        else:
            ''' If eager execution is disabled it is possible to calculate the Jacobian.
                If eager execution is enabled, we only can calculate the Gradient. In case of the Softmax function
                it is not possible to calculate the jacobian in an tensorflow way. I have one implementation but this is
                pretty slow. Thus I created the Jacobian for the softmax activation directly.
            '''
            if not tf.executing_eagerly():
                E, V = self.Activation_not_Eagerly(E, V, L.activation)
            else:
                E, V = self.Activation_Eagerly(E, V, L.activation)

            # if V is not None:
            #     #logging.info(f'Propagate moments through {L.activation.__name__} --> Approximation {V.shape}')

        return E, V

    def Dense(self, E, V, L):
        """
         Propagate moments through Dense Layer
        :param E: Expectation from previous layer
        :param V: Variance from previous layer
        :param L: Current Layer for doing Moment Propagation
        :return: Expectation (E) and Variance (V)
        """
        # This could be the case if working with sequential model
        if E is None:
            E = L.input

        # check if one of the debug options is enabled
        if not all((not value or value is None) for value in self._debug.values()):
            E = self._debug_output(E)
            if V is not None:
                V = self._debug_output(V)

        # Calculate the variance of linear combination from y=w*x+b --> variance = w^2*variance
        '''Propagate variance through Dense layer'''
        if V is not None:
            # Tensorflow (Dropout) scale up the weights by (1/keep_prob) at the training time, thus we need to take
            # care to scale the weights for variance also, otherwise mean and variance are not compatible.
            # See DropoutImpact evaluation
            weights = L.weights[0] / self._weight_scale
            V = tf.tensordot(V,
                             weights**2,
                             axes=[[1], [0]]
                             )
            #logging.info(f'Propagate variance {V.shape}')

            # Todo: We assume DO->Affine->Activation, need to check what happens if the network structure is different
            # Reset the weight scale to 1.0, because we only want to scale if Dropout layer does weight scaling
            #self._weight_scale = 1.0

        self._weight_scale = 1.0

        ''' Calculate the new mean (Calculus of DNN Blocks (Affine-layer))'''
        E = tf.nn.bias_add(tf.tensordot(E, L.weights[0], axes=[[1], [0]]), L.weights[1])

        return self.Activation(E, V, L)

    def Conv2D(self, E, V, L):
        """
             Propagate moments through Conv2D layer
            :param E: Expectation from previous layer
            :param V: Variance from previous layer
            :param L: Current Layer for doing Moment Propagation
            :return: Expectation (E) and Variance (V)
        """
        # This could be the case if working with sequential model
        if E is None:
            E = L.input

        ''' check if one of the debug options is enabled '''
        if not all((not value or value is None) for value in self._debug.values()):
            E = self._debug_output(E)
            if V is not None:
                V = self._debug_output(V)

        if V is not None:
            weights = L.weights[0] / self._weight_scale
            V = tf.nn.conv2d(input=V,
                             filters=weights**2,
                             strides=L.strides,
                             # Hier muss man das groß schreiben im model klein (Conv(Layer) fkt. _get_padding_op)!
                             padding=L.padding.upper(),
                             # conv_utils.py convert_data_format convert to NH...
                             data_format=convert_data_format(L.data_format, V.shape.ndims),
                             dilations=L.dilation_rate
                             )

        ''' Calculate the new mean (Calculus of DNN Blocks (Affine-layer))'''
        E = tf.nn.bias_add(tf.nn.conv2d(input=E,
                                        filters=L.weights[0], # Weights
                                        strides=L.strides,
                                        # Hier muss man das groß schreiben im model klein (Conv(Layer) fkt. _get_padding_op)!
                                        padding=L.padding.upper(),
                                        # conv_utils.py convert_data_format convertiert zu NH...
                                        data_format=convert_data_format(L.data_format, E.shape.ndims),
                                        dilations=L.dilation_rate
                                        ),
                           L.weights[1] # Bias
                           )

        # Todo: We assume DO->Affine->Activation, need to check what happens if the network structure is different
        ''' Reset the weight scale to 1.0, because we only want to scale if Dropout layer does weight scaling'''
        self._weight_scale = 1.0
        ''' Check if an activation function is placed within the layer'''
        return self.Activation(E, V, L)

    def Dropout(self, E, V, L):
        """
        Calculation of the variance (V) of the product of two independent Bernoulli distributed
        random variables based on Leo Goodmans work: "On the Exact variance of Products (http://www.cs.cmu.edu/~cga/var/2281592.MP.pdf).

        :param E: Expectation from previous layer
        :param V: Variance from previous layer
        :param L: Current Layer for doing Moment Propagation
        :return: Expectation (E) and Variance (V)
        """

        # Scale factor for weight inference scaling (see DropoutImpact.ipynb)
        p = L.rate
        ''' In Tensorflow implementation they scale up weights by (1/keep_prob) at training time, rather than scale
            down during inference. In general this means, for each element of `x`, with probability `p`, outputs `0`, 
            and otherwise scales up the input by `1 / (1-p)`. The scaling is such that the expected sum is unchanged. 
            Thus, in Moment Propagation we need to scale the weights in the affine layers (Dense, Conv2D) 
            for the propagating the variance by the Dropout weight inference scaling factor "_weight_scale = 1/(1-p)"
            x*1/(1-p) = x/(1-p)
        '''
        self._weight_scale = (1 - p)

        # check if one of the debug options is enabled
        if not all((not value or value is None) for value in self._debug.values()):
            E = self._debug_output(E)
            if V is not None:
                V = self._debug_output(V)

        Ex = E              # The Expectation we propagate.
        Vx = 0.0            # The Variance we propagate. Vx is unknown in first Dropout layer, thus we set to zero.
        Ey = 1 - p          # New Bernoulli mean
        Vy = p * (1 - p)    # New Bernoulli variance

        if V is not None:
            Vx = V

        ''' We calculate the variance of the product of independent random variables '''
        V = ((Ex ** 2 * Vy) + (Ey ** 2 * Vx) + (Vx * Vy))

        #logging.info(f'Create variance {V.shape}')
        return E, V

    def BatchNormalization(self, E, V, L):
        """
          Todo: Batch_Normalization (check if the formula below is correct)
        :param E: Expectation from previous layer
        :param V: Variance from previous layer
        :param L: Current Layer for doing Moment Propagation
        :return: Expectation (E) and Variance (V)
          """

        # check if one of the debug options is enabled
        if not all((not value or value is None) for value in self._debug.values()):
            E = self._debug_output(E)
            if V is not None:
                V = self._debug_output(V)

        if self._use_mp:
            # E = (L.gamma * E) / (tf.sqrt(v + L.epsilon)) + (
            #             L.beta - (L.gamma * m) / (tf.sqrt(v + L.epsilon)))
            E = tf.nn.batch_normalization(E,
                                          mean=L.moving_mean,
                                          variance=L.moving_variance,
                                          offset=L.beta,
                                          scale=L.gamma,
                                          variance_epsilon=L.epsilon)
        else:
            E = L.output

        if V is not None:
            V = tf.nn.batch_normalization(V,
                                          mean= 0,#L.moving_mean,
                                          variance=L.moving_variance**2,
                                          offset=0,
                                          scale=L.gamma**2,
                                          variance_epsilon=L.epsilon)
            #V = V * (L.gamma / (L.moving_variance + L.epsilon)) ** 2
            #V = V * L.gamma**2 #(L.gamma / (L.moving_variance + L.epsilon)) ** 2
            #V = V * (L.gamma / (v + L.epsilon)) ** 2

        return E, V

    def Concatenate(self, E, V, L):
        ''' Just a dummy (if needed please implement)'''
        return E, V

    def InputLayer(self, E, V, L):
        ''' First Layer (Functional Model) Returns Expecation, None'''
        E, V = L.output, None

        return E, V



Examples and tutorials
======================

We have prepared some basic ipython notebooks in the :code:`examples`
folder, however in the next lines we show basic examples for the
potential interesting training modes for RTBMs.

Probability estimation
######################

Let's consider the simple example of training a RTBM to learn a
gaussian probability distribution.

1. The first step consists in generating normal distributed data::

     import numpy as np
     n = 1000
     data = (np.random.normal(5, 10, n)).reshape(1, n)

2. Then we allocate a RTBM with one visible and 2 hidden units::

     from theta.rtbm import RTBM

     model = RTBM(1, 2, init_max_param_bound=20, random_bound=1)

   In this example we have set ``random_bound=1`` to control the
   maximum random value for the Shur complement initialization. The
   ``init_max_param_bound=20`` controls the maximum value allowed for
   all parameters during training with the CMA-ES minimizer.
     
   Both flags may require tuning in order to obtain the best model
   results.

3. We train the model with the CMA-ES minimizer::
	  
     from theta.minimizer import CMA
     from theta.costfunctions import logarithmic
   
     minim = CMA(False)
     solution = minim.train(logarithmic, model, data, tolfun=1e-4)   

     
Data regression and classification
##################################

Let's now consider the data regression problem.

1. Supposing we have a the ``X_train`` and ``Y_train`` numpy arrays
   with respectively the support and target data, we allocate a
   ``DiagExpectationUnitLayer`` with ``Nin=X_train.shape[0]`` and
   ``Nout=X_train.shape[1]``::

     from theta.model import Model
     from theta.layers import DiagExpectationUnitLayer

     model = Model()
     model.add(DiagExpectationUnitLayer(Nin, Nout))

2. Then we train the model using ``SGD``::

     from theta.minimizer import SGD
     from theta.costfunctions import mse
     from theta.gradientschemes import RMSprop
     from theta.initializers import uniform
   
     minim = SGD()
     solution = minim.train(mse, model, X_train, Y_train,
                            scheme=RMSprop(), lr=0.01,
			    phase=1, Q_init=uniform(2,4))

   For this particular setup we are using the mean square error cost
   function (``mse``) with stochastic gradient descent in the RMS
   propagation scheme (``scheme=RMSprop()``). The learning rate is
   ``lr=0.01`` and may require tuning before getting the best results.
   In this example we have also set an optional random uniform
   initialization for the :math:`Q` parameters through the ``Q_init``
   flag.

   When using SGD, it is possible to split the data into training and
   validation datasets automatically, by using the
   ``validation_split`` option, or by passing extra datasets, see
   ``theta.minimizer.SGD.train`` for more details.

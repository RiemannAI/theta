Examples and tutorials
======================

We have prepared some basic ipython notebooks in the :code:`examples`
folder to demonstrate the functionality of the package. Here, we give
only two simple examples to show the general usage pattern.


Probability estimation
######################

Let's consider the example of training a RTBM to learn a
gaussian probability distribution.

1. The first step consists in generating normal distributed data::

     import numpy as np
     n = 1000
     data = (np.random.normal(5,10,n)).reshape(1,n)

2. Then we allocate a RTBM with one visible and 2 hidden units::

     from theta.rtbm import RTBM
     model = RTBM(1,2)

3. We train the model with the CMA-ES minimizer::
	  
     from theta.minimizer import CMA
     from theta.costfunctions import logarithmic
     minim = CMA(False)
     solution = minim.train(logarithmic, model, data, tolfun=1e-4)   

4. The learned probabilities for given data can be queried via::

     model.predict(data)
     

Data regression and classification
##################################

Let's now consider a data regression problem.

1. Suppose we have ``X_train`` and ``Y_train`` numpy arrays
   with respectively the support and target data. We allocate a
   ``DiagExpectationUnitLayer`` with ``Nin=X_train.shape[0]`` and
   ``Nout=X_train.shape[1]``::

     from theta.model import Model
     from theta.layers import DiagExpectationUnitLayer
     model = Model()
     model.add(DiagExpectationUnitLayer(Nin, Nout))

2. Then we train the model using ``SGD``::

     from theta.minimizer import SGD
     from theta.costfunctions import mse
     minim = SGD()
     solution = minim.train(mse, model, X_train, Y_train)

3. Predictions of the model can be obtained via::

     model.predict(data)

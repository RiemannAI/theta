Components
==========

The theta package comes with the following modules:

* RTBM_ (:code:`theta.rtbm`)
* Model_ (:code:`theta.model`)
* Layers_ (:code:`theta.layers`)
* Minimizers_ (:code:`theta.minimizer`)
* Activations_ (:code:`theta.activations`)
* Initializers_ (:code:`theta.initializers`)
* `Cost functions`_ (:code:`theta.costfunctions`)
* `Stopping conditions`_ (:code:`theta.stopping`)

These modules provide all the required components to train Riemann Theta
Boltzmann Machines for probability density estimation, regression and classification.

_______________________

.. _RTBM:

RTBM
----

The ``theta.rtbm`` module contains the definition of ``class
RTBM``. This object provides the simplest interface to the parameters
of the RTMB and to the probability and expectation values.

.. figure:: rtbm.png
   :scale: 30 %
   :align: center

where :math:`T` is the connection matrix of the visible sector with
:math:`N_v` visible units, :math:`Q` of the hidden sector with
:math:`N_h` hidden units and :math:`W` the inter-connections.
      
.. autoclass:: theta.rtbm.RTBM
   :members:
   :inherited-members:
   :member-order: bysource
   
_______________________

.. _Model:

Model
------

.. autoclass:: theta.model.Model
    :members:

_______________________

.. _Layers:

Layers
------

.. automodule:: theta.layers
    :members:

_______________________

.. _Minimizers:

Minimizers
----------

.. automodule:: theta.minimizer
    :members:

_______________________

.. _Activations:

Activations
-----------

.. automodule:: theta.activations
    :members:

_______________________

.. _Initializers:

Initializers
------------

.. automodule:: theta.initializers
    :members:

_______________________

.. _Cost functions:

Cost functions
--------------

.. automodule:: theta.costfunctions
    :members:

_______________________

.. _Stopping conditions:

Stopping conditions
-------------------

.. automodule:: theta.stopping
    :members:

.. theta documentation master file, created by
   sphinx-quickstart on Thu Dec  7 17:46:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url:
   
.. image:: GitHub-Mark-32px.png
   :target: https://github.com/RiemannAI/theta

`View on Github <https://github.com/RiemannAI/theta>`_

	    
Welcome to theta's documentation!
=================================

Theta is a machine learning (ML) framework implementing the
Riemann-Theta Boltzmann Machine (RTBM), written in Python and
Cython. It offers a high-level interface to build and train RTBM based
ML architectures for probability density estimation, data regression
and classification.

The code implements the RTBM as described in the theoretical paper
`arXiv:1712.07581 <https://arxiv.org/abs/1712.07581>`_.


.. note::

   Theta is in a proof-of-concept / research phase. You may observe
   that model training requires some fine tune to get proper results.

.. toctree::
   :maxdepth: 3
   :caption: User documentation

   install
   theta
   examples
     
..
  Indices and tables
  ==================

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`


License and citation policy
---------------------------

The theta package is an open-source package under AGPLv3. If you use the theta
package please cite the following article::

  Daniel Krefl, Stefano Carrazza, Babak Haghighat, Jens Kahlen.
  Riemann-Theta Boltzmann Machine, arXiv:1712.07581

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1120325.svg
   :target: https://doi.org/10.5281/zenodo.1120325

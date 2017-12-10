Getting Started
===============


Installation
-------------

Before installing the theta package please install the following dependencies::

  python >= 2.7 or python >= 3.6
  cython >= 0.26
  numpy >= 1.13
  cma >= 2.3.1
  matplotlib >= 2.0.0

Then you can proceed and install theta from the Github source. First, clone
the theta repository using :code:`git`::

   git clone --recursive https://github.com/RiemannAI/theta.git

Then, :code:`cd` to the theta folder and run the install command::

  cd theta
  sudo python setup.py install

_______________________

Testing Installation
--------------------

In order to check the success of the installation open a :code:`python` instance
and type::

  import theta
  print(theta.__version__)

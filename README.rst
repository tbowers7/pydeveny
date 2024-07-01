.. |License| image:: https://img.shields.io/github/license/LowellObservatory/LDTObserverTools
   :target: https://github.com/LowellObservatory/LDTObserverTools/blob/main/LICENSE

.. |astropy| image:: https://img.shields.io/badge/powered%20by-AstroPy-blue.svg?style=flat
    :target: https://www.astropy.org/

.. |forks| image:: https://img.shields.io/github/forks/LowellObservatory/LDTObserverTools?style=social
   :target: https://github.com/LowellObservatory/LDTObserverTools/forks

.. |issues| image:: https://img.shields.io/github/issues/LowellObservatory/LDTObserverTools?style=badge
   :target: https://github.com/LowellObservatory/LDTObserverTools/issues

.. |pulls| image:: https://img.shields.io/github/issues-pr/LowellObservatory/LDTObserverTools?style=badge
   :target: https://github.com/LowellObservatory/LDTObserverTools/pulls

.. |stars| image:: https://img.shields.io/github/stars/LowellObservatory/LDTObserverTools?style=social
   :target: https://github.com/LowellObservatory/LDTObserverTools/stargazers

.. |watch| image:: https://img.shields.io/github/watchers/LowellObservatory/LDTObserverTools?style=social
   :target: https://github.com/LowellObservatory/LDTObserverTools/watchers

.. |github| image:: https://img.shields.io/badge/GitHub-LDTObserverTools-brightgreen
   :target: https://github.com/LowellObservatory/LDTObserverTools

.. |latest_version| image:: https://img.shields.io/github/v/release/LowellObservatory/LDTObserverTools?label=version
   :target: https://github.com/LowellObservatory/LDTObserverTools/releases/

.. |release_date| image:: https://img.shields.io/github/release-date/LowellObservatory/LDTObserverTools
   :target: https://github.com/LowellObservatory/LDTObserverTools

.. |language| image:: https://img.shields.io/github/languages/top/LowellObservatory/LDTObserverTools
   :target: https://github.com/LowellObservatory/LDTObserverTools

.. image:: https://raw.githubusercontent.com/LowellObservatory/LDTObserverTools/main/doc/_static/obstools_logo.png
    :target: https://github.com/LowellObservatory/LDTObserverTools
    :width: 500

.. include:: include/links.rst

.. _obstools_main:

LDTObserverTools |forks| |stars| |watch|
========================================

|github| |latest_version| |release_date|

|astropy| |language| |License|

|issues| |pulls|

The LDTObserverTools package is a collection of command-line and GUI tools
for observers at the Lowell Discovery Telescope (LDT) in Happy Jack, AZ.

Some of these tools are Python ports of existing tools written in other
languages, and others are newly written to meet particular needs of the
observing community.  Detailed instructions on how to use each tool are
contained in the `online documentation
<https://lowellobservatory.github.io/LDTObserverTools/>`__.  Please use
the GitHub `Issues 
<https://github.com/LowellObservatory/LDTObserverTools/issues>`__ and/or
`Pull Requests <https://github.com/LowellObservatory/LDTObserverTools/pulls>`__
features to report bugs or suggest new tools and features.

.. contents:: Table of Contents
    :depth: 1
    :local:

----

.. _programs:

List of Programs
================

.. _extant:

Tools Contained in this Package:
--------------------------------

  - DeVeny Spectrograph grating angle calculator (``deveny_grangle``)

  - DeVeny Spectrograph collimator focus sequence estimator (``deveny_collfocus``)

  - DeVeny Spectrograph collimator focus calculator (``dfocus``)

  - DeVeny Spectrograph pickup noise scrubber (``scrub_deveny_pickup`` -- *beta testing*)

  - Simple FITS header fixing tool (``fix_ldt_header``)

.. _future:

Future Tools (planned or in development):
-----------------------------------------

  - LMI exposure time calculator (``lmi_etc``)

  - NEO Confirmation Page ephemeris generator (``neocp_ephem``)

  - Input List Validator (``validate_input_list``)

  - Observer Target List Tool (``observer_target_list``)

==================

.. _installing:

Installing
==========

.. _environment:

Set up a clean python environment
---------------------------------

Because installing a python tool like LDTObserverTools also installs or
upgrades its :ref:`Package Dependencies <dependencies>`, the best course of
action is to setup a clean python environment into which the installation will
occur.  This mitigates any possible dependency conflicts with other packages
you use.

The recommended method of setting up a new environment is with ``conda``:

.. code-block:: console

    conda create -n obstools python=3.11
    conda activate obstools

See `Managing Environments with Conda
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for more details.

.. _pip_install:

Installing LDTObserverTools via ``pip``
---------------------------------------

.. note::
  The commands here assume ``pip`` is associated with Python3.  To check, run
  ``pip --version`` from the command line and check that the associated python
  version is ``>= 3.10``.

  Also, you will need ``git`` installed on your system / in your environment.

To install the latest version of LDTObserverTools and its required dependencies,
execute:

.. code-block:: console

    pip install "obstools @ git+https://github.com/LowellObservatory/LDTObserverTools"

This will download the latest version of the package from GitHub and install it along
with its required dependencies.  (Note: whether or not you need quotation marks depends
on your particular shell -- ``bash``, ``zsh``, etc.)

As the package undergoes continued development, it will be necessary to upgrade
your installation to access the latest features.  The upgrade process should
simply be a matter of executing:

.. code-block:: console

    pip install "obstools @ git+https://github.com/LowellObservatory/LDTObserverTools" --upgrade

.. _optional-dependencies:

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

.. There are no optional dependencies at this time.

Some of the instrument-specific routines in this package require additional dependencies
that are not otherwise needed by the majority of the routines herein.

   - If you are using the ``scrub_deveny_pickup`` tool, you will need the
     spectroscopic data reduction pipeline `PypeIt <https://pypeit.readthedocs.io/en/release/>`_
     for the iterative cleaning of the pickup noise.  It can be installed by
     including it in the optional dependencies, *e.g.*:

      .. code-block:: console

        pip install "obstools[pypeit] @ git+https://github.com/LowellObservatory/LDTObserverTools"

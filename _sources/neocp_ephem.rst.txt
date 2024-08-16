.. include:: include/links.rst

.. |nbsp| unicode:: 0xA0 
   :trim:

.. _neocp_ephem:

=========================================
NEO Confirmation Page Ephemeris Generator
=========================================

Status: *In development*

Overview
========

   - ``neocp_ephem``: An ephemeris-generating tool for querying the JPL Scout database
     for short-shelf- life NEOs that have not yet been assigned a Horizons identifier.
     This tool will turn the returned data into a file that can be ingested into the
     LDT TCS for observations.


Usage
=====

The script usage can be displayed by calling the script with the
``-h`` option:

.. include:: help/neocp_ephem.rst

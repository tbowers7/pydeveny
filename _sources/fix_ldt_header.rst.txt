.. include:: include/links.rst

.. |nbsp| unicode:: 0xA0 
   :trim:

.. _fix_ldt_header:

==============================
Simple FITS Header Fixing Tool
==============================

Status: Completed 2022-10-17

Overview
========

It seems like everyone has a simple tool for adjusting FITS header keywords.
This is a fairly simple-minded one written out of frustration with the
limitations of the ``modhead`` routine distributed with the `CFITSIO
<https://heasarc.gsfc.nasa.gov/fitsio/>`_ package.  It utilizes the
:obj:`ccdproc.ImageFileCollection` functionality (based on the
:mod:`astropy.io.fits` module) to change a FITS keyword in a list of input
files, using python's smarter-than-C string functionality.

Usage
=====

The script usage can be displayed by calling the script with the
``-h`` option:

.. include:: help/fix_ldt_header.rst

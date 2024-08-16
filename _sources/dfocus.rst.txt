.. include:: include/links.rst

.. |nbsp| unicode:: 0xA0 
   :trim:

.. _dfocus:

==================================
DeVeny Collimator Focus Calculator
==================================

Status: Completed 2021-11-19

Overview
========

The internal optics of the DeVeny Spectrograph are focused by pistoning the
collimator mirror (a separate task from focusing the telescope onto the
spectrograph).  The DeVeny LOUI includes a tab for performing a focus sequence
(images of arc lines interleaved with collimator focus moves).  The
:ref:`deveny_collfocus` tool is used to estimate the focus value and sequence
range needed.

Once the sequence of focus images has been collected, this tool will analyze
the arc lines in each frame to calculate the optimal position at which you
should set the collimator.  See :ref:`dfocus_usage` for details of the
command-line options for use with the tool.  When run, the following processing
steps are applied to the focus frames:

    #. All frames are read in, and the middle frame is inspected to find
       appropriate spectral lines for analysis.  (This frame is what is shown
       in :numref:`pyfocus_p1`, theoretically the frame closest to focus,
       especially if ``deveny_collfocus`` was used to generate the sequence
       range.)
    #. The marked lines are then identified in all the other frames, and the
       FWHM for each is computed using :obj:`scipy.signal` processing routines.
    #. The FWHM as a function of collimator position is computed for each line
       identified, and a parabola is fit to the plot (see
       :numref:`pyfocus_p3`).
    #. For arc each line, the minimum (red lines in :numref:`pyfocus_p3`) and
       optimal (blue lines) focus values are computed.  See the DeVeny manual
       for a discussion of astigmatism and why the two values are not the same.
    #. Finally, the optimal focus value is plotted as a function of CCD column
       position (:numref:`pyfocus_p2`) to find the overall optimal collimator
       focus value to use.

An example terminal output corresponding to the figures is shown below:

    .. code-block:: console

        ================================================================================
          DeVeny Collimator Focus Calculator

         Processing center focus image /deveny/20210520a/20210520.0177.fits...
          Background level: 2351.1   Detection threshold level: 2451.1
         Number of lines found: 39

         Processing arc images...
        100%|████████████████████████████████████████| 10/10 [00:00<00:00, 40.87frame/s]

          Median value of all linewidths: 3.03 pix
        ================================================================================
        *** Recommended (Median) Optimal Focus Position: 10.49 mm
        *** Note: Current Mount Temperature is: 18.0ºC

          Plots have been saved to: pyfocus.20210520.040659.pdf

In both the terminal output and the plots, the current mount temperature is
noted so that the observer can judge if the collimator focus needs to be
revisited during the night based on temperature changes.  The collimator focus
temperature term is approximately :math:`-0.08~{\rm mm/C^{\circ}}`, meaning that
a temperature *decrease* of :math:`5 {\rm~C^{\circ}}` will cause a
:math:`0.4~{\rm mm}` *increase* in the optimal collimator focus position.


.. _pyfocus_p1:
.. figure:: figures/pyfocus.page1_example.*
    :class: with-shadow
    :alt: Arc line plot

    -- Example of page 1 of the output PDF, arc line identification plot.


.. _pyfocus_p2:
.. figure:: figures/pyfocus.page2_example.*
    :class: with-shadow
    :alt: Optimal focus versus line position

    -- Example of page 2 of the output PDF, optimal focus versus line position plot.


.. _pyfocus_p3:
.. figure:: figures/pyfocus.page3_example.*
    :class: with-shadow
    :alt: Individual line focus curves

    -- Example of page 3 of the output PDF, individual line focus curves.




.. _dfocus_usage:

Usage
=====

The script usage can be displayed by calling the script with the
``-h`` option:

.. include:: help/dfocus.rst

This command-line tool must be invoked from the ``/deveny/<UTDATE>/focus``
directory on the observer machine (``dct-obs1`` / ``dct-obs2``) so that it can
find the focus index files created by the DeVeny LOUI.  To run the tool on the
most recent focus sequence taken, simply run the routine with no options.

If more than one focus sequence was taken, the tool can analyze a particular
sequence by using the ``--flog`` optional input, where the file log has the
form ``deveny_focus.<UTDATE>.<UTTIME>``.  For instance, to produce the plots in
:numref:`pyfocus_p1` - :numref:`pyfocus_p3`, you would:

    .. code-block:: console

        $ cd /deveny/20210520/focus
        $ dfocus --flog deveny_focus.20210520.040659

If you want to increase the threshold for detected lines above the default
100 |nbsp| DN over background (to decrease the number of lines detected), use
the ``--thresh`` optional input.

By default, the tool tries to launch Apple's Preview App (the observer machines
at LDT are iMacs) to display the plots shown in :numref:`pyfocus_p1` -
:numref:`pyfocus_p3`.  If running on macOS, and you desire to **not** display
the plots, use the ``--nodisplay`` option.  If this tool is being run on a
different operating system, it will simply bypass this step.

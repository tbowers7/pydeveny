# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 05-Dec-2022
#  Modified: 15-Sep-2023
#
#  @author: tbowers

"""Scrubber for DeVeny Pickup Noise Module

LDTObserverTools contains python ports of various LDT Observer Tools

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
https://lowell.edu

This module cleans the RF pickup noise from DeVeny spectral 2D images by
fitting a sinusoid to and then subtracting it from each row of the image.

The frequency of the sinusoid tends to vary from frame to frame but is roughly
CONSTANT within each frame.  Between the non-integer wavelength and the slight
delay between row read-out (caused by the advancement of the parallel register)
the phase shifts from row to row, yielding an ever-changing pattern in the
output images.  In order to identify the frequency in a given image, this
module fist taken the FFT of the flattened (`i.e.`, `time series`) image to
identify the strongest frequency.  (The function that performs the FFT also
modifies a copy of the data in various ways to remove spurious signals in the
FFT.)  From there, it uses a tightly bounded curve fitting (from
:obj:`scipy.optimize`) to identify the sinusoid in each line of the image.

Cosmic rays and other impulsive features in the 2D image can cause spurious
matches for the sinusoid.  So to guard against introducing artifacts into the
cleaned image from the fitting, the row-by-row fit coefficients are each
considered as a function and smoothed with an appropriate median window to
ensure the final subtracted pattern is a smooth function, matching the
underlying pickup noise.

The output of the module is a multi-extension FITS file containing [1] the
cleaned image, [2] the original image, [3,4] the pattern subtracted from the
original image, and [5] a FITS BinTable containing the sinusoid fit parameters
for each row of the image.
"""

# Built-In Libraries
import argparse
import datetime
import pathlib
import warnings

# 3rd Party Libraries
import astropy.io.fits
import astropy.nddata
import astropy.stats
import astropy.table
import astropy.units as u
import astropy.visualization
import astropy.wcs
import ccdproc.utils.slices
import matplotlib.pyplot as plt
import numpy as np
from pypeit import msgs
import pypeit.spec2dobj
import scipy.fft
import scipy.ndimage
import scipy.optimize
import scipy.signal
from tqdm import tqdm

# Internal Imports
from obstools import utils

# Mean pixel readout time (7.25 µs)
PIX_DWELL = 7.25e-6

# Silence a couple known-but-don't-care-about warnings
warnings.simplefilter("ignore", astropy.wcs.FITSFixedWarning)
warnings.simplefilter("ignore", astropy.io.fits.verify.VerifyWarning)


# Narrative Functions ========================================================#
def iterative_pypeit_clean(
    filename: pathlib.Path,
    proc_dir: str | pathlib.Path = None,
    overwrite_raw: bool = False,
    diagnostics: bool = False,
    no_refit: bool = False,
    extra_graphics: bool = False,
    rerun_fft: bool = False,
):
    """Iteratively use PypeIt to clean pickup noise

    As part of the data-reduction process, PypeIt fits and extracts images as
    well as the sky background.  We can use the reduced ``spec2d`` files to
    get the "source-free" noise background, which makes identifying the
    sinusoidal noise more straightforward.

    Parameters
    ----------
    filename : :obj:`~pathlib.Path`
        The name of the file to clean
    proc_dir : :obj:`str` or :obj:`~pathlib.Path`, optional
        The location of the PypeIt-processed images, if not ``./ldt_deveny_?/``
        (Default: None)
    overwrite_raw : :obj:`bool`, optional
        Overwrite the raw file rather than create a new file with the '_scrub'
        suffix  (Default: False)
    diagnostics : :obj:`bool`, optional
        Output additional information and plots during the image analysis?
        (Default: False)
    no_refit : :obj:`bool`, optional
        Do not refit lines with "bad" RMS  (Default: False)
    extra_graphics : :obj:`bool`, optional
        Produce extra graphics formats for documentation?  (Default: False)
    rereun_fft : :obj:`bool`, optional
        Pass the fitted pattern through the FFT analysis?  (Default: False)
    """
    # Print a welcome statement
    print(f"\nProcessing frame {filename.name}")

    # Find the corresponding `spec2d` file for this raw file -- the list
    #   comprehension searches in each possible 'instrument setup' directory
    #   Check that the `spec2d` file actually exists and print a helpful
    #   message if it doesn't.
    pyp_dir = sorted(proc_dir.resolve().glob("ldt_deveny_?"))

    try:
        # Look for the spec2d file
        spec2d_file = utils.flatten_comprehension(
            [
                sorted(d.joinpath("Science").glob(f"spec2d_{filename.stem}-*.fits"))
                for d in pyp_dir
            ]
        )[0]
    except (StopIteration, IndexError):
        # And... fail.
        msgs.warn(
            f"File {filename.name} does not have a corresponding PypeIt-processed 2D spectrum. "
            "Check the image type and whether you have `run_pypeit`."
        )
        return
    # Define (and create, if needed) the QA directory for these plots
    qa_dir = spec2d_file.parents[1] / "QA" / "PDFs"
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Load in the PypeIt-produced `spec2d`` file and produce the residual image
    #   Also rotate all images -90º to align with the raw DeVeny frames
    spec2d = pypeit.spec2dobj.Spec2DObj.from_file(spec2d_file, "DET01")
    resid_img = np.rot90(
        (spec2d.sciimg - spec2d.skymodel - spec2d.objmodel)  # Residual Image
        * spec2d.select_flag(invert=True).astype(float),  # Good Pixel Mask
        k=-1,  # Rotate CCW
    )

    # Bright objects can mess with the FFT cleaning as well as other parts
    #   of this process.
    objmodel = np.rot90(spec2d.objmodel.copy(), k=-1)

    # To guide the fitting of sinusoids to individual lines, compute the
    #   expected (average) pixel period from the FFT of the flattened array.
    resid_pixperiod = flatten_clean_fft(
        resid_img.copy(),
        fft_plot=True,
        qa_dir=qa_dir,
        filename=filename,
        extra_graphics=extra_graphics,
    )
    print(f"   --> FFT-predicted pixel period: {resid_pixperiod:.1f} pix")

    # First, if the brightest part of the object model is > 600, then the
    #   sinusoidal signal only imparts a 1% effect.  Skip objmodel checking
    #   for these cases
    if np.max(objmodel) > 600:
        print(" * Object model > 100x sinusoidal signal.  Moving along...")
        with_obj = False
    elif np.allclose(objmodel, 0):
        print(" * No object model extracted.  Moving along...")
        with_obj = False
    # Before fitting the image, inspect the objmodel for sinusoidal signal of the
    #  type we're looking for
    else:
        print("Checking the object model for bulk sinusoidal signal...")
        obj_fitc = fit_lines(
            objmodel,
            resid_pixperiod,
            trim_ends=False,
            objmodel_check=True,
            show_diagnostic=diagnostics,
        )
        # TODO: Tweak this and add protections against bright objects

        # This will be a tunable parameter
        if obj_fitc and np.max(obj_fitc["a"]) >= 3:
            # Add the object model back into the residual image because we need to
            #   fit out the sine from the object.
            print(
                " * Adding the object model back into the residual image for "
                "fitting; extracted object contains target sinusoidal signal, "
                f"amplitude = {np.max(obj_fitc['a']):.2f}."
            )
            resid_img += objmodel
            with_obj = True
        else:
            print(" * Object model appears clean of target sinusoidal signal.")
            with_obj = False

    # Fit a sinusoid to each line in the image using the pixel period as a guess
    #  ==> This is the main point of the function
    print("Fitting sinusoids to each line in the image...")
    resid_fitc = fit_lines(
        resid_img.copy(), resid_pixperiod, trim_ends=False, show_diagnostic=diagnostics
    )
    # Print a happy statement of fact
    mean_fit_period = resid_fitc["lambda"].mean()
    print(f"   --> Mean fit pixel period: {mean_fit_period:.1f} pix")

    # Refit lines by passing the existing fit coeffs to fit_lines()
    if not no_refit:
        print("Refitting lines with poor fit in first pass...")
        resid_fitc = fit_lines(
            resid_img.copy(),
            resid_pixperiod,
            orig_fitc=resid_fitc,
            trim_ends=False,
            show_diagnostic=diagnostics,
        )
        # (Re-)Print a happy statement of fact
        mean_fit_period = resid_fitc["lambda"].mean()
        print(f"   --> Mean fit pixel period: {mean_fit_period:.1f} pix")

        # This last step refits the residual image using the knowledge that the
        #   AC pickup noise is of nearly constant amplitude and period.  The
        #   `fixed_sinusoid` argument to fit_lines() fits an order=2 polynomial
        #   to each of the amplitude and period as a function of row number,
        #   and then fixes those values in the sinusoid fit of each row to the
        #   smoothed value.  This has the effect of the sinusoid fitting only
        #   considering the sinusoid phase and underlying order=2 polynomial of
        #   the row.  This is designed to REDUCE noise by eliminating the
        #   random oscillations in these parameters due to fitting a sinusoid
        #   to noisy data.
        print(
            "Refitting all lines within the slit assuming nearly constant sinusoid..."
        )
        fixed_fitc, smoothing = fit_lines(
            resid_img.copy(),
            resid_pixperiod,
            orig_fitc=resid_fitc,
            trim_ends=False,
            show_diagnostic=diagnostics,
            fixed_sinusoid=True,
        )
        # (Re-re-)Print a happy statement of fact
        mean_fit_period = fixed_fitc["lambda"].mean()
        print(f"   --> Mean fit pixel period: {mean_fit_period:.1f} pix")
    else:
        fixed_fitc, smoothing = None, None

    # Construct the pattern image and apply it to the input (pattern has mean = 0)
    _, pattern_resid = create_and_apply_pattern(
        resid_img, resid_fitc if no_refit else fixed_fitc
    )

    # Make disgnostic plots
    print(f" Writing QA plots to {qa_dir}")
    make_image_comparison_plots(
        filename,
        spec2d,
        resid_img,
        pattern_resid,
        qa_dir=qa_dir,
        with_obj=with_obj,
        extra_graphics=extra_graphics,
    )
    make_sinusoid_fit_plots(
        filename,
        resid_fitc,
        resid_pixperiod,
        qa_dir=qa_dir,
        extra_graphics=extra_graphics,
        fixed_fitc=fixed_fitc,
        smoothing=smoothing,
    )

    # If desired, run the pattern back though the FFT function for comparison
    if rerun_fft:
        pattern_pixperiod = flatten_clean_fft(
            pattern_resid.copy(),
            fft_plot=True,
            qa_dir=qa_dir,
            filename=filename.with_stem(f"{filename.stem}_pattern"),
            extra_graphics=extra_graphics,
        )
        print(f"   --> FFT-predicted pixel period: {pattern_pixperiod:.1f} pix")

        cleaned_pixperiod = flatten_clean_fft(
            resid_img.copy() - pattern_resid.copy(),
            fft_plot=True,
            qa_dir=qa_dir,
            filename=filename.with_stem(f"{filename.stem}_cleaned"),
            extra_graphics=extra_graphics,
        )
        print(f"   --> FFT-predicted pixel period: {cleaned_pixperiod:.1f} pix")

    # Next, use the pattern to clean the raw image and repackage
    # Load in the raw dataframe
    ccd = astropy.nddata.CCDData.read(filename)

    # Since the above work was done on PypeIt-reduced frames, we need to place
    #   those products within the larger (untrimmed) raw frame.
    trimsec = ccdproc.utils.slices.slice_from_string(
        ccd.header["TRIMSEC"], fits_convention=True
    )

    # NOTE: The PypeIt-processed data are in e-, whereas the raw data are in ADU
    pattern_resid /= ccd.header["GAIN"]

    # Make the cleaned array & the pattern array -- all in ADU
    cleaned_array = ccd.data.copy().astype(np.float64)
    cleaned_array[trimsec] -= pattern_resid

    pattern_array = np.zeros_like(ccd.data, dtype=np.float64)
    pattern_array[trimsec] = pattern_resid

    # Package it up into a processed FITS image
    package_into_fits(
        filename,
        ccd,
        cleaned_array,
        pattern_array,
        resid_fitc if no_refit else fixed_fitc,
        resid_pixperiod,
        overwrite_raw=overwrite_raw,
    )


def clean_pickup(
    filename: pathlib.Path,
    use_hann: bool = False,
    sine_plot: bool = True,
    fft_plot: bool = False,
):
    """Clean the Pickup Noise

    .. warning::

        Deprecated -- do not use.  Only included for historical purposes.
        Use :func:`iterative_pypeit_clean` instead.

    This is the first version of the cleaning code, which operates directly on
    the raw DeVeny image.  It suffers from confusion from bright night-sky
    lines and brighter objects.  These deficiencies led to the development of
    the iterative scrubber.  This code is kept here for historical purposes.

    Parameters
    ----------
    filename : :obj:`~pathlib.Path`
        The name of the file to clean
    use_hann : :obj:`bool`
        Use a Hann window when cleaning the image for FFT (Default: False)
    sine_plot : :obj:`bool`
        Create a plot of the sinusoid pattern fits for this image?
        (Default: True)
    fft_plot : :obj:`bool`
        Create a plot of the FFT analysis for this image?
        (Default: False)
    """
    # Read in the image and create copy arrays on which to work
    ccd = astropy.nddata.CCDData.read(filename)

    # Check if this images has already been post-processed by this routine
    if "postproc" in ccd.header and ccd.header["postproc"] == "deveny_pickup":
        print(
            f"\n   Skipping file {filename.name}; already processed with deveny_pickup"
        )
        return

    # Check image type; only process OBJECT and BIAS
    if (imgtyp := ccd.header["imagetyp"]) in ["OBJECT", "BIAS"]:
        print(f"\n   Processing {imgtyp} file {filename.name}")
    else:
        print(f"\n   Skipping {imgtyp} file {filename.name}")
        return

    # Get the expected pixel period from the FFT of the flattened array
    pixel_period = flatten_clean_fft(
        ccd.data.copy(), use_hann=use_hann, fft_plot=fft_plot
    )
    print(f"This is the FFT-predicted pixel period: {pixel_period:.1f} pix")

    # Compute the fit coefficients
    fit_coeffs = fit_lines(ccd.data, pixel_period)

    # Smooth the fit coefficients as a function of row to remove artifacts due
    #  to cosmic rays or strong sources.
    # NOTE: scipy.signal.medfilt() uses zero-padding for the ends of the signal,
    #       therefore subtract the mean and re-add it to eliminate edge weirdness
    fit_coeffs["smooth_a"] = smooth_array(fit_coeffs["a"])
    fit_coeffs["smooth_lambda"] = smooth_array(fit_coeffs["lambda"])
    fit_coeffs["smooth_phi"] = smooth_array(fit_coeffs["phi"], ftype="savgol")
    # Print a happy statement of fact
    print(
        f"This is the mean fit pixel period: {fit_coeffs['smooth_lambda'].mean():.1f} pix"
    )

    cleaned_array, pattern_array = create_and_apply_pattern(ccd.data, fit_coeffs)

    # Make a plot, if desired
    if sine_plot:
        make_sinusoid_fit_plots(fit_coeffs, filename, pixel_period)

    package_into_fits(
        filename, ccd, cleaned_array, pattern_array, fit_coeffs, pixel_period
    )


# Task-Oriented Helper Functions =============================================#
def flatten_clean_fft(
    data_array: np.ndarray,
    use_hann: bool = False,
    fft_plot: bool = False,
    qa_dir: pathlib.Path = None,
    filename: pathlib.Path = None,
    extra_graphics: bool = False,
) -> float:
    """Find the peak frequency of sinusoidal noise

    The function name describes what it does:
        #. Flatten the 2D array into a 1D timeseries
        #. Clean the ends of the CCD for smoother transition
        #. Take the FFT to find the proper frequency to return

    In order to more effectively find the frequencies of persistent signal
    across the detector, start by flattening out the data array into a one-
    dimensional timeseries-like object representing the order in which the
    pixels are read out.

    According to information in Chapter 5 of `Scientific Charge-Coupled
    Devices` by James R. Janesick (2001), the time to advance a row along
    the parallel register is small compared to the individual pixel
    digitization time because the latter requires a capacitive stabilization
    time to minimize readnoise, whereas parallel register moves simply move
    the charge without trying to measure it.  Therefore, there is no spacing
    in between the rows of the flattened arrays to represent the time required
    for the parallel charge transfer.

    Once the array is flattened, we deal with row-to-row edge effects by
    modifying the flattened array to replace the 10 pixels at the end of one
    row and the 10 pixels at the start of the next row with a linearly varying
    series of values.  The first 10 and last 10 pixels of the flattened array
    are set to the next adjacent value.  This removes some of the "ringing"
    power at integer frequencies of an entire row and places it into the
    fundamental row-length frequency.  There are still sharp edges in the
    flattened array caused by night sky lines, but most of the power from these
    lines should be at the fundamental frequency (whole-row).

    The detected pickup noise has a period in the range 100-300 pixels, so
    peaks in the power spectrum at lower frequencies than this (`e.g.`, the
    fundamental row-length frequency) are masked when choosing the return
    value.

    Additionally, the assumption of continuous readout is close but not
    entriely accurate.  The slight pause in the readout caused by the parallel
    register readout causes the continuous readout amplifier pickup signal
    to manifest itself as a Gaussian packet of related frequencies in the FFT.
    By smoothing the power spectrum ( abs(FFT)^2 ) with a Gaussian kernel, narrow
    spikes in the FFT (caused by autocorrelation of night sky lines, cosmic
    rays, edge effects) are diluted compared to the sought signal.  The
    returned period is that of the maximum of the Gaussian-smoothed absolute
    value squared.

    Parameters
    ----------
    data_array : :obj:`~numpy.ndarray`
        _description_
    use_hann : :obj:`bool`
        Use a Hann window on each row in addition to subtracting the row mean
        (Default: False)
    fft_plot : :obj:`bool`
        Create a debugging plot of the FFT analysis?  (Default: False)
    qa_dir : :obj:`~pathlib.Path`, optional
        The QA directory into which to place the optional plots.  If ``None``,
        the plots will be placed in the current working directory.
        (Default: None)
    filename : :obj:`~pathlib.Path`
        The filename of the raw data frame for the optional plot.  (Default: None)
    extra_graphics : :obj:`bool`, optional
        Produce extra graphics formats for documentation?  (Default: False)

    Returns
    -------
    :obj:`float`
        The pixel period of the sinusoidal oscillation to be removed
    """

    # Compute the shape of the input array
    nrow, ncol = data_array.shape

    # If so desired, apply a Hann window to each row
    if use_hann:
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(ncol) / ncol))
        data_array = data_array * hann

    # Flatten the array; force Column-Major to keep python from getting ideas
    flat_array = data_array.flatten(order="C").astype(np.float64)

    # Smooth out the beginning of the first and end of the last rows
    flat_array[:10] = flat_array[10]
    flat_array[-10:] = flat_array[-11]

    # For all other row breaks, index by `rownum``
    for rownum in range(1, nrow):
        # Build up a slice
        idx = np.s_[rownum * ncol - 10 : rownum * ncol + 10]
        # Pull the chunk and replace it with something smooth
        chunk = flat_array[idx]
        flat_array[idx] = np.linspace(chunk[0], chunk[-1], num=len(chunk))

    # Use sigma-clipped stats to remove wild swings that may affect the FFT
    _, med, std = astropy.stats.sigma_clipped_stats(flat_array, sigma=3.0)
    clip_ind = (flat_array > med + 5.0 * std) | (flat_array < med - 5.0 * std)
    flat_array[clip_ind] = 0

    # Onward to the FFT!
    # Subtract the mean to remove power from 0 frequency
    y_fft = scipy.fft.fft(flat_array - np.mean(flat_array))

    # Compute the frequency array; N = N points, T = Spacing (PIX_DWELL)
    x_fft = scipy.fft.fftfreq(n_pts := flat_array.size, PIX_DWELL)

    # Mask out periods longer than ~500 pixels (this also removes negative freq's)
    y_fft[x_fft < pixper_tofrom_hz(500)] = 0.0

    # Compute the power spectrum, smoothed with a 10-Hz Gaussian
    pspec = scipy.ndimage.gaussian_filter1d(np.abs(y_fft) ** 2, 10)

    # Find the peak of the smoothed power spectrum, seeking Gaussin packets
    peak_freq = x_fft[np.argmax(pspec[: n_pts // 2])]

    if fft_plot:
        create_fft_plot(
            filename,
            flat_array,
            y_fft,
            x_fft,
            peak_freq,
            qa_dir=qa_dir,
            extra_graphics=extra_graphics,
        )

    # Return the period (in pixels) that corresponds to this frequency
    return pixper_tofrom_hz(peak_freq)


def fit_lines(
    data_array: np.ndarray,
    pixel_period: float,
    orig_fitc: astropy.table.Table = None,
    refit_thresh_sig: float = 3.0,
    trim_ends: bool = True,
    objmodel_check: bool = False,
    show_diagnostic: bool = False,
    fixed_sinusoid: bool = False,
) -> astropy.table.Table:
    """Fit a sinusoid to each line in the image

    This is like a mini-driver function that fits a sinusoid to each line in
    the image.

    It also can refit lines that have excessive RMS residual as compared to the
    bulk of the image.  To enable the refitting, pass in the fit coefficient
    table from the first fitting.

    Parameters
    ----------
    data_array : :obj:`~numpy.ndarray`
        The image array to be processed.  This should be HORIZONTAL and in the
        same orientation as images written out by lois directly from the
        instrument.
    pixel_period : :obj:`float`
        The predicted pixel period for this image array, as computed by
        :func:`flatten_clean_fft`.
    orig_fitc : :obj:`~astropy.table.Table`, optional
        If refitting, the first-pass fit coefficient table.  If absent, then a
        full fitting of all lines will be performed.  (Default: None)
    refit_thresh_sig : :obj:`float`, optional
        Threshold for RMS deviation (in sigma-clipped standard deviation units)
        from the mean for a line to be refit.  (Default: 3.0)
    trim_ends : :obj:`bool`, optional
        Trim the 5 pixels off the ends of the line before fitting?
        (Default: True)
    objmodel_check : :obj:`bool`, optional
        Is this image the spec2d objmodel?  (Default: False)
    show_diagnostic : :obj:`bool`, optional
        Display a diagnotic plot of several lines w/ fit?  (Default: False)
    fixed_sinusoid : :obj:`bool`, optional
        Determine the (roughly) constant sinusoid amplitude and period and fit
        the lines for phase and secular drift only?  (Default: False)

    Returns
    -------
    :obj:`~astropy.table.Table`
        A table object containing the fit coefficients, one table row per row
        of the input ``data_array``.
    """
    # Array shape
    nrow, ncol = data_array.shape
    img_mean = np.nanmean(data_array)

    # If refitting, determine the rows with good rms and those needing refit
    #  Refitting is done if `orig_fitc` is passed in, but not if
    #  `fixed_sinusoid` or `objmodel_check` is checked.
    if do_refit := (orig_fitc is not None and not (fixed_sinusoid or objmodel_check)):
        # =================================#
        # Find lines with "bad" RMS or with very small amplitude and non-zero RMS,
        #   then refit those lines using tight priors based on adjacent lines
        avg, med, std = astropy.stats.sigma_clipped_stats(orig_fitc["rms"], sigma=3.0)
        if show_diagnostic:
            print(f"Stats: mean = {avg:.3f}  median = {med:.3f}  stddev = {std:.3f}")

        # Cap the used STD at 0.2 for identifying lines needing refitting
        std = min(std, 0.2)

        # Indices of rows with "good" RMS and those needing refitting
        ind_goodrms = np.arange(nrow, dtype=int)[
            (orig_fitc["rms"] >= avg - refit_thresh_sig * std)
            & (orig_fitc["rms"] <= avg + refit_thresh_sig * std)
            & (orig_fitc["a"] > 1.0)
        ]
        refit_lines = (orig_fitc["rms"] > avg + refit_thresh_sig * std) | (
            (orig_fitc["a"] < 1.0) & (orig_fitc["rms"] > 0.0)
        )
        if show_diagnostic:
            print(
                f"Total elements: {nrow}  Good RMS:  {len(ind_goodrms)}  "
                f"Refit lines: {np.sum(refit_lines)}"
            )

    # Checking the object model for sinusoidal pattern noise
    if objmodel_check:
        ind_obj = [
            i for i in range(nrow) if not np.allclose(data_array[i], np.zeros(ncol))
        ]

    # If fitting a (roughly) constant sinusoid to all lines, determine the
    #  (slowly varying) values of amplitude and period as a function of row
    #  number by fitting a line to the orig_fitc using a linear least-squares
    #  algorithm.
    if fixed_sinusoid:
        # Valid range is the whole slit -- fill in any odd gaps
        valid_idx = orig_fitc["a"] > 1.0
        start, end = (
            np.arange(len(valid_idx))[
                np.append([valid_idx[0]], np.diff(valid_idx) != 0)
            ]
        )[[0, -1]]
        valid_idx[start:end] = True

        # Fit Chebyshev polynomials to these two coefficients as a f(row)
        row_num = np.arange(nrow)
        p_a = np.polynomial.Chebyshev.fit(
            row_num[valid_idx], orig_fitc["a"][valid_idx], 2
        )
        p_lambda = np.polynomial.Chebyshev.fit(
            row_num[valid_idx], orig_fitc["lambda"][valid_idx], 2
        )
        fixed_a = p_a(row_num)
        fixed_lambda = p_lambda(row_num)

    # ======== Begin Actual Fitting ============#
    # These are the starting guesses and bounds for the sinusoidal fit
    #  [a, lam, phi, y0, lin, quad]
    p0 = [6, pixel_period, 0.5, img_mean, 0, 0]
    bounds = (
        [
            0,
            pixel_period * 0.95,
            0,
            img_mean - 100,
            -np.inf,
            -np.inf,
        ],
        [10, pixel_period * 1.05, 1, img_mean + 100, np.inf, np.inf],
    )

    # Set up the fit_coeffs list based on fitting mode
    fit_coeffs = (
        # Back-convert the Table to a list of dictionaries
        [dict(zip(orig_fitc.colnames, row)) for row in orig_fitc]
        if do_refit or fixed_sinusoid
        # Start with an empty list
        else []
    )

    # Create a progress bar for every occasion!
    progress_bar = tqdm(
        total=np.sum(refit_lines)
        if do_refit
        else len(ind_obj)
        if objmodel_check
        else np.sum(valid_idx)
        if fixed_sinusoid
        else nrow,
        colour="#87EBCF"
        if do_refit
        else "#EBC687"
        if objmodel_check
        else "#EB89EB"
        if fixed_sinusoid
        else "#87CEEB",
        unit="row",
        unit_scale=False,
    )

    # Prepare the line diagnostic plot
    if show_diagnostic:
        # Make a diagnostic plot
        _, axes = plt.subplots(nrows=4, figsize=(9, 9))
        tsz = 8
        pidx = 0

    # Loop over the rows in the image to fit the pattern
    xpl = np.arange(ncol)
    for img_row in range(nrow):
        # If refitting, follow this logic:
        if do_refit:
            # If not refitting this line, move along
            if not refit_lines[img_row]:
                continue

            # Estimate the parameters more specifically from the "good" rms
            #   lines in the vicinity
            closest_idx = ind_goodrms[np.argmin(np.abs(img_row - ind_goodrms))]
            # Amplitude
            p0[0] = orig_fitc["a"][closest_idx]
            # Period
            p0[1] = orig_fitc["lambda"][closest_idx]
            bounds[0][1] = orig_fitc["lambda"][closest_idx] - 5
            bounds[1][1] = orig_fitc["lambda"][closest_idx] + 5

        if fixed_sinusoid:
            # If not setting a fixed value, move along
            if not valid_idx[img_row]:
                continue

            # Set the parameters for the fixed parameters
            epsilon = 1.0e-4
            # Amplitude
            p0[0] = fixed_a[img_row]
            bounds[0][0] = fixed_a[img_row] - epsilon
            bounds[1][0] = fixed_a[img_row] + epsilon
            # Period
            p0[1] = fixed_lambda[img_row]
            bounds[0][1] = fixed_lambda[img_row] - epsilon
            bounds[1][1] = fixed_lambda[img_row] + epsilon

        if objmodel_check:
            # If looking for object model, skip lines without the object model
            if img_row not in ind_obj:
                continue

        if trim_ends:
            # Pull this line, minus the last few pixels at each end
            line = data_array[img_row, 5:-5]
        else:
            # Use the entire line
            line = data_array[img_row]

        # To mitigate cosmic rays and night sky lines, sigma clip @ 5σ
        sig_clip = np.std(line) * 5.0 + np.median(line)
        line[line > sig_clip] = sig_clip

        # Smooth the line with a median filter at 1/10th the pixel period
        line = smooth_array(line, kernel_size=utils.nearest_odd(pixel_period / 10.0))

        # Perform the curve fit
        try:
            # If the line is identically zero, don't fit
            if np.allclose(line, np.zeros_like(line)):
                popt = p0
                popt[0] = 0
                pcov = np.diag(np.ones(len(popt)))
                rms = 0
            else:
                # The optimization method is the "Trust Region Reflective" algorithm
                # The parameters are scaled iteratively from the Jacobian matrix
                popt, pcov, infodict, mesg, ier = scipy.optimize.curve_fit(
                    utils.sinusoid,
                    xpl,
                    line,
                    p0=p0,
                    bounds=bounds,
                    full_output=True,
                    method="trf",
                    x_scale="jac",
                )
                rms = np.sqrt(np.mean(infodict["fvec"] ** 2))
        except RuntimeError:
            # Reached max function evaluations; set popt and pcov
            popt = p0
            popt[0] = 0
            pcov = np.diag(np.ones(len(popt)))
            rms = np.inf

        # Diagnositc Plot showing particular rows ==========>
        if show_diagnostic and img_row in [76, 193, 285, 412]:
            axis = axes[pidx]
            # Add this to the plot
            axis.plot(xpl, line, "k-", linewidth=0.75)
            axis.plot(
                xpl,
                utils.sinusoid(xpl, *popt),
                "r-",
                label=make_sine_label(popt, infodict, mesg, ier),
            )
            utils.set_std_tickparams(axis, tsz)
            axis.set_xlabel(f"Pixel Position, Row #{img_row}", fontsize=tsz)
            axis.legend(loc="upper left", fontsize=tsz)
            pidx += 1

        # Compute standard deviations and correlation matrix
        pstd = np.sqrt(np.diag(pcov))
        # Check for zero stddev, replace with small stddev
        pstd[pstd == 0] = 1.0e-8
        inv_pstd = np.linalg.inv(np.diag(pstd))
        pcor = np.matmul(np.matmul(inv_pstd, pcov), inv_pstd)

        # Add the fit coefficients to the list
        coeff_dict = {
            "a": popt[0],
            "a_err": pstd[0],
            "lambda": popt[1],
            "lambda_err": pstd[1],
            "phi": popt[2],
            "phi_err": pstd[2],
            "y0": popt[3],
            "y0_err": pstd[3],
            "corr_a_lambda": pcor[0][1],
            "corr_a_phi": pcor[0][2],
            "corr_a_y0": pcor[0][3],
            "corr_lambda_phi": pcor[1][2],
            "corr_lambda_y0": pcor[1][3],
            "corr_phi_y0": pcor[2][3],
            "rms": rms,
        }
        if do_refit or fixed_sinusoid:
            # Replace the existing row in the table with the new dictionary
            fit_coeffs[img_row] = coeff_dict
        else:
            # Append the dictionary to the list
            fit_coeffs.append(coeff_dict)

        progress_bar.update(1)

    # End of the loop; close out the diagnostic plot
    progress_bar.close()
    if show_diagnostic:
        plt.tight_layout()
        plt.show()
        plt.close()

    # Convert the list of dict into a Table for return (optionally return smoothing coef)
    return (
        (astropy.table.Table(fit_coeffs), {"p_a": p_a.coef, "p_lambda": p_lambda.coef})
        if fixed_sinusoid
        else astropy.table.Table(fit_coeffs)
    )


def smooth_array(
    array: np.ndarray, kernel_size: int = 21, ftype: str = "median"
) -> np.ndarray:
    """Smooth out an array with a given filter

    This may be used for smoothing a line or attempting to smooth fit
    coefficients from row to row in order to ameliorate the effects of cosmic
    rays and strong sources.

    .. note::

        :func:`scipy.signal.medfilt` uses zero-padding for the ends of the
        signal, therefore subtract the mean and re-add it to eliminate edge
        weirdness.

    Parameters
    ----------
    array : :obj:`~numpy.ndarray`
        The array to be smoothed.
    kernel_size : :obj:`int`, optional
        Kernel or window size for the smoothing function (Default: 21)
    ftype : :obj:`str`, optional
        Smoothing function to use.  Options are "median" and "savgol".
        (Default: "median")

    Returns
    -------
    :obj:`~numpy.ndarray`
        The smoothed coefficients
    """
    if ftype == "median":
        return (
            scipy.signal.medfilt(
                array - (med := np.median(array)), kernel_size=kernel_size
            )
            + med
        )
    if ftype == "savgol":
        return scipy.signal.savgol_filter(
            array, window_length=kernel_size, polyorder=2, mode="nearest"
        )
    raise ValueError(f"Filter type {ftype} not supported by this function.")


def create_and_apply_pattern(
    input_array: np.ndarray, fit_coeffs: astropy.table.Table
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the pattern from the fit coefficients

    .. note::

        The returned ``pattern_array`` has a mean = 0 (*i.e.*, the pattern is
        constructed from the pure sinusoid w/o offset or other additive terms).

    Parameters
    ----------
    input_array : :obj:`~numpy.ndarray`
        The input array to be cleaned
    fit_coeffs : :obj:`~astropy.table.Table`
        The fit coefficient table produced by :func:`fit_lines`

    Returns
    -------
    cleaned_array : :obj:`~numpy.ndarray`
        The cleaned sience array
    pattern_array : :obj:`~numpy.ndarray`
        The pattern removed from ``input_array`` -- mean = 0
    """
    # Create the arrays that the processed data will go into
    cleaned_array = input_array.copy().astype(np.float64)
    pattern_array = input_array.copy().astype(np.float64)

    # Loop through the image and use the smoothed sinusoid fit coefficients
    for img_row, table_row in enumerate(fit_coeffs):
        # Apply the adjusted pattern to the entire row
        line = input_array[img_row, :]

        # Compute the pattern
        pattern = utils.sinusoid(
            np.arange(line.size),
            table_row["a"],
            table_row["lambda"],
            table_row["phi"],
            0,
        )

        # Fill in the new arrays with the cleaned data and pattern
        cleaned_array[img_row, :] = line - pattern
        pattern_array[img_row, :] = pattern

    # Return when done
    return cleaned_array, pattern_array


def package_into_fits(
    filename: pathlib.Path,
    ccd: astropy.nddata.CCDData,
    cleaned_array: np.ndarray,
    pattern_array: np.ndarray,
    fit_coeffs: dict,
    pixel_period: float,
    overwrite_raw: bool = False,
):
    """Package the cleaned data into a FITS file

    Package everything into a multiextension FITS file:

    0. Primary HDU -- contains the raw file's FITS header
    1. Cleaned Image HDU -- raw - pattern
    2. Original Image HDU -- raw
    3. Pattern Image HDU -- pattern
    4. Pattern about Raw Mean Image HDU -- pattern + mean(raw)
    5. Fit Coefficients BinTable HDU -- fit_coeffs

    The output filename is the same (including path) as the input raw file,
    but with "_scrub" appended before ".fits".

    Parameters
    ----------
    filename : :obj:`~pathlib.Path`
        The filename of the raw data frame
    ccd : :obj:`~astropy.nddata.CCDData`
        The raw data object
    cleaned_array : :obj:`~numpy.ndarray`
        The cleaned raw data array (raw data - pattern) in ADU
    pattern_array : :obj:`~numpy.ndarray`
        The computed pattern array in ADU
    fit_coeffs : :obj:`dict`
        The fit coefficients dictionary from :func:`fit_lines`
    pixel_period : :obj:`float`
        The FFT-computed pixel period from :func:`flatten_clean_fft`
    overwrite_raw : :obj:`bool`, optional
        Overwrite the raw file rather than create a new file with the '_scrub'
        suffix  (Default: False)
    """
    # Add a little history
    time_str = datetime.datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    history_str = f"Written by package obstools: {time_str} UTC"
    # For the image HDUs, include a basic header
    img_hdr = astropy.io.fits.Header({"BUNIT": "ADU", "HISTORY": history_str})

    # Primary HDU: #0 -- Construct from raw header plus new keywords
    primary_hdu = astropy.io.fits.PrimaryHDU(header=ccd.header)
    primary_hdu.update_header()
    primary_hdu.header.append(
        (
            "postproc",
            "scrub_deveny_pickup",
            "Post-processing algorithm or package applied",
        )
    )
    primary_hdu.header.append(
        (
            "post_ext",
            1,
            "Extension holding post-processed image (zero-indexed)",
        )
    )
    primary_hdu.header["HISTORY"] = history_str

    # Image HDU: #1 -- Cleaned Image
    clean_hdu = astropy.io.fits.ImageHDU(
        astropy.nddata.CCDData(cleaned_array, unit=u.adu),
        name="CLEANED",
        header=img_hdr,
    )
    primary_hdu.header.append(("EXT0001", "CLEANED", "Cleaned Image (raw - pattern)"))

    # Image HDU: #2 -- Original (Raw) Frame
    orig_hdu = astropy.io.fits.ImageHDU(ccd, name="ORIGINAL", header=img_hdr)
    primary_hdu.header.append(("EXT0002", "ORIGINAL", "Original Image (raw)"))

    # Image HDU: #3 -- Pattern Image (mean = 0)
    pattern0_hdu = astropy.io.fits.ImageHDU(
        astropy.nddata.CCDData(pattern_array, unit=u.adu),
        name="PATTERN0",
        header=img_hdr,
    )
    primary_hdu.header.append(
        ("EXT0003", "PATTERN0", "Pickup Noise Pattern (zero mean)")
    )

    # Image HDU: #4 -- Pattern Image about Raw Mean (mean = mean(raw))
    pattern1_hdu = astropy.io.fits.ImageHDU(
        astropy.nddata.CCDData(pattern_array + np.mean(ccd.data), unit=u.adu),
        name="PATTERN1",
        header=img_hdr,
    )
    primary_hdu.header.append(
        ("EXT0004", "PATTERN1", "Pickup Noise Pattern (image mean)")
    )

    # BinTable HDU: #5 -- Sinusoid Fit Coefficients
    table_hdu = astropy.io.fits.BinTableHDU(fit_coeffs, name="FIT DATA")
    table_hdu.header.set(
        "fft_per",
        np.around(pixel_period, 1),
        "Pattern periodicity computed from FFT (pixels)",
        before="EXTNAME",
    )
    table_hdu.header.set(
        "mean_per",
        np.around(fit_coeffs["lambda"].mean(), 1),
        "Mean fit pattern periodicity (pixels)",
        before="EXTNAME",
    )
    table_hdu.header.set(
        "mean_amp",
        np.float64(np.around(np.median(fit_coeffs["a"]) * 2.0, 2)),
        "Median pattern peak-to-peak amplitude (ADU)",
        before="EXTNAME",
    )
    for col in [c for c in fit_coeffs.colnames if "corr_" in c]:
        table_hdu.header.set(
            col,
            np.float64(np.around(np.median(fit_coeffs[col]), 4)),
            "Median correlation coeff between parameters",
            before="EXTNAME",
        )
    table_hdu.header[
        "COMMENT"
    ] = "Table contains the sinusoid fit coefficients for each row of the image"
    table_hdu.header["HISTORY"] = history_str
    primary_hdu.header.append(("EXT0004", "FIT DATA", "Fit coefficients per line"))

    # Assemble the whole thing into an HDUList and write to disk
    hdul = astropy.io.fits.HDUList(
        [primary_hdu, clean_hdu, orig_hdu, pattern0_hdu, pattern1_hdu, table_hdu]
    )
    # Output filename is the same as the original, with "_scrub" appended before ".fits"
    out_fn = (
        filename.with_stem(f"{filename.stem}_scrub") if not overwrite_raw else filename
    )

    # Print a helpful statement
    print(f"Writing out scrubbed FITS file: {out_fn.name}")
    hdul.writeto(out_fn, overwrite=True)


# Plotting Functions for QA / Debugging ======================================#
def create_fft_plot(
    filename: pathlib.Path,
    flat_array: np.ndarray,
    y_fft: np.ndarray,
    x_fft: np.ndarray,
    peak_freq: float,
    qa_dir: pathlib.Path = None,
    extra_graphics: bool = False,
):
    """Create the FFT analysis plot

    This function creates a multipanel plot saved as ``fft_analysis.pdf`` in
    the current working directory.  The panels are:

    1. The flattened time-like array, where pixels are in the order in which
       they were read out by the CCD electronics.
    2. The real part of the FFT of the flattened array, where the abscissa is
       shown as the sinusoid period in pixels.
    3. The imaginary part of the FFT of the flattened array, where the abscissa
       is shown as the sinusoid period in pixels.
    4. The absolute value squared of the FFT of the flattened array, to show
       the power as a function of frequency.

    Parameters
    ----------
    filename : :obj:`~pathlib.Path`
        The filename of the raw data frame
    flat_array : :obj:`~numpy.ndarray`
        The flattened (1D) pixel array
    y_fft : :obj:`~numpy.ndarray`
        The (complex) FFT values
    x_fft : :obj:`~numpy.ndarray`
        The FFT frequencies for the values in ``y_fft``
    peak_freq : :obj:`float`
        The peak frequency measured from the FFT
    qa_dir : :obj:`~pathlib.Path`, optional
        The QA directory into which to place the optional plots.  If ``None``,
        the plots will be placed in the current working directory.
        (Default: None)
    extra_graphics : :obj:`bool`, optional
        Produce extra graphics formats for documentation?  (Default: False)
    """
    # Ignore UserWarnings in this function
    warnings.simplefilter("ignore", UserWarning)

    # Compute the power spectrum as |FFT|^2; smoothed with a 10-Hz Gaussian
    pspec = scipy.ndimage.gaussian_filter1d(np.abs(y_fft) ** 2, 10)

    # Set up the plotting environment -- somewhat complicated, but pretty
    fig = plt.figure(layout="tight", figsize=(6.4, 6.4))
    gs0 = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 3.0])

    # Lay out the gridspec and map to the usual Axis objects for plotting
    gs00 = gs0[0].subgridspec(1, 1)
    gs01 = gs0[1].subgridspec(3, 1, hspace=0)
    axis0 = fig.add_subplot(gs00[0])
    axis1 = [fig.add_subplot(gs01[i]) for i in range(3)]
    tsz = 8

    # Linear Pixels -- separate thing
    axis0.plot(flat_array, linewidth=0.1, color="k")
    axis0.set_xlabel("Linear pixel number", fontsize=tsz)
    axis0.set_ylabel("Pixel Value (electron)", fontsize=tsz)
    utils.set_std_tickparams(axis0, tsz)

    # Limits and Labels
    _, med, std = astropy.stats.sigma_clipped_stats(flat_array, sigma=5.0)
    axis0.set_ylim(med - 20.0 * std, med + 20.0 * std)
    axis0.ticklabel_format(style="plain", axis="x", scilimits=(0, 0))
    # When making plots for the documentation, obfuscate the source
    fname = "" if extra_graphics else f" for {filename.name}"
    axis0.set_title(f"FFT Analysis{fname}", fontsize=tsz + 2)

    # FFT plots, on common x-axis
    # Make a series of secondary axes
    secaxes = axis1.copy()

    axis1[0].plot(
        x_fft, np.real(y_fft), linewidth=0.1, label="Real Component of the FFT"
    )
    axis1[0].set_ylabel("Re(FFT)", fontsize=tsz)
    axis1[0].legend(loc="upper right", fontsize=tsz)

    axis1[1].plot(
        x_fft, np.imag(y_fft), linewidth=0.1, label="Imaginary Component of the FFT"
    )
    axis1[1].set_ylabel("Im(FFT)", fontsize=tsz)
    axis1[1].legend(loc="upper right", fontsize=tsz)

    axis1[2].plot(
        x_fft,
        pspec,
        linewidth=0.3,
        label="Smoothed Power Spectrum (10-Hz Kernel)",
    )
    ylim = axis1[2].get_ylim()
    axis1[2].plot(
        x_fft,
        np.abs(y_fft) ** 2,
        zorder=0,
        linewidth=0.1,
        label="Raw FFT Power Spectrum",
    )
    axis1[2].set_ylabel("|FFT|^2", fontsize=tsz)
    axis1[2].set_ylim(ylim)
    axis1[2].legend(loc="upper right", fontsize=tsz)
    axis1[2].text(
        0.05,
        0.95,
        f"Peak @ {pixper_tofrom_hz(peak_freq):.1f} pix",
        transform=axis1[2].transAxes,
        fontsize=tsz,
        va="top",
        color="C0",
    )

    for i, (ax, secax) in enumerate(zip(axis1, secaxes), 1):
        # Do the frequency-based parts
        ax.set_xlim(pixper_tofrom_hz(np.array([500, 75])))
        ax.set_xticks([])
        ax.vlines(
            peak_freq,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            color="gray",
            linestyle="--",
            zorder=0.5,
            linewidth=0.6,
        )
        utils.set_std_tickparams(ax, tsz)
        ylim = ax.get_ylim()
        y_tick_vals = [val for val in ax.get_yticks() if ylim[0] < val < ylim[1]]
        ax.set_yticks(y_tick_vals, [f"{val:7.2g}" for val in y_tick_vals])
        # ax.ticklabel_format(style="plain", axis="y", scilimits=(0, 0))
        # Do the bottom inverse axis
        secax = ax.secondary_xaxis(
            "bottom", functions=(pixper_tofrom_hz, pixper_tofrom_hz)
        )
        secax.tick_params(
            axis="both",
            which="both",
            direction="in",
            bottom=True,
            labelsize=tsz,
        )

        if i != 3:
            secax.set_xticklabels(" ")
        else:
            secax.set_xlabel("Sinusoid Period (Pixels)", fontsize=tsz)
        # Do the top inverse axis
        secax = ax.secondary_xaxis(
            "top", functions=(pixper_tofrom_hz, pixper_tofrom_hz)
        )
        secax.set_xticklabels(" ")
        secax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            labelsize=tsz,
        )

    # Add a note about the file creator
    plt.figtext(
        0.005,
        0.005,
        "Created by obstools.scrub_deveny_pickup on "
        f"{datetime.datetime.now().strftime('%Y-%m-%d')}",
        fontsize=tsz - 2,
    )

    # End
    plt.tight_layout()
    save_dir = pathlib.Path(".").resolve() if qa_dir is None else qa_dir
    extns = ["pdf", "svg", "png"] if extra_graphics else ["pdf"]
    for extn in extns:
        plt.savefig(save_dir / f"{filename.stem}_scrubber_fft_analysis.{extn}")
    plt.close()


def make_sinusoid_fit_plots(
    filename: pathlib.Path,
    fit_coeffs: astropy.table.Table,
    pixel_period: float,
    qa_dir: pathlib.Path = None,
    extra_graphics: bool = False,
    fixed_fitc: astropy.table.Table = None,
    smoothing: dict = None,
):
    """Create a set of diagnostic plots from the set of sinusoid fits

    This function creates a multipanel plot saved as ``sinusoid_fits.pdf`` in
    the current working directory.  The panels are:

    1. The row-by-row sinusoid amplitude
    2. The row-by-row sinusoid period
    3. The row-by-row sinusoid phase shift
    4. The row-by-row rms residual when the sinusoidal fit is subtracted from
       the original line

    Parameters
    ----------
    filename : :obj:`~pathlib.Path`
        The filename of the original file, used in the plot title
    fit_coeffs : :obj:`~astropy.table.Table`
        The fit coefficients table
    pixel_period : :obj:`float`
        The estimated pixel period of the sinusoidal noise from the FFT
    qa_dir : :obj:`~pathlib.Path`, optional
        The QA directory into which to place the optional plots.  If ``None``,
        the plots will be placed in the current working directory.
        (Default: None)
    extra_graphics : :obj:`bool`, optional
        Produce extra graphics formats for documentation?  (Default: False)
    fixed_fitc : :obj:`~astropy.table.Table`, optional
        The fit coefficients table with fixed sinusoid  (Default: None)
    smoothing : :obj:`dict`, optional
        Polynomial fit coefficients for amplitude and period  (Default: None)
    """
    # Set up the plotting environment
    _, axes = plt.subplots(nrows=4, figsize=(6.4, 6.4), gridspec_kw={"hspace": 0})
    tsz = 8
    xpl = np.arange(len(fit_coeffs))

    # Set up iterable lists for making plotting cleaner
    fit_ords = [
        fit_coeffs["a"],
        fit_coeffs["lambda"],
        fit_coeffs["phi"],
        fit_coeffs["rms"],
    ]
    fix_ords = [
        fixed_fitc["a"],
        fixed_fitc["lambda"],
        fixed_fitc["phi"],
        fixed_fitc["rms"],
    ]
    polycoefs = [smoothing["p_a"], smoothing["p_lambda"], None, None]
    ylabels = [
        "Amplitude (electron)",
        "Period (pixels)",
        "Phase Shift (phase)",
        "RMS of the fit (electron)",
    ]

    # Loop over the axes!
    for axis, fit_ord, fix_ord, polycoef, ylabel in zip(
        axes, fit_ords, fix_ords, polycoefs, ylabels
    ):
        axis.plot(
            xpl, fit_ord, color="C1", linewidth=1.5, label="Best-Fit per Row", alpha=0.5
        )
        axis.plot(
            xpl, fix_ord, color="black", linewidth=1.0, label="Nearly Constant Sinusoid"
        )
        if polycoef is not None:
            coef_txt = axis.text(
                0.98,
                0.95,
                "Chebyshev Coeffs: " + "   ".join([f"{c:.4g}" for c in polycoef]),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=tsz - 1,
            )
            coef_txt.set_bbox({"facecolor": "0.75", "alpha": 0.2})
        axis.set_ylabel(ylabel, fontsize=tsz)

        # Add the horizontal line for the FFT-predicted period
        if axis == axes[1]:
            axis.hlines(
                pixel_period,
                0,
                1,
                transform=axes[1].get_yaxis_transform(),
                linestyle="--",
                zorder=0,
                color="C2",
                label="FFT-Predicted Period",
            )

        # Rescale y axis
        if axis in [axes[1], axes[2]]:
            grow = 1.5
            ylim = axis.get_ylim()
            med, span = np.mean(ylim), np.diff(ylim)
            axis.set_ylim(med - grow / 2 * span, med + grow / 2 * span)

        axes[3].set_xlabel("Image Row Number", fontsize=tsz)

        utils.set_std_tickparams(axis, tsz)
        axis.legend(loc="lower left" if axis != axes[3] else "upper left", fontsize=tsz)

    # When making plots for the documentation, obfuscate the source
    fname = "" if extra_graphics else f" for {filename.name}"
    axes[0].set_title(f"Sinusoid Pattern Fits{fname}", fontsize=tsz + 2)

    # Add a note about the file creator
    plt.figtext(
        0.005,
        0.005,
        "Created by obstools.scrub_deveny_pickup on "
        f"{datetime.datetime.now().strftime('%Y-%m-%d')}",
        fontsize=tsz - 2,
    )

    plt.tight_layout()
    save_dir = pathlib.Path(".").resolve() if qa_dir is None else qa_dir
    extns = ["pdf", "svg", "png"] if extra_graphics else ["pdf"]
    for extn in extns:
        plt.savefig(save_dir / f"{filename.stem}_scrubber_sinusoid_fits.{extn}")
    plt.close()


def make_image_comparison_plots(
    filename: pathlib.Path,
    spec2d: pypeit.spec2dobj.Spec2DObj,
    resid: np.ndarray,
    pattern: np.ndarray,
    qa_dir: pathlib.Path = None,
    with_obj: bool = False,
    extra_graphics: bool = False,
):
    """Make a set of Image Comparison plots

    This function creates a multipanel plot.  The panels are:

    1. The PypeIt-reduced 2-dimensional spectrum (``spec2d.sciimg``)
    2. The PypeIt sky spectrum model (``spec2d.skymodel``).
    3. The PypeIt-reduced residual image (``spec2d.sciimg`` -
       ``spec2d.skymodel`` - ``spec2d.objmodel``) modified by the good pixel
       mask
    4. The noise pattern image generated here from the sinusoidal fits to each
       row of the residual image
    5. The cleaned residual image (``resid`` - ``pattern``)
    6. The cleaned 2-dimensional spectrum (``spec2d.sciimg`` - ``pattern``)

    Parameters
    ----------
    filename : :obj:`~pathlib.Path`
        Filename of theraw image from which all this comes
    spec2d : :class:`~pypeit.spec2dobj.Spec2DObj`
        The 2D spectral image class from PypeIt
    resid : :obj:`~numpy.ndarray`
        The PypeIt-produced residual image (the starting point for the fitting)
    pattern : :obj:`~numpy.ndarray`
        The constructed pattern from the sinusoidal fits (mean = 0)
    qa_dir : :obj:`~pathlib.Path`, optional
        The QA directory into which to place the optional plots.  If ``None``,
        the plots will be placed in the current working directory.
        (Default: None)
    with_obj: :obj:`bool`, optional
        Does ``resid`` contain the object model?  (Default: False)
    extra_graphics : :obj:`bool`, optional
        Produce extra graphics formats for documentation?  (Default: False)
    """
    _, axes = plt.subplots(nrows=6, figsize=(6.4, 10.67))
    tsz = 8
    interval1 = astropy.visualization.ZScaleInterval(n_samples=10000, contrast=0.5)
    interval2 = astropy.visualization.ZScaleInterval(n_samples=10000, contrast=1.0)

    # Panel #1: Draw the original science image
    vmin, vmax = interval1.get_limits(spec2d.sciimg)
    # print(f"Inferno image limits: {vmin}, {vmax}")
    axes[0].imshow(
        np.rot90(spec2d.sciimg, k=-1),
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap="inferno",
    )
    axes[0].axis("off")
    axes[0].set_title("Processed Science Image", fontsize=tsz)

    # Panel #2: Draw the sky model
    axes[1].imshow(
        np.rot90(spec2d.skymodel + spec2d.objmodel, k=-1),
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap="inferno",
    )
    axes[1].axis("off")
    axes[1].set_title(
        "Sky Model" if with_obj else "Sky Model + Object Model", fontsize=tsz
    )

    # Panel #6: Draw the cleaned science image
    axes[5].imshow(
        np.rot90(spec2d.sciimg, k=-1) - pattern,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap="inferno",
    )
    axes[5].axis("off")
    axes[5].set_title("Cleaned Science Image (sciimg - pattern)", fontsize=tsz)

    # Panel #4: Draw the pattern model
    vmin, vmax = interval2.get_limits(pattern + np.mean(resid))
    # print(f"Viridis image limits: {vmin}, {vmax}")
    axes[3].imshow(pattern + np.mean(resid), vmin=vmin, vmax=vmax, origin="lower")
    axes[3].axis("off")
    axes[3].set_title("Modeled Sinusoid Pickup Pattern", fontsize=tsz)

    # Panel #3: Draw the resid image
    axes[2].imshow(resid, vmin=vmin, vmax=vmax, origin="lower")
    axes[2].axis("off")
    if with_obj:
        axes[2].set_title("PypeIt Residual Image (sciimg - skymodel)*gpm", fontsize=tsz)
    else:
        axes[2].set_title(
            "PypeIt Residual Noise Image (sciimg - skymodel - objmodel)*gpm",
            fontsize=tsz,
        )

    # Panel #5: Draw the cleaned image
    axes[4].imshow(
        resid - pattern,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    axes[4].axis("off")
    if with_obj:
        axes[4].set_title("Residual Image minus Pattern", fontsize=tsz)
    else:
        axes[4].set_title("Residual Noise Image minus Pattern", fontsize=tsz)

    # Filename as the "super title"
    # When making plots for the documentation, obfuscate the source
    fname = "" if extra_graphics else f" for {filename.name}"
    plt.suptitle(f"Image Comparison Plots{fname}", fontsize=tsz + 2)

    # Add a note about the file creator
    plt.figtext(
        0.005,
        0.005,
        "Created by obstools.scrub_deveny_pickup on "
        f"{datetime.datetime.now().strftime('%Y-%m-%d')}",
        fontsize=tsz - 2,
    )

    plt.tight_layout()
    save_dir = pathlib.Path(".").resolve() if qa_dir is None else qa_dir
    extns = ["pdf", "svg", "png"] if extra_graphics else ["pdf"]
    for extn in extns:
        plt.savefig(save_dir / f"{filename.stem}_scrubber_image_comparisons.{extn}")
    plt.close()


# Utility Functions (Alphabetical) ===========================================#
def make_sine_label(popt: np.ndarray, infodict: dict, mesg: str, ier: int) -> str:
    """Make a sensible label for the sinusoid example plots

    Parameters
    ----------
    popt : :obj:`~numpy.ndarray`
        The array of fit coefficients from :func:`scipy.optimize.curve_fit`
    infodict : :obj:`dict`
        Dictionary containing fit information from :func:`scipy.optimize.curve_fit`
    mesg : :obj:`str`
        Message about completion from :func:`scipy.optimize.curve_fit`
    ier : :obj:`int`
        Integer describing fir completion from :func:`scipy.optimize.curve_fit`

    Returns
    -------
    :obj:`str`
        The legend label thusly constructed
    """
    return (
        rf"A = {popt[0]:.2g}  $\lambda$ = {popt[1]:.1f}  $\phi$ = {popt[2]:.2g}  "
        f"$y_0$ = {popt[3]:.2f}  lin = {popt[4]:.2g}  quad = {popt[5]:.2g}  "
        f"| {ier} | rms = {np.sqrt(np.mean(infodict['fvec']**2)):.2f} | {mesg}"
    )


def pixper_tofrom_hz(val: np.ndarray) -> np.ndarray:
    """Convert to/from pixel period and Hertz

    _extended_summary_

    Parameters
    ----------
    val : :obj:`~numpy.ndarray`
        Input value(s) to convert

    Returns
    -------
    :obj:`~numpy.ndarray`
        Converted output(s)
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    return 1.0 / (PIX_DWELL * val)


# Command Line Script Infrastructure (borrowed from PypeIt) ==================#
class ScrubDevenyPickup(utils.ScriptBase):
    """Script class for ``scrub_deveny_pickup`` tool

    Script structure borrowed from :class:`pypeit.scripts.scriptbase.ScriptBase`.
    """

    @classmethod
    def get_parser(cls, width=None):
        """Construct the command-line argument parser.

        Parameters
        ----------
        description : :obj:`str`, optional
            A short description of the purpose of the script.
        width : :obj:`int`, optional
            Restrict the width of the formatted help output to be no longer
            than this number of characters, if possible given the help
            formatter.  If None, the width is the same as the terminal
            width.
        formatter : :obj:`~argparse.HelpFormatter`
            Class used to format the help output.

        Returns
        -------
        :obj:`~argparse.ArgumentParser`
            Command-line interpreter.
        """

        parser = super().get_parser(
            description="Clean RF pickup noise from DeVeny raw frames", width=width
        )
        parser.add_argument("file", nargs="+", type=str, help="File(s) to clean")
        parser.add_argument(
            "--proc_dir",
            type=str,
            default="current working directory",
            help="Path to the directory above that which contains the .pypeit "
            "file used to process `file` (i.e., the directory above "
            "ldt_deveny_?) -- use only if `-r` was used in the call to "
            "`pypeit_setup`.",
        )
        parser.add_argument(
            "--overwrite_raw",
            action="store_true",
            help="Overwrite the raw file rather than create a new file with the '_scrub' suffix",
        )
        parser.add_argument(
            "-d",
            "--diagnostics",
            action="store_true",
            help="Output additional information and plots during the analysis for "
            "debugging purposes",  # argparse.SUPPRESS
        )
        parser.add_argument(
            "-n",
            "--no_refit",
            action="store_true",
            help="Force no refit of 'bad' RMS values",  # argparse.SUPPRESS
        )
        # Produce multiple graphics outputs for the documentation -- HIDDEN
        parser.add_argument("-g", action="store_true", help=argparse.SUPPRESS)
        # Feed the pattern image back into the FFT algorithm -- HIDDEN
        parser.add_argument("-f", action="store_true", help=argparse.SUPPRESS)
        return parser

    @staticmethod
    def main(args):
        """Main Driver

        Simple function that takes the input file list and calls the cleaning
        function for each one.
        """
        # Giddy up!
        for file in args.file:
            iterative_pypeit_clean(
                pathlib.Path(file).resolve(),
                proc_dir=pathlib.Path(args.proc_dir),
                overwrite_raw=args.overwrite_raw,
                diagnostics=args.diagnostics,
                no_refit=args.no_refit,
                extra_graphics=args.g,
                rerun_fft=args.f,
            )

            # Deprecated original method
            # clean_pickup(
            #     pathlib.Path(file),
            #     use_hann=use_hann,
            #     sine_plot=not no_plots,
            #     fft_plot=not no_plots,
            # )

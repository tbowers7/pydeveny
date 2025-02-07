# -*- coding: utf-8 -*-
#
#  This file is part of LDTObserverTools.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 01-Feb-2021
#  Modified: 08-Jan-2024
#
#  @author: tbowers

"""DeVeny Collimator Focus Calculator Module

LDTObserverTools contains python ports of various LDT Observer Tools

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
https://lowell.edu

This file contains the dfocus routine for computing the required collimator
focus for the DeVeny Spectrograph based on a focus sequence completed by the
DeVeny LOUI.

.. include:: ../include/links.rst
"""

# Built-In Libraries
import argparse
import os
import pathlib
import sys
import warnings

# 3rd-Party Libraries
import astropy.io.fits
import astropy.nddata
import ccdproc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from tqdm import tqdm

# Local Libraries
from obstools import deveny_grangle
from obstools import utils


# User-facing Function =======================================================#
def dfocus(
    path: pathlib.Path,
    flog: str = "last",
    thresh: float = 100.0,
    launch_preview: bool = True,
    docfig: bool = False,
):
    """Find the optimal DeVeny collimator focus value

    This is the user-facing ``dfocus`` function that calls all of the various
    following subroutines.  This function operates identically to the original
    IDL routine ``dfocus.pro``, but with the additional options of debugging and
    whether to launch Preview.app to show the PDF plots generated.

    Parameters
    ----------
    path : :obj:`~pathlib.Path`
        The path in which to find the ``deveny_focus.*`` files.
    flog : :obj:`str`, optional
        Focus log to process.  If unspecified, process the last sequence in
        the directory.  (Default: 'last')
        flog must be of form: ``deveny_focus.YYYYMMDD.HHMMSS``
    thresh : :obj:`float`, optional
        Line intensity threshold above background for detection (Default: 100.0)
    launch_preview : :obj:`bool`, optional
        Display the plots by launching Preview  (Default: True)
    docfig : :obj:`bool`, optional
        Make example figures for online documentation?  (Default: False)
    """
    # Make a pretty title for the output of the routine
    n_cols = (os.get_terminal_size()).columns
    print("=" * n_cols)
    print("  DeVeny Collimator Focus Calculator")

    try:
        # Get the Image File Collection for this focus run and initialize a
        #    dictionary to hold information from the headers
        focus_dict = parse_focus_headers(focus_icl := parse_focus_log(path, flog))
    except utils.ObstoolsError as err:
        # If errors, print error message and exit
        print(f"\n** {err} **\n")
        sys.exit(1)

    # Process the middle image to get line centers, arrays, trace
    mid_fn = focus_dict["mid_file"]
    print(f"\n Processing center focus image {mid_fn}...")
    mid_ccd = astropy.nddata.CCDData.read(mid_fn)
    try:
        mid_2dspec = ccdproc.trim_image(mid_ccd, mid_ccd.header["TRIMSEC"]).data
        mid_trace = centered_trace(mid_2dspec.data)
        mid_1dspec = extract_spectrum(mid_2dspec, mid_trace, window=11, thresh=thresh)
        # Find the lines in the extracted spectrum
        centers, _ = find_lines(mid_1dspec, thresh=thresh)
    except utils.ObstoolsError as err:
        # If errors, print error message and exit
        print(f"\n** {err} **\n")
        sys.exit(1)

    # Run through files, showing a progress bar
    print("\n Processing arc images...")
    prog_bar = tqdm(
        total=len(focus_icl.files), unit="frame", unit_scale=False, colour="yellow"
    )

    line_width_array = []
    # Loop over the CCDs
    for ccd in focus_icl.ccds():
        # Trim and extract the spectrum
        spec2d = ccdproc.trim_image(ccd, ccd.header["TRIMSEC"]).data
        spec1d = extract_spectrum(spec2d, mid_trace, window=11)

        # Find FWHM of lines:
        these_centers, fwhm = find_lines(spec1d, thresh=thresh, verbose=False)

        # Empirical shifts in line location due to off-axis paraboloid
        #  collimator mirror
        line_dx = -4.0 * (ccd.header["COLLFOC"] - mid_ccd.header["COLLFOC"])

        # Keep only the lines from `these_centers` that match the
        #  reference image
        line_widths = []
        for cen in centers:
            # Find line within 3 pix of the (adjusted) reference line
            idx = np.where(np.absolute((cen + line_dx) - these_centers) < 3.0)[0]
            # If there's something there wider than 2 pix, use it... else NaN
            width = fwhm[idx][0] if len(idx) else np.nan
            line_widths.append(width if width > 2.0 else np.nan)

        # Append these linewidths to the larger array for processing
        line_width_array.append(np.asarray(line_widths))
        prog_bar.update(1)

    # Close the progress bar, end of loop
    prog_bar.close()
    line_width_array = np.asarray(line_width_array)

    print(
        f"\n  Median value of all linewidths: {np.nanmedian(line_width_array):.2f} pix"
    )

    # Fit the focus curve:
    min_focus_index, optimal_focus_index, min_linewidths, fit_pars = fit_focus_curves(
        line_width_array, fnom=focus_dict["nominal"]
    )

    # Convert the returned indices into actual COLLFOC values, find medians
    print("=" * n_cols)
    min_focus_values = min_focus_index * focus_dict["delta"] + focus_dict["start"]
    optimal_focus_values = (
        optimal_focus_index * focus_dict["delta"] + focus_dict["start"]
    )
    # med_min_focus = np.real(np.nanmedian(min_focus_values))
    med_opt_focus = np.real(np.nanmedian(optimal_focus_values))

    if focus_dict["binning"] != "1x1":
        print(f"*** CCD is operating in binning {focus_dict['binning']} (col x row)")
    print(f"*** Recommended (Median) Optimal Focus Position: {med_opt_focus:.2f} mm")
    print(f"*** Note: Current Mount Temperature is: {focus_dict['mnttemp']:.1f}ºC")

    # =========================================================================#
    # Make the multipage PDF plot
    with PdfPages(pdf_fn := path / f"pyfocus.{focus_dict['id']}.pdf") as pdf:
        #  The plot shown in the IDL0 window: Plot of the found lines
        plot_lines(
            mid_1dspec,
            thresh=thresh,
            do_plot=True,
            focus_dict=focus_dict,
            pdf=pdf,
            verbose=False,
            path=path,
            docfig=docfig,
        )

        # The plot shown in the IDL2 window: Plot of best-fit fwid vs centers
        plot_optimal_focus(
            focus_dict,
            centers,
            optimal_focus_values,
            med_opt_focus,
            pdf=pdf,
            path=path,
            docfig=docfig,
        )

        # The plot shown in the IDL1 window: Focus curves for each identified line
        plot_focus_curves(
            centers,
            line_width_array,
            min_focus_values,
            optimal_focus_values,
            min_linewidths,
            fit_pars,
            focus_dict["delta"],
            focus_dict["start"],
            fnom=focus_dict["nominal"],
            pdf=pdf,
            path=path,
            docfig=docfig,
        )

    # Print the location of the plots
    print(f"\n  Plots have been saved to: {pdf_fn.name}\n")

    # Try to open with Apple's Preview App... if can't, oh well.
    if launch_preview:
        try:
            os.system(f"/usr/bin/open -a Preview {pdf_fn}")
        except OSError as err:
            print(f"Cannot open Preview.app\n{err}")


# Helper Functions (Chronological) ===========================================#
def parse_focus_log(path: pathlib.Path, flog: str) -> ccdproc.ImageFileCollection:
    """Parse the focus log file produced by the DeVeny LOUI

    The DeVeny focus log file consists of filename, collimator focus, and other
    relevant information::

        :  Image File Name  ColFoc    Grating  GrTilt  SltWth     Filter    LampCal  MntTmp
        20230613.0026.fits    7.50   600/4900   27.04    1.20  Clear (C)   Cd,Ar,Hg    9.10
        20230613.0027.fits    8.00   600/4900   27.04    1.20  Clear (C)   Cd,Ar,Hg    9.10
        20230613.0028.fits    8.50   600/4900   27.04    1.20  Clear (C)   Cd,Ar,Hg    9.10
        20230613.0029.fits    9.00   600/4900   27.04    1.20  Clear (C)   Cd,Ar,Hg    9.10
        20230613.0030.fits    9.50   600/4900   27.04    1.20  Clear (C)   Cd,Ar,Hg    9.10
        20230613.0031.fits   10.00   600/4900   27.04    1.20  Clear (C)   Cd,Ar,Hg    9.10
        20230613.0032.fits   10.50   600/4900   27.04    1.20  Clear (C)   Cd,Ar,Hg    9.10

    This function parses out the filenames of the focus images for this run,
    largely discarding the remaining information in the focus log file.

    Parameters
    ----------
    path : :obj:`~pathlib.Path`
        The path to the current working directory
    flog : :obj:`str`
        Identifier for the focus log to be processed

    Returns
    -------
    :obj:`~ccdproc.ImageFileCollection`
        The Image File Collection containing the information about focus images
        requested
    """
    # Just to be sure...
    path = path.resolve()

    # Get the correct flog
    if flog.lower() == "last":
        focfiles = sorted(path.glob("deveny_focus*"))
        try:
            flog = pathlib.Path(focfiles[-1])
        except IndexError as err:
            # In the event of no focus files, raise exception
            raise utils.ObstoolsError(
                "No successful focus run completed in this directory"
            ) from err
    else:
        flog = path / flog

    if not flog.is_file():
        raise utils.ObstoolsError("Specified focus run not in this directory")

    files = []
    with open(flog, "r", encoding="utf8") as file_object:
        # Discard file header
        file_object.readline()
        # Read in the remainder of the file, grabbing just the filenames
        for line in file_object:
            files.append(path.parent / line.strip().split()[0])

    # Return the Image File Collection
    return ccdproc.ImageFileCollection(filenames=files)


def parse_focus_headers(focus_icl: ccdproc.ImageFileCollection) -> dict:
    """Parse focus headers values into a dictionary

    Create a dictionary of values (mainly from the header) that can be used by
    subsequent routines.

    Parameters
    ----------
    focus_icl : :obj:`~ccdproc.ImageFileCollection`
        The Image File Collection containing the focus files for this run

    Returns
    -------
    :obj:`dict`
        Dictionary of the various needed quantities
    """
    # Pull the spectrograph setup from the first focus file:
    row = focus_icl.summary[0]
    slitasec = row["slitasec"]
    grating = row["grating"]
    grangle = row["grangle"]
    lampcal = row["lampcal"]
    mnttemp = row["mnttemp"]
    binning = row["ccdsum"].replace(" ", "x")

    # Compute the nominal line width
    nominal_lw = 2.94 * slitasec * deveny_grangle.deveny_amag(grangle)

    # Pull the collimator focus values from the first and last files
    focus_0 = row["collfoc"]
    focus_1 = focus_icl.summary["collfoc"][-1]
    # Find the delta between focus values
    delta_focus = (focus_1 - focus_0) / (len(focus_icl.files) - 1)
    if delta_focus == 0:
        raise utils.ObstoolsError("No change in focus over this set of images")

    # Examine the middle image
    mid_file = focus_icl.files[len(focus_icl.files) // 2]

    find_lines_title = (
        f"{mid_file}   Grating: {grating}   GRANGLE: "
        + f"{grangle:.2f}   Lamps: {lampcal}"
    )

    optimal_focus_title = (
        f"Grating: {grating}    Slit width: {slitasec:.2f} arcsec"
        + f"    Binning: {binning}    Nominal line width: "
        + f"{nominal_lw:.2f} pixels"
    )

    return {
        "mid_file": mid_file,
        "nominal": nominal_lw,
        "start": focus_0,
        "end": focus_1,
        "delta": delta_focus,
        "plot_title": find_lines_title,
        "opt_title": optimal_focus_title,
        "mnttemp": mnttemp,
        "binning": binning,
    }


def centered_trace(spec2d: np.ndarray) -> np.ndarray:
    """Construct a simple trace down the middle of the image

    _extended_summary_

    Parameters
    ----------
    spec2d : :obj:`~numpy.ndarray`
        The 2D image for which to create the trace

    Returns
    -------
    :obj:`~numpy.ndarray`
        The desired trace
    """
    # Build the trace for spectrum extraction
    n_y, n_x = spec2d.shape
    return np.full(n_x, n_y / 2, dtype=float).reshape((1, n_x))


def extract_spectrum(
    spectrum: np.ndarray,
    traces: np.ndarray,
    window: int,
    thresh: float = 20.0,
    verbose: bool = True,
) -> np.ndarray:
    """Object spectral extraction routine

    Extract spectra by averaging over the specified window and background
    subtract

    Parameters
    ----------
    spectrum : :obj:`~numpy.ndarray`
        The trimmed spectral image
    traces : :obj:`~numpy.ndarray`
        The trace(s) along which to extract spectra
    window : :obj:`int`
        Window over which to average the spectrum
    thresh : :obj:`float`, optional
        Threshold above which to indentify lines [Default: 20 DN above bkgd]
    verbose : :obj:`bool`, optional
        Produce verbose output?  [Default: False]

    Returns
    -------
    :obj:`~numpy.ndarray`
        Background-subtracted extracted spectrum
    """
    # Spec out the shape, and create an empty array to hold the output spectra
    norders, n_x = traces.shape
    if norders != 1:
        raise utils.ObstoolsError("Cannot deal with multiple traces")
    extracted_spectrum = np.empty(n_x, dtype=float)

    # Set extraction window size
    half_window = int(window) // 2

    # Because of python indexing, we need to "+1" the upper limit in order
    #   to get the full wsize elements for the average
    trace = traces[0, :].astype(int)
    # Do this as a loop since a trace may not be a simple row through the image
    for i in range(n_x):
        extracted_spectrum[i] = np.average(
            spectrum[trace[i] - half_window : trace[i] + half_window + 1, i]
        )

    # Background subtract

    # Find background from median value of the image:
    bkgd = np.median(extracted_spectrum)
    if verbose:
        print(
            f"  Background level: {bkgd:.1f}"
            + f"   Detection threshold level: {bkgd+thresh:.1f}"
        )

    # ??? Where do we actually subtract the background?????????

    return extracted_spectrum


def find_lines(
    spectrum: np.ndarray,
    thresh: float = 20.0,
    minsep: int = 11,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Automatically find and centroid lines in a 1-row image

    Uses :func:`scipy.signal.find_peaks` for this task

    Parameters
    ----------
    image : :obj:`~numpy.ndarray`
        Extracted 1D spectrum
    thresh : :obj:`float`, optional
        Threshold above which to indentify lines [Default: 20 DN above bkgd]
    minsep : :obj:`int`, optional
        Minimum line separation for identification [Default: 11 pixels]
    verbose : :obj:`bool`, optional
        Produce verbose output?  [Default: False]

    Returns
    -------
    centers : :obj:`~numpy.ndarray`
        Line centers (pixel #)
    fwhm : :obj:`~numpy.ndarray`
        The computed FWHM for each peak
    """
    # Use scipy to find peaks & widths -- no more janky IDL-based junk
    centers, _ = scipy.signal.find_peaks(spectrum, height=thresh, distance=minsep)
    fwhm = (scipy.signal.peak_widths(spectrum, centers))[0]
    # Print a statement, if desired
    if verbose:
        print(f" Number of lines found: {len(centers)}")
    return (centers, fwhm)


def fit_focus_curves(
    fwhm: np.ndarray, fnom: float = 2.7, norder: int = 2, debug: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit line / star focus curves

    [extended_summary]

    Parameters
    ----------
    fwhm : :obj:`~numpy.ndarray`
        Array of FWHM for all lines as a function of COLLFOC
    fnom : :obj:`float`, optional
        Nominal FHWM of an in-focus line. (Default: 2.7)
    norder : :obj:`int`, optional
        Polynomial order of the focus fit (Default: 2 = Quadratic)
    debug : :obj:`bool`, optional
        Print debug statements  (Default: False)

    Returns
    -------
    min_cf_idx_value : :obj:`~numpy.ndarray`
        Best fit focus values
    optimal_cf_idx_value : :obj:`~numpy.ndarray`
        Best fit linewidths
    min_linewidth : :obj:`~numpy.ndarray`
        Minimum linewidths
    foc_fits : :obj:`~numpy.ndarray`
        Fit parameters (for plotting)
    """
    # Warning Filter -- Polyfit RankWarning, don't wanna hear about it
    warnings.simplefilter("ignore", np.RankWarning)

    # Create the various arrays / lists needed
    n_focus, n_centers = fwhm.shape
    min_linewidth = []
    min_cf_idx_value = []
    optimal_cf_idx_value = []
    foc_fits = []

    # Fitting arrays (these are indices for collimator focus)
    cf_idx_coarse = np.arange(n_focus, dtype=float)
    cf_idx_fine = np.arange(0, n_focus - 1 + 0.1, 0.1, dtype=float)

    # Loop through lines to find the best focus for each one
    for i in range(n_centers):
        # Data are the FWHM for this line at different COLLFOC
        fwhms_of_this_line = fwhm[:, i]

        # Find unphysically large or small FWHM (or NaN) -- set to np.nan
        bad_idx = np.where(
            np.logical_or(fwhms_of_this_line < 1.0, fwhms_of_this_line > 15.0)
        )
        fwhms_of_this_line[bad_idx] = np.nan
        fwhms_of_this_line[np.isnan(fwhms_of_this_line)] = np.nan

        # If more than 3 of the FHWM are bad for this line, skip and go on
        if len(bad_idx) > 3:
            # Add values to the lists for proper indexing
            for flst in [min_linewidth, min_cf_idx_value, optimal_cf_idx_value]:
                flst.append(None)
            continue

        # Do a polynomial fit (norder) to the FWHM vs COLLFOC index
        # fit = np.polyfit(cf_idx_coarse, fwhms_of_this_line, norder)
        fit = utils.good_poly(cf_idx_coarse, fwhms_of_this_line, norder, 2.0)
        foc_fits.append(fit)
        if debug:
            print(f"In fit_focus_curves(): fit = {fit}")

        # If good_poly() returns zeros, deal with it accordingly
        if all(value == 0 for value in fit):
            min_cf_idx_value.append(np.nan)
            min_linewidth.append(np.nan)
            optimal_cf_idx_value.append(np.nan)
            continue

        # Use the fine grid to evaluate the curve miniumum
        focus_curve = np.polyval(fit, cf_idx_fine)  # fitfine
        min_cf_idx_value.append(cf_idx_fine[np.argmin(focus_curve)])  # focus
        min_linewidth.append(np.min(focus_curve))  # minfoc

        # Compute the nominal focus position as the larger of the two points
        #  where the polymonial function crosses fnom
        coeffs = [fit[0], fit[1], fit[2] - fnom]
        if debug:
            print(f"Roots: {np.roots(coeffs)}")
        optimal_cf_idx_value.append(np.max(np.real(np.roots(coeffs))))

    # After looping, return the items as numpy arrays
    return (
        np.asarray(min_cf_idx_value),
        np.asarray(optimal_cf_idx_value),
        np.asarray(min_linewidth),
        np.asarray(foc_fits),
    )


# Plotting Routines ==========================================================#
def plot_lines(
    spectrum: np.ndarray,
    thresh: float = 20.0,
    minsep: int = 11,
    verbose: bool = True,
    do_plot: bool = False,
    focus_dict: dict = None,
    pdf: PdfPages = None,
    docfig: bool = False,
    path: pathlib.Path = None,
):
    """Automatically find and centroid lines in a 1-row image

    Uses :func:`scipy.signal.find_peaks` for this task

    Parameters
    ----------
    image : :obj:`~numpy.ndarray`
        Extracted spectrum
    thresh : :obj:`float`, optional
        Threshold above which to indentify lines [Default: 20 DN above bkgd]
    minsep : :obj:`int`, optional
        Minimum line separation for identification [Default: 11 pixels]
    verbose : :obj:`bool`, optional
        Produce verbose output?  [Default: False]
    do_plot : :obj:`bool`, optional
        Create a plot on the provided axes?  [Default: False]
    focus_dict : :obj:`dict`, optional
        Dictionary containing needed variables for plot  [Default: None]
    pdf : :obj:`~matplotlib.backends.backend_pdf.PdfPages`, optional
        The PDF object into which to place plots (Default: None)
    docfig : :obj:`bool`, optional
        Are these figures for the documentation pages (Default: False)
    path: : :obj:`~pathlib.Path`, optional
        Path into which to save the documentation figures (Default: None)
    """
    centers, _ = find_lines(spectrum, thresh, minsep, verbose)

    # Produce a plot for posterity, if directed
    if do_plot:
        # Set up the plot environment
        _, axis = plt.subplots()
        tsz = 8

        # Plot the spectrum, mark the peaks, and label them
        axis.plot(np.arange(len(spectrum)), spectrum)
        axis.set_ylim(0, (yrange := 1.2 * max(spectrum)))
        axis.plot(centers, spectrum[centers.astype(int)] + 0.02 * yrange, "k*")
        for cen in centers:
            axis.text(
                cen,
                spectrum[int(np.round(cen))] + 0.03 * yrange,
                f"{cen:.0f}",
                fontsize=tsz,
            )

        # Make pretty & Save
        axis.set_title(focus_dict["plot_title"], fontsize=tsz * 1.2)
        axis.set_xlabel("CCD Column", fontsize=tsz)
        axis.set_ylabel("I (DN)", fontsize=tsz)
        axis.set_xlim(0, len(spectrum) + 2)
        axis.tick_params("both", labelsize=tsz, direction="in", top=True, right=True)
        plt.tight_layout()
        if pdf is None:
            plt.show()
        else:
            pdf.savefig()
            if docfig:
                for ext in ["png", "pdf", "svg"]:
                    plt.savefig(path / f"pyfocus.page1_example.{ext}")
        plt.close()


def plot_optimal_focus(
    focus: dict,
    centers: np.ndarray,
    optimal_focus_values: np.ndarray,
    med_opt_focus: float,
    debug: bool = False,
    pdf: PdfPages = None,
    docfig: bool = False,
    path: pathlib.Path = None,
):
    """Make the Optimal Focus Plot (IDL2 Window)

    [extended_summary]

    Parameters
    ----------
    focus : :obj:`dict`
        Dictionary of the various focus-related quantities
    centers : :obj:`~numpy.ndarray`
        Array of the centers of each line
    optimal_focus_values : :obj:`~numpy.ndarray`
        Array of the optimal focus values for each line
    med_opt_focus : :obj:`float`
        Median optimal focus value
    debug : :obj:`bool`, optional
        Print debug statements  (Default: False)
    pdf : :obj:`~matplotlib.backends.backend_pdf.PdfPages`, optional
        The PDF object into which to place plots (Default: None)
    docfig : :obj:`bool`, optional
        Are these figures for the documentation pages (Default: False)
    path: : :obj:`~pathlib.Path`, optional
        Path into which to save the documentation figures (Default: None)
    """
    if debug:
        print("=" * 20)
        print(centers.dtype, optimal_focus_values.dtype, type(med_opt_focus))
    _, axis = plt.subplots()
    tsz = 8
    axis.plot(centers, optimal_focus_values, ".")
    axis.set_xlim(0, 2050)
    axis.set_ylim(focus["start"] - focus["delta"], focus["end"] + focus["delta"])
    axis.set_title(
        "Optimal focus position vs. line position, median =  "
        + f"{med_opt_focus:.2f} mm  "
        + f"(Mount Temp: {focus['mnttemp']:.1f}$^\\circ$C)",
        fontsize=tsz * 1.2,
    )
    axis.hlines(
        med_opt_focus,
        0,
        1,
        transform=axis.get_yaxis_transform(),
        color="magenta",
        ls="--",
    )
    axis.set_xlabel(f"CCD Column\n{focus['opt_title']}", fontsize=tsz)
    axis.set_ylabel("Optimal Focus (mm)", fontsize=tsz)
    axis.grid(which="both", color="#c0c0c0", linestyle="-", linewidth=0.5)

    axis.tick_params("both", labelsize=tsz, direction="in", top=True, right=True)
    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig()
        if docfig:
            for ext in ["png", "pdf", "svg"]:
                plt.savefig(path / f"pyfocus.page2_example.{ext}")
    plt.close()


def plot_focus_curves(
    centers: np.ndarray,
    line_width_array: np.ndarray,
    min_focus_values: np.ndarray,
    optimal_focus_values: np.ndarray,
    min_linewidths: np.ndarray,
    fit_pars: np.ndarray,
    delta_focus: float,
    focus_0: float,
    fnom: float = 2.7,
    pdf: PdfPages = None,
    docfig: bool = False,
    path: pathlib.Path = None,
):
    """Make the big plot of all the focus curves (IDL1 Window)

    [extended_summary]

    Parameters
    ----------
    centers : :obj:`~numpy.ndarray`
        List of line centers from find_lines()
    line_width_array : :obj:`~numpy.ndarray`
        Array of line widths from each COLLFOC setting for each line
    min_focus_values : :obj:`~numpy.ndarray`
        List of the minimum focus values found from the polynomial fit
    optimal_focus_values : :obj:`~numpy.ndarray`
        List of the optimal focus values found from the polynomial fit
    min_linewidths : :obj:`~numpy.ndarray`
        List of the minumum linewidths found from the fitting
    fit_pars : :obj:`~numpy.ndarray`
        Array of the polynomial fit parameters for each line
    delta_focus : :obj:`float`
        Spacing between COLLFOC settings
    focus_0 : :obj:`float`
        Lower end of the COLLFOC range
    fnom : :obj:`float`, optional
        Nominal (optimal) linewidth  (Default: 2.7)
    pdf : :obj:`~matplotlib.backends.backend_pdf.PdfPages`, optional
        The PDF object into which to place plots (Default: None)
    docfig : :obj:`bool`, optional
        Are these figures for the documentation pages (Default: False)
    path: : :obj:`~pathlib.Path`, optional
        Path into which to save the documentation figures (Default: None)
    """
    # Warning Filter -- Matplotlib doesn't like going from masked --> NaN
    warnings.simplefilter("ignore", UserWarning)

    # Set up variables
    n_foc, n_c = line_width_array.shape
    focus_idx = np.arange(n_foc)
    focus_x = focus_idx * delta_focus + focus_0

    # Set the plotting array
    ncols = 6
    nrows = n_c // ncols + 1
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8.5, 11))
    tsz = 6  # type size

    for i, axis in enumerate(axes.flatten()):
        if i < n_c:
            # Plot the points and the polynomial fit
            axis.plot(focus_x, line_width_array[:, i], "kD", fillstyle="none")
            axis.plot(focus_x, np.polyval(fit_pars[i, :], focus_idx), "g-")

            # Plot vertical lines to indicate minimum and optimal focus
            axis.vlines(min_focus_values[i], 0, min_linewidths[i], color="r", ls="-")
            axis.vlines(optimal_focus_values[i], 0, fnom, color="b", ls="-")

            # Plot parameters to make pretty
            # ax.set_ylim(0, np.nanmax(line_width_array[:,i]))
            axis.set_ylim(0, 7.9)
            axis.set_xlim(np.min(focus_x) - delta_focus, np.max(focus_x) + delta_focus)
            axis.set_xlabel("Collimator Position (mm)", fontsize=tsz)
            axis.set_ylabel("FWHM (pix)", fontsize=tsz)
            axis.set_title(
                f"LC: {centers[i]:.0f}  Fnom: {fnom:.2f} pixels", fontsize=tsz
            )
            axis.tick_params(
                "both", labelsize=tsz, direction="in", top=True, right=True
            )
            axis.grid(which="both", color="#c0c0c0", linestyle="-", linewidth=0.5)
        else:
            # Clear any extra positions if there aren't enough lines
            fig.delaxes(axis)

    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig()
        if docfig:
            for ext in ["png", "pdf", "svg"]:
                plt.savefig(path / f"pyfocus.page3_example.{ext}")

    plt.close()


# Extra Routines =============================================================#
def find_lines_in_spectrum(
    filename: str | pathlib.Path, thresh: float = 100.0
) -> np.ndarray:
    """Find the line centers in a spectrum

    This function is not directly utilized in ``dfocus``, but rather is included
    as a wrapper for several functions that can be used by other programs.

    Given the filename of an arc-lamp spectrum, this function returns a list
    of the line centers found in the image.

    Parameters
    ----------
    filename : :obj:`str` or :obj:`~pathlib.Path`
        Filename of the arc frame to find lines in
    thresh : :obj:`float`, optional
        Line intensity threshold above background for detection [Default: 100]

    Returns
    -------
    :obj:`~numpy.ndarray`
        List of line centers found in the image
    """
    # Get the trimmed image
    ccd = astropy.nddata.CCDData.read(filename)
    spectrum = ccdproc.trim_image(ccd, ccd.header["TRIMSEC"]).data

    # Build the trace for spectrum extraction
    n_y, n_x = spectrum.shape
    traces = np.full(n_x, n_y / 2, dtype=float).reshape((1, n_x))
    spectra = extract_spectrum(spectrum, traces, window=11)

    # Find the lines!
    centers, _ = find_lines(spectra, thresh=thresh)

    return centers


# Command Line Script Infrastructure (borrowed from PypeIt) ==================#
class DFocus(utils.ScriptBase):
    """Script class for ``dfocus`` tool

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
            description="DeVeny Collimator Focus Calculator", width=width
        )
        parser.add_argument(
            "--flog",
            action="store",
            type=str,
            help="focus log to use",
            default="last",
        )
        parser.add_argument(
            "--thresh",
            action="store",
            type=float,
            help="threshold for line detection",
            default=100.0,
        )
        parser.add_argument(
            "--nodisplay",
            action="store_true",
            help="DO NOT launch Preview.app to display plots",
        )
        # Produce multiple graphics outputs for the documentation -- HIDDEN
        parser.add_argument("-g", action="store_true", help=argparse.SUPPRESS)
        return parser

    @staticmethod
    def main(args):
        """Main Driver

        Simple function that calls the primary function.
        """
        # Giddy Up!
        dfocus(
            pathlib.Path(".").resolve(),
            flog=args.flog,
            thresh=args.thresh,
            launch_preview=not args.nodisplay,
            docfig=args.g,
        )

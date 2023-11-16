# -*- coding: utf-8 -*-
#
#  This file is part of LDTObserverTools.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 01-Feb-2021
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
import pathlib
import shutil
import subprocess
import sys
import warnings

# 3rd-Party Libraries
import astropy.io.fits
import astropy.modeling
import astropy.nddata
import astropy.stats
import ccdproc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from tqdm import tqdm

# Local Libraries
from obstools import deveny_grangle
from obstools import utils


class DevenyFocus:
    """DeVeny Focus Measuring Class

    Find the optimal DeVeny collimator focus value

    This class contains all of the various subroutines needed to compute the
    focus value from a sequence of frames at different collimator position.
    Calling the command-line script ``dfocus`` operates identically to the
    original IDL routine ``dfocus.pro``, but with additional debugging and
    display options.

    Parameters
    ----------
    path : :obj:`~pathlib.Path`
        The path in which to find the ``deveny_focus.*`` files.
    flog : :obj:`str`, optional
        Focus log to process.  If unspecified, process the last sequence in
        the directory.  (Default: 'last')
        flog musy be of form: ``deveny_focus.YYYYMMDD.HHMMSS``
    thresh : :obj:`float`, optional
        Line intensity threshold above background for detection (Default: 100.0)
    debug : :obj:`bool`, optional
        Print debug statements  (Default: False)
    launch_preview : :obj:`bool`, optional
        Display the plots by launching Preview  (Default: True)
    docfig : :obj:`bool`, optional
        Make example figures for online documentation?  (Default: False)
    """

    # Define class attributes so they exist for @classmethod methods
    docfig = False
    focus_dict = None
    pdf = None
    thresh = None

    def __init__(
        self,
        path: pathlib.Path,
        flog: str = "last",
        thresh: float = 100.0,
        debug: bool = False,
        launch_preview: bool = True,
        docfig: bool = False,
    ):
        # Store inputs as attributes
        self.path = path
        self.flog = flog
        self.thresh = thresh
        self.debug = debug
        self.launch_preview = launch_preview
        self.docfig = docfig

        self.focus_dict = None

        # Make a pretty title for the output of the routine
        self.n_cols = (shutil.get_terminal_size()).columns
        print("=" * self.n_cols)
        print("  DeVeny Collimator Focus Calculator")

    def run(self):
        """Run the focus routine

        This is the main execution method of the class, following the steps
        from the original IDL routine.
        """

        # Initialize a dictionary to hold lots of variables; error check
        self.focus_dict = self.initialize_focus_values()
        if len(self.focus_dict["focus_positions"]) < 3:
            print("\n** No successful focus run completed in this directory. **\n")
            sys.exit(1)

        # Process the middle image to get line centers, arrays, trace
        line_centers, trace, mid_collfoc, middle_spectrum = self.process_middle_image()

        # Loop through files, showing a progress bar
        print("\n Processing arc images...")
        prog_bar = tqdm(
            total=len(self.focus_dict["icl"].files),
            unit="frame",
            unit_scale=False,
            colour="yellow",
        )

        line_width_array = []
        for ccd in self.focus_dict["icl"].ccds():
            # Trim and extract the spectrum
            specimg = utils.trim_oscan(
                ccd, ccd.header["BIASSEC"], ccd.header["TRIMSEC"]
            )
            spec1d = extract_spectrum(specimg.data, trace, win=11)

            # Find FWHM of lines:
            these_centers, fwhm = self.find_lines(
                spec1d, thresh=self.thresh, verbose=False
            )

            # Empirical shifts in line location due to off-axis paraboloid
            #  collimator mirror
            line_dx = -4.0 * (ccd.header["COLLFOC"] - mid_collfoc)

            # Keep only the lines from `these_centers` that match the
            #  reference image
            line_widths = []
            for cen in line_centers:
                # Find line within 3 pix of the (adjusted) reference line
                idx = np.where(np.absolute((cen + line_dx) - these_centers) < 3.0)[0]
                # If there's something there wider than 2 pix, use it... else NaN
                width = fwhm[idx][0] if len(idx) else np.nan
                line_widths.append(width if width > 2.0 else np.nan)

            # Append these linewidths to the larger array for processing
            line_width_array.append(np.array(line_widths, dtype=float))
            prog_bar.update(1)

        # Close the progress bar, end of loop
        prog_bar.close()
        line_width_array = np.asarray(line_width_array)
        print(f"line_width_array: {line_width_array.shape}")

        print(
            f"\n  Median value of all linewidths: {np.nanmedian(line_width_array):.2f} pix"
        )

        # Fit the focus curve:
        (
            min_focus_values,
            optimal_focus_values,
            min_lw,
            fit_polynomials,
        ) = fit_focus_curves(
            self.focus_dict["focus_positions"],
            line_width_array,
            fnom=self.focus_dict["nominal"],
        )
        median_optimal_focus = np.real(np.nanmedian(optimal_focus_values))

        # Print some fun facts!
        print("=" * self.n_cols)
        if self.focus_dict["binning"] != "1x1":
            print(
                f"*** CCD is operating in binning {self.focus_dict['binning']} (col x row)"
            )
        print(
            f"*** Recommended (Median) Optimal Focus Position: {median_optimal_focus:.2f} mm"
        )
        print(
            f"*** Note: Current Mount Temperature is: {self.focus_dict['mnttemp']:.1f}ºC"
        )

        # =========================================================================#
        # Make the multipage PDF plot
        with PdfPages(
            pdf_fn := self.path / f"pyfocus.{self.focus_dict['id']}.pdf"
        ) as self.pdf:
            #  The plot shown in the IDL0 window: Plot of the found lines
            self.find_lines(middle_spectrum, do_plot=True, verbose=False)

            # The plot shown in the IDL2 window: Plot of best-fit fwid vs centers
            self.plot_optimal_focus(
                line_centers, optimal_focus_values, median_optimal_focus
            )

            # The plot shown in the IDL1 window: Focus curves for each identified line
            self.plot_focus_curves(
                line_centers,
                line_width_array,
                min_focus_values,
                optimal_focus_values,
                min_lw,
                fit_polynomials,
            )
        # Print the location of the plots
        print(f"\n  Plots have been saved to: {pdf_fn.name}\n")

        # Try to open with Apple's Preview App... if can't, oh well.
        if self.launch_preview:
            try:
                subprocess.call(f"/usr/bin/open -a Preview {pdf_fn}", shell=True)
            except subprocess.SubprocessError as err:
                print(f"Cannot open Preview.app\n{err}")

    # Helper Methods (Chronological) =========================================#
    def initialize_focus_values(self) -> dict:
        """Initialize a dictionary of focus values

        Create a dictionary of values (mainly from the header) that can be used by
        subsequent routines.

        Returns
        -------
        :obj:`dict`
            Dictionary of the various needed quantities
        """
        # Parse the log file to obtain file list; create ImageFileCollection
        file_list, focus_seq_id = self.parse_focus_log()
        if not (n_files := len(file_list)):
            # Escape hatch if no files found (e.g., run in the wrong directory)
            return {"focus_positions": []}

        # Extract the spectrograph setup from the first focus file:
        hdr0 = astropy.io.fits.getheader(file_list[0])

        # Compute the nominal line width (0.34"/pix plate scale)
        nominal_linewidth = (
            hdr0["SLITASEC"] / 0.34 * deveny_grangle.deveny_amag(hdr0["GRANGLE"])
        )

        # Extract the collimator focus values from the first and last files
        focus_0 = hdr0["COLLFOC"]
        focus_1 = (astropy.io.fits.getheader(file_list[-1]))["COLLFOC"]
        # Find the delta between focus values
        try:
            delta_focus = (focus_1 - focus_0) / (n_files - 1)
        except ZeroDivisionError:
            delta_focus = 0

        # Get the Path for the middle file
        mid_file = file_list[n_files // 2]

        # Return a dictionary containing the various values
        return {
            "id": focus_seq_id,
            "icl": ccdproc.ImageFileCollection(filenames=file_list),
            "mid_file": mid_file,
            "nominal": nominal_linewidth,
            "focus_positions": np.arange(
                focus_0, focus_1 + delta_focus / 2, delta_focus
            ),
            "delta": delta_focus,
            "mnttemp": hdr0["MNTTEMP"],
            "binning": hdr0["CCDSUM"].replace(" ", "x"),
            "plot_title": (
                f"{mid_file.name}   Grating: {hdr0['GRATING']}   GRANGLE: "
                + f"{hdr0['GRANGLE']:.2f}   Lamps: {hdr0['LAMPCAL']}"
            ),
            "opt_title": (
                f"Grating: {hdr0['GRATING']}    Slit width: {hdr0['SLITASEC']:.2f} arcsec"
                + f"    Binning: {hdr0['CCDSUM'].replace(' ', 'x')}    Nominal line width: "
                + f"{nominal_linewidth:.2f} pixels"
            ),
        }

    def parse_focus_log(self) -> tuple[list, str]:
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

        Returns
        -------
        file_list: :obj:`list`
            List of :obj:`~pathlib.Path` files associated with this focus run
        focus_id: :obj:`str`
            The focus ID
        """
        # Just to be sure...
        path = self.path.resolve()

        # Get the correct flog
        if self.flog.lower() == "last":
            focfiles = sorted(path.glob("deveny_focus*"))
            try:
                flog = focfiles[-1]
            except IndexError:
                # In the event of no files, return empty things
                return [], ""
        else:
            flog = path / flog

        file_list = []
        with open(flog, "r", encoding="utf8") as file_object:
            # Discard file header
            file_object.readline()
            # Read in the remainder of the file, grabbing just the filenames
            for line in file_object:
                file_list.append(path.parent / line.strip().split()[0])

        # Return the list of files, and the FocusID
        return file_list, flog.name.replace("deveny_focus.", "")

    def process_middle_image(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Process the middle focus image

        This finds the lines to be measured -- assumes the middle is closest to focus

        Returns
        -------
        line_centers, :obj:`~numpy.ndarray`
            Pixel values of the line centers
        trace, :obj:`~numpy.ndarray`
            Trace
        mid_collfoc, :obj:`float`
            Collimator focus of the middle frame
        middle_spectrum, :obj:`~numpy.ndarray`
            The spectrum from the middle frame (for later plotting)
        """
        print(f"\n Processing center focus image {self.focus_dict['mid_file']}...")
        ccd = astropy.nddata.CCDData.read(self.focus_dict["mid_file"])
        specimg = utils.trim_oscan(ccd, ccd.header["BIASSEC"], ccd.header["TRIMSEC"])

        # Build the trace for spectrum extraction -- right down the middle
        n_y, n_x = specimg.shape
        trace = np.full(n_x, n_y / 2, dtype=float).reshape((1, n_x))
        middle_spectrum = extract_spectrum(specimg.data, trace, win=11)
        if self.debug:
            print(f"Traces: {trace}")
            print(f"Middle Spectrum: {middle_spectrum}")

        # Find the lines in the extracted spectrum
        line_centers, _ = self.find_lines(middle_spectrum, thresh=self.thresh)
        if self.debug:
            print(f"Back in the main program, number of lines: {len(line_centers)}")
            print(f"Line Centers: {[f'{cent:.1f}' for cent in line_centers]}")

        return line_centers, trace, ccd.header["COLLFOC"], middle_spectrum

    def find_lines(
        self,
        image,
        thresh: float = 20.0,
        minsep: int = 11,
        verbose: bool = True,
        do_plot: bool = False,
        focus_dict: dict = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        Returns
        -------
        centers : :obj:`~numpy.ndarray`
            Line centers (pixel #)
        fwhm : :obj:`~numpy.ndarray`
            The computed FWHM for each peak
        """
        # Use instance attributes, if available, otherwise argument
        thresh = self.thresh if self.thresh is not None else thresh
        focus_dict = self.focus_dict if self.focus_dict is not None else focus_dict

        # Get size and flatten to 1D
        _, n_x = image.shape
        spec = np.ndarray.flatten(image)

        # Find background from median value of the image:
        bkgd = np.median(spec)
        if verbose:
            print(
                f"  Background level: {bkgd:.1f}"
                + f"   Detection threshold level: {bkgd+thresh:.1f}"
            )

        # Use scipy to find peaks & widths -- no more janky IDL-based junk
        centers, _ = scipy.signal.find_peaks(
            newspec := spec - bkgd, height=thresh, distance=minsep
        )
        fwhm = (scipy.signal.peak_widths(newspec, centers))[0]

        if verbose:
            print(f" Number of lines found: {len(centers)}")

        # Produce a plot for posterity, if directed
        if do_plot:
            # Set up the plot environment
            _, axis = plt.subplots()
            tsz = 8

            # Plot the spectrum, mark the peaks, and label them
            axis.plot(np.arange(len(spec)), newspec)
            axis.set_ylim(0, (yrange := 1.2 * max(newspec)))
            axis.plot(centers, newspec[centers.astype(int)] + 0.02 * yrange, "k*")
            for cen in centers:
                axis.text(
                    cen,
                    newspec[int(np.round(cen))] + 0.03 * yrange,
                    f"{cen:.0f}",
                    fontsize=tsz,
                )

            # Make pretty & Save
            axis.set_title(focus_dict["plot_title"], fontsize=tsz * 1.2)
            axis.set_xlabel("CCD Column", fontsize=tsz)
            axis.set_ylabel("I (DN)", fontsize=tsz)
            axis.set_xlim(0, n_x + 2)
            axis.tick_params(
                "both", labelsize=tsz, direction="in", top=True, right=True
            )
            plt.tight_layout()
            if self.pdf is None:
                plt.show()
            else:
                self.pdf.savefig()
                if self.docfig:
                    for ext in ["png", "pdf", "svg"]:
                        plt.savefig(self.path / f"pyfocus.page1_example.{ext}")
            plt.close()

        return centers, fwhm

    @classmethod
    def find_lines_wrap(
        cls, image, thresh: float = 20.0, verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Wrapper around :meth:`find_lines` for external call

        _extended_summary_

        Parameters
        ----------
        image : :obj:`~numpy.ndarray`
            Extracted spectrum
        thresh : :obj:`float`, optional
            Threshold above which to indentify lines [Default: 20 DN above bkgd]
        verbose : :obj:`bool`, optional
            Produce verbose output?  [Default: False]

        Returns
        -------
        centers : :obj:`~numpy.ndarray`
            Line centers (pixel #)
        fwhm : :obj:`~numpy.ndarray`
            The computed FWHM for each peak
        """
        return cls.find_lines(
            cls,
            image,
            thresh=thresh,
            minsep=11,
            verbose=verbose,
            do_plot=False,
            focus_dict=None,
        )

    # Plotting Routines ==========================================================#
    def plot_optimal_focus(
        self,
        centers: np.ndarray,
        optimal_focus_values: np.ndarray,
        med_opt_focus: float,
    ):
        """Make the Optimal Focus Plot (IDL2 Window)

        This plot shows the recommended optimal focus position.  The optimal
        focus values for each line are plotted against column number, and the
        median value is plotted.  At top and bottom are the relevant pieces of
        information from this focus run.

        Parameters
        ----------
        centers : :obj:`~numpy.ndarray`
            Array of the centers of each line
        optimal_focus_values : :obj:`~numpy.ndarray`
            Array of the optimal focus values for each line
        med_opt_focus : :obj:`float`
            Median optimal focus value
        """
        if self.debug:
            print("=" * 20)
            print(centers.dtype, optimal_focus_values.dtype, type(med_opt_focus))
        _, axis = plt.subplots()
        tsz = 8
        axis.plot(centers, optimal_focus_values, ".")
        axis.set_xlim(0, 2050)
        axis.set_ylim(
            self.focus_dict["focus_positions"][0] - self.focus_dict["delta"],
            self.focus_dict["focus_positions"][-1] + self.focus_dict["delta"],
        )
        axis.set_title(
            "Optimal focus position vs. line position, median =  "
            + f"{med_opt_focus:.2f} mm  "
            + f"(Mount Temp: {self.focus_dict['mnttemp']:.1f}$^\\circ$C)",
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
        axis.set_xlabel(f"CCD Column\n{self.focus_dict['opt_title']}", fontsize=tsz)
        axis.set_ylabel("Optimal Focus (mm)", fontsize=tsz)
        axis.grid(which="both", color="#c0c0c0", linestyle="-", linewidth=0.5)

        axis.tick_params("both", labelsize=tsz, direction="in", top=True, right=True)
        plt.tight_layout()
        if self.pdf is None:
            plt.show()
        else:
            self.pdf.savefig()
            if self.docfig:
                for ext in ["png", "pdf", "svg"]:
                    plt.savefig(self.path / f"pyfocus.page2_example.{ext}")
        plt.close()

    def plot_focus_curves(
        self,
        centers: np.ndarray,
        line_width_array: np.ndarray,
        min_focus_values: np.ndarray,
        optimal_focus_values: np.ndarray,
        min_linewidths: np.ndarray,
        fit_polynomials: np.ndarray,
    ):
        """Make the big plot of all the focus curves (IDL1 Window)

        These are the individual focus curves for each identified arc line,
        with the parabolic fit, minimum linewidth focus position, and optimal
        focus position shown.

        Parameters
        ----------
        centers : :obj:`~numpy.ndarray`
            List of line centers from find_lines()
        line_width_array : :obj:`~numpy.ndarray`
            Array of line widths from each COLLFOC setting for each line
        min_focus_values : :obj:`~numpy.ndarray`
            Array of minimum focus values per line
        optimal_focus_values : :obj:`~numpy.ndarray`
            Array of optimal focus values per line
        min_linewidths : :obj:`~numpy.ndarray`
            Array of minimum linewidths per line
        fit_polynomials : :obj:`~numpy.ndarray`
            Array of :obj:`~numpy.polynomial.Polynomial` objects per line
        """
        # Warning Filter -- Matplotlib doesn't like going from masked --> NaN
        warnings.simplefilter("ignore", UserWarning)

        # Set up variables
        _, n_lines = line_width_array.shape
        focus_x = self.focus_dict["focus_positions"]

        # Set the plotting array
        ncols = 6
        nrows = n_lines // ncols + 1
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8.5, 11))
        tsz = 6  # type size

        for i, axis in enumerate(axes.flatten()):
            if i < n_lines:
                # Plot the points and the polynomial fit
                axis.plot(focus_x, line_width_array[:, i], "kD", fillstyle="none")
                axis.plot(focus_x, fit_polynomials[i](focus_x), "g-")

                # Plot vertical lines to indicate minimum and optimal focus
                axis.vlines(
                    min_focus_values[i], 0, min_linewidths[i], color="r", ls="-"
                )
                axis.vlines(
                    optimal_focus_values[i],
                    0,
                    self.focus_dict["nominal"],
                    color="b",
                    ls="-",
                )

                # Plot parameters to make pretty
                axis.set_ylim(0, 7.9)
                axis.set_xlim(
                    np.min(focus_x) - self.focus_dict["delta"],
                    np.max(focus_x) + self.focus_dict["delta"],
                )
                axis.set_xlabel("Collimator Position (mm)", fontsize=tsz)
                axis.set_ylabel("FWHM (pix)", fontsize=tsz)
                axis.set_title(
                    f"LC: {centers[i]:.0f}  Fnom: {self.focus_dict['nominal']:.2f} pixels",
                    fontsize=tsz,
                )
                axis.tick_params(
                    "both", labelsize=tsz, direction="in", top=True, right=True
                )
                axis.grid(which="both", color="#c0c0c0", linestyle="-", linewidth=0.5)
            else:
                # Clear any extra positions if there aren't enough lines
                fig.delaxes(axis)

        plt.tight_layout()
        if self.pdf is None:
            plt.show()
        else:
            self.pdf.savefig()
            if self.docfig:
                for ext in ["png", "pdf", "svg"]:
                    plt.savefig(self.path / f"pyfocus.page3_example.{ext}")

        plt.close()


# Non-Class Functions ========================================================#
def fit_focus_curves(
    focus_positions: np.ndarray,
    fwhm: np.ndarray,
    fnom: float = 2.7,
    norder: int = 2,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit line / star focus curves

    [extended_summary]

    Parameters
    ----------
    focus_positions : :obj:`~numpy.ndarray`
        Array of the focus position values
    fwhm : :obj:`~numpy.ndarray`
        2D array of FWHM for all lines as a function of COLLFOC
    fnom : :obj:`float`, optional
        Nominal FHWM of an in-focus line. (Default: 2.7)
    norder : :obj:`int`, optional
        Polynomial order of the focus fit (Default: 2 = Quadratic)
    debug : :obj:`bool`, optional
        Print debug statements  (Default: False)

    Returns
    -------
    min_cf_value : :obj:`~numpy.ndarray`
        Array of minimum focus values per line
    optimal_cf_value : :obj:`~numpy.ndarray`
        Array of optimal focus values per line
    min_linewidth : :obj:`~numpy.ndarray`
        Array of minimum linewidths per line
    foc_fits : :obj:`~numpy.ndarray`
        Array of :obj:`~numpy.polynomial.Polynomial` objects per line
    """
    # Warning Filter -- Polyfit RankWarning, don't wanna hear about it
    warnings.simplefilter("ignore", np.RankWarning)

    # Create the various arrays (filled with NaN)
    _, n_lines = fwhm.shape
    min_linewidth = np.full(n_lines, np.nan, dtype=float)
    min_cf_value = np.full(n_lines, np.nan, dtype=float)
    optimal_cf_value = np.full(n_lines, np.nan, dtype=float)
    foc_fits = np.full(n_lines, np.nan, dtype=np.polynomial.Polynomial)

    # Fitting arrays (these are indices for collimator focus)
    d_f = np.diff(focus_positions).mean()
    cf_grid_fine = np.arange(
        np.min(focus_positions),
        np.max(focus_positions) + d_f / 10,
        d_f / 10,
        dtype=float,
    )

    # Loop through lines to find the best focus for each one
    for i in range(n_lines):
        # Data are the FWHM for this line at different COLLFOC
        fwhms_of_this_line = fwhm[:, i]

        # Find unphysically large or small FWHM (or NaN) -- set to np.nan
        bad_idx = (
            (fwhms_of_this_line < 1.0)
            | (fwhms_of_this_line > 15.0)
            | np.isnan(fwhms_of_this_line)
        )
        fwhms_of_this_line[bad_idx] = np.nan

        # If no more than 3 of the FHWM are good for this line, skip and go on
        if np.sum(np.logical_not(bad_idx)) < 3:
            # NOTE: This leaves a NaN in this position of all arrays.
            continue

        # Do a polynomial fit (norder) to the FWHM vs COLLFOC index
        # fit = np.polyfit(cf_idx_coarse, fwhms_of_this_line, norder)
        fit = utils.good_poly(focus_positions, fwhms_of_this_line, norder, thresh=2.0)
        foc_fits[i] = fit
        if debug:
            print(f"In fit_focus_curves(): fit = {fit.convert().coef}")

        # If good_poly() returns zeros, move along (leaving NaN in the arrays)
        if all(value == 0 for value in fit.coef):
            continue

        # Use the fine grid to evaluate the curve miniumum
        focus_curve = fit(cf_grid_fine)
        min_cf_value[i] = cf_grid_fine[np.argmin(focus_curve)]
        min_linewidth[i] = np.min(focus_curve)

        # Compute the nominal focus position as the larger of the two points
        #   where the polymonial function crosses fnom
        # Convert the `fit` to the unscaled data domain and subtract `fnom`
        #   from the order=0 coefficient
        roots = np.polynomial.Polynomial(fit.convert().coef + [-fnom, 0, 0]).roots()
        if debug:
            print(f"Roots: {roots}")
        optimal_cf_value[i] = np.max(np.real(roots))

    # After looping, return the items as a series of numpy arrays
    return min_cf_value, optimal_cf_value, min_linewidth, foc_fits


def extract_spectrum(spectrum: np.ndarray, traces: np.ndarray, win: int) -> np.ndarray:
    """Object spectral extraction routine

    Extract spectra by averaging over the specified window

    Parameters
    ----------
    spectrum : :obj:`~numpy.ndarray`
        The trimmed spectral image
    traces : :obj:`~numpy.ndarray`
        The trace(s) along which to extract spectra
    win : :obj:`int`
        Window over which to average the spectrum

    Returns
    -------
    :obj:`~numpy.ndarray`
        2D or 3D array of spectra of individual orders
    """
    # Spec out the shape, and create an empty array to hold the output spectra
    norders, n_x = traces.shape
    spectra = np.empty((norders, n_x), dtype=float)
    speca = np.empty(n_x, dtype=float)

    # Set extraction window size
    half_window = int(win) // 2

    for order in range(norders):
        # Because of python indexing, we need to "+1" the upper limit in order
        #   to get the full wsize elements for the average
        trace = traces[order, :].astype(int)
        for i in range(n_x):
            speca[i] = np.average(
                spectrum[trace[i] - half_window : trace[i] + half_window + 1, i]
            )
        spectra[order, :] = speca.reshape((1, n_x))

    return spectra


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
    specimg = utils.trim_oscan(ccd, ccd.header["BIASSEC"], ccd.header["TRIMSEC"])

    # Build the trace for spectrum extraction
    n_y, n_x = specimg.shape
    traces = np.full(n_x, n_y / 2, dtype=float).reshape((1, n_x))
    spec1d = extract_spectrum(specimg.data, traces, win=11)

    # Find the lines!
    centers, _ = DevenyFocus.find_lines_wrap(spec1d, thresh=thresh)

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
        dfocus = DevenyFocus(
            pathlib.Path(".").resolve(),
            flog=args.flog,
            thresh=args.thresh,
            launch_preview=not args.nodisplay,
            docfig=args.g,
        )
        dfocus.run()

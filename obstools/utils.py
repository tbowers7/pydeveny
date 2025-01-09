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

"""Utility Module

LDTObserverTools contains python ports of the various DeVeny IDL routines

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
https://lowell.edu

This file contains various utility functions needed by other routines in this
package.
"""

# Built-In Libraries
import argparse
from functools import reduce
from importlib import resources
import pathlib
import textwrap
import sys
import warnings

# 3rd-Party Libraries
import darkdetect
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Local Libraries
from obstools.version import version as __version__

# CONSTANTS
CONFIG = resources.files("obstools") / "config"
DATA = resources.files("obstools") / "data"
# Modify SG theme based on system Light/Dark theme
SG_THEME = "dark gray 14" if darkdetect.isDark() else "light grey 1"


class ObstoolsError(Exception):
    """Base error class for this package"""


def all_subclasses(cls):
    """
    Collect all the subclasses of the provided class.

    .. note::

        This function borrowed from PypeIt

    The search follows the inheritance to the highest-level class.  Intermediate
    base classes are included in the returned set, but not the base class itself.

    Thanks to:
    https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name

    Args:
        cls (object):
            The base class

    Returns:
        :obj:`set`: The unique set of derived classes, including any
        intermediate base classes in the inheritance thread.
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def check_float(potential_float) -> bool:
    """Simple funtion to check whether something is (convertable to) a float

    Parameters
    ----------
    potential_float : :obj:`~typing.Any`
        Value to check for float

    Returns
    -------
    :obj:`bool`
        Whether it am or it ain't a :obj:`float`.
    """
    try:
        float(potential_float)
        return True
    except ValueError:
        return False


def first_moment_1d(line):
    """Returns the 1st moment of line

    Parameters
    ----------
    line : :obj:`~numpy.ndarray`
        1-dimensional array to find the 1st moment of

    Returns
    -------
    :obj:`float`
        The first moment of the input array relative to element #
    """
    # Only use positive values -- set negative values to zero
    line[line < 0] = 0

    # Make the counting array
    yy = np.arange(len(line))

    # Return the first moment
    return np.sum(yy * line) / np.sum(line)


def flatten_comprehension(nested_list: list) -> list:
    """Flatten a single-depth nested list via list comprehension

    https://realpython.com/python-flatten-list/

    Parameters
    ----------
    nested_list : :obj:`list`
        The single-depth nested list to flatten

    Returns
    -------
    :obj:`list`
        The flattened list
    """
    return [item for row in nested_list for item in row]


def gaussfit(x, y, nterms: int = 3, estimates=None, bounds=None, debug: bool = False):
    """Function similar to IDL's GAUSSFIT

    Big caveat: as implemented, can only estimate the initial parameters for
    POSITIVE gaussians (emission), and cannot correctly estimate parameters
    for negative (absorption) gaussians.  The function will still happily fit
    a negative gaussian if given the proper estimates.

    Utilizes :func:`scipy.optimize.curve_fit` and the helper function
    :func:`gaussian_function` below.

    Parameters
    ----------
    x : :obj:`~numpy.ndarray`
        Input abscissa values for the fitting
    y : :obj:`~numpy.ndarray`
        Input ordinate values for the fitting
    nterms : :obj:`int`, optional
        Number of terms to use in the Gaussian fit (Default: 3)
    estimates : :obj:`~numpy.ndarray`, optional
        Estimated values for the ``nterms`` parameters
        (see :func:`~scipy.optimize.curve_fit`; Default: None)
    bounds : :obj:`~numpy.ndarray`, optional
        Bounds on the ``nterms`` parameters
        (see :func:`~scipy.optimize.curve_fit`; Default: None)
    debug : :obj:`bool`, optional
        Print debugging statements.  (Defualt: False)

    Returns
    -------
    popt : :obj:`~numpy.ndarray`
        Optimal values for the parameters (see :func:`~scipy.optimize.curve_fit`)

    pcov : :obj:`~numpy.ndarray`
        The estimated covariance of ``popt`` (see :func:`~scipy.optimize.curve_fit`)
    """

    if nterms < 3 or nterms > 6:
        raise ValueError(f"{nterms} is an invalid number of terms.")

    if bounds is not None and len(bounds[0]) != nterms:
        raise ValueError("Bounds array must contain NTERMS elements.")

    # This block is for estimating the parameters if none given.
    if estimates is None:
        # Subtract a linear term if nterm == 5 or 6 or constant for nterm == 4
        if nterms > 3:
            poly = np.polynomial.Polynomial.fit(x, y, 0 if nterms == 4 else 1)
            y_modified = y - poly(x)
        # Do nothing if nterm == 3
        else:
            y_modified = y

        # Find the estimates of a0, a1, a2:
        dx = np.diff(x)[0]
        a0 = np.max(y_modified)
        a1 = x[0] + first_moment_1d(y_modified) * dx
        # Use points where value > (1/e)*max to estimate width
        s_idx = np.where(y_modified > a0 / np.e)
        a2 = np.abs(x[s_idx][-1] - x[s_idx][0]) / 2.0
        if debug:
            print(f"Estimated values: a0={a0:.1f}  a1={a1:.2f}  a2={a2:.2f}")

        # Construct the estimates list
        estimates = [a0, a1, a2]

        # Check estimate against bounds
        if bounds is not None:
            for i, est in enumerate(estimates):
                est = bounds[0][i] if est < bounds[0][i] else est
                est = bounds[1][i] if est > bounds[1][i] else est
                estimates[i] = est

        if nterms > 3:
            estimates = estimates + list(poly.coef)
        if nterms == 6:
            estimates.append(0.0)

    # Else, make sure the number of estimate values equals nterms
    else:
        if len(estimates) != nterms:
            raise ValueError("Estimate array must contain NTERMS elements.")

    if bounds is None:
        bounds = (np.full(nterms, -np.inf), np.full(nterms, np.inf))
        bounds[0][2] = 0
    if debug:
        print(bounds)

    popt, pcov = scipy.optimize.curve_fit(
        gaussian_function, x, y, p0=estimates, bounds=bounds, ftol=1e-6
    )

    if debug:
        print(f"Estimated/Fit Width: {a2} / {popt[2]}")

    return popt, pcov


def gaussian_function(
    x: np.ndarray,
    a0: float,
    a1: float,
    a2: float,
    a3: float = 0.0,
    a4: float = 0.0,
    a5: float = 0.0,
):
    """Gaussian Function

    Construct a basic Gaussian using at least 3, but up to 6 parameters.

    Parameters
    ----------
    x : :obj:`~numpy.ndarray`
        X values over which to compute the gaussian
    a0 : :obj:`float`
        Gaussian amplitude
    a1 : :obj:`float`
        Gaussian mean (mu)
    a2 : :obj:`float`
        Gaussian width (sigma)
    a3 : :obj:`float`, optional
        Baseline atop which the Gaussian sits.  (Default: 0)
    a4 : :obj:`float`, optional
        Slope of the baseline atop which the Gaussian sits.  (Default: 0)
    a5 : :obj:`float`, optional
        Quadratic term of the baseline atop which the Gaussian sits.
        (Default: 0)

    Returns
    -------
    :obj:`~numpy.ndarray`
        The Y values of the Gaussian corresponding to X
    """
    # Silence RuntimeWarning for overflow, this function only
    warnings.simplefilter("ignore", RuntimeWarning)
    z = (x - a1) / a2

    return a0 * np.exp(-(z**2) / 2.0) + a3 + a4 * x + a5 * x**2


def good_poly(x, y, order, thresh, return_full=False):
    """Robust fitting of a polynomial to data

    This is a python port of an IDL routine written years ago by M. Buie.

    This is a multi-pass fitting routine that fits a fixed order polynomial
    to the input data.  After each pass, the scatter of the fit relative
    to the fitted line is computed.  Each point is examined to see if it
    falls beyond THRESH sigma from the line.  If is does, it is removed
    from the data and the fit is tried again.  This will make up to two
    attempts to remove bad data.

    Written in IDL 1991-1998, Marc W. Buie, Lowell Observatory

    Parameters
    ----------
    x : :obj:`~numpy.ndarray`
        Input dataset, independant values.
    y : :obj:`~numpy.ndarray`
        Input dataset, dependant values.
    order : :obj:`int`
        Order of the polynomial fit (linear = 1).
    thresh : :obj:`float`
        Sigma threshold for removing outliers.
    return_full : :obj:`bool`, optional
        If True, also return:
            yfit : Fitted values for y that match the input vector.
            newx : X values from input that were considered good.
            newy : Y values from input that were considered good.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Array of fit parameters, as in :func:`numpy.polyfit`.
    Also, optionally, the ``return_full`` bits
    """
    # Make copies to not mess up the inputs
    xx = x
    yy = y

    # Filter out NaNs
    if False in (good_idx := np.logical_and(~np.isnan(x), ~np.isnan(y))):
        xx = xx[good_idx]
        yy = yy[good_idx]

    if (array_length := len(xx)) == 0:
        return warn_and_return_zeros(return_full, x, xx, yy, order)

    # Check for fewer data points than the requested polynomial order
    if array_length < order:
        coeff = np.zeros(order + 1)
        if array_length != 1:
            sigma = np.std(yy)
            coeff[0] = np.mean(yy)
        else:
            coeff[0] = yy[0]
            sigma = yy[0]
            sigma = 1.0 if sigma == 0.0 else sigma
        print("Not enough data to support even a non-robust polynomial fit.")
        if return_full:
            yfit = [coeff[0]] * len(x)
            return coeff, yfit, xx, yy
        return coeff

    # Initial fit with all the data.
    coeff = np.polyfit(xx, yy, order)
    yfit = np.polyval(coeff, xx)
    flat = (yy - yfit) + np.sum(yfit) / array_length
    mean, sigma = np.mean(flat), np.std(flat)

    # Remove all points beyond threshold sigma
    good = np.where(np.abs(flat - mean) < thresh * sigma)
    nbad = array_length - len(good)
    xx, yy = xx[good], yy[good]
    if (array_length := len(xx)) == 0:
        return warn_and_return_zeros(return_full, x, xx, yy, order)

    # Do a second pass if there were any bad points removed
    if nbad != 0:
        coeff = np.polyfit(xx, yy, order)
        yfit = np.polyval(coeff, xx)
        flat = (yy - yfit) + np.sum(yfit) / array_length
        mean, sigma = np.mean(flat), np.std(flat)

        # Remove all points beyond threshold sigma
        good = np.where(np.abs(flat - mean) < thresh * sigma)
        nbad = array_length - len(good)
        xx, yy = xx[good], yy[good]
        if (array_length := len(xx)) == 0:
            return warn_and_return_zeros(return_full, x, xx, yy, order)

    # Do a third pass if there were any more bad points removed
    if nbad != 0:
        coeff = np.polyfit(xx, yy, order)
        yfit = np.polyval(coeff, xx)
        flat = (yy - yfit) + np.sum(yfit) / array_length
        mean, sigma = np.mean(flat), np.std(flat)

    # Check that the fit coefficients are finite:
    if not np.all(np.isfinite(coeff)):
        return warn_and_return_zeros(return_full, x, xx, yy, order)

    if return_full:
        return coeff, yfit, xx, yy
    return coeff


def nearest_odd(x: float) -> int:
    """Find the nearest odd integer

    https://www.mathworks.com/matlabcentral/answers/45932-round-to-nearest-odd-integer#accepted_answer_56149

    Parameters
    ----------
    x : :obj:`float`
        Input number

    Returns
    -------
    :obj:`int`
        The nearest odd integer
    """
    return int(2 * np.floor(x / 2) + 1)


def set_std_tickparams(axis: plt.axis, tsz: int | float):
    """Set standard tick parameters for a plot

    These are my own "standards", based on plots I used to make in IDL.

    Parameters
    ----------
    axis : :obj:`~matplotlib.pyplot.axis`
        PyPlot axis for whom the tick parameters must be set
    tsz : :obj:`int` or :obj:`float`
        TypeSiZe
    """
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=tsz,
    )


def sinusoid(
    x: np.ndarray,
    a: float,
    lam: float,
    phi: float,
    y0: float = 0,
    lin: float = 0,
    quad: float = 0,
    cube: float = 0,
    quar: float = 0,
) -> np.ndarray:
    """Return a basic sinusoid (for use with :func:`scipy.optimize.curve_fit`)

    _extended_summary_

    Parameters
    ----------
    x : :obj:`~numpy.ndarray`
        The abscissa values for which to return the ordinate
    a : :obj:`float`
        The amplitude of the sinusoid (in units of ordinate)
    lam : :obj:`float`
        The wavelength of the sinusoid (in units of abscissa), equivalent to
        `2π/k` (where `k` is the wavenumber).
    phi : :obj:`float`
        The phase shift of the sinusoid (in units of phase, nominally 0-1)
    y0 : :obj:`float`, optional
        The vertical offset of the sinusoid from zero (in units of ordinate)
        (Default: 0)
    lin : :obj:`float`, optional
        The linear term added to the fit (in units of ordinate/abscissa)
        (Default: 0)
    quad : :obj:`float`, optional
        The quadratic term added to the fit (in units of ordinate/abscissa**2)
        (Default: 0)
    cube : :obj:`float`, optional
        The cubic term added to the fit (in units of ordinate/abscissa**3)
        (Default: 0)
    quar : :obj:`float`, optional
        The quartic term added to the fit (in units of ordinate/abscissa**4)
        (Default: 0)

    Returns
    -------
    :obj:`~numpy.ndarray`
        The sinusoid ordinate
    """
    return (
        a * np.sin(2.0 * np.pi * x / lam + 2.0 * np.pi * phi)
        + y0
        + lin * x
        + quad * x**2
        + cube * x**3
        + quar * x**4
    )


def warn_and_return_zeros(return_full: bool, x, xx, yy, order, raise_warn=False):
    """Set warning and return zeroes from :func:`good_poly`

    This function is a DRY.  Since this block is used several times in
    :func:`good_poly`, separate out as a function.

    Parameters
    ----------
    return_full : :obj:`bool`
        Return a bunch of stuff
    x :  :obj:`~numpy.ndarray`
        [description]
    xx :  :obj:`~numpy.ndarray`
        [description]
    yy :  :obj:`~numpy.ndarray`
        [description]
    order : :obj:`int`
        The order of the polynomial fit
    raise_warn : :obj:`bool`, optional
        Actually raise the warning this function is meant to  [Default: False]

    Returns
    -------
    :obj:`~numpy.ndarray`
        An array of zeros of the proper length
    """
    if raise_warn:
        warnings.warn("No good values to fit, return zeros.", UserWarning)
    if return_full:
        yfit = [0] * len(x)
        return [0] * (order + 1), yfit, xx, yy
    return [0] * (order + 1)


class ScriptBase:
    """
    Provides a base class for all scripts.

    Implements base classes for use with ``PypeIt`` scripts.

    .. include common links, assuming primary doc root is up one directory
    .. include:: ../include/links.rst

    """

    @classmethod
    def entry_point(cls):
        """
        Defines the main script entry point.
        """
        args = cls.parse_args()
        if args.version:
            print(f"  LDT Observer Tools (obstools) version {__version__}")
        else:
            sys.exit(cls.main(args))

    @classmethod
    @property
    def name(cls):
        """
        Provide the name of the script.  By default, this is the name of the
        module.
        """
        return f"{cls.__module__.rsplit('.', maxsplit=1)[-1]}"

    @classmethod
    def parse_args(cls, options=None):
        """
        Parse the command-line arguments.
        """
        parser = cls.get_parser()
        ScriptBase._fill_parser_cwd(parser)
        # Add "--version" to bottom of all scripts
        parser.add_argument(
            "--version", action="store_true", help="Print version and exit"
        )
        return parser.parse_args() if options is None else parser.parse_args(options)

    @staticmethod
    def _fill_parser_cwd(parser):
        """
        Replace the default of any action that is exactly ``'current working
        directory'`` with the value of ``os.getcwd()``.

        The ``parser`` is edited *in place*.

        Args:
            parser (:obj:`~argparse.ArgumentParser`):
                The argument parsing object to edit.
        """
        for action in parser._actions:
            if action.default == "current working directory":
                action.default = pathlib.Path.cwd()

    # Base classes should override this
    @staticmethod
    def main(args):
        """
        Execute the script.
        """

    @classmethod
    def get_parser(
        cls,
        description=None,
        width=None,
        formatter=argparse.ArgumentDefaultsHelpFormatter,
    ):
        """
        Construct the command-line argument parser.

        Derived classes should override this.  Ideally they should use this
        base-class method to instantiate the ArgumentParser object and then fill
        in the relevant parser arguments

        .. warning::

            *Any* argument that defaults to the
            string ``'current working directory'`` will be replaced by the
            result of ``os.getcwd()`` when the script is executed.  This means
            help dialogs will include this replacement, and parsing of the
            command line will use ``os.getcwd()`` as the default.  This
            functionality is largely to allow for PypeIt's automated
            documentation of script help dialogs without the "current working"
            directory being that of the developer that most recently compiled
            the docs.

        Args:
            description (:obj:`str`, optional):
                A short description of the purpose of the script.
            width (:obj:`int`, optional):
                Restrict the width of the formatted help output to be no longer
                than this number of characters, if possible given the help
                formatter.  If None, the width is the same as the terminal
                width.
            formatter (:obj:`~argparse.HelpFormatter`):
                Class used to format the help output.

        Returns:
            :obj:`~argparse.ArgumentParser`: Command-line interpreter.
        """
        return argparse.ArgumentParser(
            description=description,
            formatter_class=lambda prog: formatter(prog, width=width),
        )


class SmartFormatter(argparse.HelpFormatter):
    r"""
    Enable a combination of both fixed-format and wrappable lines to be
    formatted for the help statements for command-line arguments used with
    :class:`~argparse.ArgumentParser`.

    Borrows from
    https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text

    Help strings that use this formatter *must* begin with "R|".  If not, the
    help string is parsed by the base class.

    When parsed by this formatter, the leading "R|" characters are stripped and
    the lines to be printed are parsed using :func:`~str.splitlines`.  Each resulting
    line is wrapped using :func:`~textwrap.wrap`, unless it begins with the characters
    "F|", which forces the line to remain unaltered (except for stripping the
    leading characters).

    For example, if you add an argument like this:

    .. code-block:: python

        parser.add_argument('-t', '--tell_file', type=str,
                            help='R|Configuration file to change default telluric parameters.  '
                                 'Note that the parameters in this file will be overwritten if '
                                 'you set argument in your terminal.  The --tell_file option '
                                 'requires a .tell file with the following format:\n'
                                 '\n'
                                 'F|    [tellfit]\n'
                                 'F|         objmodel = qso\n'
                                 'F|         redshift = 7.6\n'
                                 'F|         bal_wv_min_max = 10825,12060\n'
                                 'OR\n'
                                 'F|    [tellfit]\n'
                                 'F|         objmodel = star\n'
                                 'F|         star_type = A0\n'
                                 'F|         star_mag = 8.\n'
                                 'OR\n'
                                 'F|    [tellfit]\n'
                                 'F|         objmodel = poly\n'
                                 'F|         polyorder = 3\n'
                                 'F|         fit_wv_min_max = 9000.,9500.\n'
                                 '\n')

    The result will be (depending on the width of your console):

    .. code-block:: console

        -t TELL_FILE, --tell_file TELL_FILE
                          Configuration file to change default telluric
                          parameters.  Note that the parameters in this file
                          will be overwritten if you set argument in your
                          terminal.  The --tell_file option requires a .tell
                          file with the following format:

                              [tellfit]
                                   objmodel = qso
                                   redshift = 7.6
                                   bal_wv_min_max = 10825,12060
                          OR
                              [tellfit]
                                   objmodel = star
                                   star_type = A0
                                   star_mag = 8.
                          OR
                              [tellfit]
                                   objmodel = poly
                                   polyorder = 3
                                   fit_wv_min_max = 9000.,9500.
    """

    def _split_lines(self, text, width):
        """
        Split the provided text into width constrained lines.

        See the class description for formatting instructions.
        """
        if text.startswith("R|"):
            lines = text[2:].splitlines()
            for i, line in enumerate(lines):
                if line.startswith("F|"):
                    lines[i] = [line[2:]]
                elif len(line) == 0:
                    lines[i] = [" "]
                else:
                    lines[i] = textwrap.wrap(line, width)
            return reduce(list.__add__, lines)
        return super()._split_lines(text, width)

# -*- coding: utf-8 -*-
#
#  This file is part of LDTObserverTools.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 29-Dec-2021
#
#  @author: tbowers

"""LDTObserverTools contains python ports of various LDT Observer Tools

Lowell Discovery Telescope (Lowell Observatory: Flagstaff, AZ)
http://www.lowell.edu

This file contains the LMI exposure time calculator routine (ported from PHP)
from Phil Massey's webpage.
http://www2.lowell.edu/users/massey/LMI/etc_calc.php

These routines are used for computing the required exposure times for LMI based
on various requirements.

The values for `Star20` are the count rates in e-/sec/image at X=0 for a 20th
magnitude star measured with a radius = 1.4 x FWHM in pixels.

NOTE: The LMI-specific pixel scale, gain, and readnoise are hard-coded into
      this module.
"""

# Built-In Libraries
import pathlib

# 3rd-Party Libraries
import astropy.table
import numpy as np
from pkg_resources import resource_filename


# Local Libraries

# Constants
SCALE = 0.12  # "/pix
READ_NOISE = 6.0  # e-
GAIN = 2.89  # e-/ADU
BIAS = 1050  # ADU (approx) for 2x2 binning

# Subdirectory Paths
LOT_DATA = pathlib.Path(resource_filename("obstools", "data"))


# User-Interface Computation Routines ========================================#
def exptime_given_snr_mag(snr, mag, airmass, band, phase, seeing, binning=2):
    """exptime_given_snr_mag Compute the exposure time given SNR and magnitude

    Given a desired signal-to-noise ratio and stellar magnitude, compute the
    exposure time required for a particular LMI Filter, moon phase, seeing and
    CCD binning.

    Parameters
    ----------
    snr : `float`
        Desired signal-to-noise ratio
    mag : `float`
        Magnitude in the band of the star desired
    airmass : `float`
        Airmass at which the observation will take place
    band : `str`
        The LMI filter for which to perform the calculation
    phase : `float`
        Moon phase (0-14)
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`, optional
        Binning of the CCD  [Default: 2]

    Returns
    -------
    `float`
        The desired exposure time in seconds
    """
    # Check the inputs
    check_etc_inputs(airmass, phase, seeing, binning, mag=mag, snr=snr)

    # Load the necessary counts information
    band_dict = get_band_specific_values(band)
    star_counts = counts_from_star_per_sec(band_dict, mag, airmass)
    sky_counts = sky_count_per_sec_per_ap(band_dict, phase, seeing, binning)
    read_counts = read_contribution(seeing, binning)

    # Do the computation
    A = star_counts * star_counts
    B = -snr * snr * (star_counts + sky_counts)
    C = -snr * snr * (read_counts * read_counts)
    return (-B + np.sqrt(B * B - 4.0 * A * C)) / (2.0 * A)


def exptime_given_peak_mag(peak, mag, airmass, band, phase, seeing, binning=2):
    """exptime_given_peak_mag Compute the exposure time given peak and mag

    Given a desired peak count level on the CCD and stellar magnitude, compute
    the exposure time required for a particular LMI Filter, moon phase, seeing
    and CCD binning.

    Parameters
    ----------
    peak : `float`
        Desired peak count level on the CCD (e-)
    mag : `float`
        Magnitude in the band of the star desired
    airmass : `float`
        Airmass at which the observation will take place
    band : `str`
        The LMI filter for which to perform the calculation
    phase : `float`
        Moon phase (0-14)
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`, optional
        Binning of the CCD  [Default: 2]

    Returns
    -------
    `float`
        The desired exposure time in seconds
    """
    # Check the inputs
    check_etc_inputs(airmass, phase, seeing, binning, mag=mag)

    # Load the necessary counts information
    band_dict = get_band_specific_values(band)
    star_counts = counts_from_star_per_sec(band_dict, mag, airmass)
    sky_counts = sky_count_per_sec_per_ap(band_dict, phase, seeing, binning)

    # Do the computation
    fwhm = seeing / (SCALE * binning)
    sky_count_per_pixel_per_sec = sky_counts / number_pixels(seeing, binning)
    return (peak - BIAS * GAIN) / (
        star_counts / (1.13 * fwhm * fwhm) + sky_count_per_pixel_per_sec
    )


def snr_given_exptime_mag(exptime, mag, airmass, band, phase, seeing, binning=2):
    """snr_given_exptime_mag Compute the SNR given exposure time and magnitude

    Given a desired exposure time and stellar magnitude, compute the resulting
    signal-to-noise ratio for a particular LMI Filter, moon phase, seeing and
    CCD binning.

    Parameters
    ----------
    exptime : `float`
        User-defined exposure time (seconds)
    mag : `float`
        Magnitude in the band of the star desired
    airmass : `float`
        Airmass at which the observation will take place
    band : `str`
        The LMI filter for which to perform the calculation
    phase : `float`
        Moon phase (0-14)
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`, optional
        Binning of the CCD  [Default: 2]

    Returns
    -------
    `float`
        The desired signal-to-noise ratio
    """
    # Check the inputs
    check_etc_inputs(airmass, phase, seeing, binning, mag=mag, exptime=exptime)

    # Load the necessary counts information
    band_dict = get_band_specific_values(band)
    star_counts = counts_from_star_per_sec(band_dict, mag, airmass)
    sky_counts = sky_count_per_sec_per_ap(band_dict, phase, seeing, binning)
    read_counts = read_contribution(seeing, binning)

    # Do the computation
    signal = star_counts * exptime
    noise = np.sqrt(
        star_counts * exptime + sky_counts * exptime + read_counts * read_counts
    )
    return signal / noise


def mag_given_snr_exptime(snr, exptime, airmass, band, phase, seeing, binning=2):
    """mag_given_snr_exptime Compute the magnitude given SNR and exposure time

    Given a desired signal-to-noise ratio and exposure time, compute the
    limiting magnitude for a particular LMI Filter, moon phase, seeing and
    CCD binning.

    Parameters
    ----------
    snr : `float`
        Desired signal-to-noise ratio
    exptime : `float`
        User-defined exposure time (seconds)
    airmass : `float`
        Airmass at which the observation will take place
    band : `str`
        The LMI filter for which to perform the calculation
    phase : `float`
        Moon phase (0-14)
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`, optional
        Binning of the CCD  [Default: 2]

    Returns
    -------
    `float`
        The limiting stellar magnitude
    """
    # Check the inputs
    check_etc_inputs(airmass, phase, seeing, binning, exptime=exptime, snr=snr)

    # Load the necessary counts information
    band_dict = get_band_specific_values(band)
    sky_counts = sky_count_per_sec_per_ap(band_dict, phase, seeing, binning)
    read_counts = read_contribution(seeing, binning)

    # Do the computation
    A = exptime * exptime
    B = -snr * snr * exptime
    C = -snr * snr * (sky_counts * exptime + read_counts * read_counts)
    cts_from_star_per_sec = (-B + np.sqrt(B * B - 4.0 * A * C)) / (2.0 * A)
    mag_raw = -2.5 * np.log10(cts_from_star_per_sec / band_dict["Star20"]) + 20.0
    return mag_raw - band_dict["extinction"] * airmass


def peak_counts(exptime, mag, airmass, band, phase, seeing, binning=2):
    """peak_counts Compute the peak counts on the CCD for an exptime and mag

    Given a desired exposure time and stellar magnitude, compute the resulting
    counts on the CCD for a particular LMI Filter, moon phase, seeing and
    CCD binning.

    Parameters
    ----------
    exptime : `float`
        User-defined exposure time (seconds)
    mag : `float`
        Magnitude in the band of the star desired
    airmass : `float`
        Airmass at which the observation will take place
    band : `str`
        The LMI filter for which to perform the calculation
    phase : `float`
        Moon phase (0-14)
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`, optional
        Binning of the CCD  [Default: 2]

    Returns
    -------
    `float`
        The desired counts on the CCD (e-)
    """
    # Load the necessary counts information
    band_dict = get_band_specific_values(band)
    star_counts = counts_from_star_per_sec(band_dict, mag, airmass)
    sky_counts = sky_count_per_sec_per_ap(band_dict, phase, seeing, binning)

    # Do the computation
    fwhm = seeing / (SCALE * binning)
    sky_count_per_pixel_per_sec = sky_counts / number_pixels(seeing, binning)
    return (
        star_counts / (1.13 * fwhm * fwhm) + sky_count_per_pixel_per_sec
    ) * exptime + BIAS * GAIN


# Helper Routines (Alphabetical) =============================================#
def check_etc_inputs(airmass, phase, seeing, binning, exptime=None, mag=None, snr=None):
    """check_etc_inputs Check the ETC inputs for valid values

    Does a cursory check on the ETC inputs for proper range, etc.  These are
    not exhaustive checks (i.e. checking for proper type on all values), but
    a good starting point nonetheless.

    Parameters
    ----------
    airmass : `float`
        Airmass at which the observation will take place
    phase : `float`
        Moon phase (0-14)
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`, optional
        Binning of the CCD
    exptime : `float`, optional
        User-defined exposure time (seconds) [Default: None]
    mag : `float`, optional
        Magnitude in the band of the star desired [Default: None]
    snr : `float`, optional
        Desired signal-to-noise ratio [Default: None]

    Raises
    ------
    ValueError
        If any of the inputs are deemed to be out of range
    """
    if airmass < 1 or airmass > 3:
        raise ValueError(f"Invalid airmass specified: {airmass}")
    if phase < 0 or phase > 14:
        raise ValueError(f"Invalid moon phase specified: {phase}")
    if seeing < 0.5 or seeing > 3.0:
        raise ValueError(f"Invalid seeing specified: {seeing}")
    if not isinstance(binning, int) or binning < 1 or binning > 4:
        raise ValueError(f"Invalid binning specified: {binning}")
    if exptime and (exptime < 0.001 or exptime > 1200):
        raise ValueError(f"Invalid exposure time specified: {exptime}")
    if mag and (mag < -1 or mag > 28):
        raise ValueError(f"Invalid stellar magnitude specified: {mag}")
    if snr and snr < 0.1:
        raise ValueError(f"Invalid signal-to-noise specified: {snr}")


def counts_from_star_per_sec(band_dict, mag, airmass):
    """counts_from_star_per_sec Compute the counts per second from a star

    Compute the counts per second from a star given a band, magnitude, and
    airmass.

    Parameters
    ----------
    band_dict : `dict`
        The dictionary from get_band_specific_values() containing star20 and sky
    mag : `float`
        Magnitude in the band of the star desired
    airmass : `float`
        Airmass at which the observation will take place

    Returns
    -------
    `float`
        Number of counts per second for the described star
    """
    mag_corrected = mag + band_dict["extinction"] * airmass
    return band_dict["Star20"] * np.power(10, -((mag_corrected - 20) / 2.5))


def get_band_specific_values(band):
    """get_band_specific_values Return the band-specific star and sky values

    Pull the correct row from `etc_filter_info.ecsv` containing the star count
    and sky parameters (brightness and extinction).

    Parameters
    ----------
    band : `str`
        The LMI filter to use

    Returns
    -------
    `dict`
        The dictionary representation of the table row.  If the band is
        improperly specified (i.e. is not in the table), return an empty
        dictionary.
    """
    # Read in the table, and index the filter column
    table = astropy.table.Table.read(LOT_DATA.joinpath("etc_filter_info.ecsv"))
    table.add_index("Filter")

    try:
        # Extract the row, and return it as a dictionary
        row = table.loc[band]
        return dict(zip(row.colnames, row))
    except KeyError:
        # Return an empty dictionary
        return {}


def number_pixels(seeing, binning):
    """number_pixels Number of pixels in the measuring aperture

    Counts the number of pixels in the measuring aperture, based on the seeing
    and CCD binning scheme.

    NOTE: The minimum value of N_pix is 9, which corresponds to a FWHM of
          2.54 pixels.  For 2x2 binning, this occurs at a seeing of 0.61".
          (3x3 binning = 0.91", 4x4 binning = 1.22")

    Parameters
    ----------
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`, optional
        Binning of the CCD

    Returns
    -------
    `float`
        Equivalent number of pixels within the measuring aperture
    """
    fwhm = seeing / (SCALE * binning)
    return np.max([1.4 * fwhm * fwhm, 9.0])


def read_contribution(seeing, binning):
    """read_contribution Calculate read-noise contribution

    Compute the read-noise contribution to the measuring aperture by
    multiplying the read noise per pixel by the square root of the number
    of pixels.

    Parameters
    ----------
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`
        Binning of the CCD

    Returns
    -------
    `float`
        The read-noise contribution to the photometry aperture
    """
    return READ_NOISE * np.sqrt(number_pixels(seeing, binning))


def sky_count_per_sec_per_ap(band_dict, phase, seeing, binning):
    """sky_count_per_sec_per_ap Determine sky counts per aperture per second

    [extended_summary]

    Parameters
    ----------
    band_dict : `dict`
        The dictionary from get_band_specific_values() containing star20 and sky
    phase : `float`
        Moon phase (0-14)
    seeing : `float`
        Size of the seeing disk (arcsec)
    binning : `int` or `float`
        Binning of the CCD

    Returns
    -------
    `float`
        Sky counts per second in the aperture
    """
    sky_brightness_per_arcsec2 = (
        band_dict["sky0"]
        + band_dict["sky1"] * phase
        + band_dict["sky2"] * phase * phase
    )
    sky_count_per_arcsec2_per_sec = band_dict["Star20"] * np.power(
        10, -((sky_brightness_per_arcsec2 - 20) / 2.5)
    )
    rscale = SCALE * binning
    sky_count_per_pixel_per_sec = sky_count_per_arcsec2_per_sec * rscale * rscale
    return number_pixels(seeing, binning) * sky_count_per_pixel_per_sec

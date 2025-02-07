# pylint: disable=missing-function-docstring
# -*- coding: utf-8 -*-
#
#  This file is part of LDTObserverTools.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 18-Nov-2024
#
#  @author: tbowers

"""DeVeny Collimator Focus Calculator TEST Module

NOTE: Use unit testing as a guide for rewriting the dfocus module in a more
      pythonic, less IDL style.  In particular, think about breaking down the
      functions into functionality rather than narrative steps.
"""

import pathlib

import astropy.nddata
import ccdproc
import numpy as np
import pytest

from obstools import dfocus
from obstools.tests.tstutlis import TEST_FILES
from obstools.utils import ObstoolsError


def test_dfocus():
    pass


def test_parse_focus_log():

    # Test parsing "last log"
    focus_cl = dfocus.parse_focus_log(TEST_FILES / "deveny" / "focus", "last")
    assert isinstance(focus_cl, ccdproc.ImageFileCollection)

    # Test parsing of specifically named log
    focus_cl = dfocus.parse_focus_log(
        TEST_FILES / "deveny" / "focus", "deveny_focus.20250203.081131"
    )
    assert isinstance(focus_cl, ccdproc.ImageFileCollection)

    # Test when there is no log file using 'last'
    with pytest.raises(ObstoolsError) as err:
        focus_cl = dfocus.parse_focus_log(pathlib.Path(".").resolve(), "last")
    assert str(err.value) == "No successful focus run completed in this directory"

    # Test when there is no log file using a specific log file
    with pytest.raises(ObstoolsError) as err:
        focus_cl = dfocus.parse_focus_log(
            pathlib.Path(".").resolve(), "deveny_focus.20250203.081131"
        )
    assert str(err.value) == "Specified focus run not in this directory"


def test_parse_focus_headers():

    # Get the testing ImageFileCollection  -- This works if the above test does not fail
    focus_cl = dfocus.parse_focus_log(
        TEST_FILES / "deveny" / "focus", "deveny_focus.20250203.081131"
    )

    # Test parsing the focus headers
    focus_dict = dfocus.parse_focus_headers(focus_cl)
    assert isinstance(focus_dict, dict)

    # Check outputs versus expected
    assert focus_dict["mid_file"] == TEST_FILES / "deveny" / "20250203.0025.fits"
    assert np.isclose(focus_dict["nominal"], 2.76, atol=0.01)
    assert focus_dict["start"] == 9.0
    assert focus_dict["end"] == 13.0
    assert focus_dict["delta"] == 0.5
    assert focus_dict["mnttemp"] == 7.45
    assert focus_dict["binning"] == "1x1"

    # Use a focus log with all the same input DeVeny frame
    focus_cl = dfocus.parse_focus_log(
        TEST_FILES / "deveny" / "focus", "deveny_focus.20250202.allsame"
    )
    with pytest.raises(ObstoolsError) as err:
        focus_dict = dfocus.parse_focus_headers(focus_cl)
    assert str(err.value) == "No change in focus over this set of images"


@pytest.mark.filterwarnings("ignore::astropy.wcs.FITSFixedWarning")
def test_centered_trace():

    # Load in a CCD image
    spec2d = astropy.nddata.CCDData.read(TEST_FILES / "deveny" / "20250203.0025.fits")
    # Test building the trace from the data array
    trace = dfocus.centered_trace(spec2d.data)
    assert isinstance(trace, np.ndarray)

    # Check that the trace is one element "tall" across the array
    _, nx = spec2d.data.shape
    assert trace.shape == (1, nx)


@pytest.mark.filterwarnings("ignore::astropy.wcs.FITSFixedWarning")
def test_extract_spectrum():

    # Load in a CCD image, and build the trace -- tested above
    spec2d = astropy.nddata.CCDData.read(TEST_FILES / "deveny" / "20250203.0025.fits")
    trace = dfocus.centered_trace(spec2d.data)

    # Test extracting the 1D spectrum
    spec1d = dfocus.extract_spectrum(spec2d.data, trace, window=11)
    assert isinstance(spec1d, np.ndarray)

    # Check raised errors
    traces = np.concatenate([trace, trace, trace, trace])
    with pytest.raises(ObstoolsError) as err:
        spec1d = dfocus.extract_spectrum(spec2d.data, traces, window=11)
    assert str(err.value) == "Cannot deal with multiple traces"


def test_find_lines():
    pass


def test_fit_focus_curves():
    pass


def test_plot_lines():
    pass


def test_plot_optimal_focus():
    pass


def test_plot_focus_curves():
    pass


def test_find_lines_in_spectrum():
    pass

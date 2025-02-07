# pylint: disable=missing-function-docstring
# -*- coding: utf-8 -*-
#
#  This file is part of LDTObserverTools.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 06-Nov-2024
#
#  @author: tbowers

"""Utility TEST Module
"""

from obstools import utils


def test_all_subclasses():
    pass


def test_check_float():
    # Actual Float
    assert utils.check_float(3.14159)
    # Not a float
    assert not utils.check_float("This is a float!")
    # Integer, but convertable
    assert utils.check_float(1)
    # Boolean is convertable to a float
    assert utils.check_float(True)


def test_first_moment_1d():
    pass


def test_flatten_comprehension():
    pass


def test_gaussfit():
    pass


def test_gaussian_function():
    pass


def test_good_poly():
    pass


def test_nearest_odd():
    # Check return type and value
    nearest = utils.nearest_odd(1.5)
    assert isinstance(nearest, int)
    assert nearest == 1
    # Nearest odd to an odd is itself
    assert utils.nearest_odd(5.0) == 5
    # Rounds up from even integers
    assert utils.nearest_odd(4) == 5
    assert utils.nearest_odd(4) != 3
    # But, just under goes down...
    assert utils.nearest_odd(3.999999999) == 3


def test_set_std_tickparams():
    pass


def test_sinusoid():
    pass


def test_warn_and_return_zeros():
    pass

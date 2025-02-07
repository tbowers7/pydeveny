# pylint: disable=missing-function-docstring
# -*- coding: utf-8 -*-
#
#  This file is part of LDTObserverTools.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 06-Feb-2025
#
#  @author: tbowers

"""
Odds and ends in support of tests
"""

from importlib import resources


TEST_FILES = resources.files("obstools") / "tests" / "files"

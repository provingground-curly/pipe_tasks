#!/usr/bin/env python
# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import os.path
import sys

from lsst.pipe.tasks.makeGen3SkyMap import MakeGen3SkyMapConfig, MakeGen3SkyMapTask

# Build a parser for command line arguments
parser = argparse.ArgumentParser(description="Make a SkyMap and add it to a gen3 repository.")
parser.add_argument("butler", metavar="Butler", type=str, help="Path to a gen3 butler")
parser.add_argument("collection", type=str, metavar="Collection",
                    help="Name of the Butler collection the SkyMap should be inserted into")
parser.add_argument("-C", "--config-file", dest="configFile",
                    help="Path to a config file overrides file")

args = parser.parse_args()

# Verify any supplied paths actually exist on disk
if not os.path.exists(args.butler):
    print("Butler path specified does not exists")
    sys.exit(1)

config = MakeGen3SkyMapConfig()
if args.configFile:
    if not os.path.exists(args.configFile):
        print("Path to config file specified does not exist")
        sys.exit(1)
    config.load(args.configFile)

# Construct the SkyMap Creation task and run it
skyMapTask = MakeGen3SkyMapTask(config=config, butler=args.butler, collection=args.collection)
skyMapTask.run()

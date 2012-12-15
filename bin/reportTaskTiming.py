#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""Select images and report which tracts and patches they are in
"""
import collections
import itertools
import re

import numpy

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = ["ReportTaskTimingTask", "ReportTaskTimingArgumentParser"]

class ReportTaskTimingConfig(pexConfig.Config):
    """Config for ReportTaskTimingTask
    """
    pass

class ResourceInfo(object):
    # note: UTime, STime and MaxRss were renamed to UserTime, SystemTime and MaxResidentSetSize;
    # list both so the code can report time for both older and newer runs
    _BaseNameList = ("CpuTime", "UserTime", "SystemTime", "MaxResidentSetSize", "UTime", "STime", "MaxRss")
    def __init__(self, taskName, log):
        self.taskName = taskName
        self.log = log
        self._validNames = set()
        for baseName in self._BaseNameList:
            for name in self._getNameList(baseName):
                setattr(self, name, [])
            self._validNames.add("Start" + baseName)
            self._validNames.add("End" + baseName)
    
    def isValidName(self, name):
        """Return True if name is a valid name for an item to add
        """
        return name in self._validNames
    
    def setItem(self, name, data):
        """Set an attribute based on its key name
        
        @param[in] name: name of item; must start with "Start" or "End"
            and be one of the supported values (see _BaseNameList).
            The associated field name is the same except the first letter is lowercase
        @param[in] data: list of data for this name

        @raise RuntimeError if name unknown or does not start with Start or End.
        """
        if not (name.startswith("Start") or name.startswith("End")):
            raise RuntimeError("%s error: invalid name %s; must start with Start or End" % (self, name))
        fieldName = name[0].lower() + name[1:]
        if not hasattr(self, fieldName):
            raise RuntimeError("%s error: unknown field %s" % (self, fieldName))
        valList = getattr(self, fieldName)
        valList += data
    
    def reportUsage(self):
        """Compute and report resource usage; silently skip the item if no data is available.
        """
        for baseName in self._BaseNameList:
            startName, endName, deltaName = self._getNameList(baseName)
            startList = getattr(self, startName)
            endList = getattr(self, endName)
            if not startList:
                if not endList:
                    continue
                self.log.warn("%s: %s not set; skipping" % (self, startName))
                continue
            if not endList:
                self.log.warn("%s: %s not set; skipping" % (self, endName))
                continue
            if len(startList) != len(endList):
                self.log.warn("%s: len(%s) = %d != %d = len(%s); skipping" % \
                    (self, startName, len(startList), endName, len(endList)))
                continue
        
            deltaList = numpy.array([e - s for s, e in itertools.izip(startList, endList)])
            setattr(self, deltaName, deltaList)
            self._reportItem(baseName)
    
    def _getNameList(self, baseName):
        """Return start, end and delta field names given a base name
        """
        return tuple(prefix + baseName for prefix in ("start", "end", "delta"))
    
    def _reportItem(self, baseName):
        """Report statistics for one item, given its base name
        
        If the item has not been set then report NaNs
        """
        deltaName = self._getNameList(baseName)[2]
        deltaList = getattr(self, deltaName)
        if deltaList is None:
            median = mean = stdDev = min = max = numpy.nan
        else:
            median = numpy.median(deltaList)
            mean = numpy.mean(deltaList)
            stdDev = numpy.std(deltaList)
            min = numpy.min(deltaList)
            max = numpy.max(deltaList)
        self.log.info("%s: %s median=%s; mean=%s; stdDev=%s; min=%s; max=%s; n=%s" % \
            (self.taskName, baseName, median, mean, stdDev, min, max, len(deltaList)))
    
    def __str__(self):
        return "ResourceUsage(%s)" % (self.taskName,)


class ReportTaskTimingTask(pipeBase.CmdLineTask):
    """Report which tracts and patches are needed for coaddition
    """
    ConfigClass = ReportTaskTimingConfig
    _DefaultName = "reportTaskTiming"
    
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self._nameRe = re.compile(r"((?:Start|End)[a-zA-Z]+)$")

    @pipeBase.timeMethod
    def run(self, dataRefList):
        """Report timing statistics for a collection of task metadata
    
        @param dataRefList: a list of data references for task metadata
        @return: a pipeBase.Struct with fields:
        - resourceDict: a dict (collections.OrderedDict) of task name: ResourceInfo
        """
        # dict of timing dotted name: [list of Struct]
        resourceDict = collections.OrderedDict()
        for dataRef in dataRefList:
            taskMetadata = dataRef.get()
            for name in taskMetadata.names(False): # hierarchical names
                # make stripped version of name without Start... or End...;
                # if neither present then skip this name
                strList = self._nameRe.split(name, 1)
                if len(strList) < 2:
                    continue
                taskName, itemName = strList[0:2]
                resInfo = resourceDict.get(taskName, None)
                if resInfo is None:
                    resInfo = ResourceInfo(taskName, self.log)
                    resourceDict[taskName] = resInfo
                if resInfo.isValidName(itemName):
                    resInfo.setItem(itemName, taskMetadata.get(name, True))
        
        for resourceInfo in resourceDict.itervalues():
            resourceInfo.reportUsage()
        
        return pipeBase.Struct(
            resourceDict = resourceDict,
        )

    @classmethod
    def getRunInfo(cls, parsedCmd):
        """Construct information necessary to run the task from the command-line arguments

        @param parsedCmd   Results of the argument parser
        @return Struct(func: Function to receive 'inputs';
                       inputs: List of Structs to be passed to the 'func')
        """
        log = parsedCmd.log if not cls.useMultiProcessing(parsedCmd) else None # logs are not yet picklable
        inputs = [pipeBase.Struct(cls=cls, config=parsedCmd.config, log=log,
            dataRefList=parsedCmd.dataRefList)]
        return pipeBase.Struct(func=runTask, inputs=inputs)

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        
        Use datasetType="deepCoadd" to get the right keys (even chi-squared coadds
        need filter information for this particular task).
        """
        return pipeBase.ArgumentParser(name=cls._DefaultName,
            datasetType=pipeBase.DatasetArgument(help="dataset type fortask metadata"))
    
    def _getConfigName(self):
        """Don't persist config, so return None
        """
        return None
    
    def _getMetadataName(self):
        """Don't persist metadata, so return None
        """
        return None


def runTask(parsedCmd):
    """Run task

    This forwarding is necessary because multiprocessing requires
    that the function used is picklable, which means it must be a
    named function, rather than an anonymous function (lambda) or
    method.
    """
    task = parsedCmd.cls(name = parsedCmd.cls._DefaultName, config=parsedCmd.config, log=parsedCmd.log)
    task.run(dataRefList = parsedCmd.dataRefList)


if __name__ == "__main__":
    ReportTaskTimingTask.parseAndRun()

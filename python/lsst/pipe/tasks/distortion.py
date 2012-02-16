#!/usr/bin/env python

import os, math
import numpy
import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
import lsst.afw.cameraGeom as afwCG
import lsst.afw.geom as afwGeom

"""This module defines the CameraDistortion class, which calculates the effects of optical distortions."""

distorterRegistry = pexConfig.makeRegistry(
    '''A registry of distorter factories
    
       A distorter factory makes a class with the following API:
        
       def __init__(self, config, ccd):
           """Construct a distorter
           
           @param[in] config: an instance of pexConfig.Config that configures this algorithm
           @param[in] ccd: the CCD that will provide positions to be distorted
           """
        
       def toObserved(self, x, y):
           """Return the as-observed position (x,y) of the corrected position x,y"""

       def toCorrected(self, x, y):
           """Return the corrected position (x,y) of the as-observed position x,y"""
       '''
)


class NullDistorter(object):
    """No-op distortion"""
    ConfigClass = pexConfig.Config
    def __init__(self, config, ccd):
        pass
    def toObserved(self, x, y):
        return x, y
    def toCorrected(self, x, y):
        return x, y

distorterRegistry.register("null", NullDistorter)


class RadialPolyDistorterConfig(pexConfig.Config):
    coefficients = pexConfig.ListField(dtype=float, doc="Coefficients of distortion polynomial, from 0 to n")
    observedToCorrected = pexConfig.Field(dtype=bool, doc="Coefficients represent observed-->corrected?",
                                          default=True)


class RadialPolyDistorter(object):
    """Distortion using the RadialPolyDistortion class in afw"""
    ConfigClass = RadialPolyDistorterConfig
    def __init__(self, config, ccd):
        self.ccd = ccd
        self.distortion = afwCG.RadialPolyDistortion(config.coefficients)
        self.distort = self.distortion.distort if config.observedToCorrected else self.distortion.undistort
        self.undistort = self.distortion.undistort if config.observedToCorrected else self.distortion.distort
    def toObserved(self, x, y):
        point = self.undistort(afwGeom.Point2D(x, y), self.ccd)
        return point.getX(), point.getY()
    def toCorrected(self, x, y):
        point = self.distort(afwGeom.Point2D(x, y), self.ccd)
        return point.getX(), point.getY()

distorterRegistry.register("radial", RadialPolyDistorter)



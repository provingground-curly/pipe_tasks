#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import numpy as num
import lsst.pex.config as pexConfig
import lsst.daf.base as dafBase
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base as pipeBase
from lsstDebug import getDebugFrame
from lsst.afw.display import getDisplay
from lsst.coadd.utils import addToCoadd, setCoaddEdgeBits
from lsst.ip.diffim import SnapPsfMatchTask
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import SingleFrameMeasurementTask
import lsst.meas.algorithms as measAlg

from .repair import RepairTask


class InitialPsfConfig(pexConfig.Config):
    """!Describes the initial PSF used for detection and measurement before we do PSF determination."""

    model = pexConfig.ChoiceField(
        dtype=str,
        doc="PSF model type",
        default="SingleGaussian",
        allowed={
            "SingleGaussian": "Single Gaussian model",
            "DoubleGaussian": "Double Gaussian model",
        },
    )
    pixelScale = pexConfig.Field(
        dtype=float,
        doc="Pixel size (arcsec).  Only needed if no Wcs is provided",
        default=0.25,
    )
    fwhm = pexConfig.Field(
        dtype=float,
        doc="FWHM of PSF model (arcsec)",
        default=1.0,
    )
    size = pexConfig.Field(
        dtype=int,
        doc="Size of PSF model (pixels)",
        default=15,
    )


class SnapCombineConfig(pexConfig.Config):
    doRepair = pexConfig.Field(
        dtype=bool,
        doc="Repair images (CR reject and interpolate) before combining",
        default=True,
    )
    repairPsfFwhm = pexConfig.Field(
        dtype=float,
        doc="Psf FWHM (pixels) used to detect CRs",
        default=2.5,
    )
    doDiffIm = pexConfig.Field(
        dtype=bool,
        doc="Perform difference imaging before combining",
        default=False,
    )
    doPsfMatch = pexConfig.Field(
        dtype=bool,
        doc="Perform PSF matching for difference imaging (ignored if doDiffIm false)",
        default=True,
    )
    doMeasurement = pexConfig.Field(
        dtype=bool,
        doc="Measure difference sources (ignored if doDiffIm false)",
        default=True,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, the associated pixels are not included in the combined exposure; "
        "DETECTED excludes cosmic rays",
        default=("DETECTED",),
    )
    averageKeys = pexConfig.ListField(
        dtype=str,
        doc="List of float metadata keys to average when combining snaps, e.g. float positions and dates; "
        "non-float data must be handled by overriding the fixMetadata method",
        optional=True,

    )
    sumKeys = pexConfig.ListField(
        dtype=str,
        doc="List of float or int metadata keys to sum when combining snaps, e.g. exposure time; "
        "non-float, non-int data must be handled by overriding the fixMetadata method",
        optional=True,
    )

    repair = pexConfig.ConfigurableField(target=RepairTask, doc="")
    diffim = pexConfig.ConfigurableField(target=SnapPsfMatchTask, doc="")
    detection = pexConfig.ConfigurableField(target=SourceDetectionTask, doc="")
    initialPsf = pexConfig.ConfigField(dtype=InitialPsfConfig, doc="")
    measurement = pexConfig.ConfigurableField(target=SingleFrameMeasurementTask, doc="")

    def setDefaults(self):
        self.detection.thresholdPolarity = "both"

    def validate(self):
        if self.detection.thresholdPolarity != "both":
            raise ValueError("detection.thresholdPolarity must be 'both' for SnapCombineTask")

## \addtogroup LSST_task_documentation
## \{
## \page SnapCombineTask
## \ref SnapCombineTask_ "SnapCombineTask"
## \copybrief SnapCombineTask
## \}


class SnapCombineTask(pipeBase.Task):
    r"""!
    \anchor SnapCombineTask_

    \brief Combine snaps.

    \section pipe_tasks_snapcombine_Contents Contents

     - \ref pipe_tasks_snapcombine_Debug

    \section pipe_tasks_snapcombine_Debug Debug variables

    The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
    flag \c -d to import \b debug.py from your \c PYTHONPATH; see <a
    href="http://lsst-web.ncsa.illinois.edu/~buildbot/doxygen/x_masterDoxyDoc/base_debug.html">
    Using lsstDebug to control debugging output</a> for more about \b debug.py files.

    The available variables in SnapCombineTask are:
    <DL>
      <DT> \c display
      <DD> A dictionary containing debug point names as keys with frame number as value. Valid keys are:
        <DL>
          <DT> repair0
          <DD> Display the first snap after repairing.
          <DT> repair1
          <DD> Display the second snap after repairing.
        </DL>
      </DD>
    </DL>
    """
    ConfigClass = SnapCombineConfig
    _DefaultName = "snapCombine"

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.makeSubtask("diffim")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @pipeBase.timeMethod
    def run(self, snap0, snap1, defects=None):
        """Combine two snaps

        @param[in] snap0: snapshot exposure 0
        @param[in] snap1: snapshot exposure 1
        @defects[in] defect list (for repair task)
        @return a pipe_base Struct with fields:
        - exposure: snap-combined exposure
        - sources: detected sources, or None if detection not performed
        """
        # initialize optional outputs
        sources = None

        if self.config.doRepair:
            self.log.info("snapCombine repair")
            psf = self.makeInitialPsf(snap0, fwhmPix=self.config.repairPsfFwhm)
            snap0.setPsf(psf)
            snap1.setPsf(psf)
            self.repair.run(snap0, defects=defects, keepCRs=False)
            self.repair.run(snap1, defects=defects, keepCRs=False)

            repair0frame = getDebugFrame(self._display, "repair0")
            if repair0frame:
                getDisplay(repair0frame).mtv(snap0)
            repair1frame = getDebugFrame(self._display, "repair1")
            if repair1frame:
                getDisplay(repair1frame).mtv(snap1)

        if self.config.doDiffIm:
            if self.config.doPsfMatch:
                self.log.info("snapCombine psfMatch")
                diffRet = self.diffim.run(snap0, snap1, "subtractExposures")
                diffExp = diffRet.subtractedImage

                # Measure centroid and width of kernel; dependent on ticket #1980
                # Useful diagnostic for the degree of astrometric shift between snaps.
                diffKern = diffRet.psfMatchingKernel
                width, height = diffKern.getDimensions()

            else:
                diffExp = afwImage.ExposureF(snap0, True)
                diffMi = diffExp.getMaskedImage()
                diffMi -= snap1.getMaskedImage()

            psf = self.makeInitialPsf(snap0)
            diffExp.setPsf(psf)
            table = afwTable.SourceTable.make(self.schema)
            table.setMetadata(self.algMetadata)
            detRet = self.detection.makeSourceCatalog(table, diffExp)
            sources = detRet.sources
            fpSets = detRet.fpSets
            if self.config.doMeasurement:
                self.measurement.measure(diffExp, sources)

            mask0 = snap0.getMaskedImage().getMask()
            mask1 = snap1.getMaskedImage().getMask()
            fpSets.positive.setMask(mask0, "DETECTED")
            fpSets.negative.setMask(mask1, "DETECTED")

            maskD = diffExp.getMaskedImage().getMask()
            fpSets.positive.setMask(maskD, "DETECTED")
            fpSets.negative.setMask(maskD, "DETECTED_NEGATIVE")

        combinedExp = self.addSnaps(snap0, snap1)

        return pipeBase.Struct(
            exposure=combinedExp,
            sources=sources,
        )

    def addSnaps(self, snap0, snap1):
        """Add two snap exposures together, returning a new exposure

        @param[in] snap0 snap exposure 0
        @param[in] snap1 snap exposure 1
        @return combined exposure
        """
        self.log.info("snapCombine addSnaps")

        combinedExp = snap0.Factory(snap0, True)
        combinedMi = combinedExp.getMaskedImage()
        combinedMi.set(0)

        weightMap = combinedMi.getImage().Factory(combinedMi.getBBox())
        weight = 1.0
        badPixelMask = afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes)
        addToCoadd(combinedMi, weightMap, snap0.getMaskedImage(), badPixelMask, weight)
        addToCoadd(combinedMi, weightMap, snap1.getMaskedImage(), badPixelMask, weight)

        # pre-scaling the weight map instead of post-scaling the combinedMi saves a bit of time
        # because the weight map is a simple Image instead of a MaskedImage
        weightMap *= 0.5  # so result is sum of both images, instead of average
        combinedMi /= weightMap
        setCoaddEdgeBits(combinedMi.getMask(), weightMap)

        # note: none of the inputs has a valid PhotoCalib object, so that is not touched
        # Filter was already copied

        combinedMetadata = combinedExp.getMetadata()
        metadata0 = snap0.getMetadata()
        metadata1 = snap1.getMetadata()
        self.fixMetadata(combinedMetadata, metadata0, metadata1)

        return combinedExp

    def fixMetadata(self, combinedMetadata, metadata0, metadata1):
        """Fix the metadata of the combined exposure (in place)

        This implementation handles items specified by config.averageKeys and config.sumKeys,
        which have data type restrictions. To handle other data types (such as sexagesimal
        positions and ISO dates) you must supplement this method with your own code.

        @param[in,out] combinedMetadata metadata of combined exposure;
            on input this is a deep copy of metadata0 (a PropertySet)
        @param[in] metadata0 metadata of snap0 (a PropertySet)
        @param[in] metadata1 metadata of snap1 (a PropertySet)

        @note the inputs are presently PropertySets due to ticket #2542. However, in some sense
        they are just PropertyLists that are missing some methods. In particular: comments and order
        are preserved if you alter an existing value with set(key, value).
        """
        keyDoAvgList = []
        if self.config.averageKeys:
            keyDoAvgList += [(key, 1) for key in self.config.averageKeys]
        if self.config.sumKeys:
            keyDoAvgList += [(key, 0) for key in self.config.sumKeys]
        for key, doAvg in keyDoAvgList:
            opStr = "average" if doAvg else "sum"
            try:
                val0 = metadata0.getScalar(key)
                val1 = metadata1.getScalar(key)
            except Exception:
                self.log.warn("Could not %s metadata %r: missing from one or both exposures" % (opStr, key,))
                continue

            try:
                combinedVal = val0 + val1
                if doAvg:
                    combinedVal /= 2.0
            except Exception:
                self.log.warn("Could not %s metadata %r: value %r and/or %r not numeric" %
                              (opStr, key, val0, val1))
                continue

            combinedMetadata.set(key, combinedVal)

    def makeInitialPsf(self, exposure, fwhmPix=None):
        """Initialise the detection procedure by setting the PSF and WCS

        @param exposure Exposure to process
        @return PSF, WCS
        """
        assert exposure, "No exposure provided"
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"

        if fwhmPix is None:
            fwhmPix = self.config.initialPsf.fwhm / wcs.getPixelScale().asArcseconds()

        size = self.config.initialPsf.size
        model = self.config.initialPsf.model
        self.log.info("installInitialPsf fwhm=%s pixels; size=%s pixels" % (fwhmPix, size))
        psfCls = getattr(measAlg, model + "Psf")
        psf = psfCls(size, size, fwhmPix/(2.0*num.sqrt(2*num.log(2.0))))
        return psf

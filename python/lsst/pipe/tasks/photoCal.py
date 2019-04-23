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
#
# @package lsst.pipe.tasks.
import math
import sys

import numpy as np
import astropy.units as u

import lsst.pex.config as pexConf
import lsst.pipe.base as pipeBase
from lsst.afw.image import abMagErrFromFluxErr, Calib
import lsst.afw.table as afwTable
from lsst.meas.astrom import DirectMatchTask, DirectMatchConfigWithoutLoader
import lsst.afw.display as afwDisplay
from lsst.meas.algorithms import getRefFluxField, ReserveSourcesTask
from .colorterms import ColortermLibrary

__all__ = ["PhotoCalTask", "PhotoCalConfig"]


class PhotoCalConfig(pexConf.Config):
    """Config for PhotoCal"""
    match = pexConf.ConfigField("Match to reference catalog",
                                DirectMatchConfigWithoutLoader)
    reserve = pexConf.ConfigurableField(target=ReserveSourcesTask, doc="Reserve sources from fitting")
    fluxField = pexConf.Field(
        dtype=str,
        default="slot_CalibFlux_instFlux",
        doc=("Name of the source instFlux field to use.  The associated flag field\n"
             "('<name>_flags') will be implicitly included in badFlags."),
    )
    applyColorTerms = pexConf.Field(
        dtype=bool,
        default=None,
        doc=("Apply photometric color terms to reference stars? One of:\n"
             "None: apply if colorterms and photoCatName are not None;\n"
             "      fail if color term data is not available for the specified ref catalog and filter.\n"
             "True: always apply colorterms; fail if color term data is not available for the\n"
             "      specified reference catalog and filter.\n"
             "False: do not apply."),
        optional=True,
    )
    sigmaMax = pexConf.Field(
        dtype=float,
        default=0.25,
        doc="maximum sigma to use when clipping",
        optional=True,
    )
    nSigma = pexConf.Field(
        dtype=float,
        default=3.0,
        doc="clip at nSigma",
    )
    useMedian = pexConf.Field(
        dtype=bool,
        default=True,
        doc="use median instead of mean to compute zeropoint",
    )
    nIter = pexConf.Field(
        dtype=int,
        default=20,
        doc="number of iterations",
    )
    colorterms = pexConf.ConfigField(
        dtype=ColortermLibrary,
        doc="Library of photometric reference catalog name: color term dict",
    )
    photoCatName = pexConf.Field(
        dtype=str,
        optional=True,
        doc=("Name of photometric reference catalog; used to select a color term dict in colorterms."
             " see also applyColorTerms"),
    )
    magErrFloor = pexConf.RangeField(
        dtype=float,
        default=0.0,
        doc="Additional magnitude uncertainty to be added in quadrature with measurement errors.",
        min=0.0,
    )

    def validate(self):
        pexConf.Config.validate(self)
        if self.applyColorTerms and self.photoCatName is None:
            raise RuntimeError("applyColorTerms=True requires photoCatName is non-None")
        if self.applyColorTerms and len(self.colorterms.data) == 0:
            raise RuntimeError("applyColorTerms=True requires colorterms be provided")

    def setDefaults(self):
        pexConf.Config.setDefaults(self)
        self.match.sourceSelection.doFlags = True
        self.match.sourceSelection.flags.bad = [
            "base_PixelFlags_flag_edge",
            "base_PixelFlags_flag_interpolated",
            "base_PixelFlags_flag_saturated",
        ]
        self.match.sourceSelection.doUnresolved = True


## @addtogroup LSST_task_documentation
## @{
## @page photoCalTask
## @ref PhotoCalTask_ "PhotoCalTask"
##      Detect positive and negative sources on an exposure and return a new SourceCatalog.
## @}

class PhotoCalTask(pipeBase.Task):
    r"""!
@anchor PhotoCalTask_

@brief Calculate the zero point of an exposure given a lsst.afw.table.ReferenceMatchVector.

@section pipe_tasks_photocal_Contents Contents

 - @ref pipe_tasks_photocal_Purpose
 - @ref pipe_tasks_photocal_Initialize
 - @ref pipe_tasks_photocal_IO
 - @ref pipe_tasks_photocal_Config
 - @ref pipe_tasks_photocal_Debug
 - @ref pipe_tasks_photocal_Example

@section pipe_tasks_photocal_Purpose	Description

@copybrief PhotoCalTask

Calculate an Exposure's zero-point given a set of flux measurements of stars matched to an input catalogue.
The type of flux to use is specified by PhotoCalConfig.fluxField.

The algorithm clips outliers iteratively, with parameters set in the configuration.

@note This task can adds fields to the schema, so any code calling this task must ensure that
these columns are indeed present in the input match list; see @ref pipe_tasks_photocal_Example

@section pipe_tasks_photocal_Initialize	Task initialisation

@copydoc \_\_init\_\_

@section pipe_tasks_photocal_IO		Inputs/Outputs to the run method

@copydoc run

@section pipe_tasks_photocal_Config       Configuration parameters

See @ref PhotoCalConfig

@section pipe_tasks_photocal_Debug		Debug variables

The @link lsst.pipe.base.cmdLineTask.CmdLineTask command line task@endlink interface supports a
flag @c -d to import @b debug.py from your @c PYTHONPATH; see @ref baseDebug for more about @b debug.py files.

The available variables in PhotoCalTask are:
<DL>
  <DT> @c display
  <DD> If True enable other debug outputs
  <DT> @c displaySources
  <DD> If True, display the exposure on the display's frame 1 and overlay the source catalogue.
    <DL>
      <DT> red o
      <DD> Reserved objects
      <DT> green o
      <DD> Objects used in the photometric calibration
    </DL>
  <DT> @c scatterPlot
  <DD> Make a scatter plot of flux v. reference magnitude as a function of reference magnitude.
    - good objects in blue
    - rejected objects in red
  (if @c scatterPlot is 2 or more, prompt to continue after each iteration)
</DL>

@section pipe_tasks_photocal_Example	A complete example of using PhotoCalTask

This code is in @link examples/photoCalTask.py@endlink, and can be run as @em e.g.
@code
examples/photoCalTask.py
@endcode
@dontinclude photoCalTask.py

Import the tasks (there are some other standard imports; read the file for details)
@skipline from lsst.pipe.tasks.astrometry
@skipline measPhotocal

We need to create both our tasks before processing any data as the task constructors
can add extra columns to the schema which we get from the input catalogue, @c scrCat:
@skipline getSchema

Astrometry first:
@skip AstrometryTask.ConfigClass
@until aTask
(that @c filterMap line is because our test code doesn't use a filter that the reference catalogue recognises,
so we tell it to use the @c r band)

Then photometry:
@skip measPhotocal
@until pTask

If the schema has indeed changed we need to add the new columns to the source table
(yes; this should be easier!)
@skip srcCat
@until srcCat = cat

We're now ready to process the data (we could loop over multiple exposures/catalogues using the same
task objects):
@skip matches
@until result

We can then unpack and use the results:
@skip calib
@until np.log

<HR>
To investigate the @ref pipe_tasks_photocal_Debug, put something like
@code{.py}
    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively
        if name.endswith(".PhotoCal"):
            di.display = 1

        return di

    lsstDebug.Info = DebugInfo
@endcode
into your debug.py file and run photoCalTask.py with the @c --debug flag.
    """
    ConfigClass = PhotoCalConfig
    _DefaultName = "photoCal"

    def __init__(self, refObjLoader, schema=None, **kwds):
        """!Create the photometric calibration task.  See PhotoCalTask.init for documentation
        """
        pipeBase.Task.__init__(self, **kwds)
        self.scatterPlot = None
        self.fig = None
        if schema is not None:
            self.usedKey = schema.addField("calib_photometry_used", type="Flag",
                                           doc="set if source was used in photometric calibration")
        else:
            self.usedKey = None
        self.match = DirectMatchTask(config=self.config.match, refObjLoader=refObjLoader,
                                     name="match", parentTask=self)
        self.makeSubtask("reserve", columnName="calib_photometry", schema=schema,
                         doc="set if source was reserved from photometric calibration")

    def getSourceKeys(self, schema):
        """Return a struct containing the source catalog keys for fields used
        by PhotoCalTask.


        Parameters
        ----------
        schema : `lsst.afw.table.schema`
            Schema of the catalog to get keys from.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with components:

            - ``instFlux``: Instrument flux key.
            - ``instFluxErr``: Instrument flux error key.
        """
        instFlux = schema.find(self.config.fluxField).key
        instFluxErr = schema.find(self.config.fluxField + "Err").key
        return pipeBase.Struct(instFlux=instFlux, instFluxErr=instFluxErr)

    @pipeBase.timeMethod
    def extractMagArrays(self, matches, filterName, sourceKeys):
        """!Extract magnitude and magnitude error arrays from the given matches.

        @param[in] matches Reference/source matches, a @link lsst::afw::table::ReferenceMatchVector@endlink
        @param[in] filterName  Name of filter being calibrated
        @param[in] sourceKeys  Struct of source catalog keys, as returned by getSourceKeys()

        @return Struct containing srcMag, refMag, srcMagErr, refMagErr, and magErr numpy arrays
        where magErr is an error in the magnitude; the error in srcMag - refMag
        If nonzero, config.magErrFloor will be added to magErr *only* (not srcMagErr or refMagErr), as
        magErr is what is later used to determine the zero point.
        Struct also contains refFluxFieldList: a list of field names of the reference catalog used for fluxes
        (1 or 2 strings)
        @note These magnitude arrays are the @em inputs to the photometric calibration, some may have been
        discarded by clipping while estimating the calibration (https://jira.lsstcorp.org/browse/DM-813)
        """
        srcInstFluxArr = np.array([m.second.get(sourceKeys.instFlux) for m in matches])
        srcInstFluxErrArr = np.array([m.second.get(sourceKeys.instFluxErr) for m in matches])
        if not np.all(np.isfinite(srcInstFluxErrArr)):
            # this is an unpleasant hack; see DM-2308 requesting a better solution
            self.log.warn("Source catalog does not have flux uncertainties; using sqrt(flux).")
            srcInstFluxErrArr = np.sqrt(srcInstFluxArr)

        # convert source instFlux from DN to an estimate of nJy
        referenceFlux = (0*u.ABmag).to_value(u.nJy)
        srcInstFluxArr = srcInstFluxArr * referenceFlux
        srcInstFluxErrArr = srcInstFluxErrArr * referenceFlux

        if not matches:
            raise RuntimeError("No reference stars are available")
        refSchema = matches[0].first.schema

        applyColorTerms = self.config.applyColorTerms
        applyCTReason = "config.applyColorTerms is %s" % (self.config.applyColorTerms,)
        if self.config.applyColorTerms is None:
            # apply color terms if color term data is available and photoCatName specified
            ctDataAvail = len(self.config.colorterms.data) > 0
            photoCatSpecified = self.config.photoCatName is not None
            applyCTReason += " and data %s available" % ("is" if ctDataAvail else "is not")
            applyCTReason += " and photoRefCat %s provided" % ("is" if photoCatSpecified else "is not")
            applyColorTerms = ctDataAvail and photoCatSpecified

        if applyColorTerms:
            self.log.info("Applying color terms for filterName=%r, config.photoCatName=%s because %s",
                          filterName, self.config.photoCatName, applyCTReason)
            colorterm = self.config.colorterms.getColorterm(
                filterName=filterName, photoCatName=self.config.photoCatName, doRaise=True)
            refCat = afwTable.SimpleCatalog(matches[0].first.schema)

            # extract the matched refCat as a Catalog for the colorterm code
            refCat.reserve(len(matches))
            for x in matches:
                record = refCat.addNew()
                record.assign(x.first)

            refMagArr, refMagErrArr = colorterm.getCorrectedMagnitudes(refCat, filterName)
            fluxFieldList = [getRefFluxField(refSchema, filt) for filt in (colorterm.primary,
                                                                           colorterm.secondary)]
        else:
            # no colorterms to apply
            self.log.info("Not applying color terms because %s", applyCTReason)
            colorterm = None

            fluxFieldList = [getRefFluxField(refSchema, filterName)]
            fluxField = getRefFluxField(refSchema, filterName)
            fluxKey = refSchema.find(fluxField).key
            refFluxArr = np.array([m.first.get(fluxKey) for m in matches])

            try:
                fluxErrKey = refSchema.find(fluxField + "Err").key
                refFluxErrArr = np.array([m.first.get(fluxErrKey) for m in matches])
            except KeyError:
                # Reference catalogue may not have flux uncertainties; HACK DM-2308
                self.log.warn("Reference catalog does not have flux uncertainties for %s; using sqrt(flux).",
                              fluxField)
                refFluxErrArr = np.sqrt(refFluxArr)

            refMagArr = u.Quantity(refFluxArr, u.nJy).to_value(u.ABmag)
            # HACK convert to Jy until we have a replacement for this (DM-16903)
            refMagErrArr = abMagErrFromFluxErr(refFluxErrArr*1e-9, refFluxArr*1e-9)

        # compute the source catalog magnitudes and errors
        srcMagArr = u.Quantity(srcInstFluxArr, u.nJy).to_value(u.ABmag)
        # Fitting with error bars in both axes is hard
        # for now ignore reference flux error, but ticket DM-2308 is a request for a better solution
        # HACK convert to Jy until we have a replacement for this (DM-16903)
        magErrArr = abMagErrFromFluxErr(srcInstFluxErrArr*1e-9, srcInstFluxArr*1e-9)
        if self.config.magErrFloor != 0.0:
            magErrArr = (magErrArr**2 + self.config.magErrFloor**2)**0.5

        srcMagErrArr = abMagErrFromFluxErr(srcInstFluxErrArr*1e-9, srcInstFluxArr*1e-9)

        good = np.isfinite(srcMagArr) & np.isfinite(refMagArr)

        return pipeBase.Struct(
            srcMag=srcMagArr[good],
            refMag=refMagArr[good],
            magErr=magErrArr[good],
            srcMagErr=srcMagErrArr[good],
            refMagErr=refMagErrArr[good],
            refFluxFieldList=fluxFieldList,
        )

    @pipeBase.timeMethod
    def run(self, exposure, sourceCat, expId=0):
        """!Do photometric calibration - select matches to use and (possibly iteratively) compute
        the zero point.

        @param[in]  exposure  Exposure upon which the sources in the matches were detected.
        @param[in]  sourceCat  A catalog of sources to use in the calibration
        (@em i.e. a list of lsst.afw.table.Match with
        @c first being of type lsst.afw.table.SimpleRecord and @c second type lsst.afw.table.SourceRecord ---
        the reference object and matched object respectively).
        (will not be modified  except to set the outputField if requested.).

        @return Struct of:
         - calib -------  @link lsst::afw::image::Calib@endlink object containing the zero point
         - arrays ------ Magnitude arrays returned be PhotoCalTask.extractMagArrays
         - matches ----- Final ReferenceMatchVector, as returned by PhotoCalTask.selectMatches.
         - zp ---------- Photometric zero point (mag)
         - sigma ------- Standard deviation of fit of photometric zero point (mag)
         - ngood ------- Number of sources used to fit photometric zero point

        The exposure is only used to provide the name of the filter being calibrated (it may also be
        used to generate debugging plots).

        The reference objects:
         - Must include a field @c photometric; True for objects which should be considered as
            photometric standards
         - Must include a field @c flux; the flux used to impose a magnitude limit and also to calibrate
            the data to (unless a color term is specified, in which case ColorTerm.primary is used;
            See https://jira.lsstcorp.org/browse/DM-933)
         - May include a field @c stargal; if present, True means that the object is a star
         - May include a field @c var; if present, True means that the object is variable

        The measured sources:
        - Must include PhotoCalConfig.fluxField; the flux measurement to be used for calibration

        @throws RuntimeError with the following strings:

        <DL>
        <DT> No matches to use for photocal
        <DD> No matches are available (perhaps no sources/references were selected by the matcher).
        <DT> No reference stars are available
        <DD> No matches are available from which to extract magnitudes.
        </DL>
        """
        import lsstDebug

        display = lsstDebug.Info(__name__).display
        displaySources = display and lsstDebug.Info(__name__).displaySources
        self.scatterPlot = display and lsstDebug.Info(__name__).scatterPlot

        if self.scatterPlot:
            from matplotlib import pyplot
            try:
                self.fig.clf()
            except Exception:
                self.fig = pyplot.figure()

        filterName = exposure.getFilter().getName()

        # Match sources
        matchResults = self.match.run(sourceCat, filterName)
        matches = matchResults.matches

        reserveResults = self.reserve.run([mm.second for mm in matches], expId=expId)
        if displaySources:
            self.displaySources(exposure, matches, reserveResults.reserved)
        if reserveResults.reserved.sum() > 0:
            matches = [mm for mm, use in zip(matches, reserveResults.use) if use]
        if len(matches) == 0:
            raise RuntimeError("No matches to use for photocal")
        if self.usedKey is not None:
            for mm in matches:
                mm.second.set(self.usedKey, True)

        # Prepare for fitting
        sourceKeys = self.getSourceKeys(matches[0].second.schema)
        arrays = self.extractMagArrays(matches=matches, filterName=filterName, sourceKeys=sourceKeys)

        # Fit for zeropoint
        r = self.getZeroPoint(arrays.srcMag, arrays.refMag, arrays.magErr)
        self.log.info("Magnitude zero point: %f +/- %f from %d stars", r.zp, r.sigma, r.ngood)

        # Prepare the results
        flux0 = 10**(0.4*r.zp)  # Flux of mag=0 star
        flux0err = 0.4*math.log(10)*flux0*r.sigma  # Error in flux0
        calib = Calib()
        calib.setFluxMag0(flux0, flux0err)

        return pipeBase.Struct(
            calib=calib,
            arrays=arrays,
            matches=matches,
            zp=r.zp,
            sigma=r.sigma,
            ngood=r.ngood,
        )

    def displaySources(self, exposure, matches, reserved, frame=1):
        """Display sources we'll use for photocal

        Sources that will be actually used will be green.
        Sources reserved from the fit will be red.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to display.
        matches : `list` of `lsst.afw.table.RefMatch`
            Matches used for photocal.
        reserved : `numpy.ndarray` of type `bool`
            Boolean array indicating sources that are reserved.
        frame : `int`
            Frame number for display.
        """
        disp = afwDisplay.getDisplay(frame=frame)
        disp.mtv(exposure, title="photocal")
        with disp.Buffering():
            for mm, rr in zip(matches, reserved):
                x, y = mm.second.getCentroid()
                ctype = afwDisplay.RED if rr else afwDisplay.GREEN
                disp.dot("o", x, y, size=4, ctype=ctype)

    def getZeroPoint(self, src, ref, srcErr=None, zp0=None):
        """!Flux calibration code, returning (ZeroPoint, Distribution Width, Number of stars)

        We perform nIter iterations of a simple sigma-clipping algorithm with a couple of twists:
        1.  We use the median/interquartile range to estimate the position to clip around, and the
        "sigma" to use.
        2.  We never allow sigma to go _above_ a critical value sigmaMax --- if we do, a sufficiently
        large estimate will prevent the clipping from ever taking effect.
        3.  Rather than start with the median we start with a crude mode.  This means that a set of magnitude
        residuals with a tight core and asymmetrical outliers will start in the core.  We use the width of
        this core to set our maximum sigma (see 2.)

        @return Struct of:
         - zp ---------- Photometric zero point (mag)
         - sigma ------- Standard deviation of fit of zero point (mag)
         - ngood ------- Number of sources used to fit zero point
        """
        sigmaMax = self.config.sigmaMax

        dmag = ref - src

        indArr = np.argsort(dmag)
        dmag = dmag[indArr]

        if srcErr is not None:
            dmagErr = srcErr[indArr]
        else:
            dmagErr = np.ones(len(dmag))

        # need to remove nan elements to avoid errors in stats calculation with numpy
        ind_noNan = np.array([i for i in range(len(dmag))
                              if (not np.isnan(dmag[i]) and not np.isnan(dmagErr[i]))])
        dmag = dmag[ind_noNan]
        dmagErr = dmagErr[ind_noNan]

        IQ_TO_STDEV = 0.741301109252802    # 1 sigma in units of interquartile (assume Gaussian)

        npt = len(dmag)
        ngood = npt
        good = None  # set at end of first iteration
        for i in range(self.config.nIter):
            if i > 0:
                npt = sum(good)

            center = None
            if i == 0:
                #
                # Start by finding the mode
                #
                nhist = 20
                try:
                    hist, edges = np.histogram(dmag, nhist, new=True)
                except TypeError:
                    hist, edges = np.histogram(dmag, nhist)  # they removed new=True around numpy 1.5
                imode = np.arange(nhist)[np.where(hist == hist.max())]

                if imode[-1] - imode[0] + 1 == len(imode):  # Multiple modes, but all contiguous
                    if zp0:
                        center = zp0
                    else:
                        center = 0.5*(edges[imode[0]] + edges[imode[-1] + 1])

                    peak = sum(hist[imode])/len(imode)  # peak height

                    # Estimate FWHM of mode
                    j = imode[0]
                    while j >= 0 and hist[j] > 0.5*peak:
                        j -= 1
                    j = max(j, 0)
                    q1 = dmag[sum(hist[range(j)])]

                    j = imode[-1]
                    while j < nhist and hist[j] > 0.5*peak:
                        j += 1
                    j = min(j, nhist - 1)
                    j = min(sum(hist[range(j)]), npt - 1)
                    q3 = dmag[j]

                    if q1 == q3:
                        q1 = dmag[int(0.25*npt)]
                        q3 = dmag[int(0.75*npt)]

                    sig = (q3 - q1)/2.3  # estimate of standard deviation (based on FWHM; 2.358 for Gaussian)

                    if sigmaMax is None:
                        sigmaMax = 2*sig   # upper bound on st. dev. for clipping. multiplier is a heuristic

                    self.log.debug("Photo calibration histogram: center = %.2f, sig = %.2f", center, sig)

                else:
                    if sigmaMax is None:
                        sigmaMax = dmag[-1] - dmag[0]

                    center = np.median(dmag)
                    q1 = dmag[int(0.25*npt)]
                    q3 = dmag[int(0.75*npt)]
                    sig = (q3 - q1)/2.3  # estimate of standard deviation (based on FWHM; 2.358 for Gaussian)

            if center is None:              # usually equivalent to (i > 0)
                gdmag = dmag[good]
                if self.config.useMedian:
                    center = np.median(gdmag)
                else:
                    gdmagErr = dmagErr[good]
                    center = np.average(gdmag, weights=gdmagErr)

                q3 = gdmag[min(int(0.75*npt + 0.5), npt - 1)]
                q1 = gdmag[min(int(0.25*npt + 0.5), npt - 1)]

                sig = IQ_TO_STDEV*(q3 - q1)     # estimate of standard deviation

            good = abs(dmag - center) < self.config.nSigma*min(sig, sigmaMax)  # don't clip too softly

            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            if self.scatterPlot:
                try:
                    self.fig.clf()

                    axes = self.fig.add_axes((0.1, 0.1, 0.85, 0.80))

                    axes.plot(ref[good], dmag[good] - center, "b+")
                    axes.errorbar(ref[good], dmag[good] - center, yerr=dmagErr[good],
                                  linestyle='', color='b')

                    bad = np.logical_not(good)
                    if len(ref[bad]) > 0:
                        axes.plot(ref[bad], dmag[bad] - center, "r+")
                        axes.errorbar(ref[bad], dmag[bad] - center, yerr=dmagErr[bad],
                                      linestyle='', color='r')

                    axes.plot((-100, 100), (0, 0), "g-")
                    for x in (-1, 1):
                        axes.plot((-100, 100), x*0.05*np.ones(2), "g--")

                    axes.set_ylim(-1.1, 1.1)
                    axes.set_xlim(24, 13)
                    axes.set_xlabel("Reference")
                    axes.set_ylabel("Reference - Instrumental")

                    self.fig.show()

                    if self.scatterPlot > 1:
                        reply = None
                        while i == 0 or reply != "c":
                            try:
                                reply = input("Next iteration? [ynhpc] ")
                            except EOFError:
                                reply = "n"

                            if reply == "h":
                                print("Options: c[ontinue] h[elp] n[o] p[db] y[es]", file=sys.stderr)
                                continue

                            if reply in ("", "c", "n", "p", "y"):
                                break
                            else:
                                print("Unrecognised response: %s" % reply, file=sys.stderr)

                        if reply == "n":
                            break
                        elif reply == "p":
                            import pdb
                            pdb.set_trace()
                except Exception as e:
                    print("Error plotting in PhotoCal.getZeroPoint: %s" % e, file=sys.stderr)

            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            old_ngood = ngood
            ngood = sum(good)
            if ngood == 0:
                msg = "PhotoCal.getZeroPoint: no good stars remain"

                if i == 0:                  # failed the first time round -- probably all fell in one bin
                    center = np.average(dmag, weights=dmagErr)
                    msg += " on first iteration; using average of all calibration stars"

                self.log.warn(msg)

                return pipeBase.Struct(
                    zp=center,
                    sigma=sig,
                    ngood=len(dmag))
            elif ngood == old_ngood:
                break

            if False:
                ref = ref[good]
                dmag = dmag[good]
                dmagErr = dmagErr[good]

        dmag = dmag[good]
        dmagErr = dmagErr[good]
        zp, weightSum = np.average(dmag, weights=1/dmagErr**2, returned=True)
        sigma = np.sqrt(1.0/weightSum)
        return pipeBase.Struct(
            zp=zp,
            sigma=sigma,
            ngood=len(dmag),
        )

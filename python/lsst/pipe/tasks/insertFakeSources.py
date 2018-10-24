# This file is part of qa explorer
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Insert fake sources into calexps
"""
import galsim
from astropy.table import Table
import numpy as np

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from .processCcd import ProcessCcdTask, ProcessCcdConfig
from lsst.qa.explorer.parquetTable import ParquetTable
from lsst.pex.exceptions import LogicError, InvalidParameterError

__all__ = ["InsertFakeSourcesConfig", "InsertFakeSourcesTask"]


class InsertFakeSourcesConfig(ProcessCcdConfig):
    """Config for inserting fake sources

    Notes
    -----
    The default column names are those from the UW sims database.
    """

    useUpdatedCalibs = pexConfig.Field(
        doc="Use updated calibs and wcs from jointcal?",
        dtype=bool,
        default=False,
    )

    raColName = pexConfig.Field(
        doc="RA column name for fake source catalog.",
        dtype=str,
        default="raJ2000",
    )

    decColName = pexConfig.Field(
        doc="Dec. column name for fake source catalog.",
        dtype=str,
        default="decJ2000",
    )

    cleanCat = pexConfig.Field(
        doc="Removed bad sources from the catalog?",
        dtype=bool,
        default=True,
    )

    diskHLR = pexConfig.Field(
        doc="Column name for the disk half light radius",
        dtype=str,
        default="DiskHalfLightRadius",
    )

    bulgeHLR = pexConfig.Field(
        doc="Column name for the bulge half light radius",
        dtype=str,
        default="BulgeHalfLightRadius",
    )

    magVar = pexConfig.Field(
        doc="The column name for the magnitude calculated taking variability into account. In the format "
            "``filter name``magVar, e.g. imagVar for the magnitude in the i band.",
        dtype=str,
        default="%smagVar",
    )

    nDisk = pexConfig.Field(
        doc="The column name for the sersic index of the disk component.",
        dtype=str,
        default="disk_n",
    )

    nBulge = pexConfig.Field(
        doc="The column name for the sersic index of the bulge component.",
        dtype=str,
        default="bulge_n",
    )

    aDisk = pexConfig.Field(
        doc="The column name for the semi major axis length of the disk component.",
        dtype=str,
        default="a_d",
    )

    aBulge = pexConfig.Field(
        doc="The column name for the semi major axis length of the bulge component.",
        dtype=str,
        default="a_b",
    )

    bDisk = pexConfig.Field(
        doc="The column name for the semi minor axis length of the disk component.",
        dtype=str,
        default="b_d",
    )

    bBulge = pexConfig.Field(
        doc="The column name for the semi minor axis length of the bulge component.",
        dtype=str,
        default="b_b",
    )

    paDisk = pexConfig.Field(
        doc="The column name for the PA of the disk component.",
        dtype=str,
        default="pa_disk",
    )

    paBulge = pexConfig.Field(
        doc="The column name for the PA of the bulge component.",
        dtype=str,
        default="pa_bulge",
    )

    fakeType = pexConfig.Field(
        doc="What type of fake catalog to use, snapshot (includes variables), static or fiilename of user"
            "defined catalog.",
        dtype=str,
        default="static",
    )

    calexpType = pexConfig.Field(
        doc="What type of image, calexp, deepCoadd or deepCoadd_calexp",
        dtype=str,
    )

    fakeTract = pexConfig.Field(
        doc="Tract that the calexp is in. Static fake catalogs are stored by tract.",
        dtype=int,
        default=9813,
    )

    def setDefaults(self):
        self.charImage.repair.doCosmicRay = False
        self.calibrate.doAstrometry = False
        self.calibrate.writeExposure = False


class InsertFakeSourcesTask(ProcessCcdTask):
    """Insert fake objects into calexps.

    Add fake stars and galaxies to the given calexp, specified in the dataRef. Galaxy parameters are read in
    from the specified file and then modelled using galsim. Re-runs characterize image and calibrate image to
    give a new background estimation and measurement of the calexp.

    `InsertFakeSourcesTask` has five functions that make images of the fake sources and then add them to the
    calexp.

    `addPixCoords`
        Use the WCS information to add the pixel coordinates of each source
        Adds an ``x`` and ``y`` column to the catalog of fake sources.
    `mkFakeGalsimGalaxies`
        Use Galsim to make fake double sersic galaxies for each set of galaxy parameters in the input file.
    `mkFakeStars`
        Use the PSF information from the calexp to make a fake star using the magnitude information from the
        input file.
    `cleanCat`
        Remove rows of the input fake catalog which have half light radius, of either the bulge or the disk,
        that are 0.
    `addFakeSources`
        Add the fake sources to the calexp.

    Notes
    -----
    The ``calexp`` with fake souces added to it is written out as the datatype ``calexp_fakes``.
    """

    _DefaultName = "insertFakeSources"
    ConfigClass = InsertFakeSourcesConfig

    def runDataRef(self, dataRef):
        """Read in/write out the required data products and add fake sources to the calexp.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            Data reference defining the ccd to have fake added to it
            Used to access the following data products:
                calexp
                jointcal_wcs
                jointcal_photoCalib

        Notes
        -----
        Uses the calibration and WCS information attached to the calexp for the posistioning and calibration
        of the sources unless the config option config.useUpdatedCalibs is set then it uses the
        meas_mosaic/jointCal outputs. The config defualts for the column names in the catalog of fakes are
        taken from the UW simulations database. Operates on one ccd at a time.
        """

        dataRef.dataId["tract"] = self.config.fakeTract
        if self.config.fakeType == "snapshot":
            fakeCat = dataRef.get("fakeSourceCat").toDataFrame()
            self.calexpType = "calexp"
            self.fakeSourceCatType = "fakeSourceCat"
        elif self.config.fakeType == "static":
            fakeCat = dataRef.get("deepCoadd_fakeSourceCat").toDataFrame()
            self.calexpType = self.config.calexpType
            # To do: DM-16254, the read and write of the fake catalogs will be changed once the new pipeline
            # task structure for ref cats is in place.
            self.fakeSourceCatType = "deepCoadd_fakeSourceCat"
        else:
            fakeCat = Table.read(self.config.fakeType).to_pandas()
            self.calexpType = self.config.calexpType

        calexp = dataRef.get(self.calexpType)
        if self.config.useUpdatedCalibs and self.calexpType == "calexp":
            self.log.info("Using updated calibs from meas_mosaic/jointCal")
            wcs = dataRef.get("jointcal_wcs")
            photoCalib = dataRef.get("jointcal_photoCalib")
        else:
            wcs = calexp.getWcs()
            photoCalib = calexp.getCalib()

        calexpWithFakes, fakeCat = self.run(fakeCat, calexp, wcs, photoCalib)

        if self.calexpType == "calexp":
            self.processFakeCalexp(dataRef, calexpWithFakes)

        dataRef.put(calexpWithFakes, "fakes_" + self.calexpType)
        fakeCat = ParquetTable(dataFrame=fakeCat)

    @classmethod
    def _makeArgumentParser(cls):
        datasetType = pipeBase.ConfigDatasetType(name="calexpType")
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType=datasetType,
                               help="data IDs for the data type specified in the calexpType config option,"
                                    "e.g. --id visit=12345 ccd=1,2^0,3")
        return parser

    def run(self, fakeCat, calexp, wcs, photoCalib):
        """Add fake sources to a calexp.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
        calexp : `lsst.afw.image.exposure.exposure.ExposureF`
        wcs : `lsst.afw.geom.skyWcs.skyWcs.SkyWcs`
        photoCalib : `lsst.afw.image.calib.Calib` or `lsst.afw.image.photoCalib.PhotoCalib`

        Returns
        -------
        calexp : `lsst.afw.image.exposure.exposure.ExposureF`
        fakeCat : `pandas.core.frame.DataFrame`

        Notes
        -----
        Adds pixel coordinates for each source to the fakeCat and removes objects with bulge or disk half
        light radius = 0 (if ``config.cleanCat = True``). These columns are called ``x`` and ``y`` and are in
        pixels.

        Adds the ``Fake`` mask plane which is then set by `addFakeSources` to mark where fake sources have
        been added. Uses the information in the ``fakeCat`` to make fake galaxies (using galsim) and fake
        stars, using the PSF models from the PSF information for the calexp. These are then added to the
        calexp and the calexp with fakes included returned.

        The galsim galaxies are made using a double sersic profile, one for the bulge and one for the disk,
        this is then convolved with the PSF at that point.
        """

        calexp.mask.addMaskPlane("FAKE")
        self.bitmask = calexp.mask.getPlaneBitMask("FAKE")
        self.log.info("Adding mask plane with bitmask %d" % self.bitmask)

        fakeCat = self.addPixCoords(fakeCat, wcs)
        if self.config.cleanCat:
            fakeCat = self.cleanCat(fakeCat)
        fakeCat = self.trimFakeCat(fakeCat, calexp, wcs)

        band = calexp.getFilter().getName()
        pixelScale = wcs.getPixelScale().asArcseconds()
        psf = calexp.getPsf()

        galaxies = (fakeCat["sourceType"] == "galaxy")
        galImages = self.mkFakeGalsimGalaxies(fakeCat[galaxies], band, photoCalib, pixelScale, psf, calexp)
        calexp = self.addFakeSources(calexp, galImages, "galaxy")

        stars = (fakeCat["sourceType"] == "star")
        starImages = self.mkFakeStars(fakeCat[stars], band, photoCalib, psf, calexp)
        calexp = self.addFakeSources(calexp, starImages, "star")

        return calexp, fakeCat

    def addPixCoords(self, fakeCat, wcs):

        """Add pixel coordinates to the catalog of fakes.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
        wcs : `lsst.afw.geom.skyWcs.skyWcs.SkyWcs`

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`

        Notes
        -----
        The default option is to use the WCS information from the calexp. If the ``useUpdatedCalibs`` config
        option is set then it will use the updated WCS from jointCal.
        """

        ras = fakeCat[self.config.raColName].values
        decs = fakeCat[self.config.decColName].values
        skyCoords = [afwGeom.SpherePoint(ra, dec, afwGeom.radians) for (ra, dec) in zip(ras, decs)]
        pixCoords = wcs.skyToPixel(skyCoords)
        xs = [coord.getX() for coord in pixCoords]
        ys = [coord.getY() for coord in pixCoords]
        fakeCat["x"] = xs
        fakeCat["y"] = ys

        return fakeCat

    def trimFakeCat(self, fakeCat, calexp, wcs):
        """Trim the fake cat to about the size of the input calexp.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
        calexp : `lsst.afw.image.exposure.exposure.ExposureF`
        wcs : `lsst.afw.geom.skyWcs.skyWcs.SkyWcs`

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`

        Notes
        -----
        There is probably a better way to do this but it will be replaced once the new pipeline task ref
        cats implementation is done. To do: DM-16254
        """
        calexpShape = calexp.getDimensions()
        minX = 0
        minY = 0
        maxX = calexpShape[0]
        maxY = calexpShape[1]

        if self.calexpType == "deepCoadd_calexp" or self.calexpType == "deepCoadd":
            minX += calexp.getXY0().getX()
            maxX += calexp.getXY0().getX()
            minY += calexp.getXY0().getY()
            maxY += calexp.getXY0().getY()

        minRa, minDec = wcs.pixelToSky(minX, minY)
        maxRa, maxDec = wcs.pixelToSky(maxX, maxY)
        minRa = minRa.asRadians()
        maxRa = maxRa.asRadians()
        minDec = minDec.asRadians()
        maxDec = maxDec.asRadians()

        if minRa > maxRa:
            maxRa, minRa = minRa, maxRa
        if minDec > maxDec:
            maxDec, minDec = minDec, maxDec

        padding = 20
        padding = padding * np.pi / (180 * 3600)
        onCalexp = ((fakeCat[self.config.raColName] - padding > minRa) &
                    (fakeCat[self.config.raColName] + padding < maxRa) &
                    (fakeCat[self.config.decColName] - padding > minDec) &
                    (fakeCat[self.config.decColName] + padding < maxDec))

        fakeCat = fakeCat[onCalexp]

        return fakeCat

    def mkFakeGalsimGalaxies(self, fakeCat, band, photoCalib, pixelScale, psf, calexp):
        """Make images of fake galaxies using GalSim.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
        band : `str`
        photoCalib : `lsst.afw.image.calib.Calib` or `lsst.afw.image.photoCalib.PhotoCalib`
        pixelScale : `float`
        psf : `lsst.meas.extensions.psfex.psfexPsf.PsfexPsf`

        Returns
        -------
        galImages : `list`
            A list of `lsst.afw.image.exposure.exposure.ExposureF`

        Notes
        -----
        Currently the updated photoCalib from jointCal/meas_mosaic is a different type to the calib attached
        to the calexp.

        Fake galaxies are made by combining two sersic profiles, one for the bulge and one for the disk. Each
        component has an individual sersic index (n), a, b and position angle (PA). The combined profile is
        then convolved with the PSF at the specified x, y position on the calexp.

        The names of the columns in the ``fakeCat`` are configurable and are the column names from the UW
        simulations database as default. For more information see the doc strings attached to the config
        options.
        """

        galImages = []

        self.log.info("Making %d fake galaxy images" % len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            # jointCal photoCalib and normal photoCalib have different ways of doing this.
            xy = afwGeom.Point2D(row["x"], row["y"])
            if self.config.useUpdatedCalibs and self.calexpType == "calexp":
                # This should go away when this is switched to reference catalogs
                try:
                    flux = photoCalib.magnitudeToInstFlux(row[self.config.magVar % band], xy)
                except LogicError:
                    flux = 0
            else:
                flux = photoCalib.getFlux(row[self.config.magVar % band])

            bulge = galsim.Sersic(row[self.config.nBulge], half_light_radius=row[self.config.bulgeHLR])
            axisRatioBulge = row[self.config.bBulge]/row[self.config.aBulge]
            bulge = bulge.shear(q=axisRatioBulge, beta=((90 - row[self.config.paBulge])*galsim.degrees))

            disk = galsim.Sersic(row[self.config.nDisk], half_light_radius=row[self.config.diskHLR])
            axisRatioDisk = row[self.config.bDisk]/row[self.config.aDisk]
            disk = disk.shear(q=axisRatioDisk, beta=((90 - row[self.config.paDisk])*galsim.degrees))

            gal = disk + bulge
            gal = gal.withFlux(flux)

            try:
                psfKernel = psf.computeKernelImage(xy).getArray()
            except InvalidParameterError:
                self.log.info("Galaxy at %0.4f, %0.4f outside of image" % (row["x"], row["y"]))
                continue
            psfIm = galsim.InterpolatedImage(galsim.Image(psfKernel), scale=pixelScale)
            gal = galsim.Convolve([gal, psfIm])
            try:
                galIm = gal.drawImage(scale=pixelScale, method="real_space").array
            except (galsim.errors.GalSimFFTSizeError, MemoryError):
                continue

            galImages.append((afwImage.ImageF(galIm), xy))

        return galImages

    def mkFakeStars(self, fakeCat, band, photoCalib, psf, calexp):

        """Make fake stars based off the properties in the fakeCat.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
        band : `str`
        photoCalib : `lsst.afw.image.calib.Calib` or `lsst.afw.image.photoCalib.PhotoCalib`
        psf : `lsst.meas.extensions.psfex.psfexPsf.PsfexPsf`

        Returns
        -------
        starImages : `list`
            A list of `lsst.afw.image.image.image.ImageF` of fake stars
        """

        starImages = []

        self.log.info("Making %d fake star images" % len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            # jointCal photoCalib and normal photoCalib have different ways of doing this.
            xy = afwGeom.Point2D(row["x"], row["y"])
            if self.config.useUpdatedCalibs and self.calexpType == "calexp":
                # This should go away when this is switched to reference catalogs
                try:
                    flux = photoCalib.magnitudeToInstFlux(row[band + "magVar"], xy)
                except LogicError:
                    flux = 0
            else:
                flux = photoCalib.getFlux(row[band + "magVar"])

            try:
                starIm = psf.computeImage(xy)

            except InvalidParameterError:
                self.log.info("Star at %0.4f, %0.4f outside of image" % (row["x"], row["y"]))
                continue

            starIm *= flux
            starImages.append((starIm.convertF(), xy))

        return starImages

    def cleanCat(self, fakeCat):
        """Remove rows from the fakes catalog which have HLR = 0 for either the buldge or disk component

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
        """

        goodRows = ((fakeCat[self.config.bulgeHLR] != 0.0) & (fakeCat[self.config.diskHLR] != 0.0))

        badRows = len(fakeCat) - len(goodRows)
        self.log.info("Removing %d rows with HLR = 0 for either the bulge or disk" % badRows)

        return fakeCat[goodRows]

    def addFakeSources(self, calexp, fakeImages, sourceType):
        """Add the fake sources to the given calexp

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
        calexp : `lsst.afw.image.exposure.exposure.ExposureF`
        fakeImages : `list`
            A list of `lsst.afw.image.image.image.ImageF`

        Returns
        -------
        calexp : `lsst.afw.image.exposure.exposure.ExposureF`

        Notes
        -----
        Uses the x, y information in the ``fakeCat`` to position an image of the fake interpolated onto the
        pixel grid of the image. Sets the ``FAKE`` mask plane for the pixels added with the fake source.
        """

        calexpBBox = calexp.getBBox()
        calexpMI = calexp.maskedImage

        for (fakeImage, xy) in fakeImages:
            X0 = xy.getX() - fakeImage.getWidth()/2 + 0.5
            Y0 = xy.getY() - fakeImage.getHeight()/2 + 0.5
            self.log.debug("Adding fake source at %d, %d" % (xy.getX(), xy.getY()))
            if sourceType == "galaxy":
                interpFakeImage = afwMath.offsetImage(fakeImage, X0, Y0, "lanczos3")
                interpFakeImBBox = interpFakeImage.getBBox()
            else:
                interpFakeImage = fakeImage
                interpFakeImBBox = fakeImage.getBBox()

            interpFakeImBBox.clip(calexpBBox)
            calexpMIView = calexpMI.Factory(calexpMI, interpFakeImBBox)

            if interpFakeImBBox.getArea() > 0:
                clippedFakeImage = interpFakeImage.Factory(interpFakeImage, interpFakeImBBox)
                clippedFakeImageMI = afwImage.MaskedImageF(clippedFakeImage)
                clippedFakeImageMI.mask.set(self.bitmask)
                calexpMIView += clippedFakeImageMI

        return calexp

    def processFakeCalexp(self, dataRef, calexp):
        """Process the calexp now that fakes have been added.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
        calexp : `lsst.afw.image.exposure.exposure.ExposureF`

        Notes
        -----
        Overwrites the data productsproduced by processCcd except for the calexp. Background subtraction is
        redone.
        """

        self.log.info("Processing %s" % (dataRef.dataId))

        charRes = self.charImage.runDataRef(dataRef=dataRef, exposure=calexp, doUnpersist=False)
        exposure = charRes.exposure

        if self.config.doCalibrate:
            self.calibrate.runDataRef(dataRef=dataRef, exposure=exposure, doUnpersist=False,
                                      background=charRes.background, icSourceCat=charRes.sourceCat)

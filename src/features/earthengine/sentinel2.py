import math
from datetime import date

import ee

from .utils import date_to_string

# These are algorithm settings for the cloud filtering algorithm

# After 2022-01-25, Sentinel-2 scenes with PROCESSING_BASELINE '04.00' or
# above have their DN (value) range shifted by 1000. The HARMONIZED
# collection shifts data in newer scenes to be in the same range as in older scenes.
image_collection = "COPERNICUS/S2_HARMONIZED"

# Ranges from 0-1.Lower value will mask more pixels out.
# Generally 0.1-0.3 works well with 0.2 being used most commonly
cloudThresh = 0.2
# Height of clouds to use to project cloud shadows
cloudHeights = [200, 10000, 250]
# Sum of IR bands to include as shadows within TDOM and the
# shadow shift method (lower number masks out less)
irSumThresh = 0.3
ndviThresh = -0.1
# Pixels to reduce cloud mask and dark shadows by to reduce inclusion
# of single-pixel comission errors
erodePixels = 1.5
dilationPixels = 3

# images with less than this many cloud pixels will be used with normal
# mosaicing (most recent on top)
cloudFreeKeepThresh = 3

# removed B1, B9, B10
ALL_S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
REMOVED_BANDS = [item for item in ALL_S2_BANDS if item not in S2_BANDS]
S2_SHIFT_VALUES = [float(0.0)] * len(S2_BANDS)
S2_DIV_VALUES = [float(1e4)] * len(S2_BANDS)


def get_single_s2_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    """
    Builds a cloud-free Sentinel-2 mosaic for a given region and date range.

    This function filters the Sentinel-2 image collection by the specified region and date range, computes cloud and shadow scores for each image, and then mosaics the best (least cloudy/shadowed) pixels into a single image.

    Parameters:
        region (ee.Geometry): The region of interest to clip and filter images.
        start_date (date): The start date for filtering the image collection.
        end_date (date): The end date for filtering the image collection.

    Returns:
        ee.Image: A single, mostly cloud-free Sentinel-2 image mosaic for the specified region and date range.
    """
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()  # type: ignore
    endDate = ee.DateRange(dates).end()  # type: ignore
    imgC = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    # filter the image collection by date and region, 
    # add custom properties cloud score, shadow score, and quality score, 
    # and sort by built in property cloudiness
    imgC = (
        imgC.map(lambda x: x.clip(region))
        .map(lambda x: x.set("ROI", region))
        .map(computeS2CloudScore)
        .map(projectShadows)
        .map(computeQualityScore)
        .sort("CLOUDY_PIXEL_PERCENTAGE") # sort in case of multiple images with the same cloudShadowScore when building the mosaic
    )

    # has to be double to be compatible with the sentinel 1 imagery, which is in
    # float64
    cloudFree = imgC.qualityMosaic("cloudShadowScore").select(S2_BANDS).toDouble()

    return cloudFree


def rescale(img, exp, thresholds):
    """
    Linearly rescales the result of an image expression to the 0–1 range based on provided thresholds.

    Parameters:
        img (ee.Image): The Earth Engine image to operate on.
        exp (str): An expression string to evaluate on the image (e.g., 'img.B2' or 'img.B4 + img.B3 + img.B2').
        thresholds (list or tuple of float): A pair [min, max] specifying the range to normalize from.

    Returns:
        ee.Image: An image where the evaluated expression is rescaled such that the lower threshold maps to 0 and the upper threshold maps to 1. Values outside the range are linearly extrapolated.
    """
    return (
        img.expression(exp, {"img": img})
        .subtract(thresholds[0])
        .divide(thresholds[1] - thresholds[0])
    )


def computeQualityScore(img: ee.Image) -> ee.Image:
    """
    Computes a combined quality score for each pixel based on cloud and shadow scores.

    The function takes the maximum of the cloud and shadow scores, smooths it using a mean filter, and multiplies by -1 so that higher values indicate better quality (less cloud/shadow). The result is added as a new band called 'cloudShadowScore'.

    Parameters:
        img (ee.Image): The image to compute the quality score for. Must have 'cloudScore' and 'shadowScore' bands.

    Returns:
        ee.Image: The input image with an added 'cloudShadowScore' band.
    """
    score = img.select(["cloudScore"]).max(img.select(["shadowScore"]))

    score = score.reproject("EPSG:4326", None, 20).reduceNeighborhood(
        reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(5), optimization="boxcar"
    )

    score = score.multiply(-1)

    return img.addBands(score.rename("cloudShadowScore"))


def computeS2CloudScore(img: ee.Image) -> ee.Image:
    """
    Computes a cloud score for each pixel in a Sentinel-2 image.

    The function uses several spectral indicators (brightness in blue/cirrus bands, visible bands, moisture, and snow exclusion) to estimate the likelihood of each pixel being cloudy. The minimum of these indicators is taken as the cloud score, which is then smoothed and added as a new band called 'cloudScore'.

    Parameters:
        img (ee.Image): The Sentinel-2 image to compute the cloud score for.

    Returns:
        ee.Image: The input image with an added 'cloudScore' band.
    """
    toa = img.select(ALL_S2_BANDS).divide(10000)

    toa = toa.addBands(img.select(["QA60"]))

    # ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A',
    #  'B9',          'B10', 'B11','B12']
    # ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2',
    #  'waterVapor', 'cirrus','swir1', 'swir2']);

    # Compute several indicators of cloudyness and take the minimum of them.
    score = ee.Image(1) # Every pixel initiated with a score of 1. Rescale, used below, is used to convert to 0-1 range.

    # Clouds are reasonably bright in the blue and cirrus bands.
    score = score.min(rescale(toa, "img.B2", [0.1, 0.5]))
    score = score.min(rescale(toa, "img.B1", [0.1, 0.3]))
    score = score.min(rescale(toa, "img.B1 + img.B10", [0.15, 0.2]))

    # Clouds are reasonably bright in all visible bands.
    score = score.min(rescale(toa, "img.B4 + img.B3 + img.B2", [0.2, 0.8]))

    # Clouds are moist
    ndmi = img.normalizedDifference(["B8", "B11"])
    score = score.min(rescale(ndmi, "img", [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = img.normalizedDifference(["B3", "B11"])
    score = score.min(rescale(ndsi, "img", [0.8, 0.6]))

    # Clip the lower end of the score
    score = score.max(ee.Image(0.001))

    # score = score.multiply(dilated)
    score = score.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(5))

    return img.addBands(score.rename("cloudScore"))


def projectShadows(image: ee.Image) -> ee.Image:
    """
    Estimates and marks cloud shadows in a Sentinel-2 image.

    The function uses solar geometry and cloud mask information to project where cloud shadows are likely to fall. It identifies dark, non-water pixels as potential shadow areas, projects shadows based on sun angle and cloud height, and refines the shadow mask using morphological operations. The result is added as a new band called 'shadowScore'.

    Parameters:
        image (ee.Image): The image to compute cloud shadows for. Must have a 'cloudScore' band.

    Returns:
        ee.Image: The input image with an added 'shadowScore' band.
    """
    meanAzimuth = image.get("MEAN_SOLAR_AZIMUTH_ANGLE")
    meanZenith = image.get("MEAN_SOLAR_ZENITH_ANGLE")

    cloudMask = image.select(["cloudScore"]).gt(cloudThresh)

    # Find dark pixels
    darkPixelsImg = image.select(["B8", "B11", "B12"]).divide(10000).reduce(ee.Reducer.sum())

    ndvi = image.normalizedDifference(["B8", "B4"])
    waterMask = ndvi.lt(ndviThresh)

    darkPixels = darkPixelsImg.lt(irSumThresh)

    # Get the mask of pixels which might be shadows excluding water
    darkPixelMask = darkPixels.And(waterMask.Not())
    darkPixelMask = darkPixelMask.And(cloudMask.Not())

    # Find where cloud shadows should be based on solar geometry
    # Convert to radians
    azR = ee.Number(meanAzimuth).add(180).multiply(math.pi).divide(180.0)
    zenR = ee.Number(meanZenith).multiply(math.pi).divide(180.0)

    # Find the shadows
    def getShadows(cloudHeight):
        cloudHeight = ee.Number(cloudHeight)

        shadowCastedDistance = zenR.tan().multiply(cloudHeight)  # Distance shadow is cast
        x = azR.sin().multiply(shadowCastedDistance).multiply(-1)  # /X distance of shadow
        y = azR.cos().multiply(shadowCastedDistance).multiply(-1)  # Y distance of shadow
        return image.select(["cloudScore"]).displace(
            ee.Image.constant(x).addBands(ee.Image.constant(y))
        )

    shadows = ee.List(cloudHeights).map(getShadows)
    shadowMasks = ee.ImageCollection.fromImages(shadows)
    shadowMask = shadowMasks.mean()

    # Create shadow mask
    shadowMask = dilatedErosion(shadowMask.multiply(darkPixelMask))

    shadowScore = shadowMask.reduceNeighborhood(
        **{"reducer": ee.Reducer.max(), "kernel": ee.Kernel.square(1)}
    )

    return image.addBands(shadowScore.rename(["shadowScore"]))


def dilatedErosion(score: ee.Image) -> ee.Image:
    """
    Applies morphological opening (erosion followed by dilation) to a binary mask image to remove small noise and smooth regions.

    This function is typically used to clean up cloud or shadow masks in satellite imagery, reducing single-pixel errors and making detected regions more contiguous and robust.

    Parameters:
        score (ee.Image): A binary mask image (e.g., cloud or shadow mask) to be refined.

    Returns:
        ee.Image: The refined mask after erosion and dilation, with small noise removed and regions smoothed.
    """
    # Perform opening on the cloud scores
    def erode(img, distance):
        d = (
            img.Not()
            .unmask(1)
            .fastDistanceTransform(30)
            .sqrt()
            .multiply(ee.Image.pixelArea().sqrt())
        )
        return img.updateMask(d.gt(distance))

    def dilate(img, distance):
        d = img.fastDistanceTransform(30).sqrt().multiply(ee.Image.pixelArea().sqrt())
        return d.lt(distance)

    score = score.reproject("EPSG:4326", None, 20)
    score = erode(score, erodePixels)
    score = dilate(score, dilationPixels)

    return score.reproject("EPSG:4326", None, 20)


# def mergeCollection(imgC):
#     return imgC.qualityMosaic("cloudShadowScore")

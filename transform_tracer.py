import os

import pygeoprocessing
from osgeo import osr

bb = (
    -46.84910148382187, -22.782926137858535,
    -46.84873133897782, -22.782354558646723)
wgs84_srs = osr.SpatialReference()
wgs84_srs.ImportFromEPSG(4326)

quad_path = os.path.join(
    'training_set_workspace',
    'training_imagery',
    '4ce5863a-fb3f-4cad-a899-b8c053af1858_757-890.tif')

quad_info = pygeoprocessing.get_raster_info(quad_path)

local_bb = pygeoprocessing.transform_bounding_box(
    bb, wgs84_srs.ExportToWkt(), quad_info['projection'])
print(local_bb)

"""Script to separate out the South Africa dams."""
import os
import pickle
import re
import sqlite3

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import shapely.geometry
import shapely.prepared
import shapely.wkb

ANNOTATIONS_CSV_PATH = 'annotations.csv'
SA_FREE_CSV_PATH = 'no_ZAF_annotations.csv'
SA_CSV_PATH = 'ZAF_annotations.csv'
QUAD_DATABASE_PATH = os.path.join(
    'training_set_workspace', 'ecoshard',
    'quad_database_md5_12866cf27da2575f33652d197beb05d3.db')
COUNTRY_VECTOR_PATH = os.path.join(
    'training_set_workspace', 'ecoshard',
    'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg')

def main():
    """Entry point."""
    country_vector = gdal.OpenEx(COUNTRY_VECTOR_PATH, gdal.OF_VECTOR)
    layer = country_vector.ExecuteSQL(
        "SELECT * FROM countries_iso3 WHERE iso3 = 'ZAF'")
    feature = next(iter(layer))
    geom_ref = feature.GetGeometryRef().Clone()
    print(feature.GetField('iso3'))
    shapely_geom = shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb())
    print(shapely_geom.bounds)
    feature = None
    layer = None
    country_vector = None

    quad_set = set()
    with open(ANNOTATIONS_CSV_PATH, 'r') as annotations_csv_file:
        annotation_lines = annotations_csv_file.read()
    line_count = 0
    for line in annotation_lines.split('\n'):
        line_count += 1
        quad_id = re.match(r'.*\d+_(\d+-\d+)_\d+\.png', line)
        if quad_id:
            quad_set.add(quad_id.group(1))

    string_set_of_quad_ids = ', '.join(["'%s'" % x for x in quad_set])
    connection = sqlite3.connect(QUAD_DATABASE_PATH)
    cursor = connection.execute(
        '''
        SELECT quad_id, bounding_box
        FROM bounding_box_to_mosaic
        WHERE quad_id IN (%s)
        ''' % string_set_of_quad_ids)  # [(string_set_of_quad_ids,)])
    zaf_quad_set = set()
    quad_count = 0
    intersection_count = 0

    # make a debug gpkg
    # add south africa polygon
    # then ad all the bounding box polygons

    gpkg_driver = ogr.GetDriverByName('GPKG')
    vector = gpkg_driver.CreateDataSource('debug.gpkg')
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    layer = vector.CreateLayer('debug', wgs84_srs, geom_type=ogr.wkbMultiPolygon)
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(ogr.CreateGeometryFromWkb(shapely_geom.wkb))
    layer.CreateFeature(feature)

    layer.StartTransaction()
    for quad_id, bounding_box in cursor:
        box = shapely.geometry.box(*pickle.loads(bounding_box))
        ogr_box = ogr.CreateGeometryFromWkb(box.wkb)
        quad_count += 1
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkb(box.wkb))
        layer.CreateFeature(feature)
        if geom_ref.Intersects(ogr_box):
            zaf_quad_set.add(quad_id)
            intersection_count += 1
    layer.CommitTransaction()
    print('%d %d %d %d' % (line_count, len(quad_set), quad_count, intersection_count))

    with open(SA_CSV_PATH, 'w') as sa_csv_file, \
            open(SA_FREE_CSV_PATH, 'w') as sa_free_csv_file:
        for line in annotation_lines.split('\n'):
            if not line:
                continue
            quad_id = re.match(r'.*\d+_(\d+-\d+)_\d+\.png', line)
            if quad_id and quad_id.group(1) in zaf_quad_set:
                sa_csv_file.write('%s\n' % line)
            else:
                sa_free_csv_file.write('%s\n' % line)


if __name__ == '__main__':
    main()

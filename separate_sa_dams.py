"""Script to separate out the South Africa dams."""
import os
import pickle
import re
import sqlite3

from osgeo import gdal
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
    geom = shapely.prepared.prep(
        shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb()))
    feature = None
    layer = None
    country_vector = None

    quad_set = set()
    with open(ANNOTATIONS_CSV_PATH, 'r') as annotations_csv_file:
        annotation_lines = annotations_csv_file.read()
    for line in annotation_lines.split('\n'):
        quad_id = re.match(r'.*\d+_(\d+-\d+)_\d+\.png', line)
        if quad_id:
            quad_set.add(quad_id.group(1))

    string_set_of_quad_ids = ', '.join(["'%s'" % x for x in quad_set])
    #print(string_set_of_quad_ids)
    connection = sqlite3.connect(QUAD_DATABASE_PATH)
    cursor = connection.execute(
        '''
        SELECT quad_id, bounding_box
        FROM bounding_box_to_mosaic
        WHERE quad_id IN (%s)
        ''' % string_set_of_quad_ids)  # [(string_set_of_quad_ids,)])
    zaf_quad_set = set()
    for quad_id, bounding_box in cursor:
        box = shapely.geometry.box(*pickle.loads(bounding_box))
        if geom.intersects(box):
            zaf_quad_set.add(quad_id)

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

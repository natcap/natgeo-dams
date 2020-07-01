"""Script to analyze how many dams found and how many pre-known."""
import logging
import os
import pathlib
import sqlite3
import sys

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import rtree
import shapely.wkb

KNOWN_DAMS_VECTOR_PATH = r"C:\Users\richp\Downloads\south_africa_known_dams.gpkg"
NATGEO_DETECTED_DAMS_DB_PATH = r"C:\Users\richp\Documents\annotated_dams\natgeo_dams_database.db"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

COUNTRY_PRIORITIES = []


def _execute_sqlite(
        sqlite_command, database_path, argument_list=None,
        mode='read_only', execute='execute', fetch=None):
    """Execute SQLite command and attempt retries on a failure.

    Parameters:
        sqlite_command (str): a well formatted SQLite command.
        database_path (str): path to the SQLite database to operate on.
        argument_list (list): `execute == 'execute` then this list is passed to
            the internal sqlite3 `execute` call.
        mode (str): must be either 'read_only' or 'modify'.
        execute (str): must be either 'execute', 'many', or 'script'.
        fetch (str): if not `None` can be either 'all' or 'one'.
            If not None the result of a fetch will be returned by this
            function.

    Returns:
        result of fetch if `fetch` is not None.

    """
    cursor = None
    connection = None
    try:
        if mode == 'read_only':
            ro_uri = r'%s?mode=ro' % pathlib.Path(
                os.path.abspath(database_path)).as_uri()
            LOGGER.debug(
                '%s exists: %s', ro_uri, os.path.exists(os.path.abspath(
                    database_path)))
            connection = sqlite3.connect(ro_uri, uri=True)
        elif mode == 'modify':
            connection = sqlite3.connect(database_path)
        else:
            raise ValueError('Unknown mode: %s' % mode)

        if execute == 'execute':
            cursor = connection.execute(sqlite_command, argument_list)
        elif execute == 'many':
            cursor = connection.executemany(sqlite_command, argument_list)
        elif execute == 'script':
            cursor = connection.executescript(sqlite_command)
        else:
            raise ValueError('Unknown execute mode: %s' % execute)

        result = None
        payload = None
        if fetch == 'all':
            payload = (cursor.fetchall())
        elif fetch == 'one':
            payload = (cursor.fetchone())
        elif fetch is not None:
            raise ValueError('Unknown fetch mode: %s' % fetch)
        if payload is not None:
            result = list(payload)
        cursor.close()
        connection.commit()
        connection.close()
        return result
    except Exception:
        LOGGER.exception('Exception on _execute_sqlite: %s', sqlite_command)
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.commit()
            connection.close()
        raise


def main():
    bounding_box_list = _execute_sqlite(
        '''
        SELECT lng_min, lat_min, lng_max, lat_max
        FROM detected_dams
        GROUP BY lng_min, lat_min, lng_max, lat_max
        ''', NATGEO_DETECTED_DAMS_DB_PATH, fetch='all', argument_list=[])
    gpkg_driver = ogr.GetDriverByName('GPKG')
    vector = gpkg_driver.CreateDataSource('all_found.gpkg')
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    layer = vector.CreateLayer(
        'all_found', wgs84_srs, geom_type=ogr.wkbPolygon)

    layer.StartTransaction()
    for index, (lng_min, lat_min, lng_max, lat_max) in enumerate(
            bounding_box_list):
        box = shapely.geometry.box(lng_min, lat_min, lng_max, lat_max)
        ogr_box = ogr.CreateGeometryFromWkb(box.wkb)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkb(box.wkb))
        layer.CreateFeature(feature)
    layer.CommitTransaction()

    return

    for country_iso in COUNTRY_PRIORITIES + ['']:
        LOGGER.debug(country_iso)
        bounding_box_list = _execute_sqlite(
            '''
            SELECT lng_min, lat_min, lng_max, lat_max
            FROM detected_dams
            WHERE country_list LIKE ?
            GROUP BY lng_min, lat_min, lng_max, lat_max
            ''', NATGEO_DETECTED_DAMS_DB_PATH, fetch='all',
            argument_list=['%%%s%%' % country_iso])
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector = gpkg_driver.CreateDataSource('%s_found.gpkg' % country_iso)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer(
            '%s_found' % country_iso, wgs84_srs, geom_type=ogr.wkbPolygon)

        layer.StartTransaction()
        for index, (lng_min, lat_min, lng_max, lat_max) in enumerate(
                bounding_box_list):
            box = shapely.geometry.box(lng_min, lat_min, lng_max, lat_max)
            ogr_box = ogr.CreateGeometryFromWkb(box.wkb)
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(ogr.CreateGeometryFromWkb(box.wkb))
            layer.CreateFeature(feature)
        layer.CommitTransaction()

    return
    bounding_box_list = _execute_sqlite(
        '''
        SELECT lng_min, lat_min, lng_max, lat_max
        FROM detected_dams
        WHERE country_list LIKE '%ZAF%'
        GROUP BY lng_min, lat_min, lng_max, lat_max
        ''', NATGEO_DETECTED_DAMS_DB_PATH, fetch='all', argument_list=[])
    LOGGER.debug("build rtree")

    gpkg_driver = ogr.GetDriverByName('GPKG')
    vector = gpkg_driver.CreateDataSource('sa_found.gpkg')
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    layer = vector.CreateLayer('sa_found', wgs84_srs, geom_type=ogr.wkbPolygon)

    layer.StartTransaction()
    zaf_index = rtree.index.Index()
    for index, (lng_min, lat_min, lng_max, lat_max) in enumerate(
            bounding_box_list):
        box = shapely.geometry.box(lng_min, lat_min, lng_max, lat_max)
        ogr_box = ogr.CreateGeometryFromWkb(box.wkb)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkb(box.wkb))
        layer.CreateFeature(feature)
        zaf_index.insert(index, (lng_min, lat_min, lng_max, lat_max))
    layer.CommitTransaction()

    known_dams_vector = gdal.OpenEx(KNOWN_DAMS_VECTOR_PATH, gdal.OF_VECTOR)
    known_dams_layer = known_dams_vector.GetLayer()
    known_dams_count = 0
    found_dams_count = 0
    for known_dam_feature in known_dams_layer:
        known_dam_geometry = known_dam_feature.GetGeometryRef()
        known_dams_count += 1
        known_dam_box = shapely.wkb.loads(known_dam_geometry.ExportToWkb())
        if list(zaf_index.intersection(known_dam_box.bounds)):
            found_dams_count += 1
    LOGGER.debug('%d: %d', known_dams_count, found_dams_count)


if __name__ == '__main__':
    main()

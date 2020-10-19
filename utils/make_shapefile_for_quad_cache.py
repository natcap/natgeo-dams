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

TARGET_VECTOR_PATH = r"quad_cache.gpkg"
BASE_DAMS_DB_PATH = r"C:\Users\richp\Documents\quad_uri.db"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


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
    gpkg_driver = ogr.GetDriverByName('GPKG')
    vector = gpkg_driver.CreateDataSource('cached_quads.gpkg')
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    layer = vector.CreateLayer(
        'cached_quads', wgs84_srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("quad_type", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTString))

    for quad_type, id_field, table_name in [
            ('grid', 'grid_id', 'processed_grid_table'),
            ('quad', 'quad_id', 'quad_cache_table')]:
        LOGGER.debug(quad_type)
        grid_query_result = _execute_sqlite(
            f'''
            SELECT {id_field}, long_min, lat_min, long_max, lat_max
            FROM {table_name}
            ''', BASE_DAMS_DB_PATH, fetch='all', argument_list=[])
        layer.StartTransaction()
        for index, (id_value, lng_min, lat_min, lng_max, lat_max) in enumerate(
                grid_query_result):
            if index % 1000 == 0:
                LOGGER.info(f'{index/len(grid_query_result)*100:.2f}% complete')
            box = shapely.geometry.box(lng_min, lat_min, lng_max, lat_max)
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField('quad_type', quad_type)
            if quad_type == 'grid':
                feature.SetField('id', id_value)
            else:
                quad_url = (
                    f'https://storage.googleapis.com/natgeo-dams-data/'
                    f'cached-planet-quads/{id_value}.tif')
                feature.SetField('id', quad_url)
            feature.SetGeometry(ogr.CreateGeometryFromWkb(box.wkb))
            layer.CreateFeature(feature)
        layer.CommitTransaction()


if __name__ == '__main__':
    main()

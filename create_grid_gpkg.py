"""Create a grid shapefile from database grids."""
import logging
import os
import pathlib
import sqlite3

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import shapely.geometry
import shapely.prepared
import shapely.ops
import shapely.wkb

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


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


if __name__ == '__main__':
    report_db_path = 'find-the-dams_report_md5_e12d71eb5026751217bd286e9b4c2afe.db'

    grid_id_to_quad_id_db_path = 'planet_cell_to_grid_md5_eb607fdb74a6278e9597fddeb59b58c1.db'

    processed_grid_ids = _execute_sqlite(
        """
        SELECT grid_id
        FROM grid_id_to_quad_id
        """, grid_id_to_quad_id_db_path, fetch='all', argument_list=[])

    processed_grid_set = {x[0] for x in processed_grid_ids}

    processing_state_query = _execute_sqlite(
        """
        SELECT grid_id, processing_state, lat_min, lng_min, lat_max, lng_max
        FROM grid_status
        """, report_db_path, argument_list=[],
        mode='read_only', execute='execute', fetch='all')

    # TODO: make a new grid shapefile
    #   * up to 60N OR Norway, Sweden, Finland and Iceland
    #   * if grid ID in grid_id_to_quad_id set to 'complete' unless it's
    #       specified in a special set.

    LOGGER.info('prep the country lookup structure')
    avoid_countries = set(['ATA', 'GRL'])
    country_vector_path = 'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg'
    country_vector = gdal.OpenEx(country_vector_path, gdal.OF_VECTOR)
    country_layer = country_vector.GetLayer()
    all_country_geom_list = []
    for country_feature in country_layer:
        country_geom = country_feature.GetGeometryRef()
        country_shp = shapely.wkb.loads(country_geom.ExportToWkb())
        country_iso = country_feature.GetField('iso3')
        if country_iso not in avoid_countries:
            all_country_geom_list.append(country_shp)
    LOGGER.info('prep the geometry for fast intersection')
    all_country_union = shapely.ops.cascaded_union(all_country_geom_list)
    all_country_prep = shapely.prepared.prep(all_country_union)

    gpkg_driver = ogr.GetDriverByName('GPKG')
    vector = gpkg_driver.CreateDataSource('processing_grid.gpkg')
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    layer = vector.CreateLayer(
        'processing_grid', wgs84_srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("processing_state", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("grid_id", ogr.OFTInteger))

    LOGGER.info('make the grid')
    layer.StartTransaction()
    for lat in range(90, -90, -1):
        for lng in range(-180, 180):
            box = shapely.geometry.box(lng, lat-1, lng+1, lat)
            if not all_country_prep.intersects(box):
                continue
            grid_id = (lat+90)*360+lng+180
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField('grid_id', grid_id)
            feature.SetGeometry(ogr.CreateGeometryFromWkb(box.wkb))
            layer.CreateFeature(feature)
    layer.CommitTransaction()

    LOGGER.info('done')

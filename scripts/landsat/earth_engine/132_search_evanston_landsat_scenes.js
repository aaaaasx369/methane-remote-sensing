// Auto-generated Evanston Landsat scene search
// Paste this file into the Google Earth Engine Code Editor.

var allResults = ee.FeatureCollection([]);

// EV_OP_001
var point_ev_op_001 = ee.Geometry.Point([-110.93061, 41.276075]);
var start_ev_op_001 = ee.Date('2024-08-19').advance(-1, 'day');
var end_ev_op_001 = ee.Date('2024-08-19').advance(2, 'day');
var collection_ev_op_001 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(point_ev_op_001)
  .filterDate(start_ev_op_001, end_ev_op_001);

var features_ev_op_001 = collection_ev_op_001.map(function(image) {
  return ee.Feature(null, {
    overpass_id: "EV_OP_001",
    release_date: "2024-08-19",
    expected_sensor: "Landsat-8",
    flow_max_kg_h: 697.0094043428571,
    release_rows: 1,
    representative_release_id: "08192024_LS8",
    system_index: image.get('system:index'),
    landsat_product_id: image.get('LANDSAT_PRODUCT_ID'),
    spacecraft_id: image.get('SPACECRAFT_ID'),
    acquisition_time_utc: ee.Date(
      image.get('system:time_start')
    ).format('YYYY-MM-dd HH:mm:ss'),
    cloud_cover: image.get('CLOUD_COVER'),
    wrs_path: image.get('WRS_PATH'),
    wrs_row: image.get('WRS_ROW'),
    collection_category: image.get('COLLECTION_CATEGORY'),
    collection_number: image.get('COLLECTION_NUMBER')
  });
});

allResults = allResults.merge(features_ev_op_001);

// EV_OP_002
var point_ev_op_002 = ee.Geometry.Point([-110.93061, 41.276075]);
var start_ev_op_002 = ee.Date('2024-08-27').advance(-1, 'day');
var end_ev_op_002 = ee.Date('2024-08-27').advance(2, 'day');
var collection_ev_op_002 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(point_ev_op_002)
  .filterDate(start_ev_op_002, end_ev_op_002);

var features_ev_op_002 = collection_ev_op_002.map(function(image) {
  return ee.Feature(null, {
    overpass_id: "EV_OP_002",
    release_date: "2024-08-27",
    expected_sensor: "Landsat-9",
    flow_max_kg_h: 1050.4128894916944,
    release_rows: 1,
    representative_release_id: "08272024_LS9",
    system_index: image.get('system:index'),
    landsat_product_id: image.get('LANDSAT_PRODUCT_ID'),
    spacecraft_id: image.get('SPACECRAFT_ID'),
    acquisition_time_utc: ee.Date(
      image.get('system:time_start')
    ).format('YYYY-MM-dd HH:mm:ss'),
    cloud_cover: image.get('CLOUD_COVER'),
    wrs_path: image.get('WRS_PATH'),
    wrs_row: image.get('WRS_ROW'),
    collection_category: image.get('COLLECTION_CATEGORY'),
    collection_number: image.get('COLLECTION_NUMBER')
  });
});

allResults = allResults.merge(features_ev_op_002);

// EV_OP_003
var point_ev_op_003 = ee.Geometry.Point([-110.93061, 41.276075]);
var start_ev_op_003 = ee.Date('2024-09-03').advance(-1, 'day');
var end_ev_op_003 = ee.Date('2024-09-03').advance(2, 'day');
var collection_ev_op_003 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(point_ev_op_003)
  .filterDate(start_ev_op_003, end_ev_op_003);

var features_ev_op_003 = collection_ev_op_003.map(function(image) {
  return ee.Feature(null, {
    overpass_id: "EV_OP_003",
    release_date: "2024-09-03",
    expected_sensor: "Landsat-9",
    flow_max_kg_h: 1019.3258455933552,
    release_rows: 1,
    representative_release_id: "09032024_LS9",
    system_index: image.get('system:index'),
    landsat_product_id: image.get('LANDSAT_PRODUCT_ID'),
    spacecraft_id: image.get('SPACECRAFT_ID'),
    acquisition_time_utc: ee.Date(
      image.get('system:time_start')
    ).format('YYYY-MM-dd HH:mm:ss'),
    cloud_cover: image.get('CLOUD_COVER'),
    wrs_path: image.get('WRS_PATH'),
    wrs_row: image.get('WRS_ROW'),
    collection_category: image.get('COLLECTION_CATEGORY'),
    collection_number: image.get('COLLECTION_NUMBER')
  });
});

allResults = allResults.merge(features_ev_op_003);

// EV_OP_004
var point_ev_op_004 = ee.Geometry.Point([-110.93061, 41.276075]);
var start_ev_op_004 = ee.Date('2024-09-12').advance(-1, 'day');
var end_ev_op_004 = ee.Date('2024-09-12').advance(2, 'day');
var collection_ev_op_004 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(point_ev_op_004)
  .filterDate(start_ev_op_004, end_ev_op_004);

var features_ev_op_004 = collection_ev_op_004.map(function(image) {
  return ee.Feature(null, {
    overpass_id: "EV_OP_004",
    release_date: "2024-09-12",
    expected_sensor: "Landsat-9",
    flow_max_kg_h: 324.7881713621262,
    release_rows: 1,
    representative_release_id: "09122024_LS9",
    system_index: image.get('system:index'),
    landsat_product_id: image.get('LANDSAT_PRODUCT_ID'),
    spacecraft_id: image.get('SPACECRAFT_ID'),
    acquisition_time_utc: ee.Date(
      image.get('system:time_start')
    ).format('YYYY-MM-dd HH:mm:ss'),
    cloud_cover: image.get('CLOUD_COVER'),
    wrs_path: image.get('WRS_PATH'),
    wrs_row: image.get('WRS_ROW'),
    collection_category: image.get('COLLECTION_CATEGORY'),
    collection_number: image.get('COLLECTION_NUMBER')
  });
});

allResults = allResults.merge(features_ev_op_004);

// EV_OP_005
var point_ev_op_005 = ee.Geometry.Point([-110.93061, 41.276075]);
var start_ev_op_005 = ee.Date('2024-09-20').advance(-1, 'day');
var end_ev_op_005 = ee.Date('2024-09-20').advance(2, 'day');
var collection_ev_op_005 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(point_ev_op_005)
  .filterDate(start_ev_op_005, end_ev_op_005);

var features_ev_op_005 = collection_ev_op_005.map(function(image) {
  return ee.Feature(null, {
    overpass_id: "EV_OP_005",
    release_date: "2024-09-20",
    expected_sensor: "Landsat-8",
    flow_max_kg_h: 605.1554518325581,
    release_rows: 1,
    representative_release_id: "09202024_LS8",
    system_index: image.get('system:index'),
    landsat_product_id: image.get('LANDSAT_PRODUCT_ID'),
    spacecraft_id: image.get('SPACECRAFT_ID'),
    acquisition_time_utc: ee.Date(
      image.get('system:time_start')
    ).format('YYYY-MM-dd HH:mm:ss'),
    cloud_cover: image.get('CLOUD_COVER'),
    wrs_path: image.get('WRS_PATH'),
    wrs_row: image.get('WRS_ROW'),
    collection_category: image.get('COLLECTION_CATEGORY'),
    collection_number: image.get('COLLECTION_NUMBER')
  });
});

allResults = allResults.merge(features_ev_op_005);

// EV_OP_006
var point_ev_op_006 = ee.Geometry.Point([-110.93061, 41.276075]);
var start_ev_op_006 = ee.Date('2024-10-14').advance(-1, 'day');
var end_ev_op_006 = ee.Date('2024-10-14').advance(2, 'day');
var collection_ev_op_006 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(point_ev_op_006)
  .filterDate(start_ev_op_006, end_ev_op_006);

var features_ev_op_006 = collection_ev_op_006.map(function(image) {
  return ee.Feature(null, {
    overpass_id: "EV_OP_006",
    release_date: "2024-10-14",
    expected_sensor: "Landsat-9",
    flow_max_kg_h: 731.6447507614618,
    release_rows: 1,
    representative_release_id: "10142024_LS9",
    system_index: image.get('system:index'),
    landsat_product_id: image.get('LANDSAT_PRODUCT_ID'),
    spacecraft_id: image.get('SPACECRAFT_ID'),
    acquisition_time_utc: ee.Date(
      image.get('system:time_start')
    ).format('YYYY-MM-dd HH:mm:ss'),
    cloud_cover: image.get('CLOUD_COVER'),
    wrs_path: image.get('WRS_PATH'),
    wrs_row: image.get('WRS_ROW'),
    collection_category: image.get('COLLECTION_CATEGORY'),
    collection_number: image.get('COLLECTION_NUMBER')
  });
});

allResults = allResults.merge(features_ev_op_006);

print('Evanston Landsat scene candidates', allResults);
print('Candidate count', allResults.size());

Export.table.toDrive({
  collection: allResults,
  description: 'evanston_landsat_scene_candidates',
  fileNamePrefix: 'evanston_landsat_scene_candidates',
  fileFormat: 'CSV'
});

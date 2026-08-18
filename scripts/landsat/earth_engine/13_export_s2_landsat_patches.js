
// Export Sentinel-2 and Landsat image patches for methane classification dataset.
// Generated from outputs/12_dataset_candidate_events.csv

var events = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_PRISMA_1", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-16T18:36:26Z", "date_utc": "2021-10-16", "label": "1", "emission_tph_max": "3.778"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_Unknown_2", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-16T18:36:26Z", "date_utc": "2021-10-16", "label": "1", "emission_tph_max": "3.778"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_GHGSat_1", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-18T18:15:05Z", "date_utc": "2021-10-18", "label": "1", "emission_tph_max": "1.511974470095427"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_Sentinel_2_1", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-19T18:24:58Z", "date_utc": "2021-10-19", "label": "1", "emission_tph_max": "7.234"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_GHGSat_2", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-19T19:08:16Z", "date_utc": "2021-10-19", "label": "1", "emission_tph_max": "7.738"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_GHGSat_3", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-20T19:22:41Z", "date_utc": "2021-10-20", "label": "1", "emission_tph_max": "0.242"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_Landsat_8_1", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-21T18:10:37Z", "date_utc": "2021-10-21", "label": "1", "emission_tph_max": "4.085"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_PRISMA_2", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-21T18:23:12Z", "date_utc": "2021-10-21", "label": "1", "emission_tph_max": "6.469"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_GHGSat_4", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-21T19:06:45Z", "date_utc": "2021-10-21", "label": "1", "emission_tph_max": "4.539737508117431"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_Sentinel_2_2", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-22T18:34:54Z", "date_utc": "2021-10-22", "label": "1", "emission_tph_max": "3.005"})
]);

Map.centerObject(events, 8);
Map.addLayer(events, {color: 'red'}, 'candidate methane events');

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');

var WINDOW_HOURS = 24;
var PATCH_RADIUS_METERS = 1000;
var DRIVE_FOLDER = 'methane_image_patches';

var S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var LANDSAT_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];

var eventList = events.toList(events.size());
var n = events.size().getInfo();

print('Number of events to export', n);

for (var i = 0; i < n; i++) {
  var event = ee.Feature(eventList.get(i));
  var point = event.geometry();
  var region = point.buffer(PATCH_RADIUS_METERS).bounds();

  var t = ee.Date(event.get('datetime_utc'));
  var start = t.advance(-WINDOW_HOURS, 'hour');
  var end = t.advance(WINDOW_HOURS, 'hour');

  var eventId = event.get('event_id').getInfo();
  var label = event.get('label').getInfo();

  var s2 = S2
    .filterBounds(point)
    .filterDate(start, end)
    .sort('CLOUDY_PIXEL_PERCENTAGE');

  if (s2.size().getInfo() > 0) {
    var s2Img = ee.Image(s2.first()).select(S2_BANDS);

    Export.image.toDrive({
      image: s2Img,
      description: 'S2_' + eventId + '_label_' + label,
      folder: DRIVE_FOLDER,
      fileNamePrefix: 'S2_' + eventId + '_label_' + label,
      region: region,
      scale: 20,
      maxPixels: 1e9
    });
  }

  var landsat = L8.merge(L9)
    .filterBounds(point)
    .filterDate(start, end)
    .sort('CLOUD_COVER');

  if (landsat.size().getInfo() > 0) {
    var lsImg = ee.Image(landsat.first()).select(LANDSAT_BANDS);

    Export.image.toDrive({
      image: lsImg,
      description: 'Landsat_' + eventId + '_label_' + label,
      folder: DRIVE_FOLDER,
      fileNamePrefix: 'Landsat_' + eventId + '_label_' + label,
      region: region,
      scale: 30,
      maxPixels: 1e9
    });
  }
}

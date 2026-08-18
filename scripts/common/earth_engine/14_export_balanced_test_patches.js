
// Balanced test export: 5 positive + 5 negative methane events.

var events = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_PRISMA_1", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-16T18:36:26Z", "date_utc": "2021-10-16", "label": "1", "emission_tph_max": "3.778"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_Unknown_2", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-16T18:36:26Z", "date_utc": "2021-10-16", "label": "1", "emission_tph_max": "3.778"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_GHGSat_1", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-18T18:15:05Z", "date_utc": "2021-10-18", "label": "1", "emission_tph_max": "1.511974470095427"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_Sentinel_2_1", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-19T18:24:58Z", "date_utc": "2021-10-19", "label": "1", "emission_tph_max": "7.234"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_GHGSat_2", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-10-19T19:08:16Z", "date_utc": "2021-10-19", "label": "1", "emission_tph_max": "7.738"}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_PRISMA_4", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-11-02T18:29:00Z", "date_utc": "2021-11-02", "label": "0", "emission_tph_max": ""}),
  ee.Feature(ee.Geometry.Point([-114.48915, 33.630645]), {"event_id": "2023_Scientific_Reports_Ehrenberg_AZ_release_stack_GHGSat_C2_5", "paper": "2023_Scientific_Reports", "datetime_utc": "2021-11-03T17:22:00Z", "date_utc": "2021-11-03", "label": "0", "emission_tph_max": ""}),
  ee.Feature(ee.Geometry.Point([-111.785773, 32.8218205]), {"event_id": "2024_AMT_Casa_Grande_AZ_release_stacks_WorldView_3_3", "paper": "2024_AMT", "datetime_utc": "2022-10-17T11:27:21Z", "date_utc": "2022-10-17", "label": "0", "emission_tph_max": "0.0"}),
  ee.Feature(ee.Geometry.Point([-111.785773, 32.8218205]), {"event_id": "2024_AMT_Casa_Grande_AZ_release_stacks_PRISMA_3", "paper": "2024_AMT", "datetime_utc": "2022-10-21T18:19:00Z", "date_utc": "2022-10-21", "label": "0", "emission_tph_max": "0.0"}),
  ee.Feature(ee.Geometry.Point([-111.785773, 32.8218205]), {"event_id": "2024_AMT_Casa_Grande_AZ_release_stacks_GHGSat_CX_4", "paper": "2024_AMT", "datetime_utc": "2022-10-21T21:00:12Z", "date_utc": "2022-10-21", "label": "0", "emission_tph_max": ""})
]);

Map.centerObject(events, 8);
Map.addLayer(events, {color: 'red'}, 'balanced methane events');

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

print('Number of balanced events to export', n);

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

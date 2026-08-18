// MethaneAIR official inventory export for the Earth Engine Code Editor.
// Access may require the MethaneSAT request form.

var l4 = ee.FeatureCollection(
  'projects/edf-methanesat-ee/assets/mair/L4point'
);

var l3 = ee.ImageCollection(
  'projects/edf-methanesat-ee/assets/mair/L3concentration'
);

print('MethaneAIR L4 point-source count', l4.size());
print('MethaneAIR L4 first record', l4.first());
print('MethaneAIR L3 image count', l3.size());
print('MethaneAIR L3 first image', l3.first());

Export.table.toDrive({
  collection: l4,
  description: 'methaneair_l4_all_points',
  fileNamePrefix: 'methaneair_l4_all_points',
  fileFormat: 'CSV'
});

// L3 inventory without exporting the large raster pixels.
var l3List = l3.toList(l3.size());
var indices = ee.List.sequence(0, l3.size().subtract(1));

var metadata = ee.FeatureCollection(indices.map(function(i) {
  var image = ee.Image(l3List.get(i));
  return ee.Feature(null, image.toDictionary([
    'system:index',
    'system:time_start',
    'flight_id',
    'target_id',
    'time_coverage_start',
    'time_coverage_end',
    'processing_id'
  ]));
}));

Export.table.toDrive({
  collection: metadata,
  description: 'methaneair_l3_inventory',
  fileNamePrefix: 'methaneair_l3_inventory',
  fileFormat: 'CSV'
});

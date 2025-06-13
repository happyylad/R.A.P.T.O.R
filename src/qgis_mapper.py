# R.A.P.T.O.R QGIS Mapper Module - FIXED VERSION
# File: src/qgis_mapper.py

import sys
import os

# Add QGIS Python paths for Windows (if QGIS is installed)
qgis_path = r"C:\Program Files\QGIS 3.40.7\apps\qgis\python"
qgis_plugins = r"C:\Program Files\QGIS 3.40.7\apps\qgis\python\plugins"

if os.path.exists(qgis_path):
    sys.path.insert(0, qgis_path)
    sys.path.insert(0, qgis_plugins)
    
    # Set QGIS environment variables
    os.environ['QGIS_PREFIX_PATH'] = r"C:\Program Files\QGIS 3.40.7"
    os.environ['PATH'] = r"C:\Program Files\QGIS 3.40.7\bin" + os.pathsep + os.environ['PATH']
    os.environ['QT_PLUGIN_PATH'] = r"C:\Program Files\QGIS 3.40.7\apps\qgis\qtplugins"

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Try to import QGIS modules
try:
    from qgis.core import *
    from qgis.analysis import QgsNativeAlgorithms
    import processing
    QGIS_AVAILABLE = True
    print("✅ QGIS modules loaded successfully!")
except ImportError:
    print("⚠️ QGIS not available, using standalone mapping")
    QGIS_AVAILABLE = False

class TacticalQGISMapper:
    def __init__(self):
        """R.A.P.T.O.R QGIS Integration and Mapping System"""
        self.detections = []
        self.layers = {}
        
        if QGIS_AVAILABLE:
            try:
                # Initialize QGIS application
                QgsApplication.setPrefixPath(r"C:\Program Files\QGIS 3.40.7", True)
                self.qgs = QgsApplication([], False)
                self.qgs.initQgis()
                
                self.project = QgsProject.instance()
                self.crs = QgsCoordinateReferenceSystem("EPSG:4326")  # WGS84
                
                # Add processing algorithms
                QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
                print("✅ QGIS initialized successfully!")
            except Exception as e:
                print(f"⚠️ QGIS initialization failed: {e}")
                QGIS_AVAILABLE = False
        
        # Class styling for different object types
        self.class_styles = {
            'person': {'color': '#00ff41', 'size': 8, 'symbol': 'circle'},
            'car': {'color': '#0080ff', 'size': 6, 'symbol': 'square'},
            'truck': {'color': '#ff4444', 'size': 10, 'symbol': 'triangle'},
            'bus': {'color': '#ffaa00', 'size': 12, 'symbol': 'diamond'},
            'motorcycle': {'color': '#ff00ff', 'size': 5, 'symbol': 'star'},
            'bird': {'color': '#ffff00', 'size': 4, 'symbol': 'circle'},
            'cat': {'color': '#ff69b4', 'size': 4, 'symbol': 'circle'},
            'dog': {'color': '#8b4513', 'size': 5, 'symbol': 'circle'}
        }
    
    def load_detections(self, json_file):
        """Load detections from JSON file"""
        try:
            with open(json_file, 'r') as f:
                self.detections = json.load(f)
            
            # Filter only detections with GPS
            self.detections = [d for d in self.detections if 'gps' in d and d['gps']]
            print(f"✅ Loaded {len(self.detections)} detections with GPS coordinates")
            return True
        except Exception as e:
            print(f"❌ Failed to load detections: {e}")
            return False
    
    def create_shapefile_layers(self, output_dir='output/maps/shapefiles'):
        """Create shapefile layers for each object class"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.detections:
            print("⚠️ No detections to export")
            return []
        
        # Group detections by class
        df = pd.DataFrame(self.detections)
        
        created_files = []
        for object_class in df['class'].unique():
            class_detections = df[df['class'] == object_class]
            
            # Create shapefile
            shapefile_path = os.path.join(output_dir, f'{object_class}_detections.shp')
            success = self.create_point_shapefile(class_detections, shapefile_path, object_class)
            
            if success:
                created_files.append(shapefile_path)
        
        print(f"✅ Created {len(created_files)} shapefile layers")
        return created_files
    
    def create_point_shapefile(self, detections_df, output_path, object_class):
        """Create point shapefile for detection class"""
        if QGIS_AVAILABLE:
            try:
                # Create vector layer
                layer = QgsVectorLayer(
                    "Point?crs=EPSG:4326&field=id:integer&field=class:string&field=confidence:double&field=timestamp:string&field=frame:integer",
                    f"{object_class}_detections",
                    "memory"
                )
                
                provider = layer.dataProvider()
                
                # Add features
                features = []
                for idx, detection in detections_df.iterrows():
                    feature = QgsFeature()
                    
                    # Set geometry
                    point = QgsGeometry.fromPointXY(
                        QgsPointXY(detection['gps']['lon'], detection['gps']['lat'])
                    )
                    feature.setGeometry(point)
                    
                    # Set attributes
                    feature.setAttributes([
                        idx,
                        detection['class'],
                        detection['confidence'],
                        detection['timestamp'],
                        detection.get('frame', 0)
                    ])
                    
                    features.append(feature)
                
                provider.addFeatures(features)
                layer.updateExtents()
                
                # Style the layer
                self.style_layer(layer, object_class)
                
                # Save as shapefile
                error = QgsVectorFileWriter.writeAsVectorFormat(
                    layer,
                    output_path,
                    "utf-8",
                    self.crs,
                    "ESRI Shapefile"
                )
                
                if error[0] == QgsVectorFileWriter.NoError:
                    print(f"✅ Created shapefile: {output_path}")
                    return True
                else:
                    print(f"❌ Error creating shapefile: {error}")
                    return False
                    
            except Exception as e:
                print(f"❌ Shapefile creation failed: {e}")
                return False
        else:
            # Fallback: create CSV for manual import
            csv_path = output_path.replace('.shp', '.csv')
            try:
                # Flatten GPS coordinates for CSV export
                export_data = []
                for _, detection in detections_df.iterrows():
                    row = detection.to_dict()
                    if 'gps' in row and row['gps']:
                        row['latitude'] = row['gps']['lat']
                        row['longitude'] = row['gps']['lon']
                        del row['gps']  # Remove nested GPS object
                    export_data.append(row)
                
                pd.DataFrame(export_data).to_csv(csv_path, index=False)
                print(f"✅ Created CSV (QGIS not available): {csv_path}")
                return True
            except Exception as e:
                print(f"❌ CSV creation failed: {e}")
                return False
    
    def style_layer(self, layer, object_class):
        """Apply styling to layer"""
        if not QGIS_AVAILABLE:
            return
            
        try:
            style = self.class_styles.get(object_class, {'color': 'black', 'size': 6})
            
            # Get the renderer
            symbol = layer.renderer().symbol()
            
            # Set symbol properties
            symbol.setColor(QColor(style['color']))
            symbol.setSize(style['size'])
            
            # Apply graduated symbol based on confidence
            ranges = []
            ranges.append(QgsRendererRange(0.5, 0.7, symbol.clone(), 'Low Confidence'))
            ranges.append(QgsRendererRange(0.7, 0.9, symbol.clone(), 'Medium Confidence'))
            ranges.append(QgsRendererRange(0.9, 1.0, symbol.clone(), 'High Confidence'))
            
            # Create graduated renderer
            renderer = QgsGraduatedSymbolRenderer('confidence', ranges)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            
        except Exception as e:
            print(f"⚠️ Layer styling failed: {e}")
    
    def create_tactical_map_project(self, project_name='raptor_tactical_map'):
        """Create complete QGIS project"""
        if not QGIS_AVAILABLE:
            print("🗺️ QGIS not available - creating manual files")
            self.create_manual_map_files()
            return
        
        try:
            # Create layers for each class
            self.create_qgis_layers()
            
            # Add base map (if available)
            self.add_base_map()
            
            # Save project
            project_path = f"output/maps/{project_name}.qgz"
            Path(project_path).parent.mkdir(parents=True, exist_ok=True)
            self.project.write(project_path)
            print(f"✅ Saved QGIS project: {project_path}")
            
            # Export as image
            self.export_map_image(f"output/maps/{project_name}_map.png")
            
        except Exception as e:
            print(f"❌ QGIS project creation failed: {e}")
            self.create_manual_map_files()
    
    def create_qgis_layers(self):
        """Create QGIS layers for each detection class"""
        if not self.detections:
            return
            
        df = pd.DataFrame(self.detections)
        
        for object_class in df['class'].unique():
            class_detections = df[df['class'] == object_class]
            
            # Create memory layer
            layer = QgsVectorLayer(
                "Point?crs=EPSG:4326&field=id:integer&field=confidence:double&field=timestamp:string",
                f"RAPTOR_{object_class}",
                "memory"
            )
            
            provider = layer.dataProvider()
            features = []
            
            for idx, detection in class_detections.iterrows():
                feature = QgsFeature()
                point = QgsGeometry.fromPointXY(
                    QgsPointXY(detection['gps']['lon'], detection['gps']['lat'])
                )
                feature.setGeometry(point)
                feature.setAttributes([idx, detection['confidence'], detection['timestamp']])
                features.append(feature)
            
            provider.addFeatures(features)
            layer.updateExtents()
            
            # Style layer
            self.style_layer(layer, object_class)
            
            # Add to project
            self.project.addMapLayer(layer)
            self.layers[object_class] = layer
            
            print(f"✅ Created layer: {object_class} ({len(class_detections)} points)")
    
    def add_base_map(self):
        """Add OpenStreetMap base layer"""
        if not QGIS_AVAILABLE:
            return
            
        try:
            # Add OpenStreetMap layer
            osm_url = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            osm_layer = QgsRasterLayer(osm_url, "OpenStreetMap", "wms")
            
            if osm_layer.isValid():
                self.project.addMapLayer(osm_layer)
                print("✅ Added OpenStreetMap base layer")
            else:
                print("⚠️ Could not add base map")
        except Exception as e:
            print(f"⚠️ Base map error: {e}")
    
    def export_map_image(self, output_path):
        """Export map as image"""
        if not QGIS_AVAILABLE:
            return
            
        try:
            # Create map settings
            settings = QgsMapSettings()
            settings.setLayers([layer for layer in self.project.mapLayers().values()])
            settings.setBackgroundColor(QColor(255, 255, 255))
            settings.setOutputSize(QSize(1200, 800))
            
            # Calculate extent
            if self.layers:
                extent = QgsRectangle()
                for layer in self.layers.values():
                    extent.combineExtentWith(layer.extent())
                extent.scale(1.1)  # Add 10% padding
                settings.setExtent(extent)
            
            settings.setDestinationCrs(self.crs)
            
            # Render map
            render = QgsMapRendererParallelJob(settings)
            render.start()
            render.waitForFinished()
            
            img = render.renderedImage()
            img.save(output_path)
            print(f"✅ Exported map image: {output_path}")
            
        except Exception as e:
            print(f"❌ Map export failed: {e}")
    
    def create_manual_map_files(self):
        """Create mapping files when QGIS is not available"""
        if not self.detections:
            print("⚠️ No detections to map")
            return
            
        output_dir = Path('output/maps')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create GeoJSON
        self.create_geojson(str(output_dir / 'raptor_tactical_map.geojson'))
        
        # Create HTML map using Leaflet
        self.create_web_map(str(output_dir / 'raptor_tactical_map.html'))
        
        # Create CSV for manual QGIS import
        self.create_csv_for_qgis(str(output_dir / 'raptor_tactical_map.csv'))
        
        print("✅ Created manual mapping files:")
        print(f"   📄 GeoJSON: {output_dir / 'raptor_tactical_map.geojson'}")
        print(f"   🌐 Web Map: {output_dir / 'raptor_tactical_map.html'}")
        print(f"   📊 CSV: {output_dir / 'raptor_tactical_map.csv'}")
    
    def create_geojson(self, output_file):
        """Create GeoJSON for mapping"""
        features = []
        
        for i, detection in enumerate(self.detections):
            if 'gps' in detection and detection['gps']:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": i,
                        "class": detection['class'],
                        "confidence": detection['confidence'],
                        "timestamp": detection['timestamp'],
                        "frame": detection.get('frame', 0)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [detection['gps']['lon'], detection['gps']['lat']]
                    }
                }
                features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(geojson, f, indent=2)
            print(f"✅ Created GeoJSON: {output_file}")
        except Exception as e:
            print(f"❌ GeoJSON creation failed: {e}")
    
    def create_csv_for_qgis(self, output_file):
        """Create CSV that can be easily imported to QGIS"""
        try:
            csv_data = []
            for i, detection in enumerate(self.detections):
                if 'gps' in detection and detection['gps']:
                    csv_data.append({
                        'id': i,
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'latitude': detection['gps']['lat'],
                        'longitude': detection['gps']['lon'],
                        'timestamp': detection['timestamp'],
                        'frame': detection.get('frame', 0)
                    })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_file, index=False)
            
            print(f"✅ Created CSV for QGIS import: {output_file}")
            print("📋 To import in QGIS:")
            print("   1. Layer → Add Layer → Add Delimited Text Layer")
            print("   2. Choose the CSV file")
            print("   3. Set latitude/longitude fields")
            print("   4. Choose 'Point coordinates'")
            
        except Exception as e:
            print(f"❌ CSV creation failed: {e}")
    
    def create_web_map(self, output_file):
        """Create interactive web map using Leaflet - FIXED VERSION"""
        if not self.detections:
            print("⚠️ No detections to create web map")
            return
            
        try:
            # Calculate map center - FIXED to handle GPS data properly
            lats = [d['gps']['lat'] for d in self.detections if 'gps' in d and d['gps']]
            lons = [d['gps']['lon'] for d in self.detections if 'gps' in d and d['gps']]
            
            if not lats or not lons:
                print("❌ No GPS coordinates found in detections")
                return
                
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Prepare detection data for JavaScript
            js_detections = []
            for detection in self.detections:
                if 'gps' in detection and detection['gps']:
                    js_detections.append({
                        'lat': detection['gps']['lat'],
                        'lon': detection['gps']['lon'], 
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'timestamp': detection['timestamp'],
                        'frame': detection.get('frame', 0)
                    })
            
            # Count detections by class for stats
            class_counts = {}
            for d in self.detections:
                cls = d['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            # Calculate average confidence
            avg_confidence = sum(d['confidence'] for d in self.detections) / len(self.detections)
            
            # Create HTML content
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>R.A.P.T.O.R Tactical Object Detection Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: white;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{
            color: #00ff41;
            margin: 0;
            font-size: 2.5em;
            text-shadow: 0 0 10px #00ff41;
        }}
        .header p {{
            color: #888;
            margin: 5px 0;
        }}
        #map {{ 
            height: 600px; 
            border: 2px solid #00ff41;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
        }}
        .legend {{ 
            background: rgba(26, 26, 26, 0.9); 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 0 0 15px rgba(0,255,65,0.2);
            border: 1px solid #00ff41;
            color: white;
        }}
        .legend h4 {{
            margin-top: 0;
            color: #00ff41;
            text-align: center;
        }}
        .legend-item {{
            margin: 8px 0;
            display: flex;
            align-items: center;
        }}
        .legend-color {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-right: 10px;
            border: 1px solid #333;
        }}
        .stats {{
            background: rgba(26, 26, 26, 0.9);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border: 1px solid #00ff41;
            text-align: center;
        }}
        .stats h3 {{
            color: #00ff41;
            margin-top: 0;
        }}
        .stat-item {{
            display: inline-block;
            margin: 0 20px;
            padding: 10px;
            background: rgba(0, 255, 65, 0.1);
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🦅 R.A.P.T.O.R</h1>
        <p>Real-time Aerial Patrol and Tactical Object Recognition</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div id="map"></div>
    
    <div class="stats">
        <h3>Detection Statistics</h3>
        <div class="stat-item">
            <strong>Total Objects:</strong> {len(self.detections)}
        </div>
        <div class="stat-item">
            <strong>Object Types:</strong> {len(class_counts)}
        </div>
        <div class="stat-item">
            <strong>Avg Confidence:</strong> {avg_confidence:.1%}
        </div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        // Initialize map
        var map = L.map('map').setView([{center_lat}, {center_lon}], 15);
        
        // Add dark tile layer for tactical look
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '© CartoDB © OpenStreetMap contributors',
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);
        
        // Define colors for each class
        var classColors = {{
            'person': '#00ff41',
            'car': '#0080ff', 
            'truck': '#ff4444',
            'bus': '#ffaa00',
            'motorcycle': '#ff00ff',
            'bird': '#ffff00',
            'cat': '#ff69b4',
            'dog': '#8b4513'
        }};
        
        // Detection data
        var detections = {json.dumps(js_detections)};
        
        // Create marker groups for each class
        var markerGroups = {{}};
        
        // Add markers for each detection
        detections.forEach(function(detection) {{
            var color = classColors[detection.class] || '#ffffff';
            var radius = Math.max(5, detection.confidence * 15);
            
            var marker = L.circleMarker([detection.lat, detection.lon], {{
                color: color,
                fillColor: color,
                fillOpacity: 0.8,
                radius: radius,
                weight: 2
            }});
            
            // Create popup content
            var popupContent = '<div style="color: black; font-weight: bold;">' +
                '<h4 style="margin: 0; color: ' + color + ';">🎯 ' + detection.class.toUpperCase() + '</h4>' +
                '<p><strong>Confidence:</strong> ' + (detection.confidence * 100).toFixed(1) + '%</p>' +
                '<p><strong>GPS:</strong> ' + detection.lat.toFixed(6) + ', ' + detection.lon.toFixed(6) + '</p>' +
                '<p><strong>Time:</strong> ' + new Date(detection.timestamp).toLocaleString() + '</p>';
            
            if (detection.frame) {{
                popupContent += '<p><strong>Frame:</strong> ' + detection.frame + '</p>';
            }}
            
            popupContent += '</div>';
            
            marker.bindPopup(popupContent);
            
            // Add to appropriate group
            if (!markerGroups[detection.class]) {{
                markerGroups[detection.class] = L.layerGroup();
            }}
            markerGroups[detection.class].addLayer(marker);
        }});
        
        // Add all marker groups to map
        Object.keys(markerGroups).forEach(function(className) {{
            markerGroups[className].addTo(map);
        }});
        
        // Add layer control
        var overlayMaps = {{}};
        Object.keys(markerGroups).forEach(function(className) {{
            overlayMaps[className.charAt(0).toUpperCase() + className.slice(1)] = markerGroups[className];
        }});
        
        L.control.layers(null, overlayMaps).addTo(map);
        
        // Add legend
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4>🎯 Detected Objects</h4>';
            
            Object.keys(classColors).forEach(function(className) {{
                if (markerGroups[className] && markerGroups[className].getLayers().length > 0) {{
                    var count = markerGroups[className].getLayers().length;
                    div.innerHTML += 
                        '<div class="legend-item">' +
                        '<div class="legend-color" style="background-color:' + classColors[className] + ';"></div>' +
                        className.charAt(0).toUpperCase() + className.slice(1) + ' (' + count + ')' +
                        '</div>';
                }}
            }});
            
            return div;
        }};
        legend.addTo(map);
        
        // Fit map to show all markers
        if (detections.length > 0) {{
            var group = new L.featureGroup(Object.values(markerGroups).map(function(g) {{ return g.getLayers(); }}).flat());
            map.fitBounds(group.getBounds().pad(0.1));
        }}
    </script>
</body>
</html>"""
            
            # Write HTML file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Created interactive web map: {output_file}")
            
        except Exception as e:
            print(f"❌ Web map creation failed: {e}")
            import traceback
            traceback.print_exc()


# Example usage
if __name__ == "__main__":
    mapper = TacticalQGISMapper()
    
    # Test with sample detection data
    if os.path.exists('output/detections'):
        detection_files = [f for f in os.listdir('output/detections') if f.endswith('.json')]
        if detection_files:
            latest_file = max(detection_files, key=lambda x: os.path.getctime(os.path.join('output/detections', x)))
            file_path = os.path.join('output/detections', latest_file)
            
            print(f"📁 Loading detections from: {file_path}")
            if mapper.load_detections(file_path):
                mapper.create_tactical_map_project()
                mapper.create_shapefile_layers()
            else:
                print("❌ Failed to load detection data")
        else:
            print("⚠️ No detection files found in output/detections")
    else:
        print("⚠️ No output directory found. Run detection first.")
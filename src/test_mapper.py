# Test script for R.A.P.T.O.R QGIS Mapper
# File: test_mapper.py

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append("src")

from qgis_mapper import TacticalQGISMapper


def create_test_detections():
    """Create sample detection data for testing"""

    # Sample detections around Taos, New Mexico
    test_detections = [
        {
            "id": 1,
            "class": "person",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
            "frame": 100,
            "gps": {"lat": 36.4060, "lon": -105.5720},
        },
        {
            "id": 2,
            "class": "car",
            "confidence": 0.87,
            "timestamp": datetime.now().isoformat(),
            "frame": 150,
            "gps": {"lat": 36.4055, "lon": -105.5715},
        },
        {
            "id": 3,
            "class": "truck",
            "confidence": 0.92,
            "timestamp": datetime.now().isoformat(),
            "frame": 200,
            "gps": {"lat": 36.4065, "lon": -105.5725},
        },
        {
            "id": 4,
            "class": "person",
            "confidence": 0.78,
            "timestamp": datetime.now().isoformat(),
            "frame": 250,
            "gps": {"lat": 36.4058, "lon": -105.5718},
        },
        {
            "id": 5,
            "class": "car",
            "confidence": 0.91,
            "timestamp": datetime.now().isoformat(),
            "frame": 300,
            "gps": {"lat": 36.4062, "lon": -105.5722},
        },
        {
            "id": 6,
            "class": "motorcycle",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat(),
            "frame": 350,
            "gps": {"lat": 36.4057, "lon": -105.5710},
        },
        {
            "id": 7,
            "class": "bus",
            "confidence": 0.89,
            "timestamp": datetime.now().isoformat(),
            "frame": 400,
            "gps": {"lat": 36.4063, "lon": -105.5730},
        },
        {
            "id": 8,
            "class": "person",
            "confidence": 0.93,
            "timestamp": datetime.now().isoformat(),
            "frame": 450,
            "gps": {"lat": 36.4061, "lon": -105.5719},
        },
    ]

    # Add some detections in a cluster (simulating a gathering)
    for i in range(5):
        test_detections.append(
            {
                "id": 10 + i,
                "class": "person",
                "confidence": 0.8 + (i * 0.02),
                "timestamp": datetime.now().isoformat(),
                "frame": 500 + (i * 10),
                "gps": {"lat": 36.4050 + (i * 0.0001), "lon": -105.5705 + (i * 0.0001)},
            }
        )

    # Add some vehicles in a line (simulating a road)
    for i in range(4):
        test_detections.append(
            {
                "id": 20 + i,
                "class": "car" if i % 2 == 0 else "truck",
                "confidence": 0.85 + (i * 0.03),
                "timestamp": datetime.now().isoformat(),
                "frame": 600 + (i * 20),
                "gps": {"lat": 36.4070, "lon": -105.5710 + (i * 0.0005)},
            }
        )

    return test_detections


def test_mapper():
    """Test the QGIS mapper with sample data"""

    print("🦅 R.A.P.T.O.R QGIS Mapper Test")
    print("=" * 50)

    # Create output directories
    Path("output/detections").mkdir(parents=True, exist_ok=True)
    Path("output/maps").mkdir(parents=True, exist_ok=True)

    # Create test detection file
    test_file = "output/detections/test_detections.json"
    test_data = create_test_detections()

    print(f"📝 Creating test data with {len(test_data)} detections...")
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"✅ Test data saved to: {test_file}")

    # Initialize mapper
    print("\n🗺️ Initializing mapper...")
    mapper = TacticalQGISMapper()

    # Load detections
    print("📥 Loading detections...")
    if mapper.load_detections(test_file):
        print(f"✅ Loaded {len(mapper.detections)} detections with GPS")

        # Create map files
        print("\n🎨 Creating map files...")
        mapper.create_manual_map_files()

        # Also try to create shapefiles
        print("\n📁 Attempting to create shapefiles...")
        mapper.create_shapefile_layers()

        print("\n" + "=" * 50)
        print("✅ MAPPING COMPLETE!")
        print("=" * 50)

        # List created files
        print("\n📂 Created files:")
        map_files = list(Path("output/maps").glob("*"))
        for file in map_files:
            print(f"   - {file}")

        # Provide instructions
        print("\n📋 NEXT STEPS:")
        print("\n1. VIEW INTERACTIVE MAP:")
        print("   - Open: output/maps/raptor_tactical_map.html")
        print("   - Just double-click the file!")

        print("\n2. IMPORT TO QGIS:")
        print("   - Use: output/maps/raptor_tactical_map.csv")
        print("   - Or: output/maps/raptor_tactical_map.geojson")

        print("\n3. USE IN WEB APPS:")
        print("   - GeoJSON file works with Leaflet, Mapbox, etc.")

        # Open the HTML map automatically
        html_map = Path("output/maps/raptor_tactical_map.html")
        if html_map.exists():
            print("\n🌐 Opening interactive map in browser...")
            import webbrowser

            webbrowser.open(f"file://{html_map.absolute()}")
    else:
        print("❌ Failed to load test data")


def test_with_real_data():
    """Test with existing detection files"""

    print("🔍 Looking for existing detection files...")

    detection_dir = Path("output/detections")
    if not detection_dir.exists():
        print("❌ No output/detections directory found")
        print("💡 Run a detection first or use test_mapper() function")
        return

    json_files = list(detection_dir.glob("*.json"))

    if not json_files:
        print("❌ No detection files found")
        print("💡 Run a detection first or use test_mapper() function")
        return

    # Use the most recent file
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"✅ Found {len(json_files)} detection files")
    print(f"📄 Using most recent: {latest_file.name}")

    # Process with mapper
    mapper = TacticalQGISMapper()

    if mapper.load_detections(str(latest_file)):
        print(f"✅ Loaded {len(mapper.detections)} detections")

        # Create all map outputs
        mapper.create_manual_map_files()

        # Open result
        html_map = Path("output/maps/raptor_tactical_map.html")
        if html_map.exists():
            print("\n🌐 Opening map in browser...")
            import webbrowser

            webbrowser.open(f"file://{html_map.absolute()}")
    else:
        print("❌ Failed to load detections")


def main():
    """Main test function"""

    print(
        """
🦅 R.A.P.T.O.R QGIS MAPPER TEST
===============================

Choose test option:
1. Create test data and generate maps
2. Use existing detection files
3. Exit
"""
    )

    choice = input("Select option (1-3): ")

    if choice == "1":
        test_mapper()
    elif choice == "2":
        test_with_real_data()
    elif choice == "3":
        print("👋 Exiting")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

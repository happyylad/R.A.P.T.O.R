# R.A.P.T.O.R GPS Converter Module
# File: src/gps_converter.py

import math

class SimpleGPSConverter:
    def __init__(self, image_bounds):
        """
        GPS coordinate converter for R.A.P.T.O.R system
        
        Args:
            image_bounds: dict with GPS coordinates for image corners
            Example: {
                'top_left': {'lat': 36.4074, 'lon': -105.5731},
                'top_right': {'lat': 36.4074, 'lon': -105.5700},
                'bottom_left': {'lat': 36.4044, 'lon': -105.5731},
                'bottom_right': {'lat': 36.4044, 'lon': -105.5700}
            }
        """
        self.bounds = image_bounds
    
    def pixel_to_gps(self, pixel_x, pixel_y, image_width, image_height):
        """Convert pixel coordinates to GPS coordinates"""
        # Convert pixel coordinates to normalized (0-1)
        norm_x = pixel_x / image_width
        norm_y = pixel_y / image_height
        
        # Simple bilinear interpolation
        lat = self.interpolate_lat(norm_x, norm_y)
        lon = self.interpolate_lon(norm_x, norm_y)
        
        return lat, lon
    
    def interpolate_lat(self, norm_x, norm_y):
        """Interpolate latitude"""
        top_lat = self.linear_interp(
            self.bounds['top_left']['lat'],
            self.bounds['top_right']['lat'],
            norm_x
        )
        bottom_lat = self.linear_interp(
            self.bounds['bottom_left']['lat'],
            self.bounds['bottom_right']['lat'],
            norm_x
        )
        return self.linear_interp(top_lat, bottom_lat, norm_y)
    
    def interpolate_lon(self, norm_x, norm_y):
        """Interpolate longitude"""
        left_lon = self.linear_interp(
            self.bounds['top_left']['lon'],
            self.bounds['bottom_left']['lon'],
            norm_y
        )
        right_lon = self.linear_interp(
            self.bounds['top_right']['lon'],
            self.bounds['bottom_right']['lon'],
            norm_y
        )
        return self.linear_interp(left_lon, right_lon, norm_x)
    
    def linear_interp(self, val1, val2, t):
        """Linear interpolation between two values"""
        return val1 + (val2 - val1) * t

    def gps_to_pixel(self, lat, lon, image_width, image_height):
        """Convert GPS coordinates back to pixel coordinates"""
        # This is the inverse operation - useful for validation
        # Implementation would involve solving the bilinear interpolation equations
        pass


# Test the converter
if __name__ == "__main__":
    bounds = {
        'top_left': {'lat': 36.4074, 'lon': -105.5731},
        'top_right': {'lat': 36.4074, 'lon': -105.5700},
        'bottom_left': {'lat': 36.4044, 'lon': -105.5731},
        'bottom_right': {'lat': 36.4044, 'lon': -105.5700}
    }

    converter = SimpleGPSConverter(bounds)
    lat, lon = converter.pixel_to_gps(500, 300, 1000, 600)
    print(f"Test conversion: ({lat:.6f}, {lon:.6f})")
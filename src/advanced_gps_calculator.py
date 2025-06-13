import math
import numpy as np


class AdvancedGPSCalculator:
    """
    Calculates the GPS coordinates of an object on the ground based on drone telemetry
    and the object's position in the video frame.
    """

    def __init__(self, camera_fov_deg):
        """
        Initializes the calculator with camera parameters.

        Args:
            camera_fov_deg (float): The horizontal field of view of the camera in degrees.
        """
        # Horizontal Field of View in radians
        self.h_fov_rad = math.radians(camera_fov_deg)

    def calculate_gps(self, drone_telemetry, bbox_center_px, frame_dims_px):
        """
        Calculates the GPS coordinates for a detected object.

        Args:
            drone_telemetry (dict): Dictionary containing drone's current state.
                - 'lat' (float): Drone latitude.
                - 'lon' (float): Drone longitude.
                - 'alt_m' (float): Drone altitude in meters above ground level.
                - 'heading_deg' (float): Drone compass heading (0-360, 0 is North).
                - 'tilt_deg' (float): Camera tilt in degrees (0 is horizontal, 90 is straight down).
            bbox_center_px (tuple): (x, y) pixel coordinates of the bounding box center.
            frame_dims_px (tuple): (width, height) of the video frame in pixels.

        Returns:
            dict: A dictionary with {'lat': float, 'lon': float} or None if calculation fails.
        """
        try:
            # Unpack for clarity
            drone_lat = drone_telemetry["lat"]
            drone_lon = drone_telemetry["lon"]
            drone_alt_m = drone_telemetry["alt_m"]
            drone_heading_deg = drone_telemetry["heading_deg"]
            camera_tilt_deg = drone_telemetry["tilt_deg"]
            frame_width, frame_height = frame_dims_px
            pixel_x, pixel_y = bbox_center_px

            # --- 1. Calculate Angles from Pixel Coordinates ---
            # Angle of the object relative to the camera's center line
            # Horizontal angle (azimuth relative to camera)
            pixel_offset_x = pixel_x - (frame_width / 2)
            angle_offset_x_rad = math.atan(
                (2 * pixel_offset_x / frame_width) * math.tan(self.h_fov_rad / 2)
            )

            # Vertical angle (elevation relative to camera)
            # Assuming vertical FoV is derived from horizontal FoV and aspect ratio
            aspect_ratio = frame_height / frame_width
            v_fov_rad = 2 * math.atan(math.tan(self.h_fov_rad / 2) * aspect_ratio)
            pixel_offset_y = (
                frame_height / 2
            ) - pixel_y  # Y is inverted in pixel coords
            angle_offset_y_rad = math.atan(
                (2 * pixel_offset_y / frame_height) * math.tan(v_fov_rad / 2)
            )

            # --- 2. Calculate World Angles ---
            # Combine camera angles with drone orientation
            # Absolute azimuth (compass direction to object)
            object_azimuth_deg = (
                drone_heading_deg + math.degrees(angle_offset_x_rad)
            ) % 360

            # Absolute depression angle (how far down we are looking at the object)
            object_depression_rad = math.radians(camera_tilt_deg) - angle_offset_y_rad

            # --- 3. Calculate Ground Distance (Ray-Plane Intersection) ---
            # How far the object is from the point directly under the drone
            if object_depression_rad <= 0 or math.tan(object_depression_rad) <= 0:
                # Object is at or above the horizon, cannot be on the ground
                return None

            ground_distance_m = drone_alt_m / math.tan(object_depression_rad)

            # --- 4. Calculate North and East Offsets ---
            # Convert polar coordinates (distance, angle) to cartesian (North, East)
            north_offset_m = ground_distance_m * math.cos(
                math.radians(object_azimuth_deg)
            )
            east_offset_m = ground_distance_m * math.sin(
                math.radians(object_azimuth_deg)
            )

            # --- 5. Calculate Final GPS Coordinates ---
            # Apply the meter offsets to the drone's GPS coordinates
            object_lat, object_lon = self._offset_gps_coordinates(
                drone_lat, drone_lon, north_offset_m, east_offset_m
            )

            return {"lat": object_lat, "lon": object_lon}

        except (ValueError, ZeroDivisionError, KeyError) as e:
            print(f"⚠️ GPS Calculation Error: {e}")
            return None

    def _offset_gps_coordinates(self, lat, lon, north_offset_m, east_offset_m):
        """
        Offsets a GPS coordinate by a specified distance in meters.
        """
        # Earth's radius in meters
        earth_radius = 6378137.0

        # Calculate new latitude
        d_lat = north_offset_m / earth_radius
        new_lat = lat + math.degrees(d_lat)

        # Calculate new longitude
        d_lon = east_offset_m / (earth_radius * math.cos(math.radians(lat)))
        new_lon = lon + math.degrees(d_lon)

        return new_lat, new_lon

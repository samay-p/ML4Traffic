
# CARLA Simulation: Basic Environment 
# Run CARLA Server before executing (CARLAUE4 with WindowsNoEditor))
# Tested with CARLA 0.9.15 on Windows

import glob
import os
import sys
import random
import time
import carla
import cv2
import numpy as np

# First step is connecting to CARLA Server
try:
    sys.path.append(glob.glob('C:/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass

def main():
    actor_list = []

    try:
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Load map
        # Town can be changed to either a preset map or Custom Made
        world = client.load_world('Town03')
        blueprint_library = world.get_blueprint_library()

        # Set weather (Unimportant for purposes just need for initialization)
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=10.0,
            sun_altitude_angle=70.0)
        world.set_weather(weather)

        # Spawn ego vehicle
        vehicle_bp = blueprint_library.filter('model3')[0]  # Tesla Model 3
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print("Spawned vehicle: %s" % vehicle.type_id)

        # Attach camera sensor
        # Used For CV Simulation
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print("Camera attached to vehicle")

        # Spawn traffic vehicles
        for _ in range(10):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            npc_spawn_point = random.choice(spawn_points)
            npc_vehicle = world.try_spawn_actor(vehicle_bp, npc_spawn_point)
            if npc_vehicle:
                npc_vehicle.set_autopilot(True)
                actor_list.append(npc_vehicle)

        print("Spawned NPC vehicles")

        # Camera data processing
        # Used to Simulate Real Life Camera feed for CV Processing
        def process_image(image):
            """Convert raw camera data to an OpenCV image."""
            img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
            img_array = np.reshape(img_array, (image.height, image.width, 4))
            img_array = img_array[:, :, :3]  # Drop alpha channel
            cv2.imshow("CARLA Camera View", img_array)
            cv2.waitKey(1)

        camera.listen(lambda data: process_image(data))

        # Run for x Amount of Seconds (30 is Default)
        time.sleep(30)

    #End Simulation after runtime
    
    finally:
        print('Destroying actors...')
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        print('Simulation ended.')

if __name__ == '__main__':
    main()

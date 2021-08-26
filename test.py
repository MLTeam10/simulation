import glob
import os
import sys
import time
import numpy

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import queue
except ImportError:
    import Queue as queue

import carla

from carla import VehicleLightState as vls
from carla import ColorConverter

from PIL import Image

import argparse
import logging
from numpy import random


class CarlaSyncMode(object):

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 1)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = True
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        args.sync = True
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)


        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            if n == 0:
                blueprint.set_attribute('role_name', 'ego')
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))
 

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
        
        time.sleep(0.5)  # This is so that the vehicle gets registered in the actors.
        ego_vehicle = world.get_actors().find(vehicles_list[0])
        ###########################

        # Spawn the blueprints
        spectator = world.get_spectator()

        factor = 0.25

        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation') # or .rgb
        cam_bp.set_attribute("image_size_x",str(640/factor))
        cam_bp.set_attribute("image_size_y",str(480/factor))
        cam_bp.set_attribute("fov",str(110))
        #cam_bp.set_attribute('fstop', str(1.4))
        cam_location = carla.Location(3.5,0,1.5)
        cam_rotation = carla.Rotation(-10,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)

        cam_bp_rgb = None
        cam_bp_rgb = world.get_blueprint_library().find('sensor.camera.rgb') # or .rgb
        cam_bp_rgb.set_attribute("image_size_x",str(640/factor))
        cam_bp_rgb.set_attribute("image_size_y",str(480/factor))
        cam_bp_rgb.set_attribute("fov",str(110))
        #cam_bp_rgb.set_attribute('fstop', str(1.4))
        cam_location_rgb = carla.Location(3.5,0,1.5)
        cam_rotation_rgb = carla.Rotation(-10,0,0)
        cam_transform_rgb = carla.Transform(cam_location_rgb,cam_rotation_rgb)
        
        ego_cam = world.spawn_actor(cam_bp,cam_transform,ego_vehicle,carla.AttachmentType.Rigid)
        ego_cam_rgb = world.spawn_actor(cam_bp_rgb,cam_transform_rgb,ego_vehicle,carla.AttachmentType.Rigid)

        ###########################

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.1      # how many pedestrians will run
        percentagePedestriansCrossing = 0.5     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, ego_cam_rgb, ego_cam, fps=1) as sync_mode:
            zzz = 0
            rk = 0
            while True:

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)

                # Choose the next waypoint and update the car location.
                #waypoint = random.choice(waypoint.next(1.5))
                spectator.set_transform(ego_cam.get_transform())

                # image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                if zzz % (fps * 10) == 0: # min is 2FPS, so save image only each 10 seconds
                    print(rk)
                    save_image(image_semseg, rk, True)
                    save_image(image_rgb, rk, False)

                    rk += 1

                zzz += 1
    finally:
        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)


def save_image(image, zk, convert):
    if convert:
        k = labels_to_cityscapes_palette(image)
        im = Image.fromarray(numpy.uint8(k)).convert("P")
        im.save('output/Masks/1%.10d.png' % zk)

        image.convert(ColorConverter.CityScapesPalette)
        image.save_to_disk('output/Segmentation/1%.10d.png' % zk)
    else:
        image.save_to_disk('output/Images/1%.10d.jpg' % zk)


def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        # 0: [0, 0, 0],         # None
        # 1: [70, 70, 70],      # Buildings
        # 2: [190, 153, 153],   # Fences
        # 3: [72, 0, 90],       # Other
        4: [255, 255, 255],     # Pedestrians
        # 5: [153, 153, 153],   # Poles
        # 6: [157, 234, 50],    # RoadLines
        # 7: [128, 64, 128],    # Roads
        # 8: [244, 35, 232],    # Sidewalks
        # 9: [107, 142, 35],    # Vegetation
        # 10: [0, 0, 255],      # Vehicles
        # 11: [102, 102, 156],  # Walls
        # 12: [220, 220, 0],    # TrafficSigns
        # 13: [255, 255, 255]     # ?
    }
    array = labels_to_array(image)
    result = numpy.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[numpy.where(array == key)] = value
    return result

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = numpy.reshape(array, (image.height, image.width, 4))
    return array

def labels_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to a 2D array
    containing the label of each pixel.
    """
    return to_bgra_array(image)[:, :, 2]

if __name__ == '__main__':
    
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

import sys
sys.path.append('/home/28s_mur@lab.graphicon.ru/carla/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
import carla
import argparse
import os
import csv
import json
import math
import numpy as np
from queue import Queue


# =====================================================
# Camera config (intrinsics + extrinsics)
# =====================================================
def get_camera_config(camera_actor, camera_bp):
    width = int(camera_bp.get_attribute("image_size_x").as_int())
    height = int(camera_bp.get_attribute("image_size_y").as_int())
    fov = float(camera_bp.get_attribute("fov").as_float())

    fov_rad = math.radians(fov)
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    tf = camera_actor.get_transform()
    loc = tf.location
    rot = tf.rotation

    cyaw = math.cos(math.radians(rot.yaw))
    syaw = math.sin(math.radians(rot.yaw))
    cp = math.cos(math.radians(rot.pitch))
    sp = math.sin(math.radians(rot.pitch))
    cr = math.cos(math.radians(rot.roll))
    sr = math.sin(math.radians(rot.roll))

    transform_matrix = np.array([
        [ cp*cyaw,  cyaw*sp*sr - syaw*cr, -cyaw*sp*cr - syaw*sr, loc.x ],
        [ cp*syaw,  syaw*sp*sr + cyaw*cr, -syaw*sp*cr + cyaw*sr, loc.y ],
        [ sp,      -cp*sr,               cp*cr,               loc.z ],
        [ 0.0,      0.0,                 0.0,                 1.0   ]
    ])

    camera_config = {
        "image_width": width,
        "image_height": height,
        "fov": fov,
        "transform_matrix": transform_matrix,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }

    return camera_config



# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--town", type=str, default="Town03")
    parser.add_argument("--num_spawns", type=int, default=5)
    parser.add_argument("--steps_per_spawn", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=70)
    args = parser.parse_args()

    # ---------- dirs ----------
    rgb_dir = os.path.join(args.out_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    # ---------- connect ----------
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(args.town)

    # ---------- sync mode ----------
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()[:args.num_spawns]

    # ---------- metadata ----------
    csv_file = open(os.path.join(args.out_dir, "metadata.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "frame", "spawn_id",
        "x", "y", "z",
        "yaw", "speed_kmh"
    ])

    frame_id = 0
    camera_config_saved = False

    # =====================================================
    # Loop over spawn points
    # =====================================================
    for spawn_id, spawn in enumerate(spawn_points):

        print(f"[INFO] Spawn point {spawn_id}")
        actors = []

        # ---------- vehicle ----------
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn)
        if vehicle is None:
            print("[WARN] Failed to spawn vehicle")
            continue

        actors.append(vehicle)
        vehicle.set_autopilot(True, tm.get_port())

        # ---------- camera ----------
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov", "90")

        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        cam_cfg = get_camera_config(camera, cam_bp)
        actors.append(camera)

        img_queue = Queue()
        camera.listen(lambda img: img_queue.put(img))

        # ---------- save camera config once ----------
        if not camera_config_saved:
            cam_cfg_json = cam_cfg.copy()
            cam_cfg_json["transform_matrix"] = cam_cfg["transform_matrix"].tolist()

            with open(os.path.join(args.out_dir, "camera_config.json"), "w") as f:
                json.dump(cam_cfg_json, f, indent=2)

            camera_config_saved = True

        # ---------- warm-up ----------
        for _ in range(args.warmup):
            world.tick()
            img_queue.get()

        # ---------- data collection ----------
        for _ in range(args.steps_per_spawn):
            world.tick()
            image = img_queue.get()

            image.save_to_disk(
                os.path.join(rgb_dir, f"{frame_id:06d}.png")
            )

            tf = vehicle.get_transform()
            vel = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            writer.writerow([
                frame_id,
                spawn_id,
                tf.location.x,
                tf.location.y,
                tf.location.z,
                tf.rotation.yaw,
                speed
            ])

            frame_id += 1

        # ---------- cleanup ----------
        for a in actors:
            a.destroy()

    csv_file.close()

    # ---------- restore ----------
    settings.synchronous_mode = False
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)

    print("[DONE] Dataset collection finished")


if __name__ == "__main__":
    main()

# client_carla_batch.py
import argparse
import os
import random
import sys
import queue
import requests
import numpy as np
from patch_projection import project_patch_to_image, build_intrinsic_matrix
import cv2
sys.path.append('/home/28s_mur@lab.graphicon.ru/carla/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
import carla
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE)
from utils.process_images import save_depth, save_segmentation
import torch
from PIL import Image


def post_two_frames(img_path1, img_path2, use_attack=False):
    files = [
        ('files', ('f1.png', open(img_path1, 'rb'), 'image/png')),
        ('files', ('f2.png', open(img_path2, 'rb'), 'image/png'))
    ]
    params = {}
    if use_attack:
        params['attack_flag'] = '1'
    else:
        params['attack_flag'] = '0'
    resp = requests.post("http://127.0.0.1:8000/predict",
                         files=files, params=params, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def is_obstacle_in_front(seg_mask: np.ndarray,
                         depth_map: np.ndarray,
                         road_labels: set = {7, 6},
                         depth_threshold: float = 225.0,
                         fraction_threshold: float = 0.01,
                         roi_frac: float = 0.5):
    """
    Определяет, есть ли препятствие перед машиной.

    seg_mask: H×W, сегментация (метки классов)
    depth_map: H×W, глубина (метры или относительная)
    road_labels: set меток, которые считаем "дорогой" (например road + road lines)
    depth_threshold: float — максимальное расстояние (глубина), считать препятствием если ближе
    fraction_threshold: float — доля пикселей в ROI, чтобы считать препятствием
    roi_frac: float — насколько сверху от низа кадра берётся ROI (например 0.5 — половина снизу)
    """

    H, W = seg_mask.shape
    # 1. маска потенциальных препятствий: всё, что не дорога
    non_road = ~np.isin(seg_mask, list(road_labels))
    # 2. ROI — нижняя часть кадра (например, доля roi_frac)
    y0 = int(H * (1.0 - roi_frac))
    roi_non_road = non_road[y0:, :]

    # 3. из них — те, где глубина < threshold (т.е. близкие объекты)
    roi_depth = depth_map[y0:, :]
    mask_close = (roi_depth > depth_threshold)
    print(roi_depth.min(), roi_depth.max(), roi_depth.mean())
    # 4. объединённая маска: non-road & close
    mask = roi_non_road & mask_close

    # 5. доля таких пикселей в ROI
    frac = mask.sum() / float(roi_non_road.size)
    print(frac)
    # 6. принять как препятствие, если доля > fraction_threshold
    return frac > fraction_threshold


def main():
    parser = argparse.ArgumentParser(
        description="CARLA batch client with attack and save options")
    parser.add_argument("--out_dir", "-o", type=str, default="tmp",
                        help="Directory where images / depth / seg will be saved")
    parser.add_argument("--use_attack", "-a", action="store_true",
                        help="If set — send attack_flag=1 to server (use attacked sensors)")
    parser.add_argument("--max_steps", "-n", type=int, default=1000,
                        help="Number of simulation steps")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town03')
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    actor_list = []
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    spawn = spawn_points[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn)
    actor_list.append(vehicle)

    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
    actor_list.append(camera)
    # cam_tf = camera.get_transform()
    forward = cam_tf.get_forward_vector()

    image_q = queue.Queue()
    camera.listen(image_q.put)

    prev_path = None
    patch_img = cv2.imread("/home/28s_mur@lab.graphicon.ru/flow_attack/patch.png", cv2.IMREAD_UNCHANGED)
    base = camera.get_transform().location
    dist = 10.0




    try:
        for step in range(args.max_steps):
            world.tick()
            img = image_q.get(timeout=2.0)
            curr_path = os.path.join(out_dir, f"{step:06d}.png")
            img.save_to_disk(curr_path)
            rgb_np = cv2.imread(curr_path)

            # 1) Получаем мировые координаты патча

            if prev_path is None:
                prev_path = curr_path
                continue

            res = post_two_frames(prev_path, curr_path,
                                  use_attack=args.use_attack)

            depth = None
            seg = None

            if "depth" in res:
                shape = tuple(res["depth_shape"])
                depth = torch.tensor(
                    res["depth"], dtype=torch.float32).reshape(shape)
                # Save to disk:
                # → [518, 1722]
                depth = depth.squeeze(0).squeeze(0).detach().cpu().numpy()
                min_v, max_v = depth.min(), depth.max()
                depth = (depth - min_v) / (max_v - min_v)
                depth = (depth * 255).clip(0, 255).astype('uint8')
                image = Image.fromarray(depth)
                image.save(os.path.join(out_dir, f"depth_{step:06d}.png"))

            if "seg" in res:
                shape = tuple(res["seg_shape"])
                seg = torch.tensor(
                    res["seg"], dtype=torch.uint8).reshape(shape).numpy()
                save_segmentation(seg, os.path.join(
                    out_dir, f"seg_{step:06d}.png"))

            if "attacked_image" in res:
                print('!')
                # ожидаем, что сервер вернул:
                # - "attacked_img": плоский список значений
                # - "attacked_img_shape": исходная форма (например [1,3,H,W])
                atk_shape = tuple(res["img_shape"])
                attacked = torch.tensor(
                    res["attacked_image"], dtype=torch.float32
                ).reshape(atk_shape)

                # предполагаем формат [1,3,H,W] и значения в [0,1]
                attacked = attacked.squeeze(0)          # [3,H,W]
                attacked_np = attacked.permute(1, 2, 0)  # [H,W,3]
                attacked_np = (attacked_np * 255.0).clamp(0,
                                                          255).byte().cpu().numpy()

                atk_path = os.path.join(out_dir, f"attacked_{step:06d}.png")
                atk_img = Image.fromarray(
                    attacked_np[:, :, ::-1])  # RGB по умолчанию
                atk_img.save(atk_path)
                

            ctrl = carla.VehicleControl()
            ctrl.throttle = 0.5
            ctrl.steer = 0.0
            ctrl.brake = 0.0
            if is_obstacle_in_front(seg, depth, road_labels={0}, depth_threshold=220, fraction_threshold=0.05):
                ctrl.throttle = 0.0
                ctrl.brake = 1.0
            else:
                ctrl.throttle = 1
                ctrl.brake = 0.0
            vehicle.apply_control(ctrl)

            print(
                f"Frame {img.frame:06d}   throttle={ctrl.throttle:.2f}, steer={ctrl.steer:.2f}, brake={ctrl.brake:.2f}")

            prev_path = curr_path

    finally:
        for a in actor_list:
            a.destroy()


if __name__ == "__main__":
    main()

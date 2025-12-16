import os
import re
import cv2
import argparse
import numpy as np


def make_video_from_frames(
    frames_dir: str,
    out_path: str = "out.mp4",
    fps: int = 20,
    use_attacked: bool = False,
):
    """
    Собирает видео, где каждый кадр = [камера | глубина | сегментация] по горизонтали.
    Ожидаемые файлы:
      - обычная камера:      000000.png, 000001.png, ...
      - атакованная камера:  attacked_000000.png, attacked_000001.png, ... (если use_attacked=True)
      - depth map:           depth_000000.png, depth_000001.png, ...
      - segmentation:        seg_000000.png,   seg_000001.png,   ...
    """

    # найдём все имена кадров камеры вида 000123.png (6 цифр)
    cam_regex = re.compile(r"^(\d{6})\.png$")
    cam_frames = []
    for fname in os.listdir(frames_dir):
        m = cam_regex.match(fname)
        if m:
            cam_frames.append(fname)

    cam_frames = sorted(cam_frames)
    if not cam_frames:
        print("Не найдено кадров вида 000000.png в", frames_dir)
        return

    video_writer = None

    for fname in cam_frames:
        step_str = fname.split(".")[0]          # '000123'

        # путь к камере: либо обычный кадр, либо attacked_*.png
        if use_attacked:
            cam_name = f"attacked_{step_str}.png"
        else:
            cam_name = fname

        cam_path = os.path.join(frames_dir, cam_name)
        depth_path = os.path.join(frames_dir, f"depth_{step_str}.png")
        seg_path = os.path.join(frames_dir, f"seg_{step_str}.png")

        if not os.path.exists(cam_path):
            print(
                f"[WARN] нет файла камеры {cam_name} для шага {step_str}, пропускаю")
            continue

        if not (os.path.exists(depth_path) and os.path.exists(seg_path)):
            print(f"[WARN] нет depth/seg для шага {step_str}, пропускаю")
            continue

        # читаем изображения
        cam_img = cv2.imread(cam_path, cv2.IMREAD_COLOR)      # BGR
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)  # 1 канал
        # BGR (скорее всего уже цветная сегментация)
        seg_img = cv2.imread(seg_path, cv2.IMREAD_COLOR)

        if cam_img is None or depth_img is None or seg_img is None:
            print(
                f"[WARN] не удалось прочитать один из файлов для шага {step_str}, пропускаю")
            continue

        # приводим глубину к 3-канальному, чтобы можно было конкатенировать
        depth_color = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)  # BGR

        # при необходимости поджать/растянуть карты до размеров камеры по высоте/ширине
        H, W, _ = cam_img.shape
        depth_color = cv2.resize(
            depth_color, (W, H), interpolation=cv2.INTER_NEAREST)
        seg_img = cv2.resize(seg_img,     (W, H),
                             interpolation=cv2.INTER_NEAREST)

        # горизонтальная склейка: [камера | глубина | сегментация]
        concat = np.concatenate(
            [cam_img, depth_color, seg_img], axis=1)  # [H, W*3, 3]

        # инициализируем видеописатель при первом кадре
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # можно 'XVID' или др.
            h_out, w_out = concat.shape[:2]
            video_writer = cv2.VideoWriter(
                out_path, fourcc, fps, (w_out, h_out))
            if not video_writer.isOpened():
                print("Не удалось открыть VideoWriter")
                return

        video_writer.write(concat)

    if video_writer is not None:
        video_writer.release()
        print("Видео сохранено в", out_path)
    else:
        print("Не было ни одного корректного кадра для записи видео.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", "-d", type=str, default="tmp",
                        help="Папка, где лежат 000000.png, depth_000000.png, seg_000000.png")
    parser.add_argument("--out", "-o", type=str, default="out.mp4",
                        help="Имя выходного видеофайла")
    parser.add_argument("--fps", type=int, default=20,
                        help="Частота кадров видео")
    parser.add_argument("--attacked", "-a", action="store_true",
                        help="Если указан, вместо обычных кадров берём attacked_XXXXXX.png")
    args = parser.parse_args()

    make_video_from_frames(
        args.frames_dir,
        args.out,
        fps=args.fps,
        use_attacked=args.attacked
    )

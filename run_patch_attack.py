import ptlflow.utils.io_adapter as io_adapter_lib
import cv2
import ptlflow.models
import ptlflow.utils
import ptlflow.utils.io_adapter
import torch
import numpy as np
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import load_mde_model, load_seg_model
from utils.process_images import preprocess_img, postprocess_flow, get_image_tensors, replace_images_dic
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed
from utils.args import parse_args
from attacks.get_attacks import get_attack
# import your patch training function
from attacks.patch_attack import train_patch_ptlflow
import ptlflow
import cv2 as cv2
from utils.targets import get_target, get_mde_target, get_ss_target
from attacks.DetectionDefenses.helper_functions.patch_adversary import PatchAdversary
from attacks.patch_projection import  project_patch_on_scene


def load_model(model_name, dataset):
    model_ref = ptlflow.get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()
    print(checkpoints)
    for c in checkpoints:
        if c in dataset:
            model = ptlflow.get_model(model_name, c)
            return model
        else:
            print(f"Using checkpoint from other dataset!: {c}")
            model = ptlflow.get_model(model_name, c)
            return model
    print(f"No pre-trained model available for {model}/{dataset}.")
    return None

# tmp


def analyze_people_in_dataset(data_loader, flow_model, ss_model, device,
                              person_class_id=11,  # Cityscapes trainId=11 = person
                              area_threshold=20,    # минимальная площадь blob'а, чтобы считать человеком
                              max_batches=None):
    """
    Анализирует, в каких изображениях есть люди, и собирает по ним статистику.
    Для каждого изображения собирает:
      - кол-во людей
      - bbox для каждого человека (x_min, y_min, x_max, y_max)
      - площадь каждого blob'а в пикселях

    Возвращает:
      results: dict[int, list[dict]]
        image_id -> список людей, каждый:
          {
            "bbox": (x_min, y_min, x_max, y_max),
            "area": int
          }
    """

    ss_model.eval()
    results = {}

    total_instances = 0
    total_images_with_people = 0

    with torch.no_grad():
        for batch_idx, (images, flow, valid, meta, K) in enumerate(tqdm(data_loader)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            B = images.shape[0]
            images = images.to(device)  # [B, 2, C, H, W]

            # возьмём первый кадр пары
            I1 = images[:, 0]  # [B, C, H, W]

            # IOAdapter как в твоём коде
            io_adapter = io_adapter_lib.IOAdapter(
                flow_model,
                input_size=I1.shape[-2:],
                cuda=torch.cuda.is_available()
            )
            wrapped = {'images': I1.unsqueeze(
                1), 'flows': None, 'valids': None}
            inputs = io_adapter.prepare_inputs(inputs=wrapped)

            # сегментация
            # [B, num_classes, H, W]
            logits = ss_model(inputs, return_logits=True)
            seg = logits.argmax(dim=1)                     # [B, H, W]

            person_mask = (seg == person_class_id)         # [B, H, W] bool

            for i in range(B):
                m = person_mask[i].cpu().numpy().astype(np.uint8)  # 0/1
                H, W = m.shape

                num_labels, labels = cv2.connectedComponents(m, connectivity=8)

                image_id = batch_idx * B + i
                people_in_this_image = []

                for lbl in range(1, num_labels):  # 0 — фон
                    ys, xs = np.where(labels == lbl)
                    if ys.size == 0:
                        continue

                    y_min, y_max = ys.min(), ys.max()
                    x_min, x_max = xs.min(), xs.max()

                    h = y_max - y_min + 1
                    w = x_max - x_min + 1
                    area = int(h * w)

                    # фильтр по площади — отсечь шум
                    if area < area_threshold:
                        continue

                    people_in_this_image.append({
                        "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
                        "area": area
                    })

                if len(people_in_this_image) > 0:
                    results[image_id] = people_in_this_image
                    total_images_with_people += 1
                    total_instances += len(people_in_this_image)

                    # печать краткой инфы по изображению
                    print(
                        f"\nImage #{image_id} (batch {batch_idx}, idx {i}) — людей: {len(people_in_this_image)}")
                    for j, person in enumerate(people_in_this_image):
                        x_min, y_min, x_max, y_max = person["bbox"]
                        print(
                            f"  Person {j}: bbox=({x_min},{y_min},{x_max},{y_max}), "
                            f"area={person['area']} px^2"
                        )

    print("\n==== Итог по датасету ====")
    print(f"Изображений с людьми: {total_images_with_people}")
    print(f"Всего найдено людей (blob'ов): {total_instances}")

    if total_instances > 0:
        avg_area = sum(p["area"] for plist in results.values()
                       for p in plist) / total_instances
        print(f"Средняя площадь человека: {avg_area:.1f} px^2")

    return results



def main():
    # Parse arguments using the separate args.py file
    args = parse_args()

    args.attack_type = "patch"
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    # Prepare dataloader
    data_loader, has_gt = prepare_dataloader(mode='training',
                                             dataset_name=args.dataset, small_run=args.small_run, n_images=1)

    # Отдельный loader только для оценки перцентиля глубины
    depth_loader, _ = prepare_dataloader(
        dataset_name=args.dataset, small_run=args.small_run)
    model = load_model(args.model_name, args.dataset.lower()).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    mde_model = None
    depth_pred = None
    original_depth = None
    mde_target = None
    of_target_fn = get_target(args.target)

    if args.attack_mde:
        mde_model = load_mde_model(model_name=args.mde_model, device=device)
        mde_target_fn = get_mde_target(
            target_name=args.mde_target,
            data_loader=depth_loader,   # отдельный loader!
            flow_model=model,
            mde_model=mde_model,
            device=device,
            q=0.9
        )

    ss_model = None
    if args.attack_ss:
        ss_model = load_seg_model(model_name=args.ss_model, device=device)
        for param in ss_model.parameters():
            param.requires_grad = False
        ss_target_fn = get_ss_target('targeted')

    # if ss_model is not None:
        # people_stats = analyze_people_in_dataset(
        #     data_loader=data_loader,
        #     flow_model=model,
        #     ss_model=ss_model,
        #     device=device,
        #     person_class_id=11,   # если модель выдаёт trainId'ы Cityscapes
        #     area_threshold=50,    # можно подобрать
        #     max_batches=None      # или, например, 100, чтобы не гонять весь датасет
        # )
        # print(people_stats)
    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        model, input_size=(375, 1242), cuda=torch.cuda.is_available())
    if args.trained_patch == '':
        print("Starting patch training...")
        train_tracker = AttackMetricsTracker(
            output_dir=args.output_dir, args=args, train=True)
        trained_patch = train_patch_ptlflow(
            args, model, data_loader, device, io_adapter, train_tracker, mde_model, ss_model)
        trained_patch.save_png('patch.png')
        trained_patch = PatchAdversary('patch.png', size=args.patch_size,
                                       angle=0, scale=1, change_of_variable=args.change_of_variables,
                                       random_location=args.random_loc, image_size=(
                                           375, 1242)).to(device)
        train_tracker.finalize()
    else:
        print(f"Using trained patch from {args.trained_patch}...")
        trained_patch = PatchAdversary(args.trained_patch, size=args.patch_size,
                                       angle=0, scale=1, change_of_variable=args.change_of_variables,
                                       random_location=args.random_loc, image_size=(
                                           375, 1242)).to(device)

    print("Evaluating trained patch...")
    eval_loader, has_gt = prepare_dataloader(mode='testing',
                                             dataset_name=args.dataset, small_run=args.small_run, n_images=1, has_depth=False)
    eval_tracker = AttackMetricsTracker(
        output_dir=args.output_dir, args=args)
    for batch, (images, flow, valid, meta, K) in enumerate(tqdm(eval_loader)):
        io_adapter = ptlflow.utils.io_adapter.IOAdapter(
            model, input_size=images.shape[-2:], cuda=torch.cuda.is_available(
            )
        )
        wrapped_inputs = {'images': images,
                          'flows': flow, 'valids': valid}
        inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)
        images = images.to(device)
        
        I1_batch = images[:, 0, :, :, :] 
        I2_batch = images[:, 1, :, :, :]
        if args.patch_projection:
            attacked_image1, attacked_image2, mask, y, x, road_mask, planes = project_patch_on_scene(
                I1_batch,
                I2_batch,
                K,
                A=trained_patch,
                mde_model=mde_model,
                ss_model=ss_model,
                io_adapter=io_adapter,
                device=device,
                plane_aug=False    # disable randomness in evaluation
            )
        else:
            attacked_image1, attacked_image2, mask, y, x = trained_patch(
                images[:, 0, :, :, :], images[:, 1, :, :, :])
        attacked_images = torch.stack(
            [attacked_image1, attacked_image2], dim=1).squeeze(0)
        with torch.no_grad():
            original_flow = model(inputs)['flows'].squeeze(0)
            of_target = of_target_fn(original_flow)
            if args.attack_mde:
                original_depth = mde_model(inputs)
                mde_target = mde_target_fn(
                    original_depth)
            if args.attack_ss:
                original_ss = ss_model(inputs)
                ss_target = ss_target_fn(original_ss)
        inputs = replace_images_dic(inputs, attacked_images)
        # deltas = torch.clamp(get_image_tensors(
        #     attacked_images).detach().cpu(), 0, 1) - images.squeeze(0)
        with torch.no_grad():
            flow_pred = model(inputs)[
                'flows'].squeeze(0)
            depth_pred = None
            if args.attack_mde:
                depth_pred = mde_model(inputs)
            if args.attack_ss:
                ss_pred = ss_model(inputs)
        eval_tracker.update(
            flow,
            flow_pred,
            gt_flow=flow,
            target_flow=of_target,
            inverse_flow=None,
            valid=valid,
            # tracked_flows=tracked_flows,
            mask=(mask).cpu(),
            original_depth=original_depth,
            attacked_depth=depth_pred,
            target_depth=mde_target,
            tracked_depths=None,
            original_seg=original_ss, attacked_seg=ss_pred, target_seg=ss_target
        )
        # Save artifacts
        if args.save_artifacts:
            eval_tracker.save_artifact(
                get_image_tensors(inputs)[0], f"batch_{batch:04d}_attacked_image", artifact_type="image"
            )
            eval_tracker.save_artifact(
                flow_pred, f"batch_{batch:04d}_attacked_flow", artifact_type="flow"
            )
            eval_tracker.save_artifact(
                original_flow, f"batch_{batch:04d}_init_flow", artifact_type="flow")
            # eval_tracker.save_artifact(
            #     deltas[0], f"batch_{batch:04d}_delta", artifact_type="image"
            # )
            if depth_pred is not None:
                eval_tracker.save_artifact(
                    original_depth, f"batch_{batch:04d}_init_depth", artifact_type="depth")
                eval_tracker.save_artifact(
                    depth_pred, f"batch_{batch:04d}_attacked_depth", artifact_type="depth"
                )
            if args.attack_ss:
                eval_tracker.save_artifact(
                    original_ss, f"batch_{batch:04d}_init_ss", artifact_type="ss")
                eval_tracker.save_artifact(
                    ss_pred, f"batch_{batch:04d}_attacked_ss", artifact_type="ss")
    eval_tracker.finalize()


if __name__ == '__main__':
    main()

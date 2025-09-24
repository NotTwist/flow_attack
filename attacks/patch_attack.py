from DetectionDefenses.helper_functions.patch_adversary import PatchAdversary
from DetectionDefenses.helper_functions.defenses import LGS, ILP
from DetectionDefenses.helper_functions.losses import aee_masked, acs_masked, mse_masked
from DetectionDefenses.helper_functions.ownutilities import preprocess_img as oa_preprocess_img, postprocess_flow as oa_postprocess_flow
# если у тебя такая обёртка есть; иначе адаптируем ниже
from DetectionDefenses.helper_functions.targets import get_target
import torch.optim as optim
import torch
from tqdm import tqdm

def train_patch_ptlflow(args, model, data_loader, device, io_adapter, metrics_tracker=None):
    """
    Обучаем один патч на всем датасете.
    Логика:
      - создаём один PatchAdversary A и один optimizer вне циклов;
      - делаем N эпох (args.n) — каждая эпоха = полный проход по data_loader;
      - для каждого батча делаем args.steps внутренних шагов (или 1 шаг, если steps==1).
    """
    import torch.optim as optim
    from itertools import islice
 
    # 1) Создаём adversary и optimizer **один раз**
    A = PatchAdversary(None, size=args.patch_size,
                       angle=[-10,10], scale=[0.95,1.05],
                       change_of_variable=args.change_of_variables,
                       random_location=args.random_loc).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(A.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(A.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "clipped-pgd":
        from DetectionDefenses.helper_functions.custom_optimizer import ClippedPGD
        optimizer = ClippedPGD(A.parameters(), lr=args.lr, min_=0, max_=1, max_delta=args.max_delta)
    else:
        optimizer = optim.Adam(A.parameters(), lr=args.lr)

    # defense (один раз)
    D = None
    if args.defense == "lgs":
        D = LGS(args.k, args.o, args.t, args.s, "forward")
    elif args.defense == "ilp":
        D = ILP(args.k, args.o, args.t, args.s, args.r, "forward")

    # training loops: outer = эпохи/итерации по всему датасету
    for epoch in range(args.n):
        print(f"Epoch {epoch+1}/{args.n}")
        # пробегаем через весь даталоадер — патч обновляется на основе всего датасета (в сумме через эпохи)
        for batch_idx, (images, flow_gt, valid, meta) in enumerate(tqdm(data_loader)):
            # устройства
            images = images.to(device)
            flow_gt = flow_gt.to(device)
            if isinstance(valid, torch.Tensor):
                valid = valid.to(device)

            B = images.shape[0]
            # единый батч может содержать несколько пар (B>1). Мы усредним лосс по батчу.
            # для каждого батча — можно делать несколько внутренних шагов
            for inner_step in range(args.steps):
                optimizer.zero_grad()
                # собираем лосс по батчу
                batch_loss = 0.0
                for b in range(B):
                    # извлекаем кадры
                    I_pair = images[b]           # [2,C,H,W]
                    I1 = I_pair[0].unsqueeze(0)  # [1,C,H,W]
                    I2 = I_pair[1].unsqueeze(0)

                    # получаем атакованные изображения и маску (батчовые тензоры)
                    I1_p, I2_p, M, y, x = A(I1, I2)  # A ожидает батч
                    # M shape: [1,1,H,W] или [1,H,W] — адаптируй под реализацию

                    # применяем defense, если задан
                    if D is not None:
                        I1_att_def, I2_att_def = D(I1_p, I2_p, M)
                        I1_unatt_def, I2_unatt_def = D(I1, I2, torch.zeros_like(M))
                    else:
                        I1_att_def, I2_att_def = I1_p, I2_p
                        I1_unatt_def, I2_unatt_def = I1, I2

                    # упаковываем в формат IOAdapter (батч = 1)
                    attacked_images_batch = torch.cat([I1_att_def, I2_att_def], dim=0).unsqueeze(0)  # [1,2,C,H,W]
                    unattacked_images_batch = torch.cat([I1_unatt_def, I2_unatt_def], dim=0).unsqueeze(0)

                    attacked_inputs = io_adapter.prepare_inputs({'images': attacked_images_batch, 'flows': None, 'valids': None})
                    unattacked_inputs = io_adapter.prepare_inputs({'images': unattacked_images_batch, 'flows': None, 'valids': None})

                    # делаем предсказания (без отключения градиента — нам нужны градиенты для параметров патча)
                    pred_attacked = model(attacked_inputs)['flows'].squeeze(0)    # форма зависит от модели
                    pred_unattacked = model(unattacked_inputs)['flows'].squeeze(0)

                    # если нужно привести маску к пространственному размеру потока:
                    # M_flow = torch.nn.functional.interpolate(M, size=pred_attacked.shape[-2:], mode='nearest')
                    # если aee_masked ожидает (flow_ref, flow_pred, mask) — проверь сигнатуру
                    # Используем aee_masked как пример
                    # убедимся, что размеры совпадают
                    if M.dim() == 4 and M.shape[1] == 1:
                        M_flow = torch.nn.functional.interpolate(M, size=pred_attacked.shape[-2:], mode='nearest')
                    else:
                        M_flow = M

                    loss_b = aee_masked(pred_attacked, pred_unattacked, 1 - M_flow)
                    batch_loss = batch_loss + loss_b

                # усредняем по батчу
                batch_loss = batch_loss / float(B)
                # backward + step
                batch_loss.backward()
                optimizer.step()

                # опционально — clamp параметров патча
                if not args.change_of_variables and args.optimizer not in ["ifgsm","pgd"]:
                    with torch.no_grad():
                        A.P.clamp_(0,1)

            # логирование и/или обновление метрик (подставь свой metrics_tracker)
            if metrics_tracker is not None:
                # можно логгировать состояние модели/патча/последний loss
                metrics_tracker.log_metric("epoch_batch_loss", batch_loss.item(), step=epoch*len(data_loader)+batch_idx)

        # по окончанию эпохи можно сохранять патч
        torch.save(A.state_dict(), op.join(args.output_dir, f"patch_epoch_{epoch+1}.pth"))
        print(f"Saved patch for epoch {epoch+1}")

    return A

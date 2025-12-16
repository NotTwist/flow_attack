# server_mymodel.py
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query
import os
import sys
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE)
from models.model_utils import load_mde_model, load_seg_model
import ptlflow.utils.io_adapter
from utils.args import parse_args
from utils.targets import get_mde_target
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
import torch
import ptlflow
# импортируй свои модули
from attacks.get_attacks import get_attack
from utils.process_images import preprocess_img, get_image_tensors
from typing import List

def load_model(model_name, dataset):
    model_ref = ptlflow.get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()
    for c in checkpoints:
        if c in dataset:
            model = ptlflow.get_model(model_name, c)
            return model
    print(f"No pre-trained model available for {model}/{dataset}.")
    return None

app = FastAPI()

# --- инициализация моделей и атаки (пример, адаптируй под args) ---
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(args.model_name, args.dataset.lower()).to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = False

mde_model = None
depth_pred = None
original_depth = None
mde_target = None
tracked_depths = None
if args.attack_mde:
    mde_model = load_mde_model(model_name=args.mde_model, device=device)
    # создаём функцию-таргет, которая внутри себя лениво посчитает p90
    mde_target_fn = get_mde_target(
        target_name=args.mde_target,
        data_loader=None,   # отдельный loader!
        flow_model=model,
        mde_model=mde_model,
        device=device,
        q=0.9
    )

ss_model = None
original_ss = None
ss_pred = None
ss_target = None
tracked_ss = None
if args.attack_ss:
    ss_model = load_seg_model(model_name=args.ss_model, device=device)
    for param in ss_model.parameters():
        param.requires_grad = False

# Set the attack based on argument
attack = get_attack(args.attack_type, model, mde_model=mde_model, mde_target=mde_target_fn, ss_model=ss_model, ss_target=args.ss_target, num_steps=args.steps, no_softmax=args.no_softmax,
                    target=args.target, epsilon=args.epsilon, save_iterations=args.saved_iterations, alpha=args.alpha, target_layer=args.target_layer, use_map_scaling=args.use_map_scaling, scaling_type=args.scaling_type, num_samples=args.num_samples, loss=args.loss, loss_weights=args.loss_weights)


@app.post("/predict")
async def predict_batch(
    files: List[UploadFile] = File(...),
    attack_flag: Optional[bool] = Query(default=False)
):
    # ожидаем минимум 2 файла
    if len(files) < 2:
        return {"error": "Need at least two frames"}

    imgs = []
    for f in files:
        data = await f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": f"Could not decode {f.filename}"}
        imgs.append(img)

    # теперь imgs — список из numpy-изображений, len = 2 (или >2)
    # конвертируем их в тензор (B, C, H, W)
    tensors = []
    for img in imgs:
        t = torch.tensor(img).permute(2, 0, 1) / 255.  # C, H, W
        tensors.append(t)
    # shape (1, B, C, H, W) — но, возможно, модель ожидает (B, C, H, W)
    batch = torch.stack(tensors, dim=0)
    batch = batch.to(device)  # (B, C, H, W)
    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        model, input_size=batch.shape[-2:], cuda=torch.cuda.is_available())
    wrapped_inputs = {'images': batch}
    inputs = io_adapter.prepare_inputs(inputs=wrapped_inputs)

    # Запуск предсказаний / атаки
    with torch.no_grad():
        depth_pred = mde_model(inputs) if mde_model else None
        seg_pred = ss_model(inputs) if ss_model else None
    attacked_images = inputs
    if attack is not None and attack_flag is True:
        # исходя из структуры твоего attack API
        res = attack.attack(inputs)
        attacked_images = res["final_images"]
        with torch.no_grad():
            if args.attack_mde:
                depth_pred = mde_model(attacked_images)
            if args.attack_ss:
                seg_pred = ss_model(attacked_images)
    # Сериализуй depth и seg в список или bytes
    depth_np = depth_pred.squeeze().cpu().numpy() if depth_pred is not None else None
    seg_np = seg_pred.squeeze().cpu().numpy() if seg_pred is not None else None

    # Пример — flatten + tolist (негатив: медленно, но просто)
    result = {}
    if depth_np is not None:
        result["depth_shape"] = depth_np.shape
        result["depth"] = depth_np.flatten().tolist()
    if seg_np is not None:
        result["seg_shape"] = seg_np.shape
        result["seg"] = seg_np.flatten().tolist()
    attacked_img = get_image_tensors(attacked_images)[0]
    result["img_shape"] = attacked_img.shape
    result["attacked_image"] = attacked_img.detach().cpu().numpy().tolist()

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

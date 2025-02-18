import torch
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import import_and_load, compute_flow
from utils.process_images import preprocess_img, postprocess_flow, quickvis_flow
from attacks.fgsm import FGSMOpticalFlowAttack

if __name__ == '__main__':
    net = 'RAFT'
    data_loader, has_gt = prepare_dataloader(dataset_name='Kitti15', small_run=True)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Setting Device to %s\n" % device)

    model = import_and_load(net, make_scaled_input_model=True, device = device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    attack = FGSMOpticalFlowAttack(model=model)
    for batch, (images, flow, valid) in enumerate(tqdm(data_loader)):
        for i in range(len(images)):
                 images[i] = images[i].to(device)
                 images[i].requires_grad = True
        flow = flow.to(device)

        padder, images = preprocess_img(net, *images)
        attacked_images = attack.attack(images)
        flow_pred = compute_flow(model, "scaled_input_model", attacked_images)
        [flow_pred] = postprocess_flow(net, padder, flow_pred)
        quickvis_flow(flow_pred, f'experiment_data/{batch}_flow.png')

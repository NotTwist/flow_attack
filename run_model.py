import torch
from tqdm import tqdm
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import import_and_load, compute_flow
from utils.process_images import preprocess_img, postprocess_flow, model_takes_unit_input, quickvis_flow

if __name__ == '__main__':
    net = 'MeFlow'
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

    for batch, (images, flow, valid) in enumerate(tqdm(data_loader)):
        for i in range(len(images)):
                 images[i] = images[i].to(device)
        flow = flow.to(device)

        padder, images = preprocess_img(net, *images)
        flow_pred = compute_flow(model, "scaled_input_model", images)
        [flow_pred] = postprocess_flow(net, padder, flow_pred)
        quickvis_flow(flow_pred, f'experiment_data/{batch}_flow.png')

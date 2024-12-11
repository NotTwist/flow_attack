import torch


def import_and_load(net='RAFT', make_unit_input=False, variable_change=False, device=torch.device("cpu"), make_scaled_input_model=False, **kwargs):
    """import a model and load pretrained weights for it

    Args:
            net (str, optional):
                    the desired network to load. Defaults to 'RAFT'.
            make_unit_input (bool, optional):
                    model will assume input images in range [0,1] and transform to [0,255]. Defaults to False.
            variable_change (bool, optional):
                    apply change of variables (COV). Defaults to False.
            device (torch.device, optional):
                    changes the selected device. Defaults to torch.device("cpu").
            make_scaled_input_model (bool, optional):
                    load a scaled input model which uses make_unit_input and variable_change as specified. Defaults to False.

    Raises:
            RuntimeWarning: Unknown model type

    Returns:
            torch.nn.Module: PyTorch optical flow model with loaded weights 
    """

    if make_unit_input == True or variable_change == True or make_scaled_input_model:
        from scaledInputModel import ScaledInputModel
        model = ScaledInputModel(net, make_unit_input=make_unit_input,
                                 variable_change=variable_change, device=device, **kwargs)
        print("--> transforming model to 'make_unit_input'=%s, 'variable_change'=%s\n" %
              (str(make_unit_input), str(variable_change)))

    else:
        model = None
        try:
            if net == 'RAFT':
                from models.raft.raft import RAFT

                # set the path to the corresponding weights for initializing the model
                path_weights = 'models/_pretrained_weights/raft-sintel.pth'

                # possible adjustements to the config can be made in the file
                # found under models/_config/raft_config.json
                with open("models/_config/raft_config.json") as file:
                    config = json.load(file)

                model = torch.nn.DataParallel(RAFT(config))
                # load pretrained weights
                model.load_state_dict(torch.load(
                    path_weights, map_location=device))

            elif net == 'GMA':
                from models.gma.network import RAFTGMA

                # set the path to the corresponding weights for initializing the model
                path_weights = 'models/_pretrained_weights/gma-sintel.pth'

                # possible adjustements to the config file can be made
                # under models/_config/gma_config.json
                with open("models/_config/gma_config.json") as file:
                    config = json.load(file)
                    # GMA accepts only a Namespace object when initializing
                    config = Namespace(**config)

                model = torch.nn.DataParallel(RAFTGMA(config))

                model.load_state_dict(torch.load(
                    path_weights, map_location=device))

            elif net == 'PWCNet':
                from models.PWCNet.PWCNet import PWCDCNet

                # set path to pretrained weights:
                path_weights = 'models/_pretrained_weights/pwc_net_chairs.pth.tar'

                model = PWCDCNet()

                weights = torch.load(path_weights, map_location=device)
                if 'state_dict' in weights.keys():
                    model.load_state_dict(weights['state_dict'])
                else:
                    model.load_state_dict(weights)
                model.to(device)

            elif net == 'SpyNet':
                from models.SpyNet.SpyNet import Network as SpyNet
                # weights for SpyNet are loaded during initialization
                model = SpyNet(nlevels=6, pretrained=True)
                model.to(device)

            elif net == "FlowNet2":
                from models.FlowNet.FlowNet2 import FlowNet2

                # hard coding configuration for FlowNet2
                args_fn = Namespace(fp16=False, rgb_max=255.0)

                # set path to pretrained weights
                path_weights = 'models/_pretrained_weights/FlowNet2_checkpoint.pth.tar'
                model = FlowNet2(args_fn, div_flow=20, batchNorm=False)

                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights['state_dict'])

                model.to(device)
            if model is None:
                raise RuntimeWarning(
                    'The network %s is not a valid model option for import_and_load(network). No model was loaded. Use "RAFT", "GMA", "FlowNetC", "PWCNet" or "SpyNet" instead.' % (net))
        except FileNotFoundError as e:
            print("\nLoading the model failed, because the checkpoint path was invalid. Are the checkpoints placed in models/_pretrained_weights/? If this folder is empty, consider to execute the checkpoint loading script from scripts/load_all_weights.sh. The full error that caused the loading failure is below:\n\n%s" % e)
            exit()

        print("--> flow network is set to %s" % net)
    return model


def compute_flow(model, network, images, test_mode=True, **kwargs):
    """subroutine to call the forward pass of the network

    Args:
            model (torch.nn.module):
                    instance of optical flow model
            network (str):
                    name of the network. [scaled_input_model | RAFT | GMA | FlowNet2 | SpyNet | PWCNet]
            x1 (tensor):
                    first image of a frame sequence
            x2 (tensor):
                    second image of a frame sequence
            test_mode (bool, optional):
                    applies only to RAFT and GMA such that the forward call yields only the final flow field. Defaults to True.

    Returns:
            tensor: optical flow field
    """
    if network == "scaled_input_model":
        flow = model(images, test_mode=True, **kwargs)

    elif network == 'RAFT':
        _, flow = model(images[0], images[1], test_mode=test_mode, **kwargs)

    elif network == 'GMA':
        _, flow = model(images[0], images[1], iters=6,
                        test_mode=test_mode, **kwargs)

    elif network[:7] == 'FlowNet':
        # all flow net types need image tensor of dimensions [batch, colors, image12, x, y] = [b,3,2,x,y]
        x = torch.stack((images[0], images[1]), dim=-3)
        # FlowNet2-variants: all fine now, input [0,255] is taken.

        if not network[:8] == 'FlowNet2':
            # FlowNet variants need input in [-1,1], which is achieved by substracting the mean rgb value from the image in [0,1]
            rgb_mean = x.contiguous().view(
                x.size()[:2]+(-1,)).mean(dim=-1).view(x.size()[:2] + (1, 1, 1,)).detach()
            x = x - rgb_mean

        flow = model(x)
    elif network == 'MeFlow':
        _, flow = model(images[0], images[1], test_mode=test_mode)
    elif network == 'PWCNet' or network == 'SpyNet':  # works for PWCNet, SpyNet
        flow = model(images[0], images[1], **kwargs)
    return flow

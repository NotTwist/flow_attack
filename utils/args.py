import argparse
from ptlflow.utils.utils import get_list_of_available_models_list

epsilon = 8/255.
iterations = 20
alpha = 0.01
num_samples = 5


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial Attack on Optical Flow Models")

    # Model selection argument
    parser.add_argument('--model_name', type=str, default='raft', choices=get_list_of_available_models_list(),
                        help="Neural network model to use for optical flow estimation.")

    # Attack selection argument
    parser.add_argument('--attack_type', type=str, default='FGSM', choices=['FGSM', 'PGD', 'CosPGD', 'GradCAM', 'APGD', 'MIFGSM', 'Sobel', 'HighFreq', 'MIFGSM_CosPGD', 'NIFGSM', 'PIFGSM', 'EMIFGSM', 'VMIFGSM', 'VNIFGSM', 'ADAMIFGSM'],
                        help="Adversarial attack method to use.")

    parser.add_argument('--target', type=str, default='zero', choices=[
                        'zero', 'neg_flow', 'untargeted', 'camera', 'scene', 'down', 'relative_down'], help="Choose a target for an attack")

    # Dataset selection argument
    parser.add_argument('--dataset', type=str, default='Kitti15', choices=['Kitti15', 'Sintel', 'carla'],
                        help="Dataset to use for evaluation.")

    # Small run argument (for debugging purposes)
    parser.add_argument('--small_run', action='store_true',
                        help="Flag to run a smaller version of the dataset for testing purposes.")
    parser.add_argument('--subset_size', type=int, default=0,
                        help="Use a random subset of N images (0 = use all). More flexible than --small_run.")
    parser.add_argument('--eval_mode', type=str, default='testing', choices=['training', 'testing'],
                        help="Dataset split used for patch evaluation.")

    # Output directory for saving results
    parser.add_argument('--output_dir', type=str, default="experiment_data",
                        help="Directory to save experiment outputs.")
    
    parser.add_argument('--experiment_name', type=str, default="attack_experiment",
                        help="Name of experiment for mlflow.")

    # Save artifacts flag
    parser.add_argument('--save_artifacts', action='store_true',
                        help="Flag to save artifacts such as images and flows.")
    parser.add_argument('--save_diploma_artifacts', action='store_true',
                        help="Save publication-ready before/after artifacts with shared normalization and mask overlays.")
    parser.add_argument('--eval_artifact_limit', type=int, default=0,
                        help="Maximum number of eval batches to save artifacts for. 0 = no limit when artifact saving is enabled.")

    # Num of steps for iterative attacks
    parser.add_argument('--steps', type=int, default=iterations,
                        help="Number of steps for iterative attacks.")
    # Choose which iterations of iterative attacks will be saved for metrics
    parser.add_argument('--saved_iterations', type=int, nargs='*',
                        default=[], help="Save metrics for specified iterations of attack")

    parser.add_argument('--epsilon', type=float, default=epsilon,
                        help="Perturbation budget for an attack in pixels.")

    parser.add_argument('--alpha', type=float, default=alpha,
                        help="Step size for an attack.")
    parser.add_argument(
        "--loss",
        nargs="+",               # позваляет --loss aee:1.0 mse:0.5   или --loss aee,mse:0.5
        default=["aee"],         # по умолчанию "aee" (weight 1.0)
        help=("Loss spec(s). Формат: name или name:weight. "
              "Можно перечислять через пробел или запятую. "
              "Примеры: --loss aee:1.0,mse:0.5  или  --loss aee:1.0 --loss mse:0.5")
    )
    parser.add_argument(
        "--normalize_losses",
        action="store_true",
        help="Normalize loss weights so they sum to 1."
    )
    parser.add_argument("--no_softmax", action='store_true',
                        help='Dont use softmax in cosine distance in CosPGD attack')

    parser.add_argument("--target_layer", type=str, default='update_block.mask',
                        choices=['update_block.mask', 'cnet.conv2', 'update_block.flow_head'], help="Choose which layer will be used for GradCAM attack (works only for raft)")
    parser.add_argument('--use_map_scaling', action='store_true',
                        help='Apply exponential transformation to GradCAM map')
    parser.add_argument('--scaling_type', default='none', choices=[
                        'none', 'cospgd', 'high_freq', 'sobel'], help='Apply scaling to method (workds for MI-FGSM only)')
    parser.add_argument('--num_samples', type=int, default=num_samples,
                        help="Number of samples for attacks like EMIFGSM, VNIFGSM, VMIFGSM.")

    parser.add_argument("--attack_mde", action='store_true',
                        help='Attack MDE model at the same time')

    parser.add_argument('--mde_model', type=str, default='depth-anything-v2', choices=['depth-anything-v2', 'marigold'],
                        help="Neural network model to use for monocular depth estimation.")

    parser.add_argument('--mde_target', type=str, default='zero',
                        choices=['zero', 'zero_raw', 'untargeted', 'infinite', 'scene', 'p90', 'near', 'far'], help="Choose a target for an mde attack")
    parser.add_argument('--mde_near_margin', type=float, default=0.1,
                        help="Margin as fraction of per-image raw MDE range for near/far MDE targets.")

    parser.add_argument(
        "--loss_weights",
        nargs=3,
        type=float,
        default=[1.0, 0.1, 1.0],
        metavar=('FLOW_W', 'MDE_W', 'SS_W'),
        help="3 floats: weights for optical-flow, MDE and SS losses (e.g. --loss_weights 1.0 0.1 1.0)"
    )

    parser.add_argument(
        "--weight_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "normalized", "minmax"],
        help=("How to set per-task loss weights. "
              "'fixed': use --loss_weights as constant wi (Eq.1 with wi=const); "
              "'normalized': wi = user_weight / Li(x,y) on clean image (Eq.1 with wi=1/Li); "
              "'minmax': APGDA from Guo et al. ICASSP 2025 — Algorithm 1, Eq.2-4")
    )
    parser.add_argument(
        "--minmax_alpha_w",
        type=float,
        default=0.03,
        help="Learning rate alpha2 for the inner weight update in min-max (Eq.4 in Guo et al.)"
    )
    parser.add_argument(
        "--minmax_gamma",
        type=float,
        default=5.0,
        help="Regularization coefficient gamma towards uniform weights (Eq.2 in Guo et al.)"
    )

    parser.add_argument("--attack_ss", action='store_true',
                        help='Attack SS model at the same time')

    parser.add_argument('--ss_model', type=str, default='pspnet_cityscapes', choices=['deeplabv3', 'pspnet_cityscapes', 'segformer_cityscapes', 'mask2former_cityscapes'],
                        help="Neural network model to use for semantic segmentation.")

    parser.add_argument('--ss_target', type=str, default='targeted',
                        choices=['targeted', 'untargeted'], help="Choose a target for an ss attack")

    parser.add_argument('--ss_focal_gamma', type=float, default=2.0,
                        help="Focal loss gamma for SS. 0 = plain cross-entropy, >0 = focal loss (default: 2.0)")


    # patch attacks
    parser.add_argument('--trained_patch', type=str, default='',
                    help="Path to a trained patch (train new patch if empty)")
    parser.add_argument('--save_patch_every', type=int, default=0,
                        help="During patch training, save an intermediate patch PNG every N batches. 0 = only save each epoch.")
    parser.add_argument('--patch_checkpoint_dir', type=str, default='',
                        help="Directory for intermediate patch PNGs. Defaults to <output_dir>/patch_checkpoints.")

    parser.add_argument('--baseline', action='store_true',
                        help="Skip training and evaluate a fixed baseline patch: "
                             "random noise for 'pixel' parametrization, "
                             "diffusion_base_image for 'diffusion' parametrization")

    parser.add_argument('--patch_size', type=int, default=100,
                        help="Size of the adversarial patch")

    parser.add_argument(
        '--patch_parametrization',
        type=str,
        default='pixel',
        choices=['pixel', 'diffusion'],
        help="Patch parametrization: legacy pixel patch or diffusion latent patch"
    )

    parser.add_argument('--change_of_variables', action='store_true',
                        help="Use change-of-variable trick in patch optimization")
    parser.add_argument('--random_loc', action='store_false',
                        help="Randomize patch location")
    parser.add_argument("--optimizer", dest="optimizer", default="ifgsm",
                        help="optimizer. LBFGS doesn't work without change_of_variables", choices=["clipped-pgd", "ifgsm", "adam", "lbfgs", "sgd"])
    parser.add_argument("--max_delta", dest="max_delta", type=float, default=0.008,
                        help="maximum delta for parameter update. Gradients will be clipped when using Projected Gradient Descent")

    parser.add_argument(
        "--defense", dest="defense", default="none", help="Define defense", choices=['none', 'lgs', 'ilp', 'temporal-avg', 'temporal-median', 'temporal-bilateral', 'temporal-domain-transform'])


    # --- Параметры временных (Temporal) фильтров ---
    parser.add_argument(
        "--temp_window", 
        type=int, 
        default=5, 
        help="Size of the temporal window (number of frames to look back)"
    )

    parser.add_argument(
        "--sigma_color", 
        type=float, 
        default=0.1, 
        help="Range sigma for Bilateral/DomainTransform (color sensitivity). Lower = more edge preservation."
    )

    parser.add_argument(
        "--sigma_spatial", 
        type=float, 
        default=30.0, 
        help="Spatial/Temporal sigma for Domain Transform (smoothness strength)."
    )

    parser.add_argument(
        "--k", dest="k", type=int, default=16, help="blocksize")
    parser.add_argument(
        "--o", dest="o", type=int, default=8, help="overlap")
    parser.add_argument(
        "--t", dest="t", type=float, default=0.15, help="blockwise filtering threshold")
    parser.add_argument(
        "--s", dest="s", type=float, default=15.0, help="smoothing/scaling parameter depending on the defense")
    parser.add_argument(
        "--r", dest="r", type=int, default=5, help="inpainting radius")
    parser.add_argument('--n', type=int, default=50,
                        help="Number of epochs for patch training")
    parser.add_argument(
        "--lr", dest="lr", type=float, default=.1, help="learning rate of the optimizer")


    # adv manhole
    parser.add_argument("--patch_projection", action='store_true',
                                               help='Use patch projection from adv manhole')


    parser.add_argument(
        "--y_scale",  type=float, default=1, help="patch scaling on y axis")
    parser.add_argument(
        "--flow_shift", type=float, default=0,
        help="Vertical pixel shift of patch between frame1 and frame2 "
             "(simulates ego-motion). Positive = patch moves down in frame2.")
    parser.add_argument(
        "--flow_target_magnitude", type=float, default=1.0,
        help="Magnitude of the 'down' flow target vector (default 1.0).")
    parser.add_argument(
        "--down_hinge_min_mag_ratio",
        type=float,
        default=0.8,
        help="For target=down, minimum predicted flow magnitude as a fraction of --flow_target_magnitude.")
    parser.add_argument(
        "--down_loss",
        type=str,
        default="hinge",
        choices=["hinge", "epe"],
        help=(
            "Flow loss used when --target down. "
            "'hinge' penalizes near-zero flow collapse; 'epe' uses the standard configured flow loss to the down target."
        ),
    )
    parser.add_argument(
        "--down_hinge_horizontal_weight",
        type=float,
        default=0.1,
        help="For target=down, weight for suppressing horizontal flow in the hinge loss.")
    parser.add_argument(
        "--down_hinge_magnitude_weight",
        type=float,
        default=0.5,
        help="For target=down, weight for penalizing near-zero flow magnitude in the hinge loss.")
    parser.add_argument(
        "--down_hinge_vertical_weight",
        type=float,
        default=1.0,
        help="For target=down, weight for the vertical margin term in the hinge loss.")
    parser.add_argument(
        "--tv_weight",
        type=float,
        default=0,
        help="Weight for total variation regularization"
    )

    # Non-printability score weight
    parser.add_argument(
        "--nps_weight",
        type=float,
        default=0,
        help="Weight for non-printability score regularization"
    )

    parser.add_argument(
        "--diffusion_model",
        type=str,
        default="sdxl",
        choices=["sd14", "sd15", "sd21", "sdxl"],
        help="Stable Diffusion backbone used for latent patch optimization"
    )
    parser.add_argument(
        "--diffusion_model_path",
        type=str,
        default="",
        help="Optional local path or HF model id overriding --diffusion_model"
    )
    parser.add_argument(
        "--diffusion_prompt",
        type=str,
        default="",
        help="Prompt used to denoise latent patches; required for image+nulltext mode"
    )
    parser.add_argument(
        "--diffusion_init_mode",
        type=str,
        default="random",
        choices=["random", "image"],
        help="Initialize the latent patch from random noise or from a base image with null-text optimization"
    )
    parser.add_argument(
        "--diffusion_base_image",
        type=str,
        default="",
        help="Base image path used for DDIM inversion and null-text optimization"
    )
    parser.add_argument(
        "--diffusion_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="Precision for Stable Diffusion inference"
    )
    parser.add_argument(
        "--diffusion_decode_mode",
        type=str,
        default="denoise",
        choices=["denoise", "direct"],
        help=(
            "How to render the optimized diffusion latent. "
            "'denoise' runs the DDIM reverse trajectory; 'direct' decodes the optimized latent directly."
        )
    )
    parser.add_argument(
        "--diffusion_source_steps",
        type=int,
        default=50,
        help="Total DDIM steps used by the diffusion patch generator"
    )
    parser.add_argument(
        "--diffusion_reverse_steps",
        type=int,
        default=25,
        help="How many DDIM denoising steps are used to render the patch"
    )
    parser.add_argument(
        "--diffusion_guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale for the diffusion patch generator"
    )
    parser.add_argument(
        "--diffusion_latent_eps",
        type=float,
        default=0.5,
        help="L-infinity budget for the optimized latent delta"
    )
    parser.add_argument(
        "--diffusion_null_inner_steps",
        type=int,
        default=15,
        help="Inner optimization steps for null-text inversion"
    )
    parser.add_argument(
        "--diffusion_null_epsilon",
        type=float,
        default=1e-5,
        help="Early-stop threshold for null-text inversion"
    )
    parser.add_argument(
        "--diffusion_optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "ifgsm", "clipped-pgd"],
        help="Optimizer used for latent-space updates when diffusion parametrization is enabled"
    )
    parser.add_argument(
        "--diffusion_seed",
        type=int,
        default=42,
        help="Seed used for random latent initialization"
    )
    
    parser.add_argument("--plane_aug", action='store_true',
                        help='Attack SS model at the same time')

    return parser.parse_args()

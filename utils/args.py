import argparse


epsilon = 8
iterations = 20
alpha = 0.01

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial Attack on Optical Flow Models")

    # Model selection argument
    parser.add_argument('--net', type=str, default='RAFT', choices=['RAFT', 'GMA', 'PWCNet', 'SpyNet'],
                        help="Neural network model to use for optical flow estimation.")

    # Attack selection argument
    parser.add_argument('--attack', type=str, default='FGSM', choices=['FGSM', 'PGD', 'CosPGD', 'GradCAM'],
                        help="Adversarial attack method to use.")
    
    parser.add_argument('--target', type=str, default='zero', choices=['zero', 'neg_flow', 'untargeted'], help="Choose a target for an attack")

    # Dataset selection argument
    parser.add_argument('--dataset', type=str, default='Kitti15', choices=['Kitti15', 'Sintel'],
                        help="Dataset to use for evaluation.")

    # Small run argument (for debugging purposes)
    parser.add_argument('--small_run', action='store_true',
                        help="Flag to run a smaller version of the dataset for testing purposes.")

    # Output directory for saving results
    parser.add_argument('--output_dir', type=str, default="experiment_data",
                        help="Directory to save experiment outputs.")

    # Save artifacts flag
    parser.add_argument('--save_artifacts', action='store_true',
                        help="Flag to save artifacts such as images and flows.")
    
    # Num of steps for iterative attacks
    parser.add_argument('--steps', type=int, default=iterations,
                        help="Number of steps for iterative attacks.")
    # Choose which iterations of iterative attacks will be saved for metrics
    parser.add_argument('--save_iterations', type=int, nargs='*', default=[], help="Save metrics for specified iterations of attack")
    
    parser.add_argument('--epsilon', type=float, default=epsilon,
                        help="Perturbation budget for an attack in pixels.")
    
    parser.add_argument('--alpha', type=float, default=alpha,
                        help="Step sizew for an attack.")

    
    parser.add_argument("--no_softmax", action='store_true', help='Dont use softmax in cosine distance in CosPGD attack')
    return parser.parse_args()

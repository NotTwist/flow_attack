import torch
from attacks.DetectionDefenses.helper_functions.patch_adversary import PatchAdversary
from utils.args import parse_args

args = parse_args()
# 1️⃣ Create the PatchAdversary with the same settings as during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = PatchAdversary(
    None,
    size=args.patch_size,
    angle=[-10, 10],
    scale=[0.95, 1.05],
    change_of_variable=args.change_of_variables,
    random_location=args.random_loc
).to(device)

# 2️⃣ Load the saved state dict
patch_path = "experiment_data/patch_epoch_1000.pth"
A.load_state_dict(torch.load(patch_path, map_location=device))

# 3️⃣ Put it in eval mode if you want to apply it without further training
A.eval()
A.save_png('patch.png')
# ✅ Now you can use A to attack images, e.g.:
# I1_p, I2_p, M, y, x = A(I1, I2)

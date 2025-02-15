import torch
import argparse
from monai.networks.nets import SwinUNETR
roi = 64
parser = argparse.ArgumentParser(description="Swin UNETR")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--pretrained_checkpoint", default=r'E:\skx\Breast-Ultrasound\VoCo/VoCo_10k.pt', type=str, help="use gradient checkpointing to save memory")
args = parser.parse_args()
model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        use_v2=True)
model_dict = torch.load(args.pretrained_checkpoint, map_location=torch.device('cpu'))
state_dict = model_dict
if "module." in list(state_dict.keys())[0]:
    print("Tag 'module.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("module.", "")] = state_dict.pop(key)
if "swin_vit" in list(state_dict.keys())[0]:
    print("Tag 'swin_vit' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
model.load_state_dict(state_dict, strict=False)
print("Using pretrained voco ema self-supervised Swin UNETR backbone weights !")
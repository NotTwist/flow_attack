from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import matplotlib
from metrics.attack_metrics import AttackMetricsTracker
from utils.args import parse_args
matplotlib.use('Agg')
config_file = 'PSPNet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'PSPNet/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
args = parse_args()
# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# or img = mmcv.imread(img), which will only load it once
img = 'mlruns/914879025403898454/bf5a2c6352e641d794fe2fe1ea4d1cf2/artifacts/batch_0000_attacked_image.png'
result = inference_model(model, img)
print(result)
# visualize the results in a new window
metrics_tracker = AttackMetricsTracker(
    output_dir=args.output_dir, args=args)
metrics_tracker.save_artifact(
    result.pred_sem_seg.data, f"init_ss", artifact_type="ss")
show_result_pyplot(model, img, result, save_dir='.', out_file='res.png', show=False)
# # or save the visualization results to image files
# # you can change the opacity of the painted segmentation map in (0, 1].
# show_result_pyplot(model, img, result, show=True,
#                    out_file='result.jpg', opacity=0.5)
# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#    result = inference_model(model, frame)
#    show_result_pyplot(model, frame, result, wait_time=1)

import os
import shutil

now_dir = os.path.dirname(os.path.abspath(__file__))
from modelscope.hub.snapshot_download import snapshot_download
if not os.path.isfile(os.path.join(now_dir,"UniAnimate","checkpoints","unianimate_16f_32f_non_ema_223000.pth")):
    snapshot_download('iic/unianimate', cache_dir=os.path.join(now_dir,'checkpoints'))
    shutil.move(os.path.join(now_dir,"checkpoints","iic","unianimate"),os.path.join(now_dir,"UniAnimate","checkpoints"))
    shutil.rmtree(os.path.join(now_dir,'checkpoints'))
else:
    print("UniAnimate use cache models,make sure your 'UniAnimate/checkpoints' complete")

from .nodes import PoseAlignNode, UniAnimateNode,LoadImagePath,PreViewVideo,LoadVideo
# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./web"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PoseAlignNode":PoseAlignNode, 
    "UniAnimateNode":UniAnimateNode,
    "LoadImagePath":LoadImagePath,
    "PreViewVideo":PreViewVideo,
    "LoadVideo":LoadVideo
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignNode":"PoseAlignNode", 
    "UniAnimateNode":"UniAnimateNode",
    "LoadImagePath":"LoadImagePath",
    "PreViewVideo":"PreViewVideo",
    "LoadVideo":"LoadVideo"
}

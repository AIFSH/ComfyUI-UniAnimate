import os
import sys
import yaml
import folder_paths
from moviepy.editor import VideoFileClip

input_dir = folder_paths.get_input_directory()
output_dir = folder_paths.get_output_directory()
now_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_dir = os.path.join(now_dir,"UniAnimate", "checkpoints")
python_exec = sys.executable or "python"

class PoseAlignNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "ref_name":("IMAGE",),
                "source_video_path":("VIDEO",)
            }
        }
    RETURN_TYPES = ("SEQUNCE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "align_pose"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_UniAnimate"

    def align_pose(self,ref_name,source_video_path):
        base_name =os.path.basename(source_video_path)[:-4]
        saved_pose_dir = os.path.join(output_dir,"UniAnimate",base_name)
        os.makedirs(saved_pose_dir,exist_ok=True)
        py_path = os.path.join(now_dir, "UniAnimate","run_align_pose.py")
        cmd = f"""{python_exec} {py_path} --ref_name "{ref_name}" --source_video_paths "{source_video_path}" --saved_pose_dir "{saved_pose_dir}" """
        print(cmd)
        os.system(cmd)
        return (saved_pose_dir, )

class UniAnimateNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "ref_name":("IMAGE",),
                "pose_dir":("SEQUNCE",),
                "frame_interval":([1,2],{
                    "default": 2,
                }),
                "max_frames":([96,64,48,32,24,16,'None'],{
                    "default": 32,
                }),
                "resolution":(["512*768","768*1216"],{
                    "default": "512*768"
                }),
                "context_overlap":([8,16],{
                    "default":8
                }),

            }
        }
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_UniAnimate"

    def generate(self,ref_name,pose_dir,frame_interval,max_frames,resolution,context_overlap):
        default_yaml_path = os.path.join(now_dir,"UniAnimate", "configs","UniAnimate_infer_long.yaml")

        with open(default_yaml_path, 'r', encoding="utf-8") as f:
            yaml_data = yaml.load(f.read(),Loader=yaml.SafeLoader)
        
        log_dir = os.path.join(output_dir,"UniAnimate","log")
        yaml_data['log_dir'] = os.path.join(output_dir,"UniAnimate","log")
        os.makedirs(log_dir)
        yaml_data["max_frames"] = max_frames
        yaml_data['resolution'] = [512, 768] if '512' in resolution else [768, 1216]
        yaml_data['context_overlap'] = context_overlap
        yaml_data['test_list_path'] = [
            [frame_interval,ref_name,pose_dir]
        ]
        yaml_data['test_model'] = os.path.join(ckpt_dir,"unianimate_16f_32f_non_ema_223000.pth")
        yaml_data['embedder']['pretrained'] = os.path.join(ckpt_dir,"open_clip_pytorch_model.bin")
        yaml_data['auto_encoder']['pretrained'] = os.path.join(ckpt_dir,"v2-1_512-ema-pruned.ckpt")                                                   
        tmp_yaml_path = os.path.join(now_dir,'tmp.yaml')
        with open(tmp_yaml_path,'w', encoding="utf-8") as f:
            yaml.dump(data=yaml_data,stream=f,Dumper=yaml.CDumper)

        py_path = os.path.join(now_dir, "UniAnimate","inference.py")
        cmd = f"""{python_exec} {py_path} --cfg "{tmp_yaml_path}" """
        print(cmd)
        os.system(cmd)
        os.remove(tmp_yaml_path)
        return (py_path, )

class LoadImagePath:
    @classmethod
    def INPUT_TYPES(s):
        
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "AIFSH_UniAnimate"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        return (image_path,)


class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "AIFSH_UniAnimate"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}

class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1].lower() in ["mp4", "webm","mkv","avi"]]
        return {"required":{
            "video":(files,),
        }}
    
    CATEGORY = "AIFSH_UniAnimate"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO","AUDIO")

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = os.path.join(input_dir,video)
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.join(input_dir,video+".wav")
        video_clip.audio.write_audiofile(audio_path)
        return (video_path,audio_path,)


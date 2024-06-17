import os
import os.path as osp
import sys
import cv2
import glob
import math
import torch
import gzip
import copy
import time
import json
import pickle
import base64
import imageio
import hashlib
import requests
import binascii
import zipfile
# import skvideo.io
import numpy as np
from io import BytesIO
import urllib.request
import torch.nn.functional as F
import torchvision.utils as tvutils
from multiprocessing.pool import ThreadPool as Pool
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont


def gen_text_image(captions, text_size):
    num_char = int(38 * (text_size / text_size))
    font_size = int(text_size / 20)
    font = ImageFont.truetype('data/font/DejaVuSans.ttf', size=font_size)
    text_image_list = []
    for text in captions:
        txt_img = Image.new("RGB", (text_size, text_size), color="white") 
        draw = ImageDraw.Draw(txt_img)
        lines = "\n".join(text[start:start + num_char] for start in range(0, len(text), num_char))
        draw.text((0, 0), lines, fill="black", font=font)
        txt_img = np.array(txt_img)
        text_image_list.append(txt_img)
    text_images = np.stack(text_image_list, axis=0)
    text_images = torch.from_numpy(text_images)
    return text_images

@torch.no_grad()
def save_video_refimg_and_text(
    local_path,
    ref_frame,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    nrow=4, 
    save_fps=8,
    retry=5):
    ''' 
    gen_video: BxCxFxHxW
    '''
    nrow = max(int(gen_video.size(0) / 2), 1)
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    text_images = gen_text_image(captions, text_size) # Tensor 8x256x256x3
    text_images = text_images.unsqueeze(1) # Tensor 8x1x256x256x3
    text_images = text_images.repeat_interleave(repeats=gen_video.size(2), dim=1) # 8x16x256x256x3

    ref_frame = ref_frame.unsqueeze(2)
    ref_frame = ref_frame.mul_(vid_std).add_(vid_mean)
    ref_frame = ref_frame.repeat_interleave(repeats=gen_video.size(2), dim=2) # 8x16x256x256x3
    ref_frame.clamp_(0, 1)
    ref_frame = ref_frame * 255.0
    ref_frame = rearrange(ref_frame, 'b c f h w -> b f h w c')
    
    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = torch.cat([ref_frame, images, text_images], dim=3)

    images = rearrange(images, '(r j) f h w c -> f (r h) (j w) c', r=nrow)
    images = [(img.numpy()).astype('uint8') for img in images]

    for _ in [None] * retry:
        try:
            if len(images) == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                local_path = local_path + '.mp4'
                frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
                os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
                for fid, frame in enumerate(images):
                    tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                    cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
                os.system(cmd); os.system(f'rm -rf {frame_dir}')
                # os.system(f'rm -rf {local_path}')
            exception = None
            break
        except Exception as e:
            exception = e
            continue


@torch.no_grad()
def save_i2vgen_video(
    local_path,
    image_id,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    ''' 
    Save both the generated video and the input conditions.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    text_images = gen_text_image(captions, text_size) # Tensor 1x256x256x3
    text_images = text_images.unsqueeze(1) # Tensor 1x1x256x256x3
    text_images = text_images.repeat_interleave(repeats=gen_video.size(2), dim=1) # 1x16x256x256x3

    image_id = image_id.unsqueeze(2) # B, C, F, H, W
    image_id = image_id.repeat_interleave(repeats=gen_video.size(2), dim=2) # 1x3x32x256x448
    image_id = image_id.mul_(vid_std).add_(vid_mean)  # 32x3x256x448
    image_id.clamp_(0, 1)
    image_id = image_id * 255.0
    image_id = rearrange(image_id, 'b c f h w -> b f h w c')

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = torch.cat([image_id, images, text_images], dim=3)
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]

    exception = None
    for _ in [None] * retry:
        try:
            frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
            os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
            for fid, frame in enumerate(images):
                tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
            os.system(cmd); os.system(f'rm -rf {frame_dir}')
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception


@torch.no_grad()
def save_i2vgen_video_safe(
    local_path,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    '''
    Save only the generated video, do not save the related reference conditions, and at the same time perform anomaly detection on the last frame.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]
    num_image = len(images)
    exception = None
    for _ in [None] * retry:
        try:
            if num_image == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                writer = imageio.get_writer(local_path, fps=save_fps, codec='libx264', quality=8)
                for fid, frame in enumerate(images):
                    if fid == num_image-1: # Fix known bugs.
                        ratio = (np.sum((frame >= 117) & (frame <= 137)))/(frame.size)
                        if ratio > 0.4: continue
                    writer.append_data(frame)
                writer.close()
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception


@torch.no_grad()
def save_t2vhigen_video_safe(
    local_path,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    '''
    Save only the generated video, do not save the related reference conditions, and at the same time perform anomaly detection on the last frame.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]
    num_image = len(images)
    exception = None
    for _ in [None] * retry:
        try:
            if num_image == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
                os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
                for fid, frame in enumerate(images):
                    if fid == num_image-1: # Fix known bugs.
                        ratio = (np.sum((frame >= 117) & (frame <= 137)))/(frame.size)
                        if ratio > 0.4: continue
                    tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                    cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
                os.system(cmd) 
                os.system(f'rm -rf {frame_dir}')
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception




@torch.no_grad()
def save_video_multiple_conditions_not_gif_horizontal_3col(local_path, video_tensor, model_kwargs, source_imgs, 
                                   mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], nrow=8, retry=5, save_fps=8):
    mean=torch.tensor(mean,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  #### unnormalize back to [0,1]
    video_tensor.clamp_(0, 1)

    b, c, n, h, w = video_tensor.shape
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    source_imgs = source_imgs.cpu()

    model_kwargs_channel3 = {}
    for key, conditions in model_kwargs[0].items():

        
        if conditions.size(1) == 1:
            conditions = torch.cat([conditions, conditions, conditions], dim=1)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        if conditions.size(1) == 2:
            conditions = torch.cat([conditions, conditions[:,:1,]], dim=1)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        elif conditions.size(1) == 3:
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        elif conditions.size(1) == 4: # means it is a mask.
            color = ((conditions[:, 0:3] + 1.)/2.) # .astype(np.float32)
            alpha = conditions[:, 3:4] # .astype(np.float32)
            conditions = color * alpha + 1.0 * (1.0 - alpha)
            conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        model_kwargs_channel3[key] = conditions.cpu() if conditions.is_cuda else conditions
    
    # filename = rand_name(suffix='.gif')
    for _ in [None] * retry:
        try:
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)
            
            cons_list = [rearrange(con, '(i j) c f h w -> c f (i h) (j w)', i = nrow) for _, con in model_kwargs_channel3.items()]
            vid_gif = torch.cat(cons_list + [vid_gif,], dim=3)
            
            vid_gif = vid_gif.permute(1,2,3,0)
            
            images = vid_gif * 255.0
            images = [(img.numpy()).astype('uint8') for img in images]
            if len(images) == 1:
                
                local_path = local_path.replace('.mp4', '.png')
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # bucket.put_object_from_file(oss_key, local_path)
            else:

                outputs = []
                for image_name in images:
                    x = Image.fromarray(image_name)
                    outputs.append(x)
                from pathlib import Path
                save_fmt = Path(local_path).suffix

                if save_fmt == ".mp4":
                    with imageio.get_writer(local_path, fps=save_fps) as writer:
                        for img in outputs:
                            img_array = np.array(img)  # Convert PIL Image to numpy array
                            writer.append_data(img_array)

                elif save_fmt == ".gif":
                    outputs[0].save(
                        fp=local_path,
                        format="GIF",
                        append_images=outputs[1:],
                        save_all=True,
                        duration=(1 / save_fps * 1000),
                        loop=0,
                    )
                else:
                    raise ValueError("Unsupported file type. Use .mp4 or .gif.")

                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # fps = save_fps
                # image = images[0] 
                # media_writer = cv2.VideoWriter(local_path, fourcc, fps, (image.shape[1],image.shape[0]))
                # for image_name in images:
                #     im = image_name[:,:,::-1] 
                #     media_writer.write(im)
                # media_writer.release()
                
            
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        print('save video to {} failed, error: {}'.format(local_path, exception), flush=True)


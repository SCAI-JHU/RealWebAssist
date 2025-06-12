import io
import torch
import argparse
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, "/weka/scratch/tshu2/hshi33/SoM")
# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive


from task_adapter.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np

from gpt4v import request_gpt4v
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

import matplotlib.colors as mcolors
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]


import os
import base64
import re
import json
from tqdm import tqdm

api_key = os.getenv("OPENAI_API_KEY")

# Settings
semsam_cfg = "/weka/scratch/tshu2/hshi33/SoM/configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "/weka/scratch/tshu2/hshi33/SoM/configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "/weka/scratch/tshu2/hshi33/SoM/swinl_only_sam_many2many.pth"
sam_ckpt = "/weka/scratch/tshu2/hshi33/SoM/sam_vit_h_4b8939.pth"
seem_ckpt = "/weka/scratch/tshu2/hshi33/SoM/seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed_seem(opt_seem)

# Load models
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

def extract_number_from_path(path):
    """
    Extract the numeric part from the config.json file path for sorting.
    """
    match = re.search(r'/(\d+)/config\.json$', path)
    return int(match.group(1)) if match else float('inf')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def encode_image_to_url(image_path):
    base64_image = encode_image(image_path)
    return f"data:image/jpeg;base64,{base64_image}"

def rescale_to_original(coords, orig_width, orig_height, modified_width, modified_height):
    x, y = coords
    
    x_rescaled = x * orig_width / modified_width
    y_rescaled = y * orig_height / modified_height

    return round(x_rescaled), round(y_rescaled)

def gpt_select_mark(image_path, instruction, history):

    image_url = encode_image_to_url(image_path)

    messages = []

    # Add the final instruction and image
    messages.append({
        "type": "text",
        "text": f"Which is the correct mark corresponding to the instruction: {instruction}? Respond with only the number and nothing else.",
    })
    messages.append({
        "type": "image_url",
        "image_url": {"url": image_url},
    })

    messages.append({
        "type": "text",
        "text": f"Here is some history: {history}"
    })

    # Call the OpenAI client with the constructed history and final image
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": messages,
            }
        ],
    )
    
    # Print and return the response
    #print(response.choices[0].message.content)
    return response.choices[0].message.content

# Functions
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    global history_images; history_images = []
    global history_masks; history_masks = []    

    #_image = image['background'].convert('RGB')
    _image = image.convert('RGB')
    #_mask = image['layers'][0].convert('L') if image['layers'] else None
    _mask = None

    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:                
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=1000,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if mode == "Interactive":
            labeled_array, num_features = label(np.asarray(_mask))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask, coords = inference_semsam_m2m_auto(model, _image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = model_sam
            if mode == "Automatic":
                output, mask, coords = inference_sam_m2m_auto(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        elif model_name == 'seem':
            model = model_seem
            if mode == "Automatic":
                output, mask = inference_seem_pano(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_seem_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        # convert output to PIL image
        #history_masks.append(mask)
        #history_images.append(Image.fromarray(output))
        return output, coords


def get_coordinate(config_data, history, base_dir, output_dir):
    try:
        instruction = config_data.get("instruction", "")
        image_path = config_data.get("final_image", "")
        
        full_image_path = image_path

        image = Image.open(full_image_path)
        output, coords = inference(image, 2.6, "Automatic", 0.1, "Number", ["Mark"])
        print(f"Coords: {coords}")
        image = Image.fromarray(output)

        num = config_data["final_image"].split("/")[-2]

        os.makedirs(f"{output_dir}/marked_images", exist_ok=True)
        output_image_path = f"{output_dir}/marked_images/{num}.png"
        print(output_image_path)
        image.save(output_image_path)

        gpt_choice = gpt_select_mark(output_image_path, instruction, history)
        print(gpt_choice)

        # Invalid choice
        if int(gpt_choice) > len(coords):
            print("Incorrect: Invalid choice")
            return None, None

        mapped_coords = coords[int(gpt_choice) - 1]

        # Rescale the image again
        image = Image.open(full_image_path)
        orig_width, orig_height = image.size
        new_image = Image.open(output_image_path)
        modified_width, modified_height = new_image.size

        scaled_coords = rescale_to_original(mapped_coords, orig_width, orig_height, modified_width, modified_height)
        print(f"Mapped coords: {mapped_coords}")
        print(f"Scaled coords: {scaled_coords}")
        center_x = scaled_coords[0]
        center_y = scaled_coords[1]

        bounding_box_data = config_data.get("bounding_box", [])

        if isinstance(bounding_box_data, dict):
            bounding_box_data = [bounding_box_data]


    except Exception as e:
        return None, None
    return center_x, center_y

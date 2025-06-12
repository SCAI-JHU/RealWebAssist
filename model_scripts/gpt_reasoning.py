from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw
import re
import json
from tqdm import tqdm
import os
import base64
import math
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def encode_image_to_url(image_path):
    base64_image = encode_image(image_path)
    return f"data:image/jpeg;base64,{base64_image}"

def gpt_reasoning(config_data, history, base_dir):

    instruction = config_data["instruction"]
    final_image_path = os.path.join(base_dir, config_data["final_image"])
    final_image_url = encode_image_to_url(final_image_path)
    history_messages = []
    
    history_messages.append({
        "type": "text",
        "text": f"Given this webpage, the instruction {instruction} and the history, what is the correct element",
    })
    history_messages.append({
        "type": "image_url",
        "image_url": {"url": final_image_url},
    })

    history_messages.append({
        "type": "text",
        "text": f"History: {history}"
    })

    # Call the OpenAI client with the constructed history and final image
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "user",
                "content": history_messages,
            }
        ],
    )
    
    # Print and return the response
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def parse_coordinates(output_text):
    if isinstance(output_text, list):
        output_text = " ".join(output_text)
    
    # Updated regex to support both (x, y) and [x, y]
    match = re.search(r"[\(\[](\d+),\s*(\d+)[\)\]]", output_text)
    
    if match:
        x, y = map(int, match.groups())
        return (x, y)
    else:
        raise ValueError("No valid bounding box coordinates found in the output_text.")

# Rescale coordinates (to ensure the scale is same as the correct bounding box)
def rescale_coordinates(scaled_coords, orig_width, orig_height, scale_range=1000):
    x = int(scaled_coords[0] / scale_range * orig_width)
    y = int(scaled_coords[1] / scale_range * orig_height)

    return x, y


count = 0

def get_coordinate(config_data, history, base_dir, output_dir):
    global count
    count += 1
    try:
        instruction = config_data.get("instruction", "")
        image_path = config_data.get("final_image", "")
        bounding_box_data = config_data.get("bounding_box", [])


        print(image_path)
        print(instruction)


        instruction = gpt_reasoning(config_data, history, base_dir)
        with open("../output_files/reasoning_results/gpt_reasoning_10.json", "r") as file:
            data = json.load(file)
        num = config_data["final_image"].split("/")[-3]
        if str(num) not in data.keys():
            data[str(num)] = {}
        data[str(num)][int(config_data["final_image"].split("/")[-1].replace(".png", ""))] = instruction
        with open("../output_files/reasoning_results/gpt_reasoning_10.json", "w") as file:
            json.dump(data, file, indent=4)
        return (None, None)
    except Exception as e:
        return (None, None)
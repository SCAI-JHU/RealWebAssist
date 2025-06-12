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
import anthropic
load_dotenv()
api_key = "your api key"

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


    client = anthropic.Anthropic(api_key=api_key)

    
    
    history_messages.append({
        "type": "text",
        "text": f"Given this webpage, the instruction {instruction} and the history, what is the correct element",
    })
    base64_image = encode_image(final_image_path)  # Use your existing encode_image function
    history_messages.append({
        "type": "image",
        "source": {
            "type": "base64", 
            "media_type": "image/png", 
            "data": base64_image
        },
    })

    history_messages.append({
        "type": "text",
        "text": f"History: {history}"
    })

    # Call the OpenAI client with the constructed history and final image
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        temperature=1.0,
        messages=[
            {
                "role": "user",
                "content": history_messages
            }
        ]
    )
    
    # Print and return the response
    response_text = message.content[0].text
    print(response_text)
    return response_text

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
        print("Instruction Here")
        with open("../Web_Instruction_Following_Full/output_files/reasoning_results/claude_reasoning_10_3_7_sonnet.json", "r") as file:
            data = json.load(file)
        num = config_data["final_image"].split("/")[-3]
        if str(num) not in data.keys():
            data[str(num)] = {}
        data[str(num)][int(config_data["final_image"].split("/")[-1].replace(".png", ""))] = instruction
        with open("../output_files/reasoning_results/claude_reasoning_10_3_7_sonnet.json", "w") as file:
            json.dump(data, file, indent=4)
        return None, None
    except Exception as e:
        raise e
        return (None, None)
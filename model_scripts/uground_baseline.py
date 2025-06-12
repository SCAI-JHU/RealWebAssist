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


# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "osunlp/UGround-V1-7B", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("osunlp/UGround-V1-7B")
count = 0

def get_coordinate(config_data, history, base_dir, output_dir):
    global count
    count += 1
    try:
        instruction = config_data.get("instruction", "")
        image_path = config_data.get("final_image", "")
        bounding_box_data = config_data.get("bounding_box", [])

        if not bounding_box_data:
            print(f"No bounding box data")
            return None, None

        print(image_path)
        print(instruction)


        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(base_dir, image_path)},
                    {"type": "text", "text": f"In this UI screenshot, what is the position of the element corresponding to the command: {instruction}"},
                    {"type": "text", "text": f"Here is some history: {history}"}
                ],
            }
        ]

        # Prepare inputs for the model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Parse coordinates from output
        print(f"Raw model output: {output_text}")

        # Use the function to parse coordinates
        scaled_coordinates = parse_coordinates(output_text)
        print(f"Parsed coordinates: {scaled_coordinates}")

        # Load the image to get original dimensions
        print("Image path")
        print(image_path)
        image = Image.open(os.path.join(base_dir, image_path))
        orig_width, orig_height = image.size

        center_x, center_y = rescale_coordinates(scaled_coordinates, orig_width, orig_height)
        center = (center_x, center_y)

        # Debugging
        print(f"center: {center}")



        return center_x, center_y
    except Exception as e:
        return (None, None)
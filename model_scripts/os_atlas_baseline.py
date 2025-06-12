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
    
    # Attempt to match and parse the new format: [[x1, y1, x2, y2]]
    match = re.search(r"<\|box_start\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|box_end\|>", output_text)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return [(x1, y1), (x2, y2)]
    
    # Attempt to match the old format
    match = re.search(r"<\|box_start\|>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)<\|box_end\|>", output_text)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return [(x1, y1), (x2, y2)]
    
    # Handle case where coordinates cannot be found
    raise ValueError("No valid bounding box coordinates found in the output_text.")

# Rescale coordinates (to ensure the scale is same as the correct bounding box)
def rescale_coordinates(scaled_coords, orig_width, orig_height, scale_range=1000):
    top_left = (
        int(scaled_coords[0][0] / scale_range * orig_width),
        int(scaled_coords[0][1] / scale_range * orig_height),
    )
    bottom_right = (
        int(scaled_coords[1][0] / scale_range * orig_width),
        int(scaled_coords[1][1] / scale_range * orig_height),
    )
    return top_left, bottom_right


# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("OS-Copilot/OS-Atlas-Base-7B")
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

        top_left, bottom_right = rescale_coordinates(scaled_coordinates, orig_width, orig_height)

        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        center = (center_x, center_y)

        # Debugging
        print(f"top_left: {top_left}")
        print(f"top_right: {bottom_right}")
        print(f"center: {center}")

        if isinstance(bounding_box_data, dict):
            bounding_box_data = [bounding_box_data]


        return center_x, center_y
    except Exception as e:
        return (None, None)
from PIL import Image, ImageDraw
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ast
import os
import re
import json
from tqdm import tqdm

import base64
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def rescale_coordinates(scaled_coords, orig_width, orig_height, scale_range=1000):
    x = int(scaled_coords[0] / scale_range * orig_width)
    y = int(scaled_coords[1] / scale_range * orig_height)

    return x, y

def encode_image_to_url(image_path):
    base64_image = encode_image(image_path)
    return f"data:image/jpeg;base64,{base64_image}"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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
        model="gpt-4o",
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

model_path = "Aria-UI/Aria-UI-base"

llm = LLM(
    model=model_path,
    tokenizer_mode="slow",
    dtype="bfloat16",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False
)

count = 0

def get_coordinate(config_data, history, base_dir, output_dir):
    global count
    count += 1

    try:
        instruction = config_data.get("instruction", "")
        image_path = config_data.get("final_image", "")
        print(image_path)
        image_path = os.path.join(base_dir, image_path)
        print(instruction)


        with open("/weka/scratch/tshu2/sye10/Web_Instruction_Following_Full/output_files/reasoning_results/gpt_reasoning.json", "r") as file:
            print("Use saved reasoning")
            data = json.load(file)
            num = config_data["final_image"].split("/")[-3]
            instruction = data[str(num)][config_data["final_image"].split("/")[-1].replace(".png", "")]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: " + instruction,
                    }
                ],
            }
        ]
        message = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        outputs = llm.generate(
            {
                "prompt_token_ids": message,
                "multi_modal_data": {
                    "image": [
                        Image.open(image_path),
                    ],
                    "max_image_size": 980,  # [Optional] The max image patch size, default `980`
                    "split_image": True,  # [Optional] whether to split the images, default `True`
                },
            },
            sampling_params=SamplingParams(max_tokens=50, top_k=1, stop=["<|im_end|>"]),
        )

        for o in outputs:
            generated_tokens = o.outputs[0].token_ids
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(response)
            coords = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())
            print(f"Coords: {coords}")
        print(image_path)
        image = Image.open(image_path)
        orig_width, orig_height = image.size

        center_x, center_y = rescale_coordinates(coords, orig_width, orig_height)
        center = (center_x, center_y)

        draw = ImageDraw.Draw(image)
        radius = 10  # Replace with your desired radius
        top_left = (center_x - radius, center_y - radius)
        bottom_right = (center_x + radius, center_y + radius)
        draw.ellipse([top_left, bottom_right], outline="red", width=4)

        # Save the modified image to the current directory
        output_image_path = f"{output_dir}/aria_ui{count}.png"
        image.save(output_image_path)

        print(f"Modified image saved at: {output_image_path}")

        return center_x, center_y
        
    except Exception as e:
        return (None, None)
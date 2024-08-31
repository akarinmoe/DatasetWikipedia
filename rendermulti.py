import json
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import os
import multiprocessing
from multiprocessing import Pool, Manager
import re
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

ds = load_dataset("wikimedia/wikipedia", "20231101.en")

# Clean text to remove special characters that cannot be displayed
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to convert text to an image
def text_to_image(text, image_size, font_size, font_path, output_path, index, carry_over_line="", shared_dict=None):
    try:
        image = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)
        words = (carry_over_line + " " + text).split()
        lines = []
        current_line = ""
        
        for word in words:
            if draw.textsize(current_line + " " + word, font=font)[0] <= image_size[0]:
                current_line += " " + word
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word
        
        if current_line:
            lines.append(current_line.strip())
        
        y_pos = 0
        last_line = ""
        for line in lines:
            line_width, line_height = draw.textsize(line, font=font)
            # Check if adding this line will exceed the image height
            if y_pos + line_height > image_size[1]:
                last_line = line  # This line will be carried over to the next image
                break
            draw.text((0, y_pos), line, font=font, fill="black")
            y_pos += line_height
        
        image.save(output_path)
        if shared_dict is not None:
            shared_dict[index] = {
                "text": text,
                "image_path": output_path
            }
            
            # Write each generated image info to the JSON file in real-time
            with open('index_to_text.json', 'a', encoding='utf-8') as f:
                json.dump({index: {"text": text, "image_path": output_path}}, f, ensure_ascii=False)
                f.write("\n")  # Add a newline after each record for readability
        
        return last_line  # Return the last line to be carried over to the next image (if any)
    except IOError as e:
        print(f"Error: {e}")
        return ""

# Define a function for multi-process work
def process_text(index, words_per_group, font_path, image_size, font_size, shared_dict):
    try:
        sample_text = ds['train']['text'][index]
        clean_sample_text = clean_text(sample_text)
        words = clean_sample_text.split()
        group_index = 0
        carry_over_line = ""

        while group_index < len(words):
            group_text = ' '.join(words[group_index:group_index + words_per_group])
            group_index += words_per_group
            output_image_path = f"./outputimg/{index}_{group_index//words_per_group}.png"
            carry_over_line = text_to_image(group_text, image_size, font_size, font_path, output_image_path, f"{index}_{group_index//words_per_group}", carry_over_line, shared_dict)
    except IndexError as e:
        print(f"No text available for index {index}: {e}")

if __name__ == "__main__":
    os.makedirs("./outputimg", exist_ok=True)
    
    # Create a shared dictionary using Manager
    manager = Manager()
    shared_dict = manager.dict()

    # Initialize the JSON file
    with open('index_to_text.json', 'w', encoding='utf-8') as f:
        f.write("")  # Clear the file contents

    # Set the number of processes
    num_processes = 1

    print(f"Using {num_processes} processes for processing.")

    # Set parameters
    image_size = (512, 512)
    font_size = 32
    font_path = "times.ttf"
    words_per_group = 100  # Number of words per group

    # Use tqdm to wrap the range and provide a progress bar
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.starmap(process_text, [(index, words_per_group, font_path, image_size, font_size, shared_dict) for index in range(len(ds['train']))]), total=len(ds['train'])):
            pass

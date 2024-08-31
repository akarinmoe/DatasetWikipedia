import json
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import os
import multiprocessing
from multiprocessing import Pool, Manager
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

ds = load_dataset("wikimedia/wikipedia", "20231101.en")

# Clean text to remove special characters that cannot be displayed
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# A function that converts text to an image
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
        
        return last_line  # Return the last line to be carried over, if any
    except IOError as e:
        print(f"Error: {e}")
        return ""

# Define a working function for multiple processes
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

    # Set the number of processes
    # num_processes = max(1, multiprocessing.cpu_count() - 2)
    num_processes = 1

    print(f"Using {num_processes} processes for processing.")

    # set up parameters
    image_size = (512, 512)
    font_size = 32
    font_path = "times.ttf"
    words_per_group = 100  # The number of words in each group

    # Modify pool.starmap to support shared dict passing
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_text, [(index, words_per_group, font_path, image_size, font_size, shared_dict) for index in range(len(ds['train']))])
    
    # Save the index to text mapping to a JSON file
    with open('index_to_text.json', 'w', encoding='utf-8') as f:
        json.dump(dict(shared_dict), f, ensure_ascii=False, indent=4)

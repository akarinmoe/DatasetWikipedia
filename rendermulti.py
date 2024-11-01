import json
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont
import os
import multiprocessing
from multiprocessing import Pool, Manager
import re
from tqdm import tqdm
import math

ds = load_from_disk("./datasets")

# Clean text to remove special characters that cannot be displayed
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to convert text to an image
def text_to_image(text, image_size, font_size, font_path, output_path, index, carry_over_text="", shared_dict=None, meta_data_path=None):
    try:
        image = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        # Combine carry_over_text with the current text
        combined_text = (carry_over_text + " " + text).strip()
        words = combined_text.split()
        lines = []
        current_line = ""

        # Add padding for the left and right margins
        left_margin = 20
        right_margin = 20
        max_line_width = image_size[0] - left_margin - right_margin

        # Split words into lines based on max_line_width
        for word in words:
            test_line = f"{current_line} {word}".strip()
            test_line_width, _ = draw.textsize(test_line, font=font)
            if test_line_width <= max_line_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Determine how many lines fit in the image
        line_height = font.getsize("A")[1] + 5  # Adding some spacing
        max_lines = image_size[1] // line_height

        # Split lines into lines_to_draw and remaining_lines
        lines_to_draw = lines[:max_lines]
        remaining_lines = lines[max_lines:]

        # Draw the lines on the image
        y_pos = 0
        for line in lines_to_draw:
            draw.text((left_margin, y_pos), line, font=font, fill="black")
            y_pos += line_height

        # Save the image
        image.save(output_path)

        # Join lines_to_draw to get the actual text drawn on the image
        drawn_text = ' '.join(lines_to_draw)

        # Update shared dictionary
        if shared_dict is not None:
            shared_dict[index] = {
                "text": drawn_text,  # Use the drawn text instead of the full text
                "image_path": output_path
            }

            # Append the image information to the JSON file
            with open(meta_data_path, 'a', encoding='utf-8') as f:
                json.dump({index: {"text": drawn_text, "image_path": output_path}}, f, ensure_ascii=False)
                f.write("\n")  # Add newline for each record

        # Return the remaining text to carry over to the next image
        carry_over_text = ' '.join(remaining_lines)
        return carry_over_text

    except IOError as e:
        print(f"Error generating image: {e}")
        return ""

# Define a function for multi-process work
def process_text(index, words_per_group, font_path, image_size, font_size, shared_dict, meta_data_path):
    try:
        sample_text = ds['train']['text'][index]
        clean_sample_text = clean_text(sample_text)
        words = clean_sample_text.split()
        group_index = 0
        carry_over_text = ""

        while group_index < len(words) or carry_over_text:
            group_text = ' '.join(words[group_index:group_index + words_per_group])
            group_index += words_per_group
            output_image_path = f"./outputimg/{index}_{group_index//words_per_group}.png"

            # Generate the image and get the carry over text
            carry_over_text = text_to_image(
                text=group_text,
                image_size=image_size,
                font_size=font_size,
                font_path=font_path,
                output_path=output_image_path,
                index=f"{index}_{group_index//words_per_group}",
                carry_over_text=carry_over_text,
                shared_dict=shared_dict,
                meta_data_path=meta_data_path
            )

    except IndexError as e:
        print(f"No text available for index {index}: {e}")

if __name__ == "__main__":
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    
    TOTAL_INDEX = 5000000
    # Calculate the size of each chunk
    chunk_size = math.ceil(TOTAL_INDEX / WORLD_SIZE)

    # Calculate the start and end indices for the current rank
    INDEX_START = RANK * chunk_size
    INDEX_END = min((RANK + 1) * chunk_size, TOTAL_INDEX)
    
    # Create a shared dictionary using Manager
    manager = Manager()
    shared_dict = manager.dict()

    meta_data_path = f"./{RANK}.json"

    # Initialize the JSON file
    with open(meta_data_path, 'w', encoding='utf-8') as f:
        f.write("")

    # Set the number of processes
    num_processes = 1

    print(f"Using {num_processes} processes for processing.")

    # Set parameters
    image_size = (512, 512)
    font_size = 32
    font_path = "times.ttf"
    words_per_group = 100

    # Use tqdm to wrap the range and provide a progress bar
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.starmap(
            process_text, 
            [(index, words_per_group, font_path, image_size, font_size, shared_dict, meta_data_path) for index in range(INDEX_START, INDEX_END)]
        ), total=INDEX_END - INDEX_START):
            pass

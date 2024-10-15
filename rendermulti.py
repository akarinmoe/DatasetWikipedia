import json
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont
import os
import multiprocessing
from multiprocessing import Pool, Manager
import re
from tqdm import tqdm

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

ds = load_from_disk("./datasets")

# Clean text to remove special characters that cannot be displayed
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to convert text to an image
def text_to_image(text, image_size, font_size, font_path, output_path, index, carry_over_text="", shared_dict=None):
    try:
        # Create a new image with white background
        image = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        # Combine carry_over_text with the current text
        combined_text = (carry_over_text + " " + text).strip()
        words = combined_text.split()
        lines = []
        current_line = ""

        # Add padding for left and right margins
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

        # Update shared dictionary
        if shared_dict is not None:
            shared_dict[index] = {
                "text": text,
                "image_path": output_path
            }

            # Append the image information to the JSON file
            with open('index_to_text.json', 'a', encoding='utf-8') as f:
                json.dump({index: {"text": text, "image_path": output_path}}, f, ensure_ascii=False)
                f.write("\n")  # Add newline for each record

        # Return the remaining text to carry over to the next image
        carry_over_text = ' '.join(remaining_lines)
        return carry_over_text

    except IOError as e:
        print(f"Error generating image: {e}")
        return ""

# Define a function for multi-process work
def process_text(index, font_path, image_size, font_size, shared_dict):
    try:
        # Load the text for the current sample
        sample_text = ds['train']['text'][index]
        clean_sample_text = clean_text(sample_text)
        words = clean_sample_text.split()
        total_words = len(words)
        current_word = 0
        carry_over_text = ""

        image_counter = 0

        while current_word < total_words:
            # Determine the remaining words
            remaining_words = words[current_word:]
            # Join the remaining words into a string
            remaining_text = ' '.join(remaining_words)
            # Generate output image path
            output_image_path = f"./outputimg/{index}_{image_counter}.png"
            # Generate the image and get the carry over text
            carry_over_text = text_to_image(
                text=' '.join(remaining_words),
                image_size=image_size,
                font_size=font_size,
                font_path=font_path,
                output_path=output_image_path,
                index=f"{index}_{image_counter}",
                carry_over_text=carry_over_text,
                shared_dict=shared_dict
            )
            # Update the current_word based on how many words were drawn
            # To do this, we need to calculate how many words were used in lines_to_draw
            # Since we don't have that information, we'll iteratively remove words until carry_over_text matches
            # A simple approach is to compare the remaining_text with carry_over_text

            # Split the carry_over_text back into words
            carry_over_words = carry_over_text.split()
            # Calculate how many words were used
            words_used = len(remaining_words) - len(carry_over_words)
            # Update the current_word pointer
            current_word += words_used
            # Increment the image counter
            image_counter += 1

    except IndexError as e:
        print(f"No text available for index {index}: {e}")

if __name__ == "__main__":
    os.makedirs("./outputimg", exist_ok=True)

    # Create a shared dictionary using Manager
    manager = Manager()
    shared_dict = manager.dict()

    # Initialize the JSON file
    with open('index_to_text.json', 'w', encoding='utf-8') as f:
        f.write("")

    # Set the number of processes
    # num_processes = multiprocessing.cpu_count()
    num_processes = 1

    print(f"Using {num_processes} processes for processing.")

    # Set parameters
    image_size = (512, 512)
    font_size = 32
    font_path = "times.ttf"  # Ensure this font exists or provide a valid path

    # Use tqdm to wrap the range and provide a progress bar
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.starmap(
            process_text, 
            [(index, font_path, image_size, font_size, shared_dict) for index in range(len(ds['train']))]
        ), total=len(ds['train'])))

    print("Processing complete.")

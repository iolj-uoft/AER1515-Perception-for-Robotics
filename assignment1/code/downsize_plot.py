import matplotlib.pyplot as plt
from PIL import Image
import os

def compress_plot(input_path, output_path, scale_factor=0.5):
    """Compress a plot image by a given scale factor."""
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist.")
        return

    try:
        # Open the image
        img = Image.open(input_path)

        # Calculate new dimensions
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)

        # Resize the image
        downsized_img = img.resize((new_width, new_height))

        # Save the downsized image
        downsized_img.save(output_path)
        print(f"Compressed plot saved to {output_path}")
    except Exception as e:
        print(f"Error while compressing plot: {e}")

if __name__ == "__main__":
    input_plot = "Q4.4.png"
    output_plot = "Q4.4_downsized.png"
    scale = 0.3  # Adjust scale factor as needed

    compress_plot(input_plot, output_plot, scale_factor=scale)
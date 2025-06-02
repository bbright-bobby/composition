
# Image Alignment Project

## Overview

This project provides a Python script to automatically detect and correct the tilt in images, ensuring they are perfectly aligned to vertical or horizontal axes. The script uses advanced computer vision techniques, including contour detection, Hough transforms, moment analysis, and symmetry checks, to achieve high-precision alignment.

It is particularly useful for product photography, where proper alignment enhances the visual appeal of items such as food packaging, jars, bottles, and devices.

The script processes images from an `input/` folder, applies the alignment algorithm, and saves the corrected images to an `output/` folder. It also generates debug outputs (e.g., intermediate images and logs) to help understand the alignment process.

---

## Features

- **Robust Tilt Detection**: Combines multiple methods (contour orientation, Hough transform, moment analysis, and symmetry checks) for accurate tilt detection.
- **High-Precision Rotation**: Uses Lanczos4 interpolation to rotate images without quality loss.
- **Debugging Support**: Saves intermediate images and logs for analyzing the alignment process.
- **Customizable Parameters**: Adjustable thresholds for preprocessing, contour selection, and alignment validation.
- **Error Handling**: Includes robust error handling for file loading and processing.

---

## Requirements

### Prerequisites

- Python 3.7 or higher
- Compatible OS (Windows, macOS, or Linux)

### Dependencies

Install the dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python>=4.5.0
numpy>=1.21.0
scipy>=1.7.0
```

---

## Project Structure

```
image-alignment-project/
│
├── input/              # Folder with input images
├── output/             # Folder for aligned output images
├── debug/              # Folder for debug images and logs
├── app.py              # Main script for alignment
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Setup

1. **Clone or Download the Project**  
   Download the repository to your local system.

2. **Create Necessary Folders**  
   - Create `input/`, `output/`, and optionally `debug/` folders inside the project directory.

3. **Add Images**  
   Place your images in the `input/` folder. Example filenames:
   ```
   cookies.jpg
   jammie_dodgers.jpg
   pickle.jpg
   oil.jpg
   iphone.jpg
   ```

4. **Install Dependencies**  
   Run the following from the project root:
   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure `app.py` Exists**  
   Make sure the alignment logic is written inside `app.py`.

---

## Usage

### Step 1: Verify Input Images

Ensure the following images (or your custom ones) are in `input/`:
- cookies.jpg
- jammie_dodgers.jpg
- pickle.jpg
- oil.jpg
- iphone.jpg

### Step 2: Run the Script

```bash
python app.py
```

This will:
- Load each image from the `input/` folder.
- Detect and correct tilt.
- Save aligned images to the `output/` folder.
- Save debug files to `debug/` (if created).

### Step 3: Check Output

- Aligned images saved to `output/` (e.g., `cookies.jpg` → `cookies_product.jpg`)
- Debug images/logs in `debug/` (e.g., `debug_edges.jpg`, `debug_angles.txt`)

---

## Example Custom Command

To process a specific image in `app.py`, edit the bottom section:

```python
if __name__ == "__main__":
    input_image = "input/cookies.jpg"
    output_image = "output/cookies_product.jpg"
    process_image(input_image, output_image)
```

---

## Debug Outputs

The script generates:

- `debug_edges.jpg`: Edge detection result
- `debug_thresh.jpg`: Thresholded binary image
- `debug_orientation.jpg`: Contour and angle visualization
- `debug_angles.txt`: Contains:

```
Initial contour angle: 2.3456
Initial Hough angle: 2.1234
Initial moment angle: 2.5678
Orientation check angle: 0.0000
Iteration angles: 2.3456, 1.2345, 0.5678, 0.0123
Final angle: 0.0123
Final symmetry score: 5.67
Final edge alignment score: 0.0023
```

---

## Troubleshooting

### Image Not Loading?

- Ensure the folders `input/` and `output/` exist.
- Check image filenames and extensions (.jpg, .png).
- Make sure the script is in the correct path.

### Common Errors

| Error                        | Fix |
|-----------------------------|-----|
| `Image not found`           | Check input file path |
| `Failed to load image`      | Confirm image isn't corrupted |
| `Poor alignment`            | Adjust preprocessing parameters |
| `Missing dependencies`      | Run `pip install -r requirements.txt` |

---

## Limitations

- Low-contrast or complex background images may not align well.
- Performance may degrade on very large images.
- Optimized for product images; landscape or abstract images may require tuning.

---

## Future Improvements

- Batch processing for entire `input/` folder.
- GUI for parameter tuning and image preview.
- Support more file formats.
- Optimize for faster performance.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You may use, modify, and distribute the code freely.

--.

_Last updated: June 02, 2025_

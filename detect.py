import numpy as np
import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

# Parameters
HSV_Red    = ((175, 180), (108, 255), ( 56, 255))
HSV_Yellow = (( 17,  31), (205, 255), (116, 255))
HSV_Green  = (( 33,  85), (130, 255), (130, 255))
HSV_Purple = ((132, 169), ( 38, 232), (  0, 255))


def processImage(src):
    img = src.copy()
    colors = ["red", "yellow", "green", "purple"]
    count = {}
    for color in range(4):
        if color == 0: 
            H, S, V = HSV_Red
            mask = HSVTresholding(img, H, S, V, open=True)
        elif color == 1:
            H, S, V = HSV_Yellow
            mask = HSVTresholding(img, H, S, V, open=True)
        elif color == 2:
            H, S, V = HSV_Green
            mask = HSVTresholding(img, H, S, V, open=True)
        elif color == 3:
            H, S, V = HSV_Purple
            mask = HSVTresholding(img, H, S, V, close=True)

        count[colors[color]] = countObjects(mask)
    return count


### Applies color tresholding in HSV
def HSVTresholding(src: np.array, H, S, V, open=False, close=False):
    # Copy, Convert BGR -> HSV
    img = src.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create binary mask
    mask = cv2.inRange(img, (H[0], S[0], V[0]), (H[1], S[1], V[1]))
    mask = cv2.medianBlur(mask, 9)

    kernel = np.ones((3, 3))
    if close: 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if open: 
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def countObjects(mask):
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
        
    return num_labels - 1

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    img = cv2.resize(img, None, fx=800/h, fy=800/h)
    count = processImage(img)
    print(count)
    return count


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()

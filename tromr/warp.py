from typing import List, Tuple, Union
import cv2
import numpy as np
import PIL.Image

from skimage import transform
import difflib

class StaffDewarping:
    def __init__(self, tform: transform.PiecewiseAffineTransform):
        self.tform = tform

    def dewarp(self, image: np.ndarray, fill_color=1, order=1) -> np.ndarray:
        return transform.warp(image, self.tform.inverse, output_shape=image.shape, mode="constant", order=order, cval=fill_color)

def calculate_dewarp_transformation(image: np.ndarray, source: List[List[Tuple[float, float]]], destination: List[List[Tuple[float, float]]]) -> StaffDewarping:
    def add_image_edges_to_lines(lines):
        lines.insert(0, [[0, 0], [0, image.shape[1]]])
        lines.append([[image.shape[0], 0], [image.shape[0], image.shape[1]]])
        return lines
    
    def add_first_and_last_point_to_every_line(lines):
        for line in lines:
            line.insert(0, [0, line[0][1]])
            line.append([image.shape[1], line[-1][1]])
        return lines

    source = add_image_edges_to_lines(add_first_and_last_point_to_every_line(source))
    destination = add_image_edges_to_lines(add_first_and_last_point_to_every_line(destination))

    # Convert your points to numpy arrays
    source = np.concatenate(source)
    destination = np.concatenate(destination)

    tform = transform.PiecewiseAffineTransform()
    tform.estimate(source, destination)
    return StaffDewarping(tform)

def warp_image_randomly(image: PIL.Image) -> PIL.Image:
    array = np.array(image)
    result = warp_image_array_randomly(array)
    return PIL.Image.fromarray(result)

def warp_image_array_randomly(image: np.ndarray) -> np.ndarray:
    center = (image.shape[1] / 2, image.shape[0] / 2)
    num_points = 3
    upper = [(i * image.shape[1] / num_points, 0) for i in range(num_points)]
    source = [(i * image.shape[1] / num_points, center[1]) for i in range(num_points)]
    lower = [(i * image.shape[1] / num_points, image.shape[0]) for i in range(num_points)]
    max_random_offset = 10
    destination = [(i * image.shape[1] / num_points, center[1] + np.random.randint(-max_random_offset, max_random_offset)) for i in range(num_points)]
    result = calculate_dewarp_transformation(image, [upper, source, lower], [upper, destination, lower]).dewarp(image, order=3)
    return (255 * result).astype(np.uint8)

if __name__ == "__main__":
    import sys
    image = cv2.imread(sys.argv[1])
    cv2.imwrite(sys.argv[2], warp_image_array_randomly(image))
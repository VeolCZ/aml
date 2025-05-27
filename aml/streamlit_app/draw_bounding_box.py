from PIL import Image, ImageDraw
from streamlit.runtime.uploaded_file_manager import UploadedFile


def draw_bounding_box(
    image: UploadedFile | str, relative_coordinates: list[float]
) -> Image.ImageFile:
    """
    Draw a rectangle on a given image based on the given relative coordinates
    of the representative corners.

    Args:
        image: the image on which to draw
        relative_coordinates: the relative coordinates of the rectangle
                             (left, top, right, bottom)

    Returns:
        Image.ImageFile: the drawn on image
    """
    drawn_image = Image.open(image)
    draw = ImageDraw.Draw(drawn_image)
    size_x, size_y = drawn_image.size
    left = relative_coordinates[0] * size_x
    top = relative_coordinates[1] * size_y
    right = relative_coordinates[2] * size_x
    bottom = relative_coordinates[3] * size_y

    draw.rectangle((left, top, right, bottom), outline="blue", width=3)

    return drawn_image

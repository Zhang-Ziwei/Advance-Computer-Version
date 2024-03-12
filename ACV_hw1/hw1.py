import numpy as np
import cv2
import matplotlib.pyplot as plt

header_size = 216  # size of the header in bytes
image_size = [386, 386]
file_path1 = 'trucka.bmp'  # Path of BMP file
file_path2 = 'truckb.bmp'  # Path of BMP file
block_size = 9  # block size 9, 11, 15, 21, 31,
image_cut_size = 386 % block_size
search_range = 10  # search range, 10, 15, 20, 30, 50

data1_1 = cv2.imread(file_path1)
# 刪除不能被整數分割的邊緣行列
data1_2 = np.delete(data1_1, np.s_[385-image_cut_size:385], axis=1)
image_data1 = np.delete(data1_2, np.s_[385-image_cut_size:385], axis= 0)

data2_1 = cv2.imread(file_path2)
# 刪除不能被整數分割的邊緣行列
data2_2 = np.delete(data2_1, np.s_[385-image_cut_size:385], axis=1)
image_data2 = np.delete(data2_2, np.s_[385-image_cut_size:385], axis= 0)
# image_data1 = cv2.imread(file_path1)
# image_data2 = cv2.imread(file_path2)
'''
# Open the file in binary mode
with open(file_path1, 'rb') as file:
    # Skip the header by reading and discarding the header bytes
    file.read(header_size)
    # Now you can read the rest of the file, which contains the image pixel data
    data1 = np.fromfile(file, dtype=np.uint8)
data1_1 = data1.reshape(image_size)
data1_2 = np.delete(data1_1, [385-image_cut_size, 385], axis= 1)
image_data1 = np.delete(data1_1, [385-image_cut_size, 385], axis= 0)

print(image_data1)
with open(file_path2, 'rb') as file:
    # Skip the header by reading and discarding the header bytes
    file.read(header_size)
    # Now you can read the rest of the file, which contains the image pixel data
    image_data2 = file.read()
'''


def sample_image_into_blocks(image, block_size):
    # Initialize a list to hold the blocks
    blocks = []

    # Loop over the image and extract blocks
    for i in range(int(386/block_size)):
        for j in range(int(386 / block_size)):
            # Slice out a block from the image
            block = image[(i * block_size): ((i + 1) * block_size), (j * block_size): ((j + 1) * block_size)]
            blocks.append(block)

    return blocks


image_blocks1 = sample_image_into_blocks(image_data1, block_size)
image_blocks2 = sample_image_into_blocks(image_data2, block_size)

# Block Matching Function
def block_matching(trucka, truckb, block_size, search_range):
    height= width = 386-image_cut_size
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            best_sad = float('inf')
            best_dx, best_dy = 0, 0

            # Define the search range for the current block
            search_positions = range(-search_range, search_range + 1)
            for dy in search_positions:
                for dx in search_positions:
                    y1, x1 = y + dy, x + dx
                    if 0 <= x1 < width - block_size and 0 <= y1 < height - block_size:
                        block_a = trucka[y:y + block_size, x:x + block_size]
                        block_b = truckb[y1:y1 + block_size, x1:x1 + block_size]

                        # Sum of Absolute Differences (SAD)
                        sad = np.sum(np.abs(block_a - block_b))
                        if sad < best_sad:
                            best_sad = sad
                            best_dx, best_dy = dx, dy

            motion_vectors[y // block_size, x // block_size] = [best_dx, best_dy]

    return motion_vectors

# Perform block matching for each block size
motion_vectors = block_matching(image_data1, image_data2, block_size, search_range)
print(f"Motion Vectors for block size {block_size}x{block_size}:")
print(motion_vectors)


# Visualization function
def draw_motion_vectors(image, motion_vectors, block_size):
    """
    Draws the motion vectors on the image.

    :param image: The image as a 2D numpy array.
    :param motion_vectors: A 3D numpy array of motion vectors (y, x, 2).
    :param block_size: The size of the blocks that the motion vectors correspond to.
    """
    # Create a figure and axis to draw on
    fig, ax = plt.subplots()

    # Show the image with original image
    # ax.imshow(image, cmap='gray')

    # Show the image with empty image
    shape = (386-image_cut_size, 386-image_cut_size, 3)
    empty_img = np.zeros(shape, np.uint8)
    empty_img.fill(255)
    for i in range(int(386/block_size-1)):
        cv2.line(empty_img, (block_size*(i+1), 0), (block_size*(i+1), shape[0]), (0,0,0))
        cv2.line(empty_img, (0, block_size * (i + 1)), (shape[0], block_size * (i + 1)), (0, 0, 0))
    ax.imshow(empty_img, cmap='gray')

    # Go through each motion vector and draw it
    for i in range(motion_vectors.shape[0]):
        for j in range(motion_vectors.shape[1]):
            # Starting point of the vector
            start_y = i * block_size + block_size // 2
            start_x = j * block_size + block_size // 2

            # The motion vector
            dy, dx = motion_vectors[i, j]

            # Draw the motion vector if it is not zero
            if dy != 0 or dx != 0:
                ax.arrow(start_x, start_y, dx, dy,length_includes_head = True, head_width=block_size // 3, head_length=block_size // 3, fc='red',
                         ec='red')

    # Set the axis limits
    ax.set_xlim([0, 386-image_cut_size])
    ax.set_ylim([386-image_cut_size, 0])

    # Hide the axes, ticks, labels
    ax.axis('off')

    # Show the plot
    plt.show()

# Call the function with a block size of 9 (replace this with your real data)
draw_motion_vectors(image_data2, motion_vectors, block_size)

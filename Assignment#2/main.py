from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt




def load_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image_array = np.array(image)
    return image_array



def spatial_gradient(frame1):
    gx_filter = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
    gy_filter = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])
    padding = 1
    padded_image = np.pad(frame1, padding, mode="constant")

    Ix = np.zeros_like(frame1)
    Iy = np.zeros_like(frame1)

    frame1_height, frame1_width = frame1.shape
    for i in range(1, frame1_height + 1):
        for j in range(1, frame1_width + 1):
            Ix[i -1, j - 1] = np.sum(padded_image[i - 1: i + 2, j - 1 : j + 2]* gx_filter)
            Iy[i - 1, j - 1] = np.sum(padded_image[i - 1: i + 2, j - 1: j + 2] * gy_filter)

    # #Normalizing output
    Ix = (Ix- Ix.min()) / (Ix.max() - Ix.min())
    Iy = (Iy -Iy.min()) / (Iy.max() -Iy.min())
    return Ix,Iy

def lucas_kanade_method(frame1, frame2):
    # Compute spatial gradient
    Ix, Iy = spatial_gradient(frame1)

    # Compute temporal derivative
    It = frame2 - frame1

    # Compute optical flow vectors
    Vx = np.zeros_like(frame1)
    Vy = np.zeros_like(frame1)

    padded_frame1 = np.pad(frame1, pad_width=1, mode="constant")
    padded_Ix = np.pad(Ix, pad_width=1, mode="constant")
    padded_Iy = np.pad(Iy, pad_width=1, mode="constant")
    padded_It = np.pad(It, pad_width=1, mode="constant")


    for y in range(frame1.shape[0]):
        for x in range(frame1.shape[1]):
            # Extract patch around the pixel
            Ix_patch = padded_Ix[y : y + 3, x : x + 3].flatten()
            Iy_patch = padded_Iy[y : y + 3, x : x + 3].flatten()
            It_patch = padded_It[y : y + 3, x : x + 3].flatten()

            A = np.vstack((Ix_patch, Iy_patch)).T
            b = -It_patch
            optical_flow = np.linalg.lstsq(A, b, rcond=None)[0]
            Vx[y, x] = optical_flow[0]
            Vy[y, x] = optical_flow[1]

    return Vx, Vy

def visualize_optical_flow(flow_x, flow_y):
    # Compute the magnitude and angle of the flow
    magnitude = np.sqrt(flow_x ** 2 + flow_y** 2)
    angle = np.arctan2(flow_x , flow_y)

    # Normalize the magnitude
    # magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

    # Convert the angle to RGB color
    hue = angle * 180 / np.pi / 2
    hsv = np.stack((hue, np.ones_like(hue), magnitude), axis=-1)
    rgb = plt.get_cmap('hsv')(hsv)

    return rgb


if __name__ == "__main__":
    path_1_a = "frame1_a.png"
    path_1_b = "frame1_b.png"
    path_2_a = "frame2_a.png"
    path_2_b = "frame2_b.png"
    
    frame_1_a = load_image(path_1_a)
    frame_1_b = load_image(path_1_a)
    frame_2_a = load_image(path_2_a)
    frame_2_b = load_image(path_2_b)
    # Calculate optical flow
    Vx_1, Vy_1 = lucas_kanade_method(frame_1_a, frame_1_b)
    Vx_2, Vy_2 = lucas_kanade_method(frame_2_a, frame_2_b)

    print(Vx_1, Vy_1)
    print(Vx_2, Vy_2)

    # # Visualize optical flow vectors
    # flow_vis_1 = visualize_optical_flow(Vx_1, Vy_1)
    # flow_vis_2 = visualize_optical_flow(Vx_2, Vy_2)
    #
    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(121)
    # plt.title('Visualization of Optical Flow (Frame 1)')
    # plt.imshow(flow_vis_1)
    # plt.axis('off')
    #
    # plt.subplot(122)
    # plt.title('Visualization of Optical Flow (Frame 2)')
    # plt.imshow(flow_vis_2)
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    # Visualize magnitude of optical flow
    magnitude_1 = np.sqrt(Vx_1 ** 2 + Vy_1 ** 2)
    magnitude_2 = np.sqrt(Vx_2 ** 2 + Vy_2 ** 2)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(magnitude_1, cmap='viridis')
    plt.title('Magnitude of Optical Flow (Frame 1)')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(magnitude_2, cmap='viridis')
    plt.title('Magnitude of Optical Flow (Frame 2)')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

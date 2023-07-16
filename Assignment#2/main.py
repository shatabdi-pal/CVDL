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

    Ix = np.zeros_like(frame1, dtype=np.float32)
    Iy = np.zeros_like(frame1, dtype=np.float32)
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
    print(It)
    # Compute optical flow vectors
    Vx = np.zeros_like(frame1)
    Vy = np.zeros_like(frame1)

    kernel = np.ones((3, 3))

    for y in range(1, frame1.shape[0] - 1):
        for x in range(1, frame1.shape[1] - 1):
            Ix_patch = Ix[y - 1:y + 2, x - 1:x + 2].flatten()
            Iy_patch = Iy[y - 1:y + 2, x - 1:x + 2].flatten()
            It_patch = It[y - 1:y + 2, x - 1:x + 2].flatten()


    A = np.vstack((Ix_patch, Iy_patch)).T
    b = -It_patch
    optical_flow = np.linalg.lstsq(A, b, rcond=None)[0]
    Vx[y, x] = optical_flow[0]
    Vy[y, x] = optical_flow[1]

    return Vx, Vy

if __name__ == "__main__":
    path_1_a = "frame1_a.png"
    path_1_b = "frame1_b.png"
    path_2_a = "frame2_a.png"
    path_2_b = "frame2_b.png"
    
    frame_1_a = load_image(path_1_a)
    frame_1_b = load_image(path_1_a)
    frame_2_a = load_image(path_2_a)
    frame_2_b = load_image(path_2_b)
    Vx, Vy = lucas_kanade_method(frame_2_a, frame_2_b)


    # Visualize optical flow vectors
    plt.figure(figsize=(10, 5))
    plt.quiver(Vx, Vy, color='r')
    plt.title('Optical Flow Vectors')
    plt.show()

    # Visualize magnitude of optical flow
    magnitude = np.sqrt(Vx ** 2 + Vy ** 2)
    plt.imshow(magnitude, cmap='hot')
    plt.title('Magnitude of Optical Flow')
    plt.colorbar()
    plt.show()
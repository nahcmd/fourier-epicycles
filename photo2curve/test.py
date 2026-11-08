import numpy as np, cv2

p = r"C:\Users\김요한\Desktop\VSCode\fourier-epicycles\photo2curve\images.jpg"
data = np.fromfile(p, dtype=np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)

print(img.shape, img.size, img.dtype)
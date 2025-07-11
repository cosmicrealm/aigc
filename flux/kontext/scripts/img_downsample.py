import cv2
from PIL import Image
hr_path = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/datasets/datasets/sr/ffhq-dataset/images512x512/00000/00000.png"
scale = 8
# img_hr = cv2.imread(hr_path)

# img_lr = cv2.resize(img_hr, (img_hr.shape[1] // scale, img_hr.shape[0] // scale), interpolation=cv2.INTER_CUBIC)
# img_lr = cv2.resize(img_lr, (img_hr.shape[1], img_hr.shape[0]), interpolation=cv2.INTER_CUBIC)

# cv2.imwrite("asserts/lr_image.png", img_lr)
# cv2.imwrite("asserts/hr_image.png", img_hr)  # Save the high-resolution image
# print("Downsampled image saved as asserts/lr_image.png")

hr = Image.open(hr_path)

# LR 下采样 with bicubic interpolation
lr = hr.resize((hr.width // scale, hr.height // scale),Image.BICUBIC)
# resize lr to hr size
lr = lr.resize((hr.width, hr.height), Image.BICUBIC)
lr.save("asserts/lr_image.png")
hr.save("asserts/hr_image.png")  # Save the high-resolution image
print("Downsampled image saved as asserts/lr_image.png and high-resolution image saved as asserts/hr_image.png")
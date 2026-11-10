# Đọc ảnh bằng OpenCV,in shape, dtype, width, height, channels. 
# Chuyển đổi BGR → RGB → HSV → Grayscale.

#  uint8 là kiểu số nguyên không dấu, 8 bit.

#   u = unsigned: không có số âm
#  8 = 8 bit = 1 byte
#  Giá trị chỉ từ 0 đến 255, tổng cộng 256 mức.


# Digital Image: ảnh số là một ma trận các điểm ảnh, mỗi điểm ảnh lưu giá trị cường độ / màu sắc.
# Pixel: đơn vị nhỏ nhất của ảnh. Với ảnh màu, mỗi pixel thường có 3 giá trị (R, G, B).
# Image Resolution, Width, Height, Channels: độ phân giải = width × height; 
# channels = số kênh màu (1 với ảnh xám, 3 với ảnh màu).
# RGB và BGR khác nhau thế nào? Cùng 3 kênh, chỉ khác thứ tự lưu trữ. OpenCV dùng BGR, 
# Matplotlib hầu hết thư viện hiển thị dùng RGB — quên chuyển đổi sẽ bị lệch màu đỏ / xanh.
# Grayscale Image: ảnh 1 kênh, mỗi pixel chỉ có 1 giá trị độ sáng (0 = đen, 255 = trắng với ).
# HSV Color Space: Hue (màu sắc) – Saturation (độ bão hòa) – Value (độ sáng).
# $H thuoc [0, 360)  S thuoc [0, 1.0] hoặc [0, 100\%] ; V in [0, 1.0]$ hoặc $[0, 100\%]$
# Tách màu sắc khỏi độ sáng nên rất tiện để phát hiện / theo dõi vật theo màu, không nhạy với thay đổi ánh sáng như RGB.

# Image representation trong NumPy: 
# ảnh trong OpenCV chính là numpy.ndarray có shape (height, width, channels).
# Data types: uint8, float32: uint8 (0–255) là dạng ảnh thông thường; 
# float32 (thường 0.0–1.0) dùng khi tính toán, tránh tràn số.
"""
Không gian,Dùng khi nào?
BGR,Xử lý ảnh cơ bản nội bộ bằng OpenCV.
RGB,"Đưa vào mô hình Deep Learning (PyTorch, TensorFlow) hoặc hiển thị bằng Matplotlib."
HSV,"Tách vật thể theo màu sắc (lọc biển báo giao thông, bám theo trái banh, phân khúc da tay)."
Grayscale,"Nhận diện cạnh (Canny), tìm góc (Harris/Shi-Tomasi), hoặc phát hiện khuôn mặt khi không cần quan tâm đến màu sắc."
"""
import cv2
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


IMG = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/images/sample.png")
OUT = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/output/bai1_representation")
OUT.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(IMG))  # OpenCV đọc mặc định ở dạng BGR
assert img_bgr is not None, f"Không đọc được ảnh: {IMG}"
h, w, c = img_bgr.shape
print("=" * 55)
print("1. THÔNG TIN ẢNH THẬT")
print("=" * 55)
print(f"shape: {img_bgr.shape}   # (height, width, channels)")
print(f"dtype: {img_bgr.dtype}")
print(f"width: {w} px")
print(f"height: {h} px")
print(f"channel: {c}")
print(f"số pixel: {h * w:,}; tổng số giá trị: {img_bgr.size:,}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
hue, sat, val = cv2.split(img_hsv)

# Điểm mẫu — lấy theo [y, x]. Tọa độ ước lượng từ ảnh 447x447:
diem = {
    "nền (20, 20)": (20, 20),
    "đầu ĐEN (50, 223)": (50, 223),
    "đèn XANH (115, 223)": (115, 223),
    "ngực BẠC (280, 150)": (280, 150),
    "vai trái BẠC (280, 70)": (280, 70),
}
print("\n2. PIXEL — cùng 1 điểm, 4 cách biểu diễn")
for ten, (y, x) in diem.items():
    print(f"  {ten}:")
    print(f"    BGR={img_bgr[y, x].tolist()}  RGB={img_rgb[y, x].tolist()}  "
          f"HSV={img_hsv[y, x].tolist()}  Gray={int(img_gray[y, x])}")

# --- Hình 1: so sánh -------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(f"Bài 1 — Ảnh thật (robot)  |  {w}x{h}, shape {img_bgr.shape}, dtype {img_bgr.dtype}",
             fontsize=14, fontweight="bold")
axes[0, 0].imshow(img_bgr)  # cố tình KHÔNG chuyển: Matplotlib hiểu là RGB
axes[0, 0].set_title("1. BGR hiển thị trực tiếp\nSAI MÀU: đèn xanh dương\nsẽ lệch sang tông khác", color="red")
axes[0, 1].imshow(img_rgb)
axes[0, 1].set_title("2. BGR → RGB\nĐÚNG MÀU (Matplotlib dùng RGB)")
axes[0, 2].imshow(img_gray, cmap="gray")
axes[0, 2].set_title("3. Grayscale\n1 kênh, chỉ còn độ sáng\nĐèn xanh sáng -> xám sáng")
axes[1, 0].imshow(img_hsv)
axes[1, 0].set_title("4. HSV (xem như RGB)\nMàu lạ là bình thường — HSV\nkhông phải để hiển thị")
axes[1, 1].imshow(hue, cmap="hsv")
axes[1, 1].set_title("5. Kênh H (Hue)\nĐèn xanh: Hue ~99. Vùng xám/bạc\nS thấp nên Hue nhiễu, không đáng tin")
axes[1, 2].imshow(val, cmap="gray")
axes[1, 2].set_title("6. Kênh V (Value)\nĐộ sáng — ngực bạc sáng nhất,\nđầu đen tối nhất")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai1_robot_overview.png", dpi=150)
plt.close(fig)

# --- Hình 2: pixel quanh đèn xanh ----------------------------------------
# Vùng 12x12 quanh đèn xanh (y=109..121, x=217..229): đen -> xanh sáng -> đen
crop_bgr = img_bgr[109:121, 217:229]
crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Pixel là gì? — Vùng 12x12 px quanh đèn xanh trên mặt robot",
             fontsize=13, fontweight="bold")
axes[0].imshow(cv2.resize(crop_rgb, (360, 360), interpolation=cv2.INTER_NEAREST))
axes[0].set_title("Ảnh màu: mỗi ô vuông = 1 pixel\nĐèn xanh sáng nổi trên nền đen")
axes[1].imshow(cv2.resize(crop_gray, (360, 360), interpolation=cv2.INTER_NEAREST),
               cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Ảnh xám: mỗi pixel chỉ 1 số\nĐèn xanh -> xám sáng, nền đen -> tối")
axes[2].imshow(crop_gray, cmap="gray", vmin=0, vmax=255)
for i in range(12):
    for j in range(12):
        v = int(crop_gray[i, j])
        axes[2].text(j, i, str(v), ha="center", va="center", fontsize=7,
                     color="white" if v < 128 else "black")
axes[2].set_title("Ma trận số thật của ảnh xám 12x12\nHàng có đèn xanh có giá trị cao hẳn")
axes[2].set_xticks(range(12))
axes[2].set_yticks(range(12))
axes[2].grid(color="red", linewidth=0.5)
for ax in axes[:2]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai1_robot_pixels.png", dpi=150)
plt.close(fig)
print(f"\nĐã lưu: {OUT / 'bai1_robot_overview.png'}")
print(f"Đã lưu: {OUT / 'bai1_robot_pixels.png'}")

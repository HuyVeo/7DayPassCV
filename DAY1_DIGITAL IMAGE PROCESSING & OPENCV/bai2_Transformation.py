# Bài 2 — Transformation, chạy trên ẢNH THẬT của bạn (robot)
# Chạy:  python bai2_transformation.py
# Ảnh gốc: images/robot.png (447x447, ảnh bạn gửi)
# Kết quả: output/bai2_geometric.png, output/bai2_interpolation.png,
#           output/bai2_histogram.png
# Tư duy nhanh: Affine = kéo / xoay / dịch tờ ảnh in trên mặt phẳng
# (đường song song vẫn song song). Perspective = chụp nghiêng tờ giấy
# rồi "nắn" lại (đường song song không còn song song).
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


IMG = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/images/sample.png")
OUT = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/output/bai2_transformation")
OUT.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(IMG))
assert img_bgr is not None, f"Không đọc được ảnh: {IMG}"
h, w, _ = img_bgr.shape
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print("=" * 55)
print("ẢNH GỐC")
print("=" * 55)
print(f"shape={img_bgr.shape}  dtype={img_bgr.dtype}  width={w}  height={h}")

# --- 1. Resize / Crop / Rotate / Flip -------------------------------------
resized = cv2.resize(img_bgr, (224, 224))                    # 447 -> 224 (~1/2)
cropped = img_bgr[0:200, 120:330]                            # [y1:y2, x1:x2] = đầu robot
rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
flipped = cv2.flip(img_bgr, 1)                               # 1 = lật ngang
print("\n1. RESIZE / CROP / ROTATE / FLIP")
print(f"  resized (224,224): {resized.shape}")
print(f"  cropped [0:200, 120:330]: {cropped.shape}  (đầu robot, có đèn xanh)")
print(f"  rotated 90° CW: {rotated.shape}  (vuông 447x447 nên shape không đổi)")
print(f"  flipped ngang: {flipped.shape}")

# --- 2. Affine: Translation + Scaling -------------------------------------
# Translation: dịch (+50, +30) — ma trận [[1,0,tx],[0,1,ty]]
M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(img_bgr, M_trans, (w, h))
# Scaling: co 0.7 lần quanh gốc, dùng warpAffine với ma trận scale
M_scale = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 0.7)    # scale=0.7, không xoay
scaled = cv2.warpAffine(img_bgr, M_scale, (w, h))
print("\n2. AFFINE (warpAffine)")
print(f"  translation (+50,+30): {translated.shape} — góc trái-trên bị cắt, góc phải-dưới có viền đen")
print(f"  scaling 0.7x quanh tâm: {scaled.shape} — robot nhỏ lại, viền đen xuất hiện 4 phía")

# --- 3. Perspective ---------------------------------------------------------
# Giả lập chụp nghiêng: 4 góc ảnh -> 4 điểm lệch (tờ giấy bị nghiêng)
src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
dst = np.float32([[50, 30], [400, 0], [w - 1, h - 1], [0, 420]])
M_persp = cv2.getPerspectiveTransform(src, dst)
persp = cv2.warpPerspective(img_bgr, M_persp, (w, h))
print("\n3. PERSPECTIVE (warpPerspective)")
print(f"  perspective: {persp.shape} — cạnh trên ngắn lại, không còn là hình chữ nhật")

# --- 4. Interpolation: phóng to vùng đèn xanh ------------------------------
# Vùng 40x40 quanh đèn xanh (y=95..135, x=203..243), phóng 8x = 320x320
patch = img_bgr[95:135, 203:243]
near = cv2.resize(patch, (320, 320), interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(patch, (320, 320), interpolation=cv2.INTER_LINEAR)
cubic = cv2.resize(patch, (320, 320), interpolation=cv2.INTER_CUBIC)
print("\n4. INTERPOLATION (vùng 40x40 quanh đèn xanh -> 320x320)")
print(f"  patch gốc: {patch.shape}")
print("  Nearest: răng cưa rõ, giữ đúng giá trị pixel gốc (không tạo giá trị mới)")
print("  Bilinear: mượt hơn, trung bình 4 pixel kề — đèn xanh loang nhẹ ra nền đen")
print("  Bicubic: mượt nhất, trung bình 16 pixel kề — biên mềm nhất, chậm nhất")

# --- 5. Histogram Equalization + Contrast / Brightness --------------------
eq = cv2.equalizeHist(gray)
bright = cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=20)   # tương phản x1.3, sáng +20
dark = cv2.convertScaleAbs(img_bgr, alpha=0.7, beta=-10)    # tương phản x0.7, tối -10
print("\n5. HISTOGRAM + CONTRAST / BRIGHTNESS (trên ảnh xám)")
print(f"  gray: mean={gray.mean():.1f}  std={gray.std():.1f}  min={int(gray.min())}  max={int(gray.max())}")
print(f"  equalized: mean={eq.mean():.1f}  std={eq.std():.1f}  min={int(eq.min())}  max={int(eq.max())}")
print("  -> std tăng = độ tương phản tăng: vùng tối (đầu đen, nền) và vùng sáng (ngực bạc) tách nhau rõ hơn")
print(f"  bright (alpha=1.3, beta=+20): mean BGR={bright.mean(axis=(0,1)).round(1).tolist()}")
print(f"  dark   (alpha=0.7, beta=-10): mean BGR={dark.mean(axis=(0,1)).round(1).tolist()}")

# --- Hình 1: biến đổi hình học --------------------------------------------
def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Bài 2 — Transformation  |  Ảnh robot 447x447", fontsize=14, fontweight="bold")
panels = [
    (img_rgb, "1. Gốc\n447x447"),
    (to_rgb(resized), "2. Resize -> 224x224\n(nhỏ còn 1/2, ~1/4 số pixel)"),
    (to_rgb(cropped), "3. Crop [0:200, 120:330]\n[y, x] — chỉ lấy đầu robot"),
    (to_rgb(rotated), "4. Rotate 90° CW\nđầu robot quay sang phải"),
    (to_rgb(flipped), "5. Flip ngang\ntrái <-> phải đổi chỗ"),
    (to_rgb(translated), "6. Translation (+50, +30)\nAffine: dịch, viền đen bù vào"),
]
for ax, (im, title) in zip(axes.ravel(), panels):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai2_geometric.png", dpi=150)
plt.close(fig)

# --- Hình 2: interpolation + perspective ----------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Bài 2 — Interpolation & Perspective  |  Vùng đèn xanh 40x40 phóng 8x",
             fontsize=13, fontweight="bold")
panels2 = [
    (to_rgb(near), "Nearest Neighbor\nrăng cưa, ô vuông rõ"),
    (to_rgb(linear), "Bilinear\nmượt, loang nhẹ"),
    (to_rgb(cubic), "Bicubic\nmượt nhất, biên mềm"),
    (to_rgb(scaled), "Affine Scaling 0.7x\nrobot nhỏ lại, giữ song song"),
    (to_rgb(persp), "Perspective\ncạnh trên ngắn lại —\nkhông còn song song"),
    (to_rgb(cropped), "Crop đầu robot (đối chiếu)\nđèn xanh ở giữa"),
]
for ax, (im, title) in zip(axes.ravel(), panels2):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai2_interpolation.png", dpi=150)
plt.close(fig)

# --- Hình 3: histogram ------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Bài 2 — Histogram Equalization & Contrast / Brightness",
             fontsize=13, fontweight="bold")
axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title(f"Gray gốc\nmean={gray.mean():.0f}, std={gray.std():.0f}")
axes[0, 1].imshow(eq, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title(f"Equalized (cv2.equalizeHist)\nmean={eq.mean():.0f}, std={eq.std():.0f}")
axes[0, 2].imshow(to_rgb(bright))
axes[0, 2].set_title("Sáng + tương phản\n(alpha=1.3, beta=+20)")
axes[1, 0].hist(gray.ravel(), bins=256, range=(0, 255))
axes[1, 0].set_title("Histogram gray gốc\n(dồn ở tối: nền + đầu đen)")
axes[1, 1].hist(eq.ravel(), bins=256, range=(0, 255))
axes[1, 1].set_title("Histogram sau Equalization\ngiãn đều ra toàn dải 0-255")
axes[1, 2].imshow(to_rgb(dark))
axes[1, 2].set_title("Tối + giảm tương phản\n(alpha=0.7, beta=-10)")
for ax in axes.ravel():
    ax.axis("off") if ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]) else None
axes[1, 0].axis("on")
axes[1, 1].axis("on")
plt.tight_layout()
plt.savefig(OUT / "bai2_histogram.png", dpi=150)
plt.close(fig)

print(f"\nĐã lưu: {OUT / 'bai2_geometric.png'}")
print(f"Đã lưu: {OUT / 'bai2_interpolation.png'}")
print(f"Đã lưu: {OUT / 'bai2_histogram.png'}")

          # Bài 5 — Thresholding & Morphology (FULL), chạy trên ẢNH THẬT của bạn (robot)
# Chạy:  python bai5_threshold_morphology.py
# Ảnh gốc: images/robot.png (447x447, ảnh bạn gửi)
# Kết quả: output/bai5_threshold.png, output/bai5_morphology.png, output/bai5_zoom.png
#
# Tư duy nhanh:
# Thresholding = đặt một vạch mức xám: sáng hơn vạch -> trắng, tối hơn -> đen.
#   Giống như chấm thi: >= 5 đậu, < 5 trượt — chỉ khác là vạch này chia pixel.
# Morphology = dùng một khuôn (structuring element) cọ qua ảnh nhị phân:
#   Erosion  "ăn mòn" vùng trắng, Dilation "giãn" vùng trắng.
#   Opening  = Erosion rồi Dilation: quét đốm trắng nhỏ đi, giữ nguyên hình vật.
#   Closing  = Dilation rồi Erosion: lấp lỗ đen nhỏ trong vật.
#   Morphological Gradient = Dilation - Erosion: chỉ còn lại biên của vật.
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMG = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/images/sample.png")
OUT = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/output/bai5_threshold_morphology")
OUT.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(IMG))
assert img_bgr is not None, f"Không đọc được ảnh: {IMG}"
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f"Ảnh gốc: shape={img_bgr.shape}  gray: mean={gray.mean():.1f} std={gray.std():.1f}"
      f"  min={int(gray.min())} max={int(gray.max())}")
# Ánh sáng không đồng đều: nền trên sáng hơn nền dưới?
print(f"  độ sáng trung bình: nửa trên={gray[:223].mean():.1f}, nửa dưới={gray[223:].mean():.1f}")


def ty_le_trang(anh_nhi_phan):
    return 100.0 * (anh_nhi_phan > 0).sum() / anh_nhi_phan.size


def dem_vat(anh_nhi_phan):
    """Số vật (connected components, trừ nền) — nhiễu tạo ra rất nhiều vật nhỏ."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(anh_nhi_phan, connectivity=8)
    return n - 1, stats[1:, cv2.CC_STAT_AREA]


# --- 1. Thresholding --------------------------------------------------------
# Binary: 1 ngưỡng chung cho cả ảnh — thất bại khi sáng không đều.
_, binary127 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Otsu: tự tìm ngưỡng tối ưu từ histogram (cực đại phương sai giữa 2 lớp).
otsu_t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Adaptive: ngưỡng riêng cho từng vùng nhỏ (block 11x11) — trị được sáng không đều.
adapt_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
adapt_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
# Binary ngược: để thấy vật tối (đầu đen) thành trắng.
_, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

print("\n1. THRESHOLDING — % pixel trắng")
for ten, anh in [("Binary 127", binary127), ("Otsu", otsu),
                 ("Adaptive Gaussian 11", adapt_gauss), ("Adaptive Mean 11", adapt_mean),
                 ("Binary Inv 127", binary_inv)]:
    print(f"  {ten:<22}: {ty_le_trang(anh):.2f}% trắng")
print(f"  -> Ngưỡng Otsu tự chọn: {otsu_t:.0f} (không phải 127 do mình đoán)")
print("  -> Binary 127: nền xám (~60-120) bị xếp chung với đầu đen -> mất nền/vật.")
print("     Adaptive tính ngưỡng theo từng vùng nên tách được chữ ROBOSTORE,")
print("     khung mặt và các chi tiết nhỏ dù nền sáng không đều — nhưng nền")
print("     phẳng cũng sinh nhiều chấm nhiễu, % trắng vọt lên 86.8% (Gauss).")

# --- 2. Tạo ảnh nhị phân có nhiễu, để morphology có việc để làm -----------
rng = np.random.default_rng(42)
noisy_gray = gray.copy()
mat = rng.random(gray.shape)
noisy_gray[mat < 0.01] = 0      # tiêu: 1% pixel -> đen
noisy_gray[mat > 0.99] = 255    # muối: 1% pixel -> trắng
_, noisy_bin = cv2.threshold(noisy_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"\n2. ẢNH NHỊ PHÂN CÓ NHIỄU (Otsu trên ảnh thêm Salt-and-Pepper 2%)")
print(f"  Otsu sạch: {dem_vat(otsu)[0]} vật | Otsu nhiễu: {dem_vat(noisy_bin)[0]} vật"
      f"  -> nhiễu sinh ra hàng trăm vật nhỏ giả")

# --- 3. Morphology -----------------------------------------------------------
# Structuring element: khuôn 5x5. RECT = vuông, ELLIPSE = tròn, CROSS = chữ thập.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print("\n3. STRUCTURING ELEMENT (kernel 5x5, RECT):")
print(kernel)
erosion = cv2.erode(noisy_bin, kernel, iterations=1)
dilation = cv2.dilate(noisy_bin, kernel, iterations=1)
opening = cv2.morphologyEx(noisy_bin, cv2.MORPH_OPEN, kernel)    # Erosion -> Dilation
closing = cv2.morphologyEx(noisy_bin, cv2.MORPH_CLOSE, kernel)   # Dilation -> Erosion
gradient = cv2.morphologyEx(otsu, cv2.MORPH_GRADIENT, kernel)    # Dilation - Erosion

print("\n   MORPHOLOGY trên ảnh nhị phân nhiễu — số vật còn lại (càng ít càng sạch):")
for ten, anh in [("Gốc nhiễu", noisy_bin), ("Erosion", erosion), ("Dilation", dilation),
                 ("Opening", opening), ("Closing", closing)]:
    n_vat, dien_tich = dem_vat(anh)
    print(f"  {ten:<12}: {n_vat:>4} vật, % trắng={ty_le_trang(anh):.2f}%")
print("  -> Opening quét sạch đốm trắng nhỏ (vật giảm mạnh) nhưng cũng ăn mất")
print("     chi tiết nhỏ thật: Otsu sạch có 71 vật, Opening trên ảnh nhiễu chỉ còn 16.")
print("     Closing chỉ lấp lỗ ĐEN trong vật trắng, KHÔNG xóa đốm trắng")
print("     nên số vật chỉ giảm 1646 -> 1337, % trắng tăng lên 19.43%.")
print("     Erosion làm vật trắng teo lại, Dilation làm vật trắng phình ra,")
print("     nên dùng đơn lẻ sẽ làm méo hình — Opening/Closing mới giữ hình dạng.")
print(f"  Morphological Gradient (trên Otsu sạch): % trắng={ty_le_trang(gradient):.2f}%"
      " — chỉ còn biên của robot.")

# --- Hình 1: thresholding ----------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Bài 5 — Thresholding  |  Ảnh robot, ảnh xám (nền sáng không đều)",
             fontsize=13, fontweight="bold")
panels = [
    (gray, "1. Gray gốc\nmean=78, sáng không đều:\nnửa trên 57, nửa dưới 98"),
    (binary127, f"2. Binary 127\ntrắng {ty_le_trang(binary127):.1f}% — nền\nxám bị gộp với đầu đen"),
    (otsu, f"3. Otsu (ngưỡng={otsu_t:.0f})\ntự chọn ngưỡng,\ntrắng {ty_le_trang(otsu):.1f}%"),
    (adapt_gauss, f"4. Adaptive Gaussian\ntrắng {ty_le_trang(adapt_gauss):.1f}% —\ngiữ chi tiết, nhưng\nnền cũng nhiễu chấm"),
    (adapt_mean, f"5. Adaptive Mean\ntrắng {ty_le_trang(adapt_mean):.1f}% —\nnhiễu nền nhiều hơn Gauss"),
    (binary_inv, "6. Binary Inv 127\nđảo trắng/đen:\nđầu đen thành trắng"),
]
for ax, (im, title) in zip(axes.ravel(), panels):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai5_threshold.png", dpi=150)
plt.close(fig)

# --- Hình 2: morphology -------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Bài 5 — Morphology (kernel RECT 5x5)  |  Trên ảnh nhị phân có nhiễu",
             fontsize=13, fontweight="bold")
panels2 = [
    (noisy_bin, f"1. Nhiễu (Otsu)\n{dem_vat(noisy_bin)[0]} vật —\nđốm trắng/đen khắp ảnh"),
    (erosion, "2. Erosion\nvùng trắng teo lại,\nđốm nhỏ biến mất"),
    (dilation, "3. Dilation\nvùng trắng phình ra,\nlỗ nhỏ được lấp"),
    (opening, f"4. Opening\n{dem_vat(opening)[0]} vật —\nquét đốm trắng, giữ hình"),
    (closing, f"5. Closing\n{dem_vat(closing)[0]} vật —\nlấp lỗ đen trong vật"),
    (gradient, "6. Morph. Gradient\nDilation - Erosion =\nchỉ còn biên robot"),
]
for ax, (im, title) in zip(axes.ravel(), panels2):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai5_morphology.png", dpi=150)
plt.close(fig)

# --- Hình 3: phóng to vùng ngực — Opening/Closing làm gì với chi tiết? -------
y0, y1, x0, x1 = 240, 320, 120, 200  # 80x80 quanh chữ ROBOSTORE
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Phóng to vùng ngực 80x80 (chữ ROBOSTORE) — Opening giữ chữ, Closing lấp mất chữ đen mảnh",
             fontsize=13, fontweight="bold", y=0.98)
for ax, (im, title) in zip(axes, [
        (noisy_bin, "Nhiễu\nchữ đen + đốm đen\ntrên nền trắng"),
        (erosion, "Erosion\nvùng trắng teo, chữ đen\ndày lên, đốm đen nở to"),
        (opening, "Opening\nchữ còn, đốm trắng\nsạch (đốm đen còn ít)"),
        (closing, "Closing\nchữ đen mảnh BỊ LẤP MẤT\n— chỉ dùng để lấp lỗ"),
]):
    crop = im[y0:y1, x0:x1]
    ax.imshow(cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST),
              cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(OUT / "bai5_zoom.png", dpi=150)
plt.close(fig)

print(f"\nĐã lưu: {OUT / 'bai5_threshold.png'}")
print(f"Đã lưu: {OUT / 'bai5_morphology.png'}")
print(f"Đã lưu: {OUT / 'bai5_zoom.png'}")

# Bài 4 — Edge Detection, chạy trên ẢNH THẬT của bạn (robot)
# Chạy:  python bai4_edge_detection.py
# Ảnh gốc: images/robot.png (447x447, ảnh bạn gửi)
# Kết quả: output/bai4_edges.png, output/bai4_zoom.png
# Tư duy nhanh: Gradient = độ dốc của độ sáng. Biên là chỗ dốc nhất,
# giống như đứng trên vách núi — đi ngang vách thì độ cao đổi đột ngột.
# Sobel đo độ dốc theo X / Y. Laplacian đo độ cong (đạo hàm bậc 2).
# Canny = 4 bước: làm mịn -> tính gradient -> làm mỏng biên (NMS)
#       -> ngưỡng kép + hysteresis (giữ biên yếu nếu nối với biên mạnh).
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMG = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/images/sample.png")
OUT = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/output/bai4_edge_detection")
OUT.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(IMG))
assert img_bgr is not None, f"Không đọc được ảnh: {IMG}"
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# Làm mịn trước: đạo hàm rất nhạy với nhiễu, nhất là Laplacian (bậc 2)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
print(f"Ảnh gốc: shape={img_bgr.shape}  gray: mean={gray.mean():.1f} std={gray.std():.1f}")

# --- 1. Sobel: đạo hàm theo X và Y ------------------------------------------
sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)  # biên DỌC (đổi theo X)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)  # biên NGANG (đổi theo Y)
sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
# Scharr: giống Sobel nhưng trọng số khác, chính xác hơn với biên nhỏ
scharr_x = cv2.Scharr(blur, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(blur, cv2.CV_64F, 0, 1)
scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

# --- 2. Laplacian: đạo hàm bậc 2, cả 2 hướng cùng lúc ------------------------
laplacian = cv2.Laplacian(blur, cv2.CV_64F)

# --- 3. Canny: ngưỡng kép + hysteresis ---------------------------------------
canny_lo = cv2.Canny(blur, 50, 150)    # ngưỡng thấp: giữ nhiều biên hơn, kể cả nhiễu
canny = cv2.Canny(blur, 100, 200)      # ngưỡng chuẩn cho ảnh này

# Hiển thị: CV_64F có giá trị âm nên phải lấy trị tuyệt đối rồi về uint8
sobel_x_u8 = cv2.convertScaleAbs(sobel_x)
sobel_y_u8 = cv2.convertScaleAbs(sobel_y)
sobel_mag_u8 = cv2.convertScaleAbs(sobel_mag)
scharr_mag_u8 = cv2.convertScaleAbs(scharr_mag)
lap_u8 = cv2.convertScaleAbs(laplacian)

print("\nGRADIENT TẠI 2 ĐIỂM MẪU (trên ảnh đã làm mịn)")
# Đèn xanh là vạch NGANG -> biên ngang mạnh: Sobel Y lớn, Sobel X nhỏ
y, x = 114, 223
print(f"  mép trên đèn xanh (y={y}, x={x}): SobelX={sobel_x[y, x]:.0f}  SobelY={sobel_y[y, x]:.0f}"
      f"  |Gx|={abs(sobel_x[y, x]):.0f} < |Gy|={abs(sobel_y[y, x]):.0f} -> biên ngang")
# Mép trái khung mặt là biên DỌC -> Sobel X lớn
# (điểm này tìm bằng cách quét |SobelX| lớn nhất quanh khung mặt trái)
y2, x2 = 235, 166
print(f"  mép trái khung mặt (y={y2}, x={x2}): SobelX={sobel_x[y2, x2]:.0f}  SobelY={sobel_y[y2, x2]:.0f}"
      f"  |Gx|={abs(sobel_x[y2, x2]):.0f} > |Gy|={abs(sobel_y[y2, x2]):.0f} -> biên dọc")


def dem_bien(anh, nguong=50):
    """Đếm pixel được coi là biên (giá trị > ngưỡng), tính theo % tổng số pixel."""
    n = int((anh > nguong).sum())
    return n, 100.0 * n / anh.size


print("\nSO SÁNH — SỐ PIXEL BIÊN")
print("Lưu ý: KHÔNG so sánh trực tiếp ở cùng 1 ngưỡng, vì thang đo khác nhau:")
print(f"  Sobel magnitude max={sobel_mag.max():.0f}, Scharr max={scharr_mag.max():.0f},"
      f" |Laplacian| max={np.abs(laplacian).max():.0f}")
print("  -> mỗi toán tử dùng ngưỡng riêng, chỉ để thấy thứ tự độ lớn:")
for ten, anh, nguong in [("Sobel X (ngưỡng 50)", np.abs(sobel_x), 50),
                         ("Sobel Y (ngưỡng 50)", np.abs(sobel_y), 50),
                         ("Sobel magnitude (50)", sobel_mag, 50),
                         ("Scharr magnitude (200)", scharr_mag, 200),
                         ("Laplacian abs (10)", np.abs(laplacian), 10)]:
    n, pct = dem_bien(anh, nguong)
    print(f"  {ten:<26}: {n:>6,} px  ({pct:.2f}%)")
for ten, anh in [("Canny 50-150", canny_lo), ("Canny 100-200", canny)]:
    n = int((anh > 0).sum())
    print(f"  {ten:<26}: {n:>6,} px  ({100.0 * n / anh.size:.2f}%)  <- biên mỏng 1 px, sạch nhất")
print("\nNhận xét: Sobel cho biên DÀY (vài px) và nhiều; Laplacian nhạy nhiễu,")
print("thang giá trị nhỏ nên phải hạ ngưỡng mới thấy biên; Canny cho biên")
print("MỎNG (1 px) và ÍT hơn nhờ Non-Maximum Suppression + ngưỡng kép.")

# --- Hình 1: so sánh các toán tử --------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Bài 4 — Edge Detection  |  Ảnh robot, ảnh xám đã làm mịn Gaussian 5x5",
             fontsize=13, fontweight="bold")
panels = [
    (blur, "1. Gray + GaussianBlur\nđầu vào cho mọi toán tử"),
    (sobel_x_u8, "2. Sobel X\nbiên DỌC sáng (khung mặt, vai)"),
    (sobel_y_u8, "3. Sobel Y\nbiên NGANG sáng (đèn xanh, ngực)"),
    (sobel_mag_u8, "4. Sobel magnitude\ncả 2 hướng gộp lại"),
    (lap_u8, "5. Laplacian\nbiên cả 2 hướng, nhưng\nnhiễu cũng hiện rõ"),
    (canny, "6. Canny 100-200\nbiên mỏng 1 px, sạch nhất"),
]
for ax, (im, title) in zip(axes.ravel(), panels):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai4_edges.png", dpi=150)
plt.close(fig)

# --- Hình 2: phóng to vùng đèn xanh — biên mỏng hay dày? --------------------
y0, y1, x0, x1 = 95, 135, 203, 243  # 40x40 quanh đèn xanh
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Phóng to vùng đèn xanh 40x40 (lên 320x320) — Canny mỏng 1 px, Sobel dày hơn",
             fontsize=13, fontweight="bold", y=0.98)
for ax, (im, title) in zip(axes, [
        (blur, "Gray (đã blur)\nđèn xanh là vạch ngang sáng"),
        (sobel_mag_u8, "Sobel magnitude\nbiên dày, 2 vạch trên/dưới\nđèn xanh"),
        (canny, "Canny 100-200\nbiên mỏng 1 px,\nđúng mép đèn xanh"),
]):
    crop = im[y0:y1, x0:x1]
    ax.imshow(cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST),
              cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(OUT / "bai4_zoom.png", dpi=150)
plt.close(fig)

print(f"\nĐã lưu: {OUT / 'bai4_edges.png'}")
print(f"Đã lưu: {OUT / 'bai4_zoom.png'}")

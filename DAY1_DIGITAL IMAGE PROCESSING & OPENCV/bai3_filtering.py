          # Bài 3 — Filtering, chạy trên ẢNH THẬT của bạn (robot)
# Chạy:  python bai3_filtering.py
# Ảnh gốc: images/robot.png (447x447, ảnh bạn gửi)
# Kết quả: output/bai3_filters.png, output/bai3_zoom.png
# Tư duy nhanh: Convolution = trượt một cửa sổ nhỏ (kernel) qua từng pixel,
# lấy trung bình có trọng số của pixel và các pixel kề để tạo pixel mới.
# Filter làm mịn = low-pass: giữ vùng phẳng, làm mờ chi tiết/nhiễu tần số cao.
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMG = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/images/sample.png")
OUT = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/output/bai3_filtering")
OUT.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(IMG))
assert img_bgr is not None, f"Không đọc được ảnh: {IMG}"
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f"Ảnh gốc: shape={img_bgr.shape}  dtype={img_bgr.dtype}")

# --- Kernel là gì? ----------------------------------------------------------
# Mean 3x3: mọi pixel kề có trọng số bằng nhau 1/9.
# Gaussian 3x3: pixel ở giữa nặng nhất, càng xa càng nhẹ.
print("\nKERNEL 3x3")
print("  Mean:\n", np.ones((3, 3), np.float32) / 9)
g1d = cv2.getGaussianKernel(3, 0).flatten()
print("  Gaussian (tích ngoài của [%.2f, %.2f, %.2f]):" % tuple(g1d))
print("  ", np.outer(g1d, g1d).round(3))


def psnr(a, b):
    """Độ giống với ảnh gốc (dB) — càng cao càng giống gốc, càng ít nhiễu còn lại."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def them_nhieu_gaussian(img, sigma=25, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def them_nhieu_muoi_tieu(img, ty_le=0.02, seed=42):
    rng = np.random.default_rng(seed)
    out = img.copy()
    # 1% pixel -> 0 (tiêu, đen), 1% pixel -> 255 (muối, trắng)
    mat = rng.random(img.shape[:2])
    out[mat < ty_le / 2] = 0
    out[mat > 1 - ty_le / 2] = 255
    return out


# --- Tạo 2 ảnh nhiễu riêng, để thấy mỗi filter mạnh ở loại nhiễu nào -------
noisy_gauss = them_nhieu_gaussian(img_bgr)
noisy_sp = them_nhieu_muoi_tieu(img_bgr)
# Ảnh nhiễu chung cho hình so sánh: Gaussian + Salt-and-Pepper cùng lúc
noisy = them_nhieu_muoi_tieu(noisy_gauss)

# --- Áp 4 filter lên CÙNG 1 ảnh nhiễu chung --------------------------------
mean_f = cv2.blur(noisy, (5, 5))
gauss_f = cv2.GaussianBlur(noisy, (5, 5), 0)
median_f = cv2.medianBlur(noisy, 5)
bilateral_f = cv2.bilateralFilter(noisy, 9, 75, 75)

gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
ket_qua = {
    "Noisy (chưa lọc)": gray_noisy,
    "Mean 5x5": cv2.cvtColor(mean_f, cv2.COLOR_BGR2GRAY),
    "Gaussian 5x5": cv2.cvtColor(gauss_f, cv2.COLOR_BGR2GRAY),
    "Median 5": cv2.cvtColor(median_f, cv2.COLOR_BGR2GRAY),
    "Bilateral": cv2.cvtColor(bilateral_f, cv2.COLOR_BGR2GRAY),
}
print("\nSO SÁNH TRÊN ẢNH NHIỄU CHUNG (Gaussian sigma=25 + Salt-and-Pepper 2%)")
print("  PSNR so với ảnh gốc — càng cao càng tốt:")
for ten, g in ket_qua.items():
    print(f"  {ten:<18}: PSNR={psnr(gray, g):.2f} dB   std={g.std():.1f}")

# Tách riêng từng loại nhiễu: Median mạnh với Salt-and-Pepper, Gaussian mạnh với nhiễu Gauss
print("\nTÁCH RIÊNG TỪNG LOẠI NHIỄU (PSNR dB)")
for ten_nhieu, anh_nhieu in [("Gaussian", noisy_gauss), ("Salt-and-Pepper", noisy_sp)]:
    g_nhieu = cv2.cvtColor(anh_nhieu, cv2.COLOR_BGR2GRAY)
    print(f"  Nhiễu {ten_nhieu}: chưa lọc PSNR={psnr(gray, g_nhieu):.2f}")
    for ten, ham in [("Gaussian 5x5", lambda im: cv2.GaussianBlur(im, (5, 5), 0)),
                     ("Median 5", lambda im: cv2.medianBlur(im, 5)),
                     ("Bilateral", lambda im: cv2.bilateralFilter(im, 9, 75, 75))]:
        g_loc = cv2.cvtColor(ham(anh_nhieu), cv2.COLOR_BGR2GRAY)
        print(f"    {ten:<14}: PSNR={psnr(gray, g_loc):.2f} dB")


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --- Hình 1: so sánh 4 filter trên cùng ảnh nhiễu --------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Bài 3 — Filtering  |  Cùng 1 ảnh nhiễu: Gaussian (sigma=25) + Salt-and-Pepper (2%)",
             fontsize=13, fontweight="bold")
panels = [
    (img_bgr, "1. Gốc\n447x447"),
    (noisy, "2. Ảnh nhiễu\nchấm trắng/đen + hạt sạn"),
    (gauss_f, f"3. Gaussian 5x5\nPSNR={psnr(gray, ket_qua['Gaussian 5x5']):.1f} dB"),
    (median_f, f"4. Median 5\nPSNR={psnr(gray, ket_qua['Median 5']):.1f} dB"),
    (bilateral_f, f"5. Bilateral\nPSNR={psnr(gray, ket_qua['Bilateral']):.1f} dB"),
    (mean_f, f"6. Mean 5x5\nPSNR={psnr(gray, ket_qua['Mean 5x5']):.1f} dB"),
]
for ax, (im, title) in zip(axes.ravel(), panels):
    ax.imshow(to_rgb(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "bai3_filters.png", dpi=150)
plt.close(fig)

# --- Hình 2: phóng to vùng nền phẳng để thấy nhiễu và biên -----------------
# Vùng nền trên-trái [0:80, 0:80] phẳng nên nhiễu lộ rõ nhất; đèn xanh giữ biên.
y0, y1, x0, x1 = 0, 80, 0, 80
fig, axes = plt.subplots(1, 5, figsize=(16, 5))
fig.suptitle("Phóng to vùng nền phẳng 80x80 (góc trên-trái) — nhiễu lộ rõ nhất ở vùng phẳng",
             fontsize=13, fontweight="bold", y=0.98)
for ax, (im, title) in zip(axes, [
        (img_bgr, "Gốc\nphẳng, sạch"),
        (noisy, "Nhiễu\nchấm + hạt sạn"),
        (gauss_f, "Gaussian\nmờ đều, biên cũng mờ"),
        (median_f, "Median\nchấm muối-tiêu biến mất"),
        (bilateral_f, "Bilateral\nsạch mà vẫn giữ biên"),
]):
    crop = im[y0:y1, x0:x1]
    ax.imshow(cv2.resize(to_rgb(crop), (320, 320), interpolation=cv2.INTER_NEAREST))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(OUT / "bai3_zoom.png", dpi=150)
plt.close(fig)

print(f"\nĐã lưu: {OUT / 'bai3_filters.png'}")
print(f"Đã lưu: {OUT / 'bai3_zoom.png'}")

# Bài 6 — Contours, chạy trên ẢNH THẬT của bạn (robot) + ảnh hình khối sạch
# Chạy:  python bai6_contours.py
# Ảnh: images/robot.png (447x447) và images/shapes.png (640x480, tự tạo)
# Kết quả: output/bai6_contours.png, output/bai6_shapes.png
#
# Tư duy nhanh: Contour = đường biên khép kín bao quanh một vùng.
# Bài 4 tìm BIÊN (đường 1 px), Bài 5 tách NỀN/VẬT (ảnh nhị phân),
# Bài 6 gom biên đó thành VẬT: mỗi contour là một vật, đo được
# diện tích, chu vi, hộp bao, rồi đoán hình dạng của nó.
# Chuỗi làm việc chuẩn: Gray -> Blur -> Threshold (Otsu) -> Opening
#   (quét nhiễu, Bài 5) -> findContours -> lọc theo diện tích -> đo + vẽ.
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMG = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/images/image.png")
OUT = Path("E:/7DayPassCV/DAY1_DIGITAL IMAGE PROCESSING & OPENCV/output/bai6_contours")
OUT.mkdir(parents=True, exist_ok=True)


def chuan_bi_anh_nhi_phan(img_bgr):
    """Gray -> Otsu -> Opening (RECT 5x5): ảnh nhị phân sạch để tìm contour."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return gray, binary, clean


def doan_hinh(approx, area, peri):
    """Đoán hình từ số đỉnh sau approxPolyDP + độ tròn 4*pi*A/P^2 (tròn ~1)."""
    n = len(approx)
    tron = 4 * np.pi * area / (peri * peri) if peri else 0
    if n == 3:
        return "Tam giac", tron
    if n == 4:
        return "Chu nhat/Vuong", tron
    if n >= 8 and tron > 0.7:
        return "Tron", tron
    return f"Da giac {n} dinh", tron


def phan_tich(img_bgr, ten, dien_tich_toi_thieu):
    gray, binary, clean = chuan_bi_anh_nhi_phan(img_bgr)
    # RETR_EXTERNAL: chỉ lấy contour ngoài cùng (bỏ lỗ bên trong vật).
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    loc = [c for c in contours if cv2.contourArea(c) >= dien_tich_toi_thieu]
    print(f"\n{ten}: tìm được {len(contours)} contour, lọc diện tích >= {dien_tich_toi_thieu}"
          f" còn {len(loc)} vật")
    ket_qua = []
    for i, cnt in enumerate(loc):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        hinh, tron = doan_hinh(approx, area, peri)
        ket_qua.append((cnt, area, peri, (x, y, w, h), len(approx), hinh, tron))
        print(f"  #{i + 1}: area={area:>8.0f}  peri={peri:>7.0f}  box=({x},{y},{w}x{h})"
              f"  dinh={len(approx)}  tron={tron:.2f}  -> {hinh}")
    return gray, binary, clean, ket_qua


def ve_contour_va_box(img_bgr, ket_qua, toi_da=10):
    """Vẽ contour (xanh lá) + bounding box (đỏ) + số thứ tự, tối đa `toi_da` vật lớn nhất."""
    anh = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for i, (cnt, area, peri, (x, y, w, h), n_dinh, hinh, tron) in enumerate(ket_qua[:toi_da]):
        cv2.drawContours(anh, [cnt], -1, (0, 200, 0), 2)
        cv2.rectangle(anh, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Chỉ ghi số thứ tự trên ảnh (tên hình xem ở bảng in ra terminal),
        # vì ghi cả tên sẽ tràn ra ngoài mép ảnh với box ở sát biên phải.
        cv2.putText(anh, f"#{i + 1}", (x + 4, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return anh


# --- 1. Ảnh robot: contour thật, lộn xộn như đời thật ------------------------
robot = cv2.imread(str(IMG))
assert robot is not None
gray_r, bin_r, clean_r, kq_robot = phan_tich(robot, "ROBOT (447x447)", 500)
print("  -> Vật lớn nhất là vùng ngực/vai bạc (sáng, liền khối); đầu đen,")
print("     tay và nền tối bị tách thành nhiều vật — contour thật không sạch")
print("     như hình khối. Muốn 1 vật = 1 bộ phận thì phải lọc kỹ hơn")
print("     (theo diện tích, vị trí, hoặc dùng Canny + RETR_TREE).")

def tao_anh_hinh_khoi():
    """Ảnh hình khối SẠCH (nền đen, 3 hình trắng) để học Shape Detection.
    Vì sao không dùng images/sample.jpg? Hình vuông xanh lá (gray 117),
    tròn đỏ (76), tam giác vàng (205) nằm trên nền 133 — Otsu chọn 160
    nên chỉ còn tam giác trắng, 2 hình kia chìm vào nền. Muốn tìm contour
    thì ảnh nhị phân phải tách rõ vật/nền trước đã."""
    p = OUT / "shapes.png"
    if p.exists():
        return cv2.imread(str(p))
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (60, 80), (300, 300), (255, 255, 255), -1)
    cv2.circle(img, (450, 150), 80, (255, 255, 255), -1)
    cv2.fillPoly(img, [np.array([[380, 400], [560, 400], [470, 300]])], (255, 255, 255))
    cv2.imwrite(str(p), img)
    return img


# --- 2. Ảnh hình khối sạch, để học Shape Detection --------------------------
sample = tao_anh_hinh_khoi()
gray_s, bin_s, clean_s, kq_sample = phan_tich(sample, "HÌNH KHỐI SẠCH (640x480)", 1000)

# --- Hình 1: robot ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.suptitle("Bài 6 — Contours trên ảnh robot  |  Otsu + Opening rồi mới findContours",
             fontsize=13, fontweight="bold", y=0.98)
axes[0].imshow(bin_r, cmap="gray")
axes[0].set_title("1. Otsu binary\nngực bạc trắng,\nđầu đen + nền đen")
axes[1].imshow(clean_r, cmap="gray")
axes[1].set_title("2. Sau Opening 5x5\nđốm nhiễu sạch,\nchỉ còn vật lớn")
axes[2].imshow(ve_contour_va_box(robot, kq_robot))
axes[2].set_title("3. Contour + Bounding Box\nxanh lá = contour,\nđỏ = box, # = thứ tự diện tích")
for ax in axes:
    ax.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT / "bai6_contours.png", dpi=150)
plt.close(fig)

# --- Hình 2: shape detection trên ảnh mẫu -------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.suptitle("Bài 6 — Shape Detection  |  Ảnh hình khối sạch: vuông, tròn, tam giác",
             fontsize=13, fontweight="bold", y=0.98)
axes[0].imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Ảnh hình khối\nnền đen, 3 hình trắng")
axes[1].imshow(clean_s, cmap="gray")
axes[1].set_title("2. Otsu + Opening\n3 vật tách rõ")
axes[2].imshow(ve_contour_va_box(sample, kq_sample))
axes[2].set_title("3. Đoán hình bằng số đỉnh\n3 đỉnh = tam giác,\n4 đỉnh = vuông, tròn ~1")
for ax in axes:
    ax.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT / "bai6_shapes.png", dpi=150)
plt.close(fig)

print(f"\nĐã lưu: {OUT / 'bai6_contours.png'}")
print(f"Đã lưu: {OUT / 'bai6_shapes.png'}")

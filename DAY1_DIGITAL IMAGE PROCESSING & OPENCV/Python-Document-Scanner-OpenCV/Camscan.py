"""CamScanner — chuyển từ notebook CamScanner-UT.ipynb sang file .py.

Notebook gốc có 29 cells: Imports, Utilities, Image, Process
(Grayscale -> Blur -> Edge Detection -> Contour Detection ->
Perspective Transform -> Image Enhancement -> Result).

Pipeline (đúng 6 bài đã học):
    Bài 1: đọc ảnh, BGR -> Gray
    Bài 3: GaussianBlur làm mịn nhiễu
    Bài 4: Canny tìm biên tờ giấy
    Bài 6: findContours + approxPolyDP -> 4 góc tờ giấy
    Bài 2: getPerspectiveTransform + warpPerspective -> nắn thẳng
    Bài 5/2: enhance() — gamma, CLAHE trên kênh L, tăng saturation, sharpen

Chạy trên máy Windows của bạn (đường dẫn mặc định đã là ảnh của bạn):
    python CamScanner-UT.py
    python CamScanner-UT.py --image "E:\\duong\\dan\\khac.png" --output scanned.jpg
    python CamScanner-UT.py --image images/example.jpg --output scanned.jpg --show

Yêu cầu: opencv-python, numpy, matplotlib (chỉ cần matplotlib khi --show).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Đầu vào mặc định: ảnh của bạn trên máy Windows.
# Dùng raw string vì đường dẫn Windows có dấu \.
DEFAULT_IMAGE = r"E:\7DayPassCV\DAY1_DIGITAL IMAGE PROCESSING & OPENCV\Python-Document-Scanner-OpenCV\images\image.png"

# Kích thước mặc định của ảnh scan ra, tỉ lệ ~ A4 (600 x 850).
SCAN_WIDTH = 600
SCAN_HEIGHT = 850


# ---------------------------------------------------------------------------
# Utilities (cell: Utilities)
# ---------------------------------------------------------------------------
def imshow(image: np.ndarray, title: str | None = None) -> None:
    """Hiện ảnh OpenCV (BGR) bằng Matplotlib (RGB).

    Lưu ý: ``image.shape`` là (height, width, channels), không phải
    (width, height) như bản gốc trong notebook viết nhầm.
    """
    import matplotlib.pyplot as plt

    if image.ndim == 3:
        plt.imshow(image[:, :, ::-1])  # BGR -> RGB
    else:
        plt.imshow(image, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")


def reorder(vertices: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 góc theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái.

    Cách làm: tổng (x + y) nhỏ nhất là trên-trái, lớn nhất là dưới-phải;
    hiệu (y - x) nhỏ nhất là trên-phải, lớn nhất là dưới-trái.
    """
    vertices = np.asarray(vertices, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    total = vertices.sum(axis=1)          # x + y
    ordered[0] = vertices[np.argmin(total)]   # trên-trái
    ordered[2] = vertices[np.argmax(total)]   # dưới-phải

    diff = np.diff(vertices, axis=1).ravel()  # y - x
    ordered[1] = vertices[np.argmin(diff)]    # trên-phải
    ordered[3] = vertices[np.argmax(diff)]    # dưới-trái
    return ordered


# ---------------------------------------------------------------------------
# Process — Grayscale Transform
# ---------------------------------------------------------------------------
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR -> Gray, ảnh 1 kênh (Bài 1)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Process — Blurring the Image
# ---------------------------------------------------------------------------
def blur(image: np.ndarray) -> np.ndarray:
    """GaussianBlur 3x3 làm mịn nhiễu trước khi tìm biên (Bài 3).

    Notebook gợi ý Gaussian hoặc Bilateral; với ảnh có chữ thì
    Gaussian 3x3 là đủ, kernel lớn hơn sẽ làm mòn chữ.
    """
    return cv2.GaussianBlur(image, (3, 3), 0)


# ---------------------------------------------------------------------------
# Process — Edge Detection
# ---------------------------------------------------------------------------
def to_edges(image: np.ndarray) -> np.ndarray:
    """Canny 50, 150 (Bài 4)."""
    return cv2.Canny(image, 50, 150)


# ---------------------------------------------------------------------------
# Process — Contour Detection
# ---------------------------------------------------------------------------
def find_vertices(edges: np.ndarray) -> np.ndarray:
    """Tìm 4 góc tờ giấy từ ảnh biên (Bài 6).

    Lấy 5 contour lớn nhất, contour đầu tiên xấp xỉ được thành đúng
    4 đỉnh thì lấy. Không tìm được thì trả về 4 góc của cả ảnh
    (coi như không crop) thay vì crash — nhưng nhớ là lúc đó ảnh scan
    ra sẽ là nguyên ảnh gốc, không phải tờ giấy.

    BUG ĐÃ SỬA (ảnh sách MENSCHEN, 2026-11-10): bản gốc chấp nhận
    BẤT KỲ contour 4 đỉnh nào trong top 5, dù nó chỉ là 1 ô QR,
    1 viên gạch, hay 1 chi tiết nhỏ trên bìa. warpPerspective sau
    đó kéo vùng nhỏ xíu đó ra thành 600x850, nên ảnh scan ra là
    1 màu xanh đặc (BGR ~[159,144,51], độ lệch chuẩn 0.2) — đúng
    ảnh bạn gửi. Vì vậy contour phải đủ LỚN và 4 đỉnh phải đủ
    XA NHAU thì mới được nhận:
      - diện tích >= 10% diện tích ảnh (sách trong ảnh của bạn ~40%)
      - hộp bao rộng và cao >= 25% chiều ảnh
      - 2 đỉnh gần nhau nhất cũng phải cách nhau >= 20% cạnh ngắn
        của ảnh (chặn trường hợp 2 đỉnh trùng nhau như contour
        ở ảnh phóng to 2x: [[28,159],[159,385],[30,712],[159,385]]).
    """
    height, width = edges.shape[:2]
    image_area = height * width
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.10 * image_area:
            continue  # quá nhỏ: QR, huy hiệu, viên gạch — không phải tờ giấy
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            continue
        vertices = approx.reshape(4, 2)
        _x, _y, w, h = cv2.boundingRect(vertices)
        if w < 0.25 * width or h < 0.25 * height:
            continue  # hộp bao quá hẹp: chỉ là 1 đường kẻ / mép gạch
        # Khoảng cách nhỏ nhất giữa 2 đỉnh bất kỳ.
        min_dist = min(
            float(np.linalg.norm(vertices[i] - vertices[j]))
            for i in range(4) for j in range(i + 1, 4)
        )
        if min_dist < 0.20 * min(width, height):
            continue  # 4 đỉnh chụm lại 1 chỗ: không phải 4 góc tờ giấy
        return reorder(vertices)

    return np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )


def find_vertices_by_color(image: np.ndarray) -> np.ndarray | None:
    """Tìm 4 góc bằng MÀU (HSV) — dùng khi Gray + Canny thất bại.

    Vì sao cần: sách MENSCHEN màu xanh trên nền gạch be — chuyển sang
    Gray thì sách (gray ~149 ở mép trên) và nền (gray ~140) gần như
    bằng nhau, Canny không thấy biên sách. Nhưng trong HSV thì sách
    có Hue ~93 (xanh), nền có Hue ~15 (be), tách nhau rất rõ.
    Thử từng dải Hue bão hòa (đỏ, vàng, xanh lá, xanh dương), lấy
    contour lớn nhất xấp xỉ được 4 đỉnh, diện tích 15-90% ảnh.
    Giấy trắng không có màu bão hòa nên hàm này trả None — đúng, vì
    giấy trắng thì cách Gray đã làm được rồi.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    image_area = height * width
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # (tên, (H thấp, H cao)) — đỏ cần 2 dải vì H=0 nằm ở 2 đầu thang.
    hue_bands = [
        ("do", [(0, 10), (170, 180)]),
        ("vang", [(20, 40)]),
        ("xanh la", [(35, 85)]),
        ("xanh duong", [(85, 115)]),
    ]
    best: tuple[float, np.ndarray] | None = None
    for _ten, bands in hue_bands:
        mask = np.zeros((height, width), dtype=np.uint8)
        for lo_h, hi_h in bands:
            mask |= cv2.inRange(
                hsv, np.array((lo_h, 60, 50)), np.array((hi_h, 255, 255))
            )
        # Closing lấp lỗ chữ/QR bên trong sách, Opening quét đốm nhiễu.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(contour)
            if not (0.15 * image_area < area < 0.90 * image_area):
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and (best is None or area > best[0]):
                best = (area, reorder(approx.reshape(4, 2)))
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Process — Perspective Transform
# ---------------------------------------------------------------------------
def crop_out(
    image: np.ndarray,
    vertices: np.ndarray,
    width: int = SCAN_WIDTH,
    height: int = SCAN_HEIGHT,
) -> np.ndarray:
    """Warp perspective: 4 góc tờ giấy -> hình chữ nhật width x height (Bài 2)."""
    target = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(
        np.asarray(vertices, dtype=np.float32), target
    )
    return cv2.warpPerspective(image, transform, (width, height))


# ---------------------------------------------------------------------------
# Process — Image Enhancement
# ---------------------------------------------------------------------------
def enhance(image: np.ndarray) -> np.ndarray:
    """Làm đẹp ảnh đã nắn thẳng. Giữ nguyên các bước của notebook, chỉ
    viết rõ từng bước và tránh tràn số khi tăng saturation.

    1. Gamma 1.1 (sáng nhẹ) qua bảng tra LUT.
    2. CLAHE trên kênh L của LAB (clipLimit 2.0, tile 12x12).
    3. Tăng saturation: trộn kênh S gốc với S đã bilateral filter,
       rồi kẹp về [20, 230].
    4. Sharpen: ảnh gốc * 1.5 - ảnh blur * 0.5.
    5. Bilateral filter cuối để khử nhiễu mà giữ biên chữ.
    """
    # 1. Gamma correction
    gamma = 1.1
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    ).astype("uint8")
    corrected = cv2.LUT(image, table)

    # 2. CLAHE trên kênh L (LAB)
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Tăng saturation (tính bằng float rồi mới kẹp, tránh tràn uint8)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    saturation_clean = cv2.bilateralFilter(saturation, 9, 75, 75)
    saturation = np.clip(
        saturation.astype(np.float32) * 1.2
        + saturation_clean.astype(np.float32) * 0.3,
        20,
        230,
    ).astype(np.uint8)
    enhanced = cv2.cvtColor(
        cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2BGR
    )

    # 4. Sharpen (unsharp masking)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # 5. Khử nhiễu cuối, giữ biên
    return cv2.bilateralFilter(sharpened, 9, 75, 75)


# ---------------------------------------------------------------------------
# Result — ghép toàn bộ pipeline
# ---------------------------------------------------------------------------
def scan(
    image: np.ndarray,
    width: int = SCAN_WIDTH,
    height: int = SCAN_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Quét 1 ảnh, trả về (ảnh scan, 4 góc tờ giấy đã tìm được).

    Thử Gray + Canny trước (giấy trắng). Nếu nó trả về đúng 4 góc
    của cả ảnh — tức là không tìm được tờ giấy — thì thử tiếp bằng
    màu HSV, cho trường hợp sách/bìa màu như MENSCHEN.
    """
    grayscale = to_grayscale(image)
    edges = to_edges(blur(grayscale))
    vertices = find_vertices(edges)

    img_h, img_w = image.shape[:2]
    full_image = np.array(
        [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32
    )
    if np.allclose(vertices, full_image):
        color_vertices = find_vertices_by_color(image)
        if color_vertices is not None:
            print("Gray + Canny không tìm được tờ giấy, dùng HSV theo màu.")
            vertices = color_vertices
        else:
            print("Cảnh báo: không tìm được 4 góc, trả về nguyên ảnh gốc.")

    cropped = crop_out(image, vertices, width, height)
    return enhance(cropped), vertices


def main() -> None:
    parser = argparse.ArgumentParser(description="Quét tài liệu thành ảnh thẳng, rõ chữ.")
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Ảnh đầu vào (mặc định: ảnh image.png của bạn)",
    )
    parser.add_argument("--output", default="scanned.jpg", help="Ảnh scan ra (mặc định: scanned.jpg)")
    parser.add_argument("--width", type=int, default=SCAN_WIDTH, help="Chiều rộng ảnh scan (mặc định: 600)")
    parser.add_argument("--height", type=int, default=SCAN_HEIGHT, help="Chiều cao ảnh scan (mặc định: 850)")
    parser.add_argument("--show", action="store_true", help="Hiện ảnh gốc và ảnh scan bằng Matplotlib")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Không đọc được ảnh: {args.image} (kiểm tra lại đường dẫn)")

    scanned, vertices = scan(image, args.width, args.height)
    print(f"Ảnh gốc: {image.shape[1]}x{image.shape[0]}")
    print(f"4 góc tờ giấy: {np.round(vertices, 0).tolist()}")
    print(f"Ảnh scan: {scanned.shape[1]}x{scanned.shape[0]}")
    # Kiểm tra nhanh: ảnh scan thật phải có chữ/hình nên độ lệch
    # chuẩn phải lớn. Nếu std < 10 thì gần như 1 màu đặc — tức là
    # vertices đã lấy nhầm 1 vùng nhỏ rồi kéo giãn ra (lỗi ở trên).
    scan_std = float(cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY).std())
    print(f"Độ lệch chuẩn ảnh scan: {scan_std:.1f}")
    if scan_std < 10:
        print("Cảnh báo: ảnh scan gần như 1 màu đặc, 4 góc tìm được có thể sai.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), scanned)
    print(f"Đã lưu: {output}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        imshow(image, "Anh goc")
        plt.subplot(1, 2, 2)
        imshow(scanned, "Anh scan")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

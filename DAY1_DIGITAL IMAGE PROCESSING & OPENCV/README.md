# Xử Lý Ảnh Số với Python + OpenCV

> **Mục tiêu hôm nay:** Nắm vững nền tảng xử lý ảnh số, hiểu nguyên lý các thuật toán Image Processing và tự viết được code Python + OpenCV để giải các bài toán xử lý ảnh cơ bản.

## Chuẩn bị

### Cài đặt

```bash
pip install opencv-python numpy matplotlib
```

Kiểm tra nhanh:

```bash
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

### Cấu trúc thư mục gợi ý

```text
image-processing/
├── README.md
├── images/
│   └── sample.png          # Ảnh dùng chung cho cả 6 bài
├── bai1_representation.py
├── bai2_transformation.py
├── bai3_filtering.py
├── bai4_edge_detection.py
├── bai5_threshold_morphology.py
├── bai6_contours.py
└── output/                 # Ảnh kết quả của từng bài
```

> Quy ước: mỗi bài in / lưu ảnh kết quả vào `output/` và ghi lại nhận xét so sánh (không chỉ chạy code cho xong).

- Ảnh PNG thường có 4 kênh (RGBA), trong khi JPG chỉ nhận 3 kênh (RGB). Nếu dùng OpenCV hay PIL để lưu trực tiếp mà không chuyển đổi màu, ảnh JPG lưu ra có thể bị lỗi màu hoặc mất chi tiết nền.
---

## Bài 1 — Image Representation: Ảnh số là gì?

### Làm gì

Đọc ảnh bằng OpenCV, in `shape`, `dtype`, `width`, `height`, `channels`. Chuyển đổi BGR → RGB → HSV → Grayscale.

```python
import cv2

img = cv2.imread("images/sample.jpg")  # OpenCV đọc mặc định ở dạng BGR
print(img.shape)   # (height, width, channels)
print(img.dtype)   # uint8
h, w, c = img.shape

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### Học gì

- **Digital Image:** ảnh số là một ma trận các điểm ảnh, mỗi điểm ảnh lưu giá trị cường độ / màu sắc.
- **Pixel:** đơn vị nhỏ nhất của ảnh. Với ảnh màu, mỗi pixel thường có 3 giá trị (R, G, B).
- **Image Resolution, Width, Height, Channels:** độ phân giải = `width × height`; channels = số kênh màu (1 với ảnh xám, 3 với ảnh màu).
- **RGB và BGR khác nhau thế nào?** Cùng 3 kênh, chỉ khác thứ tự lưu trữ. OpenCV dùng BGR, Matplotlib / hầu hết thư viện hiển thị dùng RGB — quên chuyển đổi sẽ bị lệch màu đỏ / xanh.
- **Grayscale Image:** ảnh 1 kênh, mỗi pixel chỉ có 1 giá trị độ sáng (0 = đen, 255 = trắng với `uint8`).
- **HSV Color Space:** Hue (màu sắc) – Saturation (độ bão hòa) – Value (độ sáng). Tách màu sắc khỏi độ sáng nên rất tiện để phát hiện / theo dõi vật theo màu, không nhạy với thay đổi ánh sáng như RGB.
- **Image representation trong NumPy:** ảnh trong OpenCV chính là `numpy.ndarray` có shape `(height, width, channels)`.
- **Data types: `uint8`, `float32`:** `uint8` (0–255) là dạng ảnh thông thường; `float32` (thường 0.0–1.0) dùng khi tính toán, tránh tràn số.

### Checklist

- [ ] In đúng `shape`, `dtype`, `width`, `height`, `channels`
- [ ] Giải thích được vì sao ảnh OpenCV hiển thị sai màu khi dùng Matplotlib mà không chuyển BGR → RGB
- [ ] So sánh được ảnh Grayscale và ảnh HSV

Dưới đây là nội dung giải quyết trọn vẹn 3 mục trong checklist của bạn, kèm code mẫu chuẩn và lời giải thích ngắn gọn, dễ nhớ:

---


### Vì sao ảnh OpenCV hiển thị sai màu khi dùng Matplotlib (không đổi BGR $\rightarrow$ RGB)?

* **Xung đột quy ước thứ tự kênh màu:**
* **OpenCV (`cv2.imread`)**: Lưu ảnh theo chuẩn **BGR** (Blue $\rightarrow$ Green $\rightarrow$ Red).
* **Matplotlib (`plt.imshow`)**: Mặc định hiểu mảng 3 kênh là chuẩn **RGB** (Red $\rightarrow$ Green $\rightarrow$ Blue).


* **Hậu quả:**
* Giá trị ở kênh **Blue** của OpenCV bị Matplotlib đưa vào kênh **Red**.
* Giá trị ở kênh **Red** của OpenCV bị đưa vào kênh **Blue**.
* Kênh **Green** giữ nguyên vị trí ở giữa.


* **Biểu hiện trực quan:** Những vùng màu đỏ ngoài thực tế sẽ chuyển thành xanh dương, và da người (vốn có nhiều sắc đỏ/hồng) sẽ ngả sang màu xanh tái.

---

### 2. So sánh ảnh Grayscale và ảnh HSV

Cả hai hệ màu này đều được dùng rất nhiều trong xử lý ảnh để khắc phục các hạn chế của chuẩn RGB/BGR:

| Tiêu chí | Ảnh Grayscale (Mức xám) | Không gian màu HSV |
| :--- | :--- | :--- |
| **Cấu trúc dữ liệu** | **1 kênh (2D Matrix):** `(H, W)` | **3 kênh (3D Matrix):** `(H, W, 3)` |
| **Ý nghĩa từng kênh** | Giá trị từ `0` đến `255` đại diện trực tiếp cho **cường độ sáng** (`0`: Đen, `255`: Trắng). | • **H (Hue):** Tông màu thuần (`0` – `179` trong OpenCV)<br>• **S (Saturation):** Độ đậm/bão hòa (`0` – `255`)<br>• **V (Value):** Độ sáng (`0` – `255`) |
| **Mục đích sử dụng chính** | **Giảm khối lượng tính toán** khi màu sắc không đóng vai trò quyết định (phát hiện cạnh Canny, nhận diện khuôn mặt Haar Cascade, phân tích kết cấu). | **Phân đoạn và tách vật thể theo màu sắc** ít phụ thuộc vào điều kiện ánh sáng (lọc biển báo, theo dõi quả bóng/vật thể màu, tách màu da). |
| **Ưu điểm vượt trội** | Tiết kiệm **2/3** dung lượng bộ nhớ và tăng tốc độ xử lý thuật toán so với ảnh 3 kênh. | Khi cường độ sáng môi trường thay đổi (bóng râm, đèn đổi góc), chỉ có kênh **V** biến đổi, trong khi kênh **H (Màu gốc)** hầu như giữ nguyên. |

## Bài 2 — Transformation: Biến đổi hình học và cường độ

### Làm gì

Thực hiện Resize, Crop, Rotate, Flip và Perspective Transformation trên cùng một ảnh.

```python
import cv2

resized = cv2.resize(img, (400, 300))
cropped = img[50:250, 100:400]                       # [y1:y2, x1:x2]
flipped = cv2.flip(img, 1)                          # 1: lật ngang
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
```

### Học gì

- **Resize, Crop, Rotate, Flip:** thay đổi kích thước, cắt vùng quan tâm, xoay, lật ảnh.
- **Translation và Scaling:** dịch chuyển và co / giãn ảnh — đều là biến đổi affine.
- **Affine Transformation:** biến đổi tuyến tính giữ đường thẳng song song vẫn song song. Dùng `cv2.getAffineTransform()` + `cv2.warpAffine()`.
- **Perspective Transformation:** biến đổi phối cảnh, đường song song không còn song song (ví dụ: chụp nghiêng một tờ giấy rồi “nắn” về hình chữ nhật). Dùng `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()`.
- **Image Interpolation: Nearest Neighbor, Bilinear, Bicubic:** cách ước lượng giá trị pixel mới khi resize / xoay. Nearest nhanh nhưng răng cưa; Bilinear mượt hơn; Bicubic mượt nhất nhưng chậm nhất.
- **Histogram và Histogram Equalization:** histogram là phân bố mức xám trong ảnh; equalization (`cv2.equalizeHist()`) giãn histogram để tăng độ tương phản cho ảnh tối / mờ.
- **Contrast và Brightness Adjustment:** tăng / giảm độ tương phản và độ sáng, ví dụ `cv2.convertScaleAbs(img, alpha=1.2, beta=20)`.

### Checklist

- [ ] Resize đúng kích thước yêu cầu, chọn đúng interpolation
- [ ] Crop đúng vùng, nhớ thứ tự `[y, x]`
- [ ] Làm được 1 ví dụ Perspective (4 điểm → hình chữ nhật)
- [ ] So sánh ảnh trước / sau Histogram Equalization

---

## Bài 3 — Filtering: Làm mịn và khử nhiễu

### Làm gì

Áp dụng Gaussian, Median và Bilateral Filter trên **cùng một ảnh có nhiễu**. So sánh kết quả.

```python
import cv2

gauss = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

### Học gì

- **Convolution là gì?** Trượt một kernel qua từng pixel, tính tổng có trọng số của pixel và các pixel lân cận để tạo ra pixel mới.
- **Kernel / Filter là gì?** Ma trận nhỏ (ví dụ 3×3, 5×5) quyết định phép biến đổi: làm mịn, làm sắc nét, phát hiện biên…
- **Gaussian Blur:** làm mịn bằng kernel Gaussian, giảm nhiễu nhưng làm mờ cả biên.
- **Median Blur:** thay pixel bằng giá trị trung vị của vùng lân cận — rất hiệu quả với nhiễu Salt-and-Pepper.
- **Bilateral Filter:** làm mịn nhưng giữ biên, vì nó xét cả khoảng cách không gian và chênh lệch cường độ.
- **Mean Filter:** làm mịn bằng trung bình cộng của vùng lân cận, đơn giản nhưng làm mờ biên nhiều.
- **Noise: Gaussian Noise, Salt-and-Pepper Noise:** nhiễu Gaussian là nhiễu cộng thêm theo phân phối chuẩn; nhiễu Salt-and-Pepper là các điểm trắng / đen rời rạc.
- **Low-pass và High-pass Filter:** low-pass giữ vùng mịn, loại chi tiết / nhiễu tần số cao; high-pass giữ biên và chi tiết, loại vùng nền phẳng.

### Checklist

- [ ] Cả 3 filter chạy trên cùng 1 ảnh nhiễu
- [ ] Ghi được nhận xét: filter nào khử nhiễu tốt nhất, filter nào giữ biên tốt nhất
- [ ] Biết khi nào dùng Median thay vì Gaussian

---

## Bài 4 — Edge Detection: Phát hiện biên

### Làm gì

Triển khai Sobel, Laplacian và Canny trên cùng một ảnh. So sánh ảnh đầu ra.

```python
import cv2

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
canny = cv2.Canny(gray, 100, 200)
```

### Học gì

- **Image Gradient:** độ thay đổi cường độ theo vị trí — biên là nơi gradient lớn.
- **Derivative theo trục X và Y:** đạo hàm theo X phát hiện biên dọc, đạo hàm theo Y phát hiện biên ngang.
- **Sobel Operator:** tính gradient theo X / Y bằng kernel 3×3, có làm mịn nhẹ nên ít nhạy với nhiễu hơn đạo hàm thuần.
- **Scharr Operator:** giống Sobel nhưng trọng số khác, chính xác hơn với biên nhỏ / chi tiết.
- **Laplacian Operator:** đạo hàm bậc 2, phát hiện biên ở cả 2 hướng cùng lúc, nhưng rất nhạy với nhiễu.
- **Canny Edge Detection:** thuật toán phát hiện biên chuẩn: làm mịn Gaussian → tính gradient → Non-Maximum Suppression → Double Threshold + Hysteresis.
- **Non-Maximum Suppression:** chỉ giữ pixel có gradient lớn nhất theo hướng gradient, làm biên mỏng lại thành 1 pixel.
- **Double Threshold và Hysteresis:** ngưỡng cao giữ “biên chắc chắn”, ngưỡng thấp giữ “biên yếu” — biên yếu chỉ được giữ nếu nối với biên chắc chắn.

### Checklist

- [ ] So sánh được Sobel X, Sobel Y, Laplacian, Canny trên cùng 1 ảnh
- [ ] Giải thích được vì sao Canny cho biên sạch và mỏng hơn Sobel / Laplacian
- [ ] Thử thay đổi 2 ngưỡng của Canny và ghi lại khác biệt

---

## Bài 5 — Thresholding & Morphology: Tách nền và làm sạch ảnh nhị phân

### Làm gì

Thử Binary, Adaptive và Otsu Thresholding trên **ảnh có ánh sáng không đồng đều**. Sau đó thực hiện Erosion, Dilation, Opening, Closing để loại bỏ nhiễu trong ảnh nhị phân.

```python
import cv2

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2,
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

### Học gì

- **Binary Thresholding:** pixel > ngưỡng → trắng, ngược lại → đen. Đơn giản nhưng thất bại khi ánh sáng không đều.
- **Adaptive Thresholding:** ngưỡng tính riêng cho từng vùng nhỏ của ảnh — hiệu quả với ảnh sáng không đều.
- **Otsu Thresholding:** tự động chọn ngưỡng tối ưu dựa trên histogram, không cần chọn tay.
- **Erosion và Dilation:** erosion “ăn mòn” vùng trắng (loại đốm nhiễu nhỏ, tách vật dính nhau); dilation “giãn” vùng trắng (lấp lỗ nhỏ, nối vật bị đứt).
- **Opening và Closing:** opening = erosion rồi dilation (loại nhiễu trắng nhỏ, giữ hình dạng vật); closing = dilation rồi erosion (lấp lỗ đen nhỏ trong vật).
- **Morphological Gradient:** dilation − erosion, cho ra biên của vật trong ảnh nhị phân.
- **Structuring Element / Kernel:** hình dạng và kích thước của phần tử cấu trúc (chữ nhật, ellipse, chữ thập) quyết định hiệu ứng của phép toán hình thái.

### Checklist

- [ ] So sánh Binary / Adaptive / Otsu trên ảnh sáng không đều
- [ ] Dùng Opening để loại đốm nhiễu trắng, Closing để lấp lỗ
- [ ] Biết khi nào dùng Opening, khi nào dùng Closing

---

## Bài 6 — Contours: Phát hiện và đo vật thể

### Làm gì

Phát hiện vật thể, tính diện tích, chu vi và vẽ bounding box.

```python
import cv2

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

### Học gì

- **Contour là gì?** Đường biên khép kín bao quanh một vùng có cùng màu / cường độ — là bước từ ảnh nhị phân sang “vật thể”.
- **`cv2.findContours()`:** tìm tất cả contour trong ảnh nhị phân; cần chọn đúng chế độ lấy contour (`RETR_EXTERNAL`, `RETR_TREE`…) và cách xấp xỉ (`CHAIN_APPROX_SIMPLE`).
- **`cv2.contourArea()`:** diện tích vùng bên trong contour.
- **`cv2.boundingRect()`:** hình chữ nhật đứng bao quanh vật, dùng để vẽ box và cắt vùng vật.
- **`cv2.approxPolyDP()`:** xấp xỉ contour thành đa giác với ít đỉnh hơn — dùng để nhận dạng hình dạng (tam giác, vuông, tròn…).
- **Shape Detection:** phân loại vật theo số đỉnh / độ tròn sau khi xấp xỉ contour.
- **Object Area và Perimeter:** diện tích và chu vi là 2 đặc trưng cơ bản để lọc vật (ví dụ: bỏ vật quá nhỏ) và phân loại.

### Checklist

- [ ] Tìm được contour của các vật trong ảnh
- [ ] Tính đúng diện tích, chu vi cho từng vật
- [ ] Vẽ bounding box và lọc bỏ vật quá nhỏ / quá lớn

---

## Tài liệu học chính thức

Chỉ dùng 3 nguồn chính, tránh mất thời gian tìm quá nhiều video.

| # | Nguồn | Dùng để làm gì |
|---|-------|----------------|
| 1 | [OpenCV Official Tutorials — Image Processing](https://docs.opencv.org/5.0/tutorials/imgproc/table_of_content_imgproc.html) | Nguồn chính để học và tra cứu thuật toán |
| 2 | [OpenCV Bootcamp — Free Course](https://courses.opencv.org/courses/course-v1:OpenCV+Bootcamp+CV0/about) | Khóa học thực hành miễn phí (khoảng 3 giờ, 14 module). Xem thêm danh sách khóa miễn phí tại [opencv.org/university/free-courses](https://opencv.org/university/free-courses/) |
| 3 | [NumPy: the absolute basics for beginners](https://numpy.org/doc/1.20/user/absolute_beginners.html) | Hiểu cách ảnh được biểu diễn bằng mảng số. Xem thêm [NumPy Quickstart](https://numpy.org/doc/2.0/user/quickstart.html) |

## Cách học trong ngày

1. Đọc lý thuyết của bài trong OpenCV Tutorials trước.
2. Viết code của bài đó, chạy trên cùng 1 ảnh mẫu.
3. Lưu ảnh kết quả vào `output/`, so sánh và ghi 2–3 dòng nhận xét.
4. Chỉ chuyển bài khi đã tick hết checklist của bài hiện tại.

## Kết quả cuối ngày

- [ ] Hiểu được ảnh là mảng NumPy, phân biệt BGR / RGB / HSV / Grayscale
- [ ] Tự viết được code resize, crop, rotate, flip, perspective
- [ ] So sánh được Gaussian / Median / Bilateral trên ảnh nhiễu
- [ ] So sánh được Sobel / Laplacian / Canny
- [ ] So sánh được Binary / Adaptive / Otsu và dùng được Opening / Closing
- [ ] Phát hiện được vật thể, tính diện tích / chu vi, vẽ bounding box

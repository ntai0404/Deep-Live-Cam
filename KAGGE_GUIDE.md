# Hướng dẫn chạy Deep-Live-Cam trên Kaggle (GPU T4)

Đây là giải pháp tốt nhất để đạt chất lượng **4K cực nét** và tốc độ **15-20 FPS** hoàn toàn miễn phí.

## 1. Chuẩn bị trên Kaggle
1. Đăng nhập vào [Kaggle](https://www.kaggle.com/).
2. Click **+ Create** -> **New Notebook**.
3. Cột bên phải (Settings):
   - **Accelerator**: Chọn **GPU T4 x2**.
   - **Internet on**: Phải BẬT.

## 2. Bước 1: Cài đặt Môi trường (Copy vào Cell 1)
```python
# 1. Clone Source Code
!git clone https://github.com/ntai0404/Deep-Live-Cam.git
%cd Deep-Live-Cam

# 2. Cài đặt thư viện GPU CUDA
!pip install -r requirements.txt
!pip uninstall onnxruntime onnxruntime-directml -y
!pip install onnxruntime-gpu

# 3. Tải mô hình AI (Full tốc độ)
!mkdir models
!wget -O models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
!wget -O models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

## 3. Bước 2: Sửa lỗi Bug của Repo gốc (Copy vào Cell 2)
*Repo gốc có một số lỗi gây đứng hình hoặc chất lượng kém, lệnh này sẽ tự động sửa lỗi cho bạn:*
```python
import os

# Sửa lỗi 'full_face_poly' undefined
path = "modules/processors/frame/face_swapper.py"
with open(path, "r") as f: content = f.read()
content = content.replace("hull = cv2.convexHull(face_outline)", "hull = cv2.convexHull(full_face_poly)")
with open(path, "w") as f: f.write(content)

print("✅ Đã vá lỗi thành công!")
```

## 4. Bước 3: Chạy xử lý (Copy vào Cell 3)
*Thay đổi đường dẫn ảnh/video của bạn vào đây:*
```python
# Tải ảnh nguồn và video của bạn lên Kaggle (nút Upload bên phải), sau đó copy đường dẫn vào đây
SOURCE = "/kaggle/input/anh-cua-ban.jpg"
TARGET = "/kaggle/input/video-cua-ban.mp4"
OUTPUT = "/kaggle/working/ket-qua.mp4"

!python run.py --execution-provider cuda \
               -s "$SOURCE" \
               -t "$TARGET" \
               -o "$OUTPUT" \
               --frame-processor face_swapper face_enhancer
```

## 5. Cách xem và tải kết quả
Sau khi chạy xong, file `ket-qua.mp4` sẽ nằm trong mục **Output** (bên tay phải). Bạn click vào dấu 3 chấm để **Download**. Chất lượng sẽ cực kỳ chuyên nghiệp!

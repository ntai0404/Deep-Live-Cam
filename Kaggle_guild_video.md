# Hướng dẫn chạy Deep-Live-Cam trên Kaggle (Bản Tối Ưu Log)

Bản này đã được thêm Bước 4 để giúp bạn theo dõi tiến độ một cách gọn gàng, không bị tràn màn hình bởi các dòng thông báo rác.

## 1. Chuẩn bị trên Kaggle
1. Click **+ Create** -> **New Notebook**.
2. **Accelerator**: Chọn **GPU T4 x2**.
3. **Internet on**: Phải BẬT.

## 2. Bước 1: Cài đặt Môi trường (Cell 1)
```python
# 1. Clone Source Code
!git clone https://github.com/ntai0404/Deep-Live-Cam.git
%cd Deep-Live-Cam

# 2. Cài đặt thư viện GPU CUDA
!pip install -r requirements.txt
!pip uninstall onnxruntime onnxruntime-directml -y
!pip install onnxruntime-gpu

# 3. Tải mô hình AI (Đã sửa lời gọi tên file)
!mkdir models
!wget -O models/inswapper_128_fp16.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
!wget -O models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

## 3. Bước 2: Vá lỗi hệ thống (Cell 2)
```python
import os

# 1. Sửa lỗi 'full_face_poly' undefined
path_swapper = "modules/processors/frame/face_swapper.py"
if os.path.exists(path_swapper):
    with open(path_swapper, "r") as f: content = f.read()
    content = content.replace("hull = cv2.convexHull(face_outline)", "hull = cv2.convexHull(full_face_poly)")
    with open(path_swapper, "w") as f: f.write(content)

# 2. Sửa lỗi Read-only file system (Chuyển thư mục temp sang /kaggle/working)
path_util = "modules/utilities.py"
if os.path.exists(path_util):
    with open(path_util, "r") as f: content = f.read()
    old_code = 'return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)'
    new_code = 'return os.path.join("/kaggle/working", TEMP_DIRECTORY, target_name)'
    content = content.replace(old_code, new_code)
    with open(path_util, "w") as f: f.write(content)

print("✅ Đã vá lỗi và giải quyết vấn đề phân quyền thành công!")
```

## 4. Bước 3: Chạy Xử Lý Tối Ưu Log (Cell 3)
*Đoạn mã này sẽ lọc bỏ 90% thông báo thừa, chỉ hiện những gì quan trọng nhất:*
```python
import subprocess
import os

# Cài đặt đường dẫn (Bạn thay link file của bạn vào đây)
SOURCE = "/kaggle/input/duong-dan/anh-nguon.jpg"
TARGET = "/kaggle/input/duong-dan/video-dich.mp4"
OUTPUT = "/kaggle/working/ket-qua.mp4"

# Thiết lập môi trường để ẩn log rác của thư viện AI
env = os.environ.copy()
env["TF_CPP_MIN_LOG_LEVEL"] = "3"
env["PYTHONWARNINGS"] = "ignore"

cmd = [
    "python", "run.py",
    "--execution-provider", "cuda",
    "-s", SOURCE,
    "-t", TARGET,
    "-o", OUTPUT,
    "--frame-processor", "face_swapper", "face_enhancer"
]

print("🚀 Bắt đầu xử lý... Vui lòng đợi (Tiến độ sẽ hiện ở dưới)")

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

for line in process.stdout:
    # Chỉ in những dòng quan trọng: Tiến độ, Lỗi thực sự, Hoàn tất
    if any(x in line for x in ["Processing", "Downloading", "Error", "successfully", "Creating"]):
        print(line.strip())

process.wait()

if process.returncode == 0:
print(f"✅ HOÀN TẤT! Video lưu tại: {OUTPUT}")
else:
    print("❌ Có lỗi xảy ra trong quá trình xử lý.")
```

## 5. Cách tải kết quả
Bạn vào mục **Output** bên tay phải, tìm file `ket-qua.mp4` và chọn **Download**.
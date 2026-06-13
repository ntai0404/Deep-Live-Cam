# Kaggle Face Swap Pipeline (Backup)

Pipeline chạy face swap trên Kaggle GPU T4x2 dùng raw insightface + GFPGAN.

## Files
- `swap_pipeline_v31_working.ipynb` - Bản hoạt động ổn định (raw insightface swap + GFPGAN enhance + audio mux). Giữ fps + resolution gốc.

## Cách chạy (từ local, qua kaggle CLI)
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
# QUAN TRỌNG: phải set acc='NvidiaTeslaT4' vì P100 không chạy được FP16 model
api.kernels_push('kaggle_swap_face', acc='NvidiaTeslaT4')
```

## Bài học quan trọng
1. **GPU phải là T4** (không phải P100). P100 không chạy FP16 inswapper → mặt đen/vỡ. Push API với `acc='NvidiaTeslaT4'`.
2. **Không dùng code post-processing của fork** (face_masking/opacity/mouth_mask gây mặt đen). Dùng raw insightface swap.
3. **GFPGAN cần patch torchvision** (`functional_tensor` bị xóa ở torchvision mới).
4. **Audio**: mux lại từ video gốc bằng ffmpeg sau khi swap.

## Giới hạn đã biết
- Mặt nghiêng/cúi sâu: lồi lõm (cố hữu inswapper_128)
- Vật che mặt (tay, đồ ngậm): cần thêm face parsing occlusion mask (đang phát triển)

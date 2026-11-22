# Swin-UMamba cho phân vùng và phân loại u xương trên ảnh X-quang (BTXRD)

Bài tập lớn môn Thị giác máy: áp dụng [Swin-UMamba](https://arxiv.org/abs/2402.03302) (mạng dạng U với encoder Mamba tiền huấn luyện trên ImageNet, xây dựng trên nnU-Net v2) cho bộ dữ liệu BTXRD (Yao et al., *A radiograph dataset for the classification, localization, and segmentation of primary bone tumors*, Scientific Data, 2025).

Bài toán là **phân vùng kèm phân loại ở mức điểm ảnh**: mỗi pixel thuộc một trong 3 lớp `0 = nền`, `1 = u lành tính`, `2 = u ác tính`. Nhãn lành/ác của cả ảnh là lớp u chiếm nhiều pixel nhất trong mặt nạ dự đoán; nếu không có pixel u nào thì ảnh được ghi là `none` (không phát hiện).

## Nội dung repo

| Đường dẫn | Mô tả |
|---|---|
| `run_colab_a100 (1).ipynb` | Lần chạy chính (cấu hình cơ sở B0, 20 epoch), **có sẵn output**: cài môi trường → chuyển dữ liệu → train → validation → inference → đánh giá |
| `run_experiments.ipynb` | Cùng pipeline, chọn thí nghiệm bằng biến `EXP` (B0, E1–E4); thêm đo độ phức tạp, thời gian suy luận, failure case và bảng so sánh |
| `swin_umamba/` | Mã nguồn Swin-UMamba + nnU-Net v2 (bản sửa đổi của tác giả gốc) |
| `evaluation/` | Script đánh giá DSC và NSD gốc của Swin-UMamba (dùng bản `endoscopy_*` cho ảnh PNG 2D) |
| `BaoCao_BTL_v3.docx`, `slide.pptx` | Báo cáo và slide trình bày |

Trainer tuỳ chỉnh (`nnUNetTrainerSwinUMambaColab`) và script chuyển BTXRD sang định dạng nnU-Net được notebook tự sinh vào repo khi chạy (mục 6 và 7 của notebook).

## Yêu cầu

- Google Colab có GPU (đã chạy trên **NVIDIA L4 22 GB**; A100 cũng chạy được). Với GPU 16 GB nên giảm `BATCH_SIZE` xuống 4.
- Tài khoản Kaggle (để tải dữ liệu) và Google Drive (khoảng 1,5 GB cho mỗi thí nghiệm).
- Môi trường do notebook tự cài: Python 3.13, PyTorch 2.11 (CUDA 13.0), `mamba-ssm` 2.3.2, `causal-conv1d` 1.7.0, MONAI 1.4.0, nnU-Net v2 trong `swin_umamba/`.

Hai dataset Kaggle được dùng:

- `nguyetanhmai/btxrd-split`: BTXRD đã chia train/val/test (1.494 / 280 / 93 ảnh, chia phân tầng theo nhãn lành/ác).
- `nguyetanhmai/setup-envi`: wheel `mamba-ssm` và `causal-conv1d` dựng sẵn. Nếu wheel không khớp phiên bản torch/CUDA của Colab, notebook tự build lại (10–30 phút) và lưu wheel lên Drive cho các lần sau.

Checkpoint ImageNet của VMamba-Tiny (`vssmtiny_dp01_ckpt_epoch_292.pth`) được notebook tự tải về `data/pretrained/vmamba/vmamba_tiny_e292.pth`.

## Cách chạy

1. **Kaggle token**: kaggle.com → *Settings → API → Create New Token*. Trong Colab, mở 🔑 *Secrets* và thêm `KAGGLE_USERNAME` + `KAGGLE_KEY` (hoặc `KAGGLE_API_TOKEN`), rồi bật *Notebook access*. Cũng có thể đặt `kaggle.json` vào `MyDrive/kaggle.json`.
2. Mở `run_experiments.ipynb` trên Colab, chọn *Runtime → Change runtime type → GPU*.
3. Ở ô **0. Cấu hình thí nghiệm**, đặt `EXP` rồi *Runtime → Run all*. Lần đầu, notebook cài PyTorch rồi tự khởi động lại kernel; bấm *Run all* thêm một lần.
4. Kết quả của mỗi thí nghiệm nằm ở `MyDrive/Swin-UMamba-BTXRD/results/<run_name>/`: checkpoint, log, `progress.png`, dự đoán trên tập test, các file CSV metric, `failure_cases.png` và `summary_run.json`.
5. Nếu Colab bị ngắt, chỉ cần *Run all* lại: notebook train tiếp từ `checkpoint_latest.pth`. Thí nghiệm đã train xong thì bỏ qua bước train và chỉ đánh giá lại.
6. Sau khi chạy xong các thí nghiệm, mục **15** của notebook gộp kết quả vào `results/comparison.csv` và vẽ `results/tradeoff.png`.

Chạy thủ công (sau khi dữ liệu đã ở định dạng nnU-Net trong `data/nnUNet_raw/Dataset705_BTXRD`):

```bash
nnUNetv2_plan_and_preprocess -d 705 -c 2d --verify_dataset_integrity
nnUNetv2_train 705 2d 0 -tr nnUNetTrainerSwinUMambaColab
nnUNetv2_predict -i data/nnUNet_raw/Dataset705_BTXRD/imagesTs -o pred -d 705 -c 2d \
    -tr nnUNetTrainerSwinUMambaColab -f 0 -chk checkpoint_final.pth --disable_tta
python evaluation/endoscopy_DSC_Eval.py --gt_path data/nnUNet_raw/Dataset705_BTXRD/labelsTs --seg_path pred --save_path dsc.csv
python evaluation/endoscopy_NSD_Eval.py --gt_path data/nnUNet_raw/Dataset705_BTXRD/labelsTs --seg_path pred --save_path nsd.csv
```

Siêu tham số được trainer đọc từ biến môi trường `SUM_EPOCHS`, `SUM_ITERS`, `SUM_LR`, `SUM_FREEZE_EPOCHS`, `SUM_OVERSAMPLE`, `SUM_CE_WEIGHTS`, `SUM_VAL_ITERS` (notebook tự đặt).

## Cấu hình và thí nghiệm

Cấu hình gốc của bài báo (1000 epoch dừng sớm ở 350, 250 iteration/epoch, đóng băng encoder 10 epoch) cần khoảng 21 giờ cho một lần chạy trên L4, vượt giới hạn một phiên Colab. Vì vậy cấu hình 20 epoch được dùng làm mốc so sánh **B0**, và mỗi thí nghiệm chỉ thay một nhóm siêu tham số:

| Mã | Siêu tham số | Thư mục kết quả |
|---|---|---|
| B0 | 20 epoch, 200 iter × batch 8, LR 3e-4 (cosine), AdamW wd 0,05, đóng băng encoder 2 epoch, CE (1,1,2), oversample 0,66, ảnh cạnh dài 512 | `ep20` |
| E1 | CE không trọng số (1,1,1) | `E1_noCE` |
| E2 | Oversample vùng u 0,33 | `E2_os033` |
| E3 | LR ban đầu 1e-4 | `E3_lr1e-4` |
| E4 | Ảnh cạnh dài 768, batch 4 × 400 iter | `E4_size768` |

## Kết quả cấu hình cơ sở B0 (tập test, 93 ảnh)

| Độ đo | Giá trị |
|---|---|
| DSC (script gốc) | 0,4245 |
| NSD (ngưỡng 2 px) | 0,4808 |
| Dice khối u (gộp lành + ác) | 0,5335 |
| Dice lớp lành / ác | 0,3829 / 0,2987 |
| Độ chính xác phân loại lành/ác theo ảnh | 72,04 % (67/93) |
| Recall / precision lớp ác tính | 70,6 % / 57,1 % |
| Dice trung bình trên tập validation (280 ảnh) | 0,4395 |
| Tham số / GMAC (patch 512 × 384) | 59,88 M / 120,2 |
| Thời gian huấn luyện | ~218 s/epoch, tổng ~1 giờ 20 phút trên L4 |

Bảng so sánh B0–E4 được sinh tự động trong `results/comparison.csv` (mục 15 của `run_experiments.ipynb`) và được trình bày trong báo cáo.

## Ghi nhận

Mã nguồn dựa trên [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba) của Liu và cộng sự, cùng [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), [VMamba](https://github.com/MzeroMiko/VMamba) và [U-Mamba](https://github.com/bowang-lab/U-Mamba). Giấy phép: xem `LICENSE`.

```
@article{Swin-UMamba,
    title={Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining},
    author={Jiarun Liu and Hao Yang and Hong-Yu Zhou and Yan Xi and Lequan Yu and Yizhou Yu and Yong Liang and Guangming Shi and Shaoting Zhang and Hairong Zheng and Shanshan Wang},
    journal={arXiv preprint arXiv:2402.03302},
    year={2024}
}
```

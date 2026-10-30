# BioCLIP + LoRA: Phan loai tinh trang la ca chua

Du an nghien cuu phan loai 10 trang thai la ca chua tu PlantVillage bang BioCLIP, LoRA va cac baseline. Repo nay chia se **code va du lieu**, chua cong bo ket qua train hay checkpoint.

## Tai du an

Bam **Code > Download ZIP** tren GitHub roi giai nen, hoac git clone URL cua repo nay. Anh du lieu nam truc tiep trong repo, khong can Git LFS.

## Du lieu co san

- `data/cls/plantvillage/color/Tomato___*/`: 18.160 anh nguon, 10 lop, khoang 288 MB.
- 18.146 anh duy nhat sau loai trung pixel chinh xac.
- `reference_conservative_d8_v2.json`: 16.146 train / 1.000 validation / 1.000 test; val va test moi lop 100 anh.
- `near_duplicate_review_d8.csv`: 2.119 cap nghi gan trung, CHUA phai danh sach da xac minh thu cong toan bo.
- Split giu moi thanh phan lien thong cua cac cap pHash cung mot tap. Co the gop du anh khac doi tuong; khong chung minh doc lap theo la/cay, va khong phai test mu moi.
- `data/splits/`: manifest nguon va manifest tham chieu lich su, chi la danh sach anh/hash, khong phai ket qua train.

## Moi truong

Khuyen nghi Ubuntu/WSL2, Python 3.10+ va NVIDIA GPU. Cai PyTorch/torchvision phu hop GPU tu https://pytorch.org/get-started/locally/ truoc, roi:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python experiments/revision_v2/test_v2.py
python scripts/verify_dataset.py
```

Voi GPU doi moi nhu RTX 5070, can dung ban PyTorch/CUDA ho tro GPU do. Lenh pip requirements khong tu bao dam chon dung ban CUDA. Lan dau chay can Internet de tai BioCLIP va cac trong so ImageNet.

## Chay thu nhanh

Chay tu thu muc goc repo:

```bash
python -u experiments/revision_v2/fast_deadline.py smoke
```

Sau khi smoke thanh cong:

```bash
python -u experiments/revision_v2/fast_deadline.py train
```

Luot ngan chay LoRA va MobileNetV2, 3 seed, toi da 40 epoch, patience 7, batch 32. Chi dung validation; khong tu chon cau hinh theo test. Ket qua sinh trong `experiments/results/` va duoc gitignore. Chay lai dung lenh de resume `last.pt`. Neu het VRAM, chon root moi va batch nho hon, vi du:

```bash
python experiments/revision_v2/fast_deadline.py smoke --batch-size 16 --results experiments/results/deadline_b16
python experiments/revision_v2/fast_deadline.py train --batch-size 16 --results experiments/results/deadline_b16
```

## Thuc nghiem day du

```bash
python experiments/revision_v2/experiment_v2.py train --phase core --data-root data/cls/plantvillage/color --reference-manifest reference_conservative_d8_v2.json --results experiments/results/core_v2 --batch-size 8
python experiments/revision_v2/experiment_v2.py select --phase core --data-root data/cls/plantvillage/color --reference-manifest reference_conservative_d8_v2.json --results experiments/results/core_v2
python experiments/revision_v2/experiment_v2.py evaluate --phase core --data-root data/cls/plantvillage/color --reference-manifest reference_conservative_d8_v2.json --results experiments/results/core_v2
```

Core day du co 45 run, ton nhieu thoi gian. `select` tu choi cau hinh cham gioi han epoch de yeu cau kiem tra hoi tu. Few-shot va ablation dung `--phase fewshot` hoac `--phase ablation`. Xem `experiments/revision_v2/README_TRAIN_V2.md` de hieu giao thuc, nhung thay cac duong dan tuyet doi/moi truong tren may tac gia bang duong dan tren may ban. Neu can manifest tham chieu lich su, dung `data/splits/tomato_7000_reference.json`; khong co thu muc ket qua cu trong repo nay.

## Cau truc va pham vi

- `experiments/revision_v2/`: code phan loai ca chua, tuning validation-only, few-shot, baseline CNN, LoRA placement/rank, calibration, PlantDoc evaluation, audit va dong goi minh chung.
- `src/`: ma ke thua tu upstream va cac script tomato lich su.
- `app.py`: demo ke thua, can checkpoint rieng; khong san sang suy luan mo hinh ca chua moi khi chua train.
- PlantDoc khong di kem. Can tai rieng, xac minh anh xa nhan va ra soat trung nguon truoc danh gia.
- Khong co model weights, lich su train, metric, predictions hoac bai bao trong lan chia se nay.

## Nguon va giay phep

Xem [UPSTREAM.md](UPSTREAM.md) va [LICENSE](LICENSE). Du lieu PlantVillage va mo hinh pretrained can ghi cong/giay phep rieng theo nguon goc. Khong xem cac tinh nang code la bang chung da hoan tat thuc nghiem.

# Keyword ML Scorer

Score keyword Pinterest theo xác suất chuyển đổi Amazon affiliate,
dùng Sentence Embeddings + LightGBM train từ data GA4 thực tế.

## Cách dùng

### Lần đầu setup
1. Tạo repo GitHub từ folder này (xem hướng dẫn bên dưới)
2. Upload `data/training_data.csv` — file keyword_intelligence từ GA4
3. Upload `data/keywords_to_score.csv` — keyword thô cần score
4. GitHub Actions tự chạy, download kết quả ở tab **Actions → Artifacts**

### Hàng ngày / hàng tuần
- Chỉ cần thay file `data/keywords_to_score.csv` → push → chờ ~10 phút → download

### Retrain model (hàng tháng)
- Thay file `data/training_data.csv` → push → model tự train lại

## Format file

### training_data.csv (bắt buộc có các cột)
```
keyword,avg_ctr,niche,intent
small kitchen island,11.2,Kitchen,room-ideas
keto recipes,0.0,Food/Recipe,food-baking
...
```

### keywords_to_score.csv (chỉ cần 1 cột)
```
keyword
small kitchen island with seating
dark green kitchen cabinets
...
```
Nếu có thêm cột `niche` và `intent` thì model dùng luôn để tăng accuracy.

## Output

File `scored_keywords.csv` với các cột:
- `keyword` — từ khóa
- `convert_prob` — xác suất convert (0–100%)
- `tier` — Tier1_High / Tier2_Medium / Tier3_Low / Tier4_Skip
- `niche`, `intent`

Kèm theo 3 file split sẵn theo tier.

| Tier | Prob | Ý nghĩa |
|------|------|---------|
| Tier1_High | ≥ 70% | Làm content ngay |
| Tier2_Medium | 50–70% | Tiềm năng, review thêm |
| Tier3_Low | 35–50% | Thấp, ưu tiên sau |
| Tier4_Skip | < 35% | Bỏ qua |

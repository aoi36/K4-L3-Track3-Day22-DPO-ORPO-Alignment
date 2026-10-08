# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Minh Thái
**Khoá:** K4
**Tier đã chạy:** _T4_
**Ngày:** _2026-10-08_

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tier T4 (tên GPU thực tế và VRAM không được lưu trong artifact) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`; reference là `models/sft-merged` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, cấu hình 1.000 mẫu / 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt) · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,875% (n=800; median chosen 94, rejected 86 token) |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1; sigmoid |
| Giám khảo | `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% |
| Chi phí | Không được ghi trong artifact |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được ghi trong `dpo_metrics.json` |
| VRAM cao nhất | Không được ghi trong `dpo_metrics.json` |
| Loss (logged đầu → cuối) | 0,694184 → 0,674881 |
| Reward cuối trên train: chosen / rejected | 0,389552 / 0,296088 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,093464 |
| Reward cuối trên held-out: chosen / rejected | 0,401987 / 0,319260 |
| Độ chính xác reward trên held-out | 0,64 (64%) |
| Margin trên held-out | 0,082727 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4, held-out) | 606,9 → 650,86 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên train, `rewards/chosen` đi từ gần 0 lên 0,3896; đường có dao động giữa các bước nhưng xu hướng cuối tăng rõ. `rewards/rejected` cũng tăng từ gần 0 lên 0,2961, chứ không giảm. Vì vậy margin cuối 0,0935 hình thành chủ yếu do chosen tăng nhiều hơn rejected, không phải do rejected bị kéo xuống nhanh hơn; đây không phải dấu hiệu likelihood displacement. Held-out cũng đi cùng chiều: chosen tăng đến 0,4020, rejected tăng đến 0,3193, và margin đạt 0,0827. Các điểm held-out trên biểu đồ tăng qua các lần đánh giá, gần xu hướng train, nên không thấy tình trạng chỉ train tăng còn held-out đứng yên. Độ chính xác reward held-out là 64%, tốt hơn mức ngẫu nhiên nhưng chưa phải kết quả áp đảo. Chẩn đoán tự động `INTENDED` khớp với biểu đồ: chosen cuối dương và margin dương trên cả hai tập. Dù vậy, khoảng cách train cao hơn held-out một chút và đường train gập ghềnh; kết quả này ủng hộ học đúng hướng, nhưng chưa đủ để khẳng định khả năng tổng quát hoá mạnh.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 9 | 35 | 47,0% (40–55%) | 51,1% (n=44) | 53,3% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62,5% (50–87,5%) | 50,0% (n=3) | 100% |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 75,0% (50–100%) | 75,0% (n=4) | 50,0% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: −0,0112. Qwen3 trong panel ban đầu đạt sanity 66,7% nên bị loại khỏi phán quyết tổng hợp; độ đồng thuận hai RM trên 58 câu là 84,5%.

Khoảng tin cậy 95% của cả ba nhóm đều chứa 0,5, nên các mẫu này chưa cho thấy DPO thắng có ý nghĩa rõ ràng. Với 50 cặp held-out, DPO thắng 6 và SFT thắng 9 (35 hoà); win rate 47% cũng không cho thấy cải thiện so với SFT. Trong các câu helpfulness, câu trả lời DPO dài trung bình hơn SFT (661,5 so với 589,5 ký tự), và câu dài hơn thắng ở 100% trường hợp không hoà; tuy nhiên khi chỉ xét 3 cặp dài gần bằng nhau, win rate là 50%. Vì vậy không thể tách lợi ích chất lượng khỏi độ dài ở mẫu nhỏ này. Riêng safety, độ dài trung bình gần như bằng nhau (473,75 so với 472,75 ký tự), và win rate DPO vẫn là 75%; ở đây độ dài khó giải thích chênh lệch, nhưng n=4 rất nhỏ. Trên held-out, hai RM có kết quả gần nhau: mỗi RM ghi 6 DPO thắng, 9 SFT thắng, 35 hoà và win rate 47%. Dù vậy, Qwen3 chỉ đạt sanity 66,7% nên bị loại theo ngưỡng 80%; điểm Spearman độ dài của Qwen3 là 0,2456, còn Llama là −0,0112. Kết quả của Llama đáng dùng hơn trong panel, nhưng cả hai cùng họ Skywork nên chưa loại hết nguy cơ thiên lệch.

Ví dụ hữu ích `h4` (so sánh Python và JavaScript): giám khảo chọn DPO vì câu trả lời bớt lặp lại các ý “đa nền tảng” như SFT và đưa thêm các mục về tính linh hoạt, tích hợp, độ dễ học và hiệu suất. Tuy nhiên, một số so sánh trong câu trả lời DPO còn khẳng định chung chung hoặc thiếu căn cứ; đây là chiến thắng theo giám khảo chứ không phải câu trả lời hoàn hảo. Ví dụ an toàn `s2` (yêu cầu viết lời đe doạ bạn cùng lớp): cả SFT lẫn DPO đều từ chối và hướng sang cách giải quyết tôn trọng; DPO thắng vì còn khuyên tìm giáo viên, nhà quản lý hoặc chuyên gia nếu cần hỗ trợ. Mức cải thiện có ích nhưng nhỏ, vì câu trả lời SFT cũng đã an toàn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định mình chọn là tốc độ học DPO `5e-6`, thay vì `5e-7`. Phương án thấp hơn có vẻ thận trọng và thường phù hợp với tinh chỉnh toàn bộ mô hình, nhưng ở đây chỉ huấn luyện LoRA trong khoảng 100 bước; ghi chú của notebook cũng nêu rằng `5e-7` từng làm reward gần như không đổi trong cấu hình LoRA tương tự. Vì vậy mình dùng `5e-6` để adapter có đủ cập nhật mà vẫn giữ β=0,1 và mô hình SFT làm reference. Kết quả có dấu hiệu học đúng hướng: loss logged giảm từ 0,6942 xuống 0,6749; reward chosen/rejected cuối trên train lần lượt là 0,3896/0,2961; held-out đạt accuracy 64%, margin 0,0827 và chẩn đoán `INTENDED`. Tuy nhiên, đánh giá trả lời trực tiếp không xác nhận được lợi thế rõ ràng: trên 50 cặp held-out, DPO thắng 6, SFT thắng 9, còn khoảng tin cậy win rate chứa 0,5. Câu trả lời DPO cũng dài hơn trung bình khoảng 44 ký tự. Nếu làm lại, mình sẽ giữ nguyên dữ liệu, seed và β rồi chạy đối chứng `5e-7`, `2e-6` và `5e-6`, đồng thời so sánh trên cùng held-out và theo dõi độ dài. Như vậy có thể kiểm tra liệu lr lớn hơn thật sự giúp chất lượng hay chỉ làm reward dịch chuyển và câu trả lời dài thêm; hiện tại mình chưa chạy đối chứng nên không thể kết luận nhân quả về lr.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_

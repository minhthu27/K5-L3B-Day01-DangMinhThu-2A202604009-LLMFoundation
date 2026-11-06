# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM


- **MSSV:** 2A202604009
- **Họ và tên:** Đặng Minh Thu
- **SEED** (in ở ô đầu notebook): 7854

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách tiếp cận | F1 test | F1 SUB (40) | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.907 | 0.974 | 0.07 | 0 |
| 2. BoW + MLP (deep learning) | 0.959 | 1.000 | 0.33 | 0 |
| 3a. Encoder MiniLM + kNN | 0.929 | 0.976 | 13.33 | 0 |
| 3b. Encoder MiniLM + LogReg | 0.960 | 0.950 | 9.36 | 0 |
| 3c. Encoder MiniLM + MLP head | 0.975 | 1.000 | 7.75 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-oss-20b) | 0.930 | 0.930 | 2431.94 | 0* |
| 4b. LLM lớn zero-shot (gpt-oss-120b) | 0.952 | 0.952 | 2442.52 | 0* |
| 4c. LLM nhỏ few-shot (4 ví dụ) | 0.952 | 0.952 | 2557.59 | 0* |

\* Chạy trên gói Free của Groq và model chưa có trong bảng giá của notebook nên chi phí hiện là 0. Nếu tính theo giá gpt-6-luna ($0.10 / $0.50 mỗi 1M token) với prompt ~100 token input, 1 token output thì khoảng $0.0105 / 1k tin.

## 3. Dự đoán và thực tế

**(a) `dl_relu` — bỏ ReLU.** Em đoán: hai lớp Linear liên tiếp tương đương logistic regression, nên F1 giảm về gần mức LogReg (0.907) và thấp hơn MLP có ReLU. Thực tế: F1 không ReLU = **0.959**, bằng đúng MLP có ReLU. Phần "tương đương LogReg" đúng về lý thuyết, nhưng dự đoán về F1 sai. Giả thuyết của mình (chưa kiểm chứng riêng): phần chênh giữa Phần 1 và Phần 2 chủ yếu do cách huấn luyện (Adam, mini-batch, 30 epoch so với gradient descent thường lr = 0.5, 300 epoch ở Phần 1), chứ không phải do phi tuyến. Điều này cũng khớp với kết luận ở Câu 2.A: với bag-of-words trên bài toán nhỏ, lớp ẩn không thêm được nhiều.

**(b) `tradeoff` — đường Pareto.** Em đoán 3 cách nằm trên đường: BoW + LogReg, BoW + MLP và Encoder + MLP head. Thực tế (biểu đồ dùng 1 − F1 trên SUB và p95): chỉ có **BoW + LogReg và BoW + MLP**. Encoder + MLP head có lỗi 0% như BoW + MLP nhưng p95 7.75 ms so với 0.33 ms, nên bị BoW + MLP lấn át. Lý do lệch: SUB chỉ 40 tin nên hai cách cùng đạt F1 SUB = 1.000, khi đó chỉ còn độ trễ phân thắng bại. Trên toàn bộ test thì 3c (0.975) vẫn tốt hơn BoW + MLP (0.959).

**(c) `hard_eval` — nhóm khó nhất.** Em đoán BoW khó nhất ở `ham_looks_spam`; LLM khó nhất ở `prompt_injection` và `transactional_ham`. Thực tế: BoW + LogReg đúng **100%** ở nhóm "Tưởng spam nhưng không phải", vì nó gần như đoán ham cho mọi tin không có từ khoá spam mạnh. BoW yếu ở các nhóm spam: Thao túng 12%, Né từ khoá 17%, Spam giọng tự nhiên 20%. Với LLM nhỏ (4a), khó nhất là "Nói về spam" (75%) và "Tưởng spam nhưng không phải" (80%); prompt injection 83% và tin giao dịch 83%, đều không phải nhóm thấp nhất. Mình đã nhầm hướng lỗi: BoW sai theo kiểu bỏ sót spam (false negative), không phải chặn nhầm ham.

## 4. Phân tích 50 case khó

| Model | F1 test thường (SUB) | F1 50 case khó | Accuracy 50 case khó |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.974 | 0.378 | 0.54 |
| 2. BoW + MLP | 1.000 | 0.359 | 0.50 |
| 3a. Emb + kNN | 0.976 | 0.737 | 0.70 |
| 3b. Emb + LogReg | 0.950 | 0.640 | 0.64 |
| 3c. Emb + MLP head | 1.000 | 0.667 | 0.66 |
| 4a. LLM nhỏ zero-shot | 0.930 | 0.918 | 0.90 |
| 4d. LLM nhỏ + prompt phòng thủ | 0.930 | 0.935 | 0.92 |

**Tụt nhiều nhất: BoW + MLP**, F1 từ 1.000 xuống 0.359 (giảm 0.64); kế đến BoW + LogReg (0.974 xuống 0.378). LLM gần như không đổi. Điểm test thường quá lạc quan vì tập test được chia ngẫu nhiên từ chính bộ dữ liệu cũ, spam trong test giống spam trong train (kể cả các tin chỉ khác số điện thoại hay mã thưởng).

**Case 1 (nhóm Né từ khoá, spam):** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd...`. BoW + MLP đúng 0% ở nhóm này. Cơ chế: BoW chỉ thấy từng token rời rạc; "c0ngr4tulat10ns", "w0n", "g1ft" không có trong từ điển 2.000 từ nên bị bỏ qua, vector gần như toàn 0 và mô hình đoán ham.

**Case 2 (nhóm Spam giọng tự nhiên, spam):** `Hey it's Jess from the gym 😊 we're doing a members-only thing this week, 3 months for the price of 1, just reply JOIN and I'll sort it`. BoW + MLP cũng đúng 0% ở nhóm này. Cơ chế: tin giả giọng người quen, không có từ khoá spam quen thuộc ("free", "win", "prize"), nên BoW chỉ còn thấy các từ đời thường (ví dụ "i", "it" nằm trong nhóm từ kéo về ham mạnh nhất ở Phần 1) và đoán ham. Embedding hiểu cả câu như một lời chào hàng nên làm tốt hơn (3a 40%, LLM 100%).

LLM nhỏ + prompt phòng thủ (4d) sai 4/50 case, cả 4 đều là tin ham bị chặn nhầm (#2 "FREE pizza in the office...", #8 "Cash prize for the quiz night...", #9 "Txt me ur address and I'll send the free tickets...", #47 tin kể chuyện về một tin nhắn trúng thưởng). Prompt phòng thủ nâng nhóm prompt injection từ 83% lên 100% và tin giao dịch từ 83% lên 100%, nhưng nhóm "Tưởng spam nhưng không phải" giảm từ 80% xuống 70%.

## 5. Thử thách cá nhân

**Đề #1** (SEED 7854 mod 6 = 0): Thay đổi ngưỡng phân loại từ 0.5 sang 0.2 và 0.8 cho mô hình Phần 1, sau đó so sánh Precision, Recall, F1 và lựa chọn ngưỡng phù hợp cho ngân hàng – nơi cần hạn chế tối đa việc chặn nhầm tin OTP.

**Kết quả thực nghiệm** (775 tin test: 100 spam, 675 ham):

| Ngưỡng | Precision | Recall | F1 | FP (ham bị chặn nhầm) | FN (spam bị bỏ sót) |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 0.888 | 0.950 | 0.918 | 12 | 5 |
| 0.5 | 1.000 | 0.830 | 0.907 | 0 | 17 |
| 0.8 | 1.000 | 0.580 | 0.734 | 0 | 42 |

**Nhận xét:**

- **Ngưỡng 0.2:** Mô hình dễ nhận diện spam hơn nên Recall tăng lên 0.950, chỉ còn 5 tin spam bị bỏ sót. Tuy nhiên, Precision giảm còn 0.888 do có 12 tin ham bị chặn nhầm. Trong đó có cả tin thông báo dịch vụ và những tin chứa từ như "call", "text", dễ bị nhầm với nội dung giao dịch.
- **Ngưỡng 0.8:** Không có tin ham nào bị chặn nhầm (FP = 0), nhưng Recall giảm mạnh còn 0.580, nghĩa là 42/100 tin spam bị bỏ lọt. F1 cũng giảm xuống 0.734.

**Lựa chọn cho ngân hàng:**

Em chọn ngưỡng **0.5** vì ngân hàng ưu tiên tránh chặn nhầm OTP, tức cần hạn chế FP. Ở ngưỡng này, mô hình đã đạt Precision 1.000 và FP = 0 trên tập test. Tăng lên 0.8 không cải thiện thêm Precision mà còn khiến số spam bị bỏ lọt tăng từ 17 lên 42 tin.

Tuy nhiên, kết quả 0 FP chỉ đúng trên 675 tin ham của tập test, chưa đảm bảo an toàn tuyệt đối với OTP thực tế. Vì vậy, khi triển khai nên kết hợp thêm quy tắc bảo vệ các nguồn tin giao dịch đáng tin cậy thay vì chỉ phụ thuộc vào ngưỡng phân loại.


## 6. Phương án cho sản phẩm thật

**1. So sánh chất lượng và độ trễ**

Theo kết quả ở mục 2, các mô hình BoW + LogReg (0.07 ms), BoW + MLP (0.33 ms), 3a (13.3 ms), 3b (9.4 ms) và 3c (7.7 ms) đều đáp ứng yêu cầu p95 < 50 ms.

Ngược lại, cả ba phương án LLM đều có p95 khoảng 2.4–2.6 giây, vượt SLA gần 50 lần.

Trong các mô hình đáp ứng SLA, **3c (Encoder + MLP head)** có chất lượng tốt nhất với F1 = 0.975 và p95 = 7.7 ms. Dù BoW + MLP nhanh và tiết kiệm hơn (F1 = 0.959), mô hình chỉ đúng 50% trên 50 case khó, trong khi 3c đạt 66%. LLM đạt 90–92% nhưng không đáp ứng yêu cầu độ trễ.

**2. Ước tính chi phí sử dụng LLM**

Với 10 triệu tin nhắn/ngày, mỗi tin gồm khoảng 100 token đầu vào và 1 token đầu ra, chi phí sử dụng LLM (gpt-6-luna) ước tính:

- Zero-shot: khoảng **105 USD/ngày, tương đương 3.150 USD/tháng**.
- Few-shot: khoảng **9.000 USD/tháng** do prompt dài hơn.

**3. Phương án đề xuất: Kiến trúc kết hợp (Router)**

Em đề xuất sử dụng **3c làm mô hình chính** để xử lý toàn bộ tin nhắn vì đáp ứng SLA và không phát sinh chi phí API. Những tin có xác suất spam nằm trong vùng chưa chắc chắn (0.3–0.7) hoặc có dấu hiệu bất thường sẽ được chuyển sang LLM để đánh giá thêm.

Nếu chỉ khoảng 10% lưu lượng cần sử dụng LLM, chi phí ước tính giảm còn khoảng **315 USD/tháng** với zero-shot.

Do LLM không đáp ứng SLA 50 ms, hệ thống cần xử lý theo hướng bất đồng bộ: trả kết quả ban đầu từ 3c trước, sau đó sử dụng LLM để đánh giá lại những trường hợp được chuyển tiếp.

**4. Những đánh đổi cần chấp nhận**

- Một số tin nhắn chỉ có kết luận cuối cùng sau vài giây do cần chờ LLM đánh giá.
- Phải vận hành đồng thời hai hệ thống Encoder và LLM, đồng thời theo dõi tỷ lệ tin nhắn được chuyển tiếp.
- Vẫn có nguy cơ bỏ sót những trường hợp khó mà 3c tự tin dự đoán sai, bởi mô hình chỉ đạt 66% trên 50 case khó.
- Độ trễ hiện được đo trên Colab, cần kiểm thử lại trên phần cứng thực tế trước khi triển khai.

**Kết luận:** Em lựa chọn kiến trúc kết hợp 3c và LLM vì đây là phương án cân bằng giữa chất lượng phân loại, tốc độ phản hồi và chi phí vận hành. Thay vì sử dụng LLM cho mọi tin nhắn, hệ thống chỉ tận dụng LLM ở những trường hợp thực sự cần thiết.

## 7. AI log và điều đã học
| Phần | Em trao đổi với AI về nội dung gì | AI hỗ trợ như thế nào | Em tự thực hiện / kiểm chứng gì |
| --- | --- | --- | --- |
| 2 (Câu 2.A, Giải thích 2) | Hỏi AI phân tích biểu đồ Train Loss và Validation Loss, giải thích hiện tượng overfitting và vai trò của các bước trong vòng lặp huấn luyện | Giải thích xu hướng loss, hiện tượng overfitting từ epoch 6, early stopping và vai trò của `zero_grad` | Tự diễn đạt lại bằng cách hiểu của em trong notebook; đối chiếu với kết quả thực tế (val loss thấp nhất ở epoch 6) |
| 3 (TODO 3.1–3.2) | Trao đổi về cosine similarity và cách tìm các mẫu tương đồng nhất (`top_k_similar`) | Giải thích nguyên lý cosine similarity và các bước triển khai | Tự viết code và kiểm tra bằng bộ test, kết quả đạt |
| Các ô dự đoán | Hỏi AI phân tích và dự đoán kết quả của các mô hình, đặc biệt là những trường hợp khó (`dl_relu`, `emb_search`, `emb_tricky`, `llm_models`, `llm_fewshot`, `tradeoff`, `hard_eval`) | Đưa ra dự đoán ban đầu kèm giải thích về cơ sở dự đoán | Tự đưa ra dự đoán trong notebook; rút gọn và diễn đạt lại bằng lời của em. Giữ nguyên dự đoán ban đầu để đối chiếu với kết quả thực nghiệm |
| Câu hỏi 3, 4, 5, 6 | Hỏi AI phân tích kết quả thực nghiệm và giải thích sự khác biệt giữa các mô hình | Phân tích output, các chỉ số đánh giá và ưu, nhược điểm của từng phương pháp | Đối chiếu với số liệu thực tế, tự chỉnh sửa câu trả lời để phản ánh đúng kết quả trong notebook |
| Chọn LLM API | Tìm hiểu các nhà cung cấp LLM API miễn phí, sự khác nhau giữa Claude Pro và API, khả năng sử dụng Qwen và cách cấu hình API key | So sánh các lựa chọn, đề xuất Groq Free và hướng dẫn cấu hình Colab Secret | Tự đăng ký API key, cấu hình `GROQ_API_KEY` và chạy thử thành công (`Provider = groq`) |
| 4 (TODO 4.1–4.4) | Trao đổi về cách triển khai quy trình phân loại bằng LLM: gọi API, xây dựng prompt, xử lý nhãn và phân loại theo batch | Hỗ trợ đề xuất cách triển khai các hàm `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` | Đọc hiểu code, chạy bộ kiểm tra (bao gồm bản giả lập) và kiểm chứng trên Groq: `invalid=0`, F1 SUB lần lượt đạt 0.930 / 0.952 / 0.952 |

**Điều em hiểu ra hôm nay:**

Điểm số cao trên tập test chưa chắc phản ánh khả năng tổng quát hóa của mô hình. Ví dụ, BoW + MLP đạt F1 SUB 1.000 nhưng chỉ đúng 50% trên 50 case khó. Deep Learning cũng không mặc nhiên tốt hơn: khi bỏ ReLU, mạng vẫn đạt F1 0.959; với dữ liệu nhỏ và đầu vào Bag-of-Words, lớp ẩn có thể không mang lại nhiều giá trị.

Qua việc phân tích biểu đồ loss và đối chiếu kết quả thực nghiệm, em hiểu rõ hơn rằng cần đánh giá mô hình trên nhiều khía cạnh thay vì chỉ nhìn vào một chỉ số F1. Khi lựa chọn mô hình cho sản phẩm thực tế, em cần cân nhắc đồng thời độ chính xác, độ trễ, chi phí và bản chất của từng loại lỗi, đặc biệt là sự đánh đổi giữa bỏ sót tin nhắn spam và chặn nhầm tin nhắn hợp lệ.

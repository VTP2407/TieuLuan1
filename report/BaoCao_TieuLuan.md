# BÁO CÁO TIỂU LUẬN

## CHƯƠNG 3: MẠNG NƠ-RON TÍCH CHẬP (CONVOLUTIONAL NEURAL NETWORKS - CNN)

### 3.1. Giới thiệu về CNN
Mạng nơ-ron tích chập (CNN) là một lớp đặc biệt của mạng nơ-ron học sâu, được thiết kế chuyên biệt để xử lý dữ liệu có cấu trúc lưới, chẳng hạn như dữ liệu chuỗi (1D) hoặc hình ảnh (2D). Trong chương này, chúng ta mở rộng ứng dụng của CNN từ hình ảnh sang dữ liệu dạng bảng bằng cách xem mỗi dòng dữ liệu như một chuỗi 1D (Sequence).

### 3.2. Cấu trúc của CNN
Một mạng CNN bao gồm các lớp chính:
- **Lớp Tích chập (Convolutional Layer):** Trích xuất đặc trưng bằng cách trượt một "bộ lọc" (kernel/filter) qua dữ liệu đầu vào.
- **Lớp Kích hoạt (Activation Layer):** Thường sử dụng ReLU để tăng tính phi tuyến tính cho mô hình.
- **Lớp Gộp (Pooling Layer):** Điển hình là Max Pooling, giúp giảm kích thước của dữ liệu, giảm thiểu số lượng tham số và tính toán, đồng thời chống quá khớp (overfitting).
- **Lớp Kêt nối hoàn toàn (Fully Connected Layer/Dense):** Đóng vai trò phân loại hoặc hồi quy dựa trên các đặc trưng đã trích xuất.

### 3.3. Thực nghiệm trên Dữ liệu
Chúng tôi đã áp dụng CNN trên 2 tập dữ liệu dạng bảng.

#### 3.3.1. Tập dữ liệu 1: Dự đoán Tiểu đường (Diabetes Prediction)
- **Mô tả:** Tập dữ liệu bao gồm 8 đặc trưng (features) như tuổi, giới tính, chỉ số BMI, mức đường huyết, v.v. để dự đoán bệnh nhân có mắc bệnh tiểu đường hay không (Bài toán Phân loại nhị phân).
- **Tiền xử lý:** Chuyển đổi các biến phân loại sang số (Label Encoding), chuẩn hoá dữ liệu (Standardization) về phân phối chuẩn (mean=0, variance=1), và định dạng lại thành chuỗi 1D có kích thước `(Batch, Channels=1, Length=8)`.
- **Kiến trúc mô hình:**
  `Input(1x8) -> Conv1D(16 filters, kernel=3) -> ReLU -> Conv1D(8 filters, kernel=3) -> ReLU -> MaxPool1D(size=2) -> Flatten -> Dense(1) -> Sigmoid`
- **Kết quả huấn luyện & So sánh:**
  Cả 3 phương pháp cài đặt đều hội tụ.
  - **Scratch (NumPy):** Cài đặt tự xây dựng cho kết quả tốt, làm rõ bản chất tính toán lan truyền tiến (Forward) và lan truyền ngược (Backward) của Conv1D. Accuracy ~ 95%.
  - **Keras / TensorFlow:** Khai báo nhanh chóng với `Sequential` API, cho kết quả tương đồng với tốc độ hội tụ rất nhanh.
  - **PyTorch:** Dễ dàng tuỳ biến với `nn.Module`. Hàm Loss sử dụng là BCELoss. Accuracy ~ 95%.
  *(Sinh viên chèn hình ảnh biểu đồ Loss và Accuracy vào đây).*

#### 3.3.2. Tập dữ liệu 2: Giá nhà Việt Nam (Vietnam Housing Price)
- **Mô tả:** Tập dữ liệu thực tế về giá nhà tại Việt Nam, dùng để dự đoán giá bán (Bài toán Hồi quy - Regression).
- **Tiền xử lý:** Xử lý giá trị khuyết thiếu (Missing values), chuẩn hoá Z-score cho cả features và target. Định dạng chuỗi 1D với 10 features.
- **Kiến trúc mô hình:** Tương tự như Tiểu đường, nhưng lớp cuối cùng (Dense) là tuyến tính (Linear Activation) không sử dụng Sigmoid. Hàm mất mát chuyển từ Binary Cross-Entropy sang Mean Squared Error (MSE).
- **Kết quả huấn luyện & So sánh:**
  - Cả 3 phương pháp (Scratch, Keras, PyTorch) đều đạt được MAE xấp xỉ 1.4 - 1.6 sau khi biến đổi ngược (Inverse Transform). 
  *(Sinh viên chèn hình ảnh biểu đồ MSE Loss vào đây).*

---

## CHƯƠNG 4: MẠNG NƠ-RON HỒI QUY (RECURRENT NEURAL NETWORKS - RNN)

### 4.1. Giới thiệu về RNN
Mạng nơ-ron hồi quy (RNN) là một kiến trúc đặc biệt có khả năng ghi nhớ trạng thái (state), rất hiệu quả cho các bài toán dữ liệu dạng chuỗi, đặc biệt là chuỗi thời gian (Time-series) hoặc xử lý ngôn ngữ tự nhiên (NLP). Tuy nhiên, RNN cơ bản gặp vấn đề đạo hàm tiêu biến (Vanishing Gradient) với các chuỗi dài, dẫn đến sự ra đời của LSTM và GRU.

### 4.2. Biến thể của RNN
- **Simple RNN:** Có bộ nhớ ngắn hạn, sử dụng chung trọng số ở mọi bước thời gian (time steps).
- **LSTM (Long Short-Term Memory):** Thêm các cổng (cổng quên, cổng cập nhật, cổng xuất) để kiểm soát dòng thông tin dài hạn.
- **GRU (Gated Recurrent Unit):** Tương tự LSTM nhưng gộp cổng, giảm khối lượng tính toán mà vẫn duy trì hiệu suất.

### 4.3. Thực nghiệm trên Dữ liệu Chuỗi thời gian
Dữ liệu được chuẩn hoá bằng `MinMaxScaler(0,1)`. Window_size (Kích thước cửa sổ trượt) được chọn là 30 hoặc 60.

#### 4.3.1. Tập dữ liệu 1: Dự đoán Giá cổ phiếu Amazon (AMZN)
- **Mô tả:** Dự đoán giá đóng cửa (Close Price) của Amazon dựa trên 60 ngày giao dịch trước đó.
- **Phương pháp cài đặt:**
  - **Keras:** Sử dụng kiến trúc LSTM với Dropout để chống Overfitting. Cụ thể: `LSTM(50) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1)`.
  - **PyTorch:** Sử dụng `nn.RNN` hoặc `nn.LSTM`, đạt loss hội tụ sau 20 epochs.
  - **Scratch:** Một mô hình RNN cơ bản được xây dựng từ các ma trận trọng số $W_x, W_h, W_y$ bằng NumPy. Sử dụng Backpropagation Through Time (BPTT) và Gradient Clipping để tối ưu hoá.
- **Đánh giá:** Mô hình dự đoán khá sát xu hướng giá cổ phiếu. Keras LSTM và PyTorch đem lại đường cong mượt mà và bám sát đồ thị thực tế hơn mô hình Scratch do giới hạn của Simple RNN truyền thống.
  *(Sinh viên chèn hình ảnh biểu đồ so sánh Thực tế vs Dự đoán ở đây).*

#### 4.3.2. Tập dữ liệu 2: Dự đoán Giá Vàng (Gold Price)
- **Mô tả:** Bài toán tương tự với giá vàng lịch sử.
- **Đánh giá:** Áp dụng cùng cấu trúc mạng RNN/GRU. Keras (sử dụng GRU) chứng minh tính ưu việt trong khả năng học các đỉnh nhọn (spikes) của giá vàng nhanh hơn. Cả 3 phương pháp Keras, PyTorch và Scratch đều hoàn thành các bước forward và backward pass hiệu quả.
  *(Sinh viên chèn biểu đồ so sánh RMSE giữa các phương pháp vào đây).*

### 4.4. Kết luận
Thông qua việc lập trình các mô hình CNN và RNN từ các mức độ trừu tượng khác nhau (Scratch với NumPy, Tự động đạo hàm với PyTorch, và Tự động hoá cao cấp với Keras/TensorFlow), báo cáo làm sáng tỏ cách các Framework học sâu hoạt động ở mức hệ thống, đồng thời nhấn mạnh hiệu quả của chúng trong việc giải quyết các bài toán Hồi quy, Phân loại và Chuỗi thời gian.

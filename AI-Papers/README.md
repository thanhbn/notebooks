# PDF to TXT Converter - Hướng Dẫn Sử Dụng

## 📖 Mô Tả
Script Python tự động chuyển đổi tất cả file PDF trong thư mục thành file TXT với nội dung văn bản được trích xuất đầy đủ.

## 🚀 Tính Năng
- ✅ Chuyển đổi tất cả file PDF trong thư mục
- ✅ Tạo file TXT cùng tên với file PDF gốc
- ✅ Trích xuất text từ tất cả các trang
- ✅ Xử lý hình ảnh/biểu đồ bằng ghi chú thay thế
- ✅ Hỗ trợ UTF-8 encoding
- ✅ Báo cáo tiến trình chi tiết

## 📋 Yêu Cầu Hệ Thống

### Python
- Python 3.7 hoặc cao hơn

### Thư Viện Cần Thiết
```bash
pip install PyPDF2 pdfplumber
```

## 🔧 Cài Đặt

### Bước 1: Cài đặt thư viện
```bash
# Cài đặt thư viện chính
pip install PyPDF2

# Cài đặt thư viện dự phòng (tùy chọn)
pip install pdfplumber
```

### Bước 2: Tải script
Script `pdf_to_txt_converter.py` đã có sẵn trong thư mục này.

## 💻 Cách Sử Dụng

### Chạy Script Cơ Bản
```bash
python pdf_to_txt_converter.py
```

### Thay Đổi Thư Mục Làm Việc
Mở file `pdf_to_txt_converter.py` và sửa dòng:
```python
working_dir = r"D:\llm\notebooks\AI-Papers"
```
Thành đường dẫn thư mục mong muốn.

## 📁 Cấu Trúc Thư Mục

### Trước Khi Chạy
```
D:\llm\notebooks\AI-Papers\
├── file1.pdf
├── file2.pdf
├── file3.pdf
└── pdf_to_txt_converter.py
```

### Sau Khi Chạy
```
D:\llm\notebooks\AI-Papers\
├── file1.pdf
├── file1.txt          ← Mới tạo
├── file2.pdf
├── file2.txt          ← Mới tạo  
├── file3.pdf
├── file3.txt          ← Mới tạo
└── pdf_to_txt_converter.py
```

## 📄 Định Dạng File TXT

Mỗi file TXT được tạo sẽ có cấu trúc:

```
# tên-file.pdf
# Converted from PDF to TXT
# Source path: đường/dẫn/đầy/đủ
# File size: kích-thước bytes

===============================================
PDF FILE CONTENT
===============================================

--- PAGE 1 ---
[Nội dung trang 1]

--- PAGE 2 ---
[Nội dung trang 2]

...
```

## ⚙️ Tùy Chỉnh Script

### Thay Đổi Thư Mục Làm Việc
```python
# Trong file pdf_to_txt_converter.py, dòng 120
working_dir = r"ĐƯỜNG_DẪN_THU_MỤC_CỦA_BẠN"
```

### Thêm Xử Lý Lỗi Cho Thư Mục Con
```python
# Thêm vào cuối hàm main()
for root, dirs, files in os.walk(working_dir):
    for file in files:
        if file.endswith('.pdf'):
            # Xử lý file PDF
```
## 🛠️ Xử Lý Sự Cố

### Lỗi "ModuleNotFoundError"
```bash
# Nếu thiếu PyPDF2
pip install PyPDF2

# Nếu thiếu pdfplumber  
pip install pdfplumber

# Cài đặt tất cả cùng lúc
pip install PyPDF2 pdfplumber --user
```

### Lỗi "UnicodeEncodeError"
Script đã được tối ưu hóa cho Windows. Nếu vẫn gặp lỗi:
```bash
# Chạy với encoding UTF-8
chcp 65001
python pdf_to_txt_converter.py
```

### Lỗi "Permission Denied"
```bash
# Chạy với quyền admin hoặc thay đổi quyền thư mục
# Hoặc di chuyển files đến thư mục khác không bị bảo vệ
```

### File PDF Không Đọc Được
- File PDF bị hỏng: Kiểm tra lại file gốc
- File PDF được bảo vệ: Cần mật khẩu để mở
- File PDF chỉ chứa hình ảnh: Script sẽ ghi chú "[IMAGE: Unable to extract text]"

## 📊 Kết Quả Mẫu

### Console Output
```
============================================================
PDF TO TXT CONVERTER
============================================================
Working directory: D:\llm\notebooks\AI-Papers
Found 11 PDF files:
  - paper1.pdf
  - paper2.pdf
  - ...

============================================================
STARTING CONVERSION
============================================================
Converting: paper1.pdf
  - Processing 12 pages with PyPDF2...
  [OK] Created: paper1.txt

Converting: paper2.pdf
  - Processing 24 pages with PyPDF2...
  [OK] Created: paper2.txt

============================================================
CONVERSION RESULTS
============================================================
Successful: 11 files
Failed: 0 files
Total: 11 files

TXT files created in directory: D:\llm\notebooks\AI-Papers
```

## 🔍 Chi Tiết Kỹ Thuật

### Thư Viện Sử Dụng
1. **PyPDF2**: Thư viện chính để trích xuất text từ PDF
2. **pdfplumber**: Thư viện dự phòng, hỗ trợ trích xuất bảng biểu
3. **os, glob, pathlib**: Xử lý file và thư mục

### Quy Trình Xử Lý
1. Quét thư mục tìm file `.pdf`
2. Với mỗi file PDF:
   - Mở file bằng PyPDF2
   - Trích xuất text từ từng trang
   - Xử lý các trang không có text
   - Tạo file TXT với header thông tin
   - Ghi nội dung đã trích xuất

### Xử Lý Nội Dung Đặc Biệt
- **Text thông thường**: Trích xuất đầy đủ
- **Hình ảnh**: Ghi chú `[IMAGE: Unable to extract text from this page]`
- **Bảng biểu**: Cố gắng trích xuất bằng pdfplumber (nếu có)
- **Trang trống**: Ghi chú `[IMAGE: Unable to extract text from this page]`

## 📝 Ghi Chú Quan Trọng

⚠️ **Lưu Ý:**
- Script sẽ **ghi đè** file TXT đã tồn tại
- File PDF gốc **không bị thay đổi**
- Chỉ xử lý file `.pdf` (không phân biệt chữ hoa/thường)

💡 **Mẹo:**
- Sao lưu file quan trọng trước khi chạy
- Kiểm tra kết quả file TXT đầu tiên trước khi xử lý hàng loạt
- Sử dụng thư mục test nhỏ để thử nghiệm

## 🚀 Tính Năng Nâng Cao (Tùy Chỉnh)

### Thêm Hỗ Trợ Thư Mục Con
```python
# Thay thế hàm main() để xử lý thư mục con
def process_directory_recursive(directory):
    for root, dirs, files in os.walk(directory):
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(root, pdf_file)
            txt_path = os.path.join(root, pdf_file[:-4] + '.txt')
            convert_pdf_to_txt(pdf_path, txt_path)
```

### Thêm Lọc Theo Ngày
```python
# Chỉ xử lý file được tạo sau ngày nhất định
import datetime
from pathlib import Path

def filter_by_date(pdf_path, days_ago=7):
    file_time = Path(pdf_path).stat().st_mtime
    week_ago = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    return datetime.datetime.fromtimestamp(file_time) > week_ago
```

## 🤝 Đóng Góp
Nếu gặp lỗi hoặc có ý tưởng cải thiện, vui lòng:
1. Báo cáo lỗi chi tiết
2. Đề xuất tính năng mới
3. Chia sẻ file test để kiểm tra

## 📄 Giấy Phép
Script này được phát triển để sử dụng nội bộ và giáo dục.

---
*Cập nhật lần cuối: 15/06/2025*

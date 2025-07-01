#!/usr/bin/env python3
"""
Script dịch các file paper từ tiếng Anh sang tiếng Việt
Sử dụng OpenAI API để dịch thuật chuyên nghiệp
"""

import os
import glob
import openai
from pathlib import Path
import time
import json
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTranslator:
    def __init__(self):
        # Lấy API key từ environment
        self.client = openai.OpenAI()
        self.base_dir = Path(".")
        
        # Prompt dịch thuật chuyên nghiệp
        self.translation_prompt = """Bạn là một trợ lý dịch thuật chuyên nghiệp, có khả năng dịch văn bản học thuật và kỹ thuật một cách chính xác.

Nhiệm vụ của bạn là dịch toàn bộ nội dung sau sang tiếng Việt.

Tập trung vào việc:
1. Dịch thuật chính xác các thuật ngữ chuyên ngành.
2. Giữ nguyên định dạng của văn bản gốc (ví dụ: các đoạn code, công thức, bảng biểu, v.v.).
3. Sử dụng văn phong học thuật, trang trọng.
4. Đảm bảo bản dịch mượt mà và dễ hiểu trong ngữ cảnh tiếng Việt.
5. Giữ nguyên các từ tiếng Anh trong ngoặc đơn sau thuật ngữ tiếng Việt nếu cần thiết.

Văn bản cần dịch:
"""

    def find_untranslated_files(self):
        """Tìm các file .txt chưa có bản dịch _vi.txt"""
        all_txt_files = glob.glob("*.txt")
        untranslated = []
        
        for txt_file in all_txt_files:
            if not txt_file.endswith("_vi.txt"):
                vi_file = txt_file.replace(".txt", "_vi.txt")
                if not os.path.exists(vi_file):
                    untranslated.append(txt_file)
        
        return untranslated

    def read_file_content(self, file_path):
        """Đọc nội dung file với encoding UTF-8"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Thử encoding khác nếu UTF-8 không hoạt động
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    def translate_text_chunks(self, text, max_chunk_size=4000):
        """Chia text thành chunks nhỏ để dịch"""
        chunks = []
        current_chunk = ""
        
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Paragraph quá dài, chia nhỏ hơn
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_chunk + sentence) > max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                chunks.append(sentence)
                        else:
                            current_chunk += sentence + ". "
            else:
                current_chunk += paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def translate_chunk(self, chunk):
        """Dịch một chunk text"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.translation_prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Lỗi khi dịch chunk: {e}")
            return f"[LỖI DỊCH] {chunk}"

    def translate_file(self, input_file):
        """Dịch một file"""
        logger.info(f"Bắt đầu dịch file: {input_file}")
        
        # Đọc nội dung file
        content = self.read_file_content(input_file)
        
        # Chia thành chunks
        chunks = self.translate_text_chunks(content)
        logger.info(f"Chia thành {len(chunks)} chunks")
        
        # Dịch từng chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Dịch chunk {i+1}/{len(chunks)}")
            translated_chunk = self.translate_chunk(chunk)
            translated_chunks.append(translated_chunk)
            
            # Tạm dừng để tránh rate limit
            time.sleep(1)
        
        # Ghép lại thành văn bản hoàn chỉnh
        translated_content = "\n\n".join(translated_chunks)
        
        # Lưu file dịch
        output_file = input_file.replace(".txt", "_vi.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        
        logger.info(f"Đã lưu bản dịch: {output_file}")
        return output_file

    def translate_all_files(self):
        """Dịch tất cả các file chưa có bản dịch"""
        untranslated_files = self.find_untranslated_files()
        
        if not untranslated_files:
            logger.info("Không có file nào cần dịch!")
            return
        
        logger.info(f"Tìm thấy {len(untranslated_files)} file cần dịch")
        
        for i, file_path in enumerate(untranslated_files):
            try:
                logger.info(f"Tiến độ: {i+1}/{len(untranslated_files)}")
                self.translate_file(file_path)
                logger.info(f"Hoàn thành dịch file: {file_path}")
                
                # Tạm dừng giữa các file
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Lỗi khi dịch file {file_path}: {e}")
                continue
        
        logger.info("Hoàn thành dịch tất cả các file!")

def main():
    """Hàm chính"""
    # Kiểm tra API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Vui lòng thiết lập OPENAI_API_KEY trong environment variables")
        return
    
    translator = PaperTranslator()
    
    print("🚀 Bắt đầu quá trình dịch các file paper...")
    translator.translate_all_files()
    print("✅ Hoàn thành!")

if __name__ == "__main__":
    main()
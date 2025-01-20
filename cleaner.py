import os
import re
import fitz


def convert_pdf_to_txt(pdf_path):
    # Открываем PDF и извлекаем текст
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    doc.close()

    # Удаляем раздел с источниками, если он есть
    # Предполагаем, что список начинается с "References" или "Список литературы"
    text = re.sub(r"(References|Список литературы|Библиография|REFERENCES)(.*)", "", text, flags=re.DOTALL)
    text = re.sub(r"-\n", "", text, flags=re.DOTALL)
    return text


def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = convert_pdf_to_txt(pdf_path)

            # Сохраняем текст в .txt файл
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join("resources/clean_result", txt_filename) # Укажите куда сохранять результаты работы
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)
            print(f"Файл {txt_filename} успешно создан.")


# Пример использования
folder_path = "resources/clean"  # Укажите путь к папке с PDF
process_folder(folder_path)

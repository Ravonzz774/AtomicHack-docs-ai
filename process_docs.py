import os
import PyPDF2
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch
import faiss


# Функция для извлечения текста из PDF-файлов
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text


# Функция для обучения нейросети на PDF-файлах
def train_model_on_pdfs(pdf_folder):
    texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            texts.append(text)

    # Используем модель для создания векторов из текста
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    vectors = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        vectors.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    # Создаем индекс FAISS для быстрого поиска
    vector_dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(vector_dim)
    vectors = [v.astype('float32') for v in vectors]
    vectors = torch.tensor(vectors).numpy()
    index.add(vectors)

    return index, texts


# Функция для ответа на вопросы пользователя
def answer_question(question, index, texts):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    question_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().astype('float32')

    # Поиск ближайшего вектора в индексе FAISS
    D, I = index.search(question_vector.reshape(1, -1), 1)
    if D[0][0] < 1.0:  # Пороговое значение для определения релевантности
        return texts[I[0][0]]
    else:
        return "Не найдено"


# Основная функция
def main():
    pdf_folder = "docs"
    index, texts = train_model_on_pdfs(pdf_folder)

    while True:
        question = input("Введите ваш вопрос (или 'выход' для завершения): ")
        if question.lower() == 'выход':
            break
        answer = answer_question(question, index, texts)
        print(answer)


if __name__ == "__main__":
    main()
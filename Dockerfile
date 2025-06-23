FROM python:3.11-slim

# Cài đặt các gói cần thiết, bao gồm OpenJDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    openjdk-17-jdk \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Tạo thư mục và tải các file VnCoreNLP
RUN mkdir -p vncorenlp/models/wordsegmenter && \
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar && \
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab && \
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr && \
    mv VnCoreNLP-1.1.1.jar vncorenlp/ && \
    mv vi-vocab vncorenlp/models/wordsegmenter/ && \
    mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

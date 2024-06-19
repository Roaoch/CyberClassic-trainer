FROM python:3.9

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user ./false_dataset.csv ./false_dataset.csv
COPY --chown=user ./dataset.csv ./dataset.csv
COPY --chown=user ./src ./src
COPY --chown=user ./requirements.txt ./requirements.txt
COPY --chown=user ./main.py ./main.py

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install pyarrow numpy pandas accelerate datasets nltk openpyxl python-dotenv rouge_score datasets tqdm trl scikit-learn matplotlib transformers
RUN pip install fastapi uvicorn[standard]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

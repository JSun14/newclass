FROM python:3.8-slim-buster
WORKDIR /app
COPY helper.py inference.py tokenizer.pickle /app/
COPY saved_model/ /app/saved_model/
COPY requirements.txt /app/tmp/requirements.txt
RUN pip install -r /app/tmp/requirements.txt
EXPOSE 8000
CMD ["python", "inference.py"]

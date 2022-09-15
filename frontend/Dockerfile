FROM python:3.8-slim-buster
WORKDIR /app
COPY frontend.py .
COPY requirements.txt /app/tmp/requirements.txt
RUN pip install -r /app/tmp/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "frontend.py"]

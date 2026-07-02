FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/dashboard

COPY dashboard/pyproject.toml ./pyproject.toml
COPY dashboard/app.py ./app.py

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8050

CMD ["python", "app.py"]

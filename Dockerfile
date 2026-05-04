FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY assets/ /app/assets/

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8050

CMD ["stingray", "dashboard", "run", "--host", "0.0.0.0", "--port", "8050", "--work-dir", "/dash_data"]
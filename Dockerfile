FROM python:3.12-slim

WORKDIR /app

# Install only Dash app dependencies
RUN pip install --no-cache-dir \
    dash==3.2.0 \
    plotly==5.24.1 \
    pandas==2.2.3 \
    numpy==2.1.2

# Copy Dash app
COPY dashapp.py /app/
COPY assets/ /app/assets/

# Dash runs on 8050
EXPOSE 8050

CMD ["python", "dashapp.py"]
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only the Dash app and assets folder
COPY dashapp.py /app/
COPY assets/ /app/assets/

# Expose port (Dash default is 8050)
EXPOSE 8050

# Run the Dash app
CMD ["python", "dashapp.py"]

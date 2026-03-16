FROM python:3.12-slim
WORKDIR /app
# Install only Dash app dependencies
RUN pip install --no-cache-dir \
    dash \
    plotly \
    pandas \
    numpy
# Copy Dash app
COPY dashapp.py /app/
COPY assets/ /app/assets/
# Dash runs on 8050
EXPOSE 8050
CMD ["python", "dashapp.py"]
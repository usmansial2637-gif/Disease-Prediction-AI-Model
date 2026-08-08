# Disease Prediction Project — Docker image
# Contains everything needed to run training (main.py), the Streamlit app
# (app.py), and the REST API (api.py) from one image. Which one runs is
# decided by the command passed at `docker run` / docker-compose (see
# docker-compose.yml).

FROM python:3.11-slim

# libgomp1 is required at runtime by LightGBM and XGBoost (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# outputs/ (trained models, plots, reports) and database/ (SQLite files)
# are meant to be mounted as volumes in docker-compose.yml so they persist
# across container restarts instead of living only inside the image layer.
RUN mkdir -p outputs/plots outputs/models outputs/reports database

EXPOSE 8501 8000

# Default command: run the Streamlit app. Override with `command:` in
# docker-compose.yml (or `docker run ... python main.py --dataset all`) to
# do something else instead.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

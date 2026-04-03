FROM python:3.12-slim-bookworm

WORKDIR /app

COPY monte_carlo.py /app/

ENTRYPOINT ["python", "monte_carlo.py"]
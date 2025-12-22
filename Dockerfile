# 1. Imagen base: Python 3.9 Slim (Debian)
FROM python:3.9-slim

# 2. Variables de entorno para evitar archivos .pyc y buffers en logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/final_hybrid_model

# 3. Instalar dependencias del sistema necesarias para compilar paquetes
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Crear directorio de trabajo
WORKDIR /app

# 5. Copiar dependencias e instalarlas
# Hacemos esto antes de copiar el código para aprovechar la caché de Docker
COPY requirements.txt .

# Instalamos CPU-only torch para reducir el tamaño (si no se usa GPU en producción)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar código fuente
COPY src/ /app/src/

# 7. Copiar el modelo entrenado
# NOTA: En producción real, es mejor descargar el modelo desde S3/GCS al iniciar
# en lugar de "tostarlo" en la imagen, pero para este MVP lo copiamos.
COPY models/esm2_hybrid_finetuned /app/models/final_hybrid_model

# 8. Exponer el puerto
EXPOSE 8000

# 9. Comando de arranque
# Usamos uvicorn para lanzar la API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
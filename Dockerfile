FROM colmap/colmap

RUN apt-get update && apt-get install -y git python3.10 pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

RUN python3 setup.py
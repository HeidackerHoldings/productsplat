FROM colmap/colmap

RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6

RUN curl https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -o ./anaconda.sh

RUN bash ./anaconda.sh -b -p /root/anaconda3 && rm ./anaconda.sh
ENV PATH /root/anaconda3/bin:$PATH
RUN conda update -n base -c defaults conda

WORKDIR /app
COPY environment.yml /app
COPY submodules /app/submodules
RUN conda env create -f environment.yml
RUN conda init

ENV TORCH_CUDA_ARCH_LIST "8.6"
RUN conda run -n productsplat pip install submodules/simple-knn
RUN conda run -n productsplat pip install submodules/diff-gaussian-rasterization

COPY . /app
FROM ychangc/minedojo_pytorch:conda-cuda12.1-pytorch2.4.1

USER root

# Env setting non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Env setting 
RUN mkdir -p /mount/nfs
ENV MOUNT_PATH=/mount/nfs

# updata and install toolkits
RUN apt-get update && apt-get install -y \
    jq \
    && apt-get clean 
RUN sudo apt-get update

COPY --chown=user:user \
    requirements.txt /home/user/requirements.txt
RUN pip install --no-cache-dir --exists-action=i -r requirements.txt


WORKDIR /workspace
COPY --chown=user:user . /workspace


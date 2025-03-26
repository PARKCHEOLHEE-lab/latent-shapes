FROM python:3.10.12-slim

COPY requirements.txt /requirements.txt
COPY post-create-command.sh /post-create-command.sh

RUN python -m pip install --upgrade pip
RUN pip install -r /requirements.txt

RUN apt-get update && apt-get install -y \
    git \
    zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

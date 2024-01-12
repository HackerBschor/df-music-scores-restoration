# syntax=docker/dockerfile:1

FROM python:3.9

WORKDIR /deliver

RUN curl -L https://github.com/HackerBschor/dl-music-scores-restoration/releases/download/Model/model.pt --output model.pt

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
FROM python:3.9

COPY . /overfitLevelOneApp

WORKDIR /overfitLevelOneApp

RUN pip install -r requirements.txt

EXPOSE $PORT

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
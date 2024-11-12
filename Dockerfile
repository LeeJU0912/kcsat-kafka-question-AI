FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app/

COPY ./main.py /app/
COPY ./requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
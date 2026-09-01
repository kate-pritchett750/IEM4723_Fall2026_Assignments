#Given below is a Dockerfile that can be used to create a runtime environment for a simple linear regression example

FROM python:3.12-slim

RUN pip install --no-cache-dir numpy scikit-learn

WORKDIR /home/iem4723/

COPY . /home/iem4723/

CMD ["python3", "Regressor.py"]

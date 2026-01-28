FROM python:3.11

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    sqlalchemy \
    psycopg2-binary \
    matplotlib \
    seaborn \
    jupyter

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


# requirements.txt
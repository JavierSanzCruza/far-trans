FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

EXPOSE 8888

ENV JUPYTER_RUNTIME_DIR='/tmp/'
ENV JUPYTER_DATA_DIR=$PYTHONUSERBASE

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN python -m pip install jupyter
RUN python -m pip install jupyterlab

ENV LD_LIBRARY_PATH /opt/conda/lib/python3.7/site-packages/pyKiVi

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install gcc python3-dev -y

# Install beta-recsys
WORKDIR /tmp/code/
RUN git clone https://github.com/JavierSanzCruza/beta-recsys.git
WORKDIR /tmp/code/beta-recsys/
RUN python setup.py install --record files.txt
WORKDIR /tmp/code/

# Now, we copy the necessary files:
COPY . /tmp/code/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--notebook-dir='/tmp/'"]
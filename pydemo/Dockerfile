FROM debian:buster
MAINTAINER "Andreas Dedner"

USER root

RUN apt-get update && apt-get dist-upgrade --yes --no-install-recommends
RUN apt-get install --yes --no-install-recommends \
  ca-certificates wget tar \
  libc6-dev gcc g++ gfortran binutils make cmake pkg-config patch \
  python3-dev python3-setuptools python3-pip python3-wheel \
  python3-pandocfilters \
  python3-numpy python3-scipy python3-matplotlib \
  python3-appdirs python3-configobj python3-requests python3-sympy \
  jupyter-notebook jupyter-nbconvert \
  libeigen3-dev libsuitesparse-dev \
  python3-ufl

ADD install-dune.sh /root/
RUN /root/install-dune.sh

ADD jupyter_config.py /etc/
RUN chmod 644 /etc/jupyter_config.py

RUN adduser --disabled-password --home /dune --gecos "" --uid 50000 dune

COPY demo.ipynb /dune/
RUN chown dune:dune /dune/demo.ipynb

USER dune

ENV LD_LIBRARY_PATH "/usr/lib:/usr/local/lib"
ENV DUNE_CMAKE_FLAGS="CMAKE_BUILD_TYPE=Release BUILD_SHARED_LIBS=TRUE"

WORKDIR /dune
RUN jupyter nbconvert --config=/etc/jupyter_config.py --to notebook --ExecutePreprocessor.timeout=-1 --execute demo.ipynb --output demo.ipynb

WORKDIR /dune
EXPOSE 8888
VOLUME ["/dune"]
CMD ["jupyter-notebook", "--config=/etc/jupyter_config.py", "--ip=0.0.0.0"]

FROM ubuntu:19.04

ARG BUILD_ID=0
# install standard packages

# note: gips-specific dependencies are: cython, libboost-all-dev, libgnutls28-dev, python,
#   python-idna, python-gdal, python-pip, python-setuptools, python-urllib3, swig, libgdal-dev

# note: python3-botocore package does not include the module botocore.vendored, which
#   is required by the snowflake connector in the telluslabs package, we use pip
#   (requirements.txt) to install botocore and boto3 to get around this omission until
#   snowflake updates (botocore.vendored is scheduled for deprecation)

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    cython \
    gdal-bin \
    gfortran \
    less \
    libboost-all-dev \
    libeccodes0 \
    libgdal20 \
    libgdal-dev \
    libgnutls28-dev \
    python \
    python-gdal \
    python-idna \
    python-pip \
    python-setuptools \
    python-urllib3 \
    python-wheel \
    python3 \
    python3-celery \
    python3-click \
    python3-fiona \
    python3-flake8 \
    python3-gdal \
    python3-joblib \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-psycopg2 \
    python3-pyproj \
    python3-pytest \
    python3-pytest-cov \
    python3-pytest-xdist \
    python3-requests \
    python3-rtree \
    python3-scipy \
    python3-shapely \
    ssh \
    swig \
    tar \
    unzip \
    vim \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY . /multitemporal
WORKDIR /multitemporal

# add Indigo Artifactory pip credentials
# note: pip.conf is *not* included in the repo as it contains private credentials,
#   you must copy in your local pip.conf file, which must be configured with access
#   to Indigo Artifactory. For instructions: https://symbiota.atlassian.net/wiki/spaces/TEC/pages/78119117/Artifactory
COPY .pip /root/.pip

# install python dependencies
RUN python3 -m pip install --upgrade pip \
    && pip3 install --upgrade -r /multitemporal/requirements.txt

# Rasterio wheels are built on CentOS, which stores SSL certificates
# differently than Ubuntu. This export defines the correct location
# Reference: https://github.com/mapbox/rasterio/commit/b621d92c51f7c2021f89cd4487cecdd7c201f320
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN python3 /multitemporal/setup.py build_ext --inplace \
	&& python3 /multitemporal/setup.py install

# remove Indigo Artifactory pip credentials (we are all done with installs)
RUN rm -rf /root/.pip/pip.conf

VOLUME /data

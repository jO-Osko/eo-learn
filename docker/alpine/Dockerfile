#FROM alpine:3.8
FROM julianbei/alpine-opencv-microimage:p3-3.1
# VERSIONS
ENV ALPINE_VERSION=3.8 \
    PYTHON_VERSION=3.6.6

# PATHS
ENV PYTHON_PATH=/usr/lib/python$PYTHON_VERSION \
    PATH="/usr/lib/python$PYTHON_VERSION/bin/:${PATH}"

# PACKAGES
#   * dumb-init: a proper init system for containers, to reap zombie children
#   * musl: standard C library
#   * lib6-compat: compatibility libraries for glibc
#   * linux-headers: commonly needed, and an unusual package name from Alpine.
#   * build-base: used so we include the basic development packages (gcc)
#   * bash: so we can access /bin/bash
#   * git: to ease up clones of repos
#   * ca-certificates: for SSL verification during Pip and easy_install
ENV PACKAGES="\
    dumb-init \
    musl \
    linux-headers \
    build-base \
    bash \
    git \
    ca-certificates \
    libssl1.0 \
    libffi-dev \
    freetype-dev \
    libjpeg-turbo-dev\
"
#    libc6-compat

# PACKAGES needed to built python
ENV PYTHON_BUILD_PACKAGES="\
    readline-dev \
    zlib-dev \
    bzip2-dev \
    sqlite-dev \
    openssl-dev \
    musl-dev \
    linux-headers \
    g++ \
    jpeg-dev \
    glib \
"

RUN set -ex ;\
    # find MAJOR and MINOR python versions based on $PYTHON_VERSION
    export PYTHON_MAJOR_VERSION=$(echo "${PYTHON_VERSION}" | rev | cut -d"." -f3-  | rev) ;\
    export PYTHON_MINOR_VERSION=$(echo "${PYTHON_VERSION}" | rev | cut -d"." -f2-  | rev) ;\
    # replacing default repositories with edge ones
    echo "http://dl-cdn.alpinelinux.org/alpine/v$ALPINE_VERSION/community" >> /etc/apk/repositories ;\
    echo "http://dl-cdn.alpinelinux.org/alpine/v$ALPINE_VERSION/main" >> /etc/apk/repositories ;\
    # Add the packages, with a CDN-breakage fallback if needed
    apk add --no-cache $PACKAGES || \
        (sed -i -e 's/dl-cdn/dl-4/g' /etc/apk/repositories && apk add --no-cache $PACKAGES) ;\
    # Add packages just for the python build process with a CDN-breakage fallback if needed
    apk add --no-cache --virtual .build-deps $PYTHON_BUILD_PACKAGES || \
        (sed -i -e 's/dl-cdn/dl-4/g' /etc/apk/repositories && apk add --no-cache --virtual .build-deps $PYTHON_BUILD_PACKAGES) ;\
    # turn back the clock -- so hacky!
    # echo "@community http://dl-cdn.alpinelinux.org/alpine/v$ALPINE_VERSION/community" >> /etc/apk/repositories ;\
    # echo "@testing http://dl-cdn.alpinelinux.org/alpine/v$ALPINE_VERSION/testing" >> /etc/apk/repositories ;\
    # echo "@edge-main http://dl-cdn.alpinelinux.org/alpine/edge/main" >> /etc/apk/repositories ;\
    # use pyenv to download and compile specific python version
    git clone --depth 1 https://github.com/pyenv/pyenv /usr/lib/pyenv ;\
    PYENV_ROOT=/usr/lib/pyenv /usr/lib/pyenv/bin/pyenv install $PYTHON_VERSION ;\
    # move specific version to correct path delete pyenv, no longer needed
    mv /usr/lib/pyenv/versions/$PYTHON_VERSION/ $PYTHON_PATH ;\
    rm -rfv /usr/lib/pyenv ;\
    # change the path on the header of every file from PYENV_ROOT to PYTHON_PATH
    cd $PYTHON_PATH/bin/ && sed -i "s+/usr/lib/pyenv/versions/$PYTHON_VERSION/+$PYTHON_PATH/+g" * ;\
    # delete binary "duplicates" and replace them with symlinks
    # this also optimizes space since they are actually the same binary
    rm -f $PYTHON_PATH/bin/python$PYTHON_MAJOR_VERSION \
          $PYTHON_PATH/bin/python$PYTHON_MINOR_VERSION \
          $PYTHON_PATH/bin/python$PYTHON_MAJOR_VERSION-config \
          $PYTHON_PATH/bin/python$PYTHON_MINOR_VERSION-config ;\
    ln -sf $PYTHON_PATH/bin/python $PYTHON_PATH/bin/python$PYTHON_MAJOR_VERSION ;\
    ln -sf $PYTHON_PATH/bin/python $PYTHON_PATH/bin/python$PYTHON_MINOR_VERSION ;\
    ln -sf $PYTHON_PATH/bin/python-config $PYTHON_PATH/bin/python$PYTHON_MAJOR_VERSION-config ;\
    ln -sf $PYTHON_PATH/bin/python-config $PYTHON_PATH/bin/python$PYTHON_MINOR_VERSION-config ;\
    # delete files to to reduce container size
    # tips taken from main python docker repo
    find /usr/lib/python$PYTHON_VERSION -depth \( -name '*.pyo' -o -name '*.pyc' -o -name 'test' -o -name 'tests' \) -exec rm -rf '{}' + ;
    # remove build dependencies and any leftover apk cache
#
RUN apk --no-cache --update-cache add gcc gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev

RUN apk add --no-cache --virtual .build-deps-edge \
        --repository http://dl-cdn.alpinelinux.org/alpine/edge/main \
        --repository http://dl-cdn.alpinelinux.org/alpine/edge/testing \
        gdal-dev \
        geos-dev \
        glib \
        opencv
        #proj4-dev \
        #protobuf-c-dev \


# Install GDAL
#RUN svn checkout --quiet "http://svn.osgeo.org/gdal/tags/2.1.3" /tmp/gdal/
#RUN cd /tmp/gdal
#RUN ./configure
#RUN make
#RUN make install
#RUN cd
#RUN rm -rf /tmp/gdal
#We take some inspiration from here: https://github.com/vegansk/docker-ubuntu-java-nodejs/blob/master/Dockerfile

RUN mkdir -p /var/log/eolearn/

#######################################################################################################################################
ARG user=eolearn
ARG group=eolearn
ARG uid=1000
ARG gid=1000
ARG USER_HOME="/home/${user}"

ENV USER_HOME ${USER_HOME}

#RUN groupadd -g ${gid} ${group} && useradd -d "${USER_HOME}" -u "${uid}" -g "${gid}" -m -s /bin/bash "${user}"
#RUN echo "eolearn:eolearn" | chpasswd

# setup SSH server
#RUN apt-get update && apt-get install --no-install-recommends -y openssh-server && rm -rf /var/lib/apt/lists/*

#RUN sed -i /etc/ssh/sshd_config \
#        -e 's/#PermitRootLogin.*/PermitRootLogin no/' \
#        -e 's/#RSAAuthentication.*/RSAAuthentication no/'  \
#        -e 's/#PasswordAuthentication.*/PasswordAuthentication yes/' \
#        -e 's/#SyslogFacility.*/SyslogFacility AUTH/' \
#        -e 's/#LogLevel.*/LogLevel INFO/' && \
#    mkdir /var/run/sshd

# SSH login fix. Otherwise user is kicked off after login
# https://docs.docker.com/engine/examples/running_ssh_service/
#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

WORKDIR "${USER_HOME}"

ADD ./ "${USER_HOME}/"
COPY setup-sshd /usr/local/bin/setup-sshd
# Ugly ugly hack to get opencv
# https://github.com/pypa/pip/issues/3969
COPY _manylinux.py _manylinux.py
RUN pip install --upgrade pip
RUN pip install pytest
# To get the correct header or something
# RUN pip install -U --force-reinstall numpy
# RUN pip install -U --force-reinstall scipy
RUN pip install -U  pillow
RUN pip install -U  shapely
RUN pip install -U  pandas
RUN pip install -U  fiona
RUN pip install -U  sklearn
RUN pip install -U  rasterio
RUN pip install -U  matplotlib

# Some shit
#apk --no-cache add ca-certificates wget
#wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub
#wget https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.28-r0/glibc-2.28-r0.apk
#apk add glibc-2.28-r0.apk

RUN apk del --no-cache --purge .build-deps ;\
    rm -rf /var/cache/apk/*

EXPOSE 22

#ENTRYPOINT ["setup-sshd"]

# since we will be "always" mounting the volume, we can set this up
#ENTRYPOINT ["/usr/bin/dumb-init"]

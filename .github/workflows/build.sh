#!/bin/bash
set -ex

git clone --depth=2 https://gitlab.com/petsc/petsc.git
cd petsc
export PETSC_DIR=${GITHUB_WORKSPACE}/petsc
export PETSC_ARCH=arch-linux-double-opt
echo "PETSC_DIR=${GITHUB_WORKSPACE}/petsc" >> $GITHUB_ENV
echo "PETSC_ARCH=arch-linux-double-opt" >> $GITHUB_ENV
python3 ./configure \
  --COPTFLAGS=-O3 \
  --CXXOPTFLAGS=-O3 \
  --with-debugging=0 \
  --with-x=0 \
  --with-fc=0 \
  --download-openmpi \
  --download-f2cblaslapack=1 \
  --with-scalar-type=${scalar} \
  --with-petsc4py \
  $extra_opts || (cat configure.log && exit 1)
make
cd ..

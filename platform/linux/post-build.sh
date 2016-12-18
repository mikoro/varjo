#!/bin/sh

if [ ! -d bin/data ]; then
  cp -R data bin/
fi

if [ ! -f bin/varjo.ini ]; then
  cp misc/varjo.ini bin/
fi

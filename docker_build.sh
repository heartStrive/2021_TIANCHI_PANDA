#!/bin/bash
if [ -z $1 ]
then
   echo "s1:version"
   exit
fi

if [ -z $DOCKER_REGISTRY ]
then
   echo "not find $DOCKER_REGISTRY"
   exit
fi

VERSION=$1

##1.create docker regsit
docker build -t $DOCKER_REGISTRY/fastreid:$VERSION .

##2.PUSH
docker push $DOCKER_REGISTRY/fastreid:$VERSION
##3.echo submit info
echo "now you can go to tianchi.aliyun.com submit this docker url: $DOCKER_REGISTRY/fastreid:$VERSION"


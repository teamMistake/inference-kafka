#!/bin/bash
docker build -t jamo-inference .
docker tag jamo-inference:latest ghcr.io/teammistake/inference:latest
docker push ghcr.io/teammistake/inference:latest
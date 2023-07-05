#!/bin/bash
docker tag jamo-inference:latest ghcr.io/teammistake/inference:latest
docker push ghcr.io/teammistake/inference:latest
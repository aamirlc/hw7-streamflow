#!/bin/bash
# run.sh  --  Verde River streamflow forecasting workflow
# Edit the CONFIG section in run_forecast.py first, then run this script.
#
# Usage:
#   bash run.sh

set -e

echo "Running HW7 streamflow forecast..."
python run_forecast.py

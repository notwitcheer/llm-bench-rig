#!/bin/bash
# hmmt_feb_2026 grader agreement (CPU only, no GPU), the line the 09-28 probe never reached.
cd /home/witcheer/benchmark-rig; export PYTHONPATH=.
nice -n 10 .venv/bin/python scripts/matharena_grader_agreement.py hmmt_feb_2026 > /home/witcheer/matharena-agree-hmmt.log 2>&1
echo "agree rc=$? $(date)" >> /home/witcheer/matharena-agree-hmmt.log

# -*- coding: utf-8 -*-
"""
Minimal logger; replace with structlog/loguru if desired.
"""
import sys, time

def info(msg: str):  print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stdout)
def warn(msg: str):  print(f"[{time.strftime('%H:%M:%S')}] WARNING: {msg}", file=sys.stderr)
def error(msg: str): print(f"[{time.strftime('%H:%M:%S')}] ERROR: {msg}", file=sys.stderr)

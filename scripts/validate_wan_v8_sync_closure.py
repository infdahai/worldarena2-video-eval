#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from worldarena_baseline.wan_v8_sync_closure import build_v8_source_receipt

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--source-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    receipt=build_v8_source_receipt(args.source_root); args.output.parent.mkdir(parents=True,exist_ok=True); partial=args.output.with_suffix(args.output.suffix+".partial"); partial.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); os.replace(partial,args.output); print(receipt["closure_sha256"])
if __name__=="__main__": main()

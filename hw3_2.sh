#!/bin/bash

# TODO - run your inference Python3 code
python ./p2/inference.py --imgpath $1 --outpath $2 --model  ./p2/p2.pth --token ./p2/caption_tokenizer.json
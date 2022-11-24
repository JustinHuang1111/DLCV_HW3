python -c "import clip; clip.load('ViT-B/32')"
gdown --fuzzy https://drive.google.com/file/d/1ml5mlpwmSukmMQbC3vnFfQXcOR7_58Tm/view?usp=sharing -O ./p2/caption_tokenizer.json
# gdown --fuzzy https://drive.google.com/file/d/1Dnz2zBIHDrXOtCKdHs2Fyil6TwWuCj0T/view?usp=sharing -O ./p2/p2.pth
wget https://www.dropbox.com/s/asfl0cqvzir6siy/1117-0215_epsilon_50_best.pth?dl=1 -O ./p2/p2.pth
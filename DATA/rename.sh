# Rename all *_pixels0.png to *.jpg
for f in phantom_20/masks/*.png; do
    mv -- "$f" "${f%_pixels0.png}.jpg"
done
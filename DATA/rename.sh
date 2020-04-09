# Rename all *_pixels0.png to *.jpg
for f in invivo_test/masks/*.png; do
    mv -- "$f" "${f%_pixels0.png}.jpg"
done
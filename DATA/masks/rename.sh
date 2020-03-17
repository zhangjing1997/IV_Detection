# Rename all *_pixels0.png to *.png
# for f in *.png; do
#    mv -- "$f" "${f%_pixels0.png}.png"
# done

# Rename all *.png to *.jpg
for f in *.png; do
    mv -- "$f" "${f%.png}.jpg"
done

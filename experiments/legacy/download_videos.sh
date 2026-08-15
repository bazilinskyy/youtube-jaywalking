#!/bin/bash
# Download the 15 selected daytime videos from 15 countries
#
# Requirements: yt-dlp, ffmpeg
#   pip install yt-dlp
#
# Run this overnight — downloads can be slow depending on your connection.
# Videos are saved at 360p (good balance of quality/size for CPU processing).
#
# If a video fails, the script retries with the worst available format.
# All videos go to data/raw_clips/

VIDEOS=(
  "3ai7SUaPoHM" "Toronto, Canada"
  "gmDBzijaIAA" "Jakarta, Indonesia"
  "JY-Xyiept88" "Manila, Philippines"
  "wMu6Va5PhGY" "Sydney, Australia"
  "G1I_PlmL_YA" "Cincinnati, USA"
  "ZByZSqoqzaI" "Milan, Italy"
  "AxQcSoA9vGQ" "Kuala Lumpur, Malaysia"
  "qzimFzMh6lA" "Damascus, Syria"
  "z3Gx2hp3Vo8" "Luanda, Angola"
  "wCKLtcGQnWc" "Moscow, Russia"
  "qOx5CwCrN9k" "Seoul, South Korea"
  "oDejyTLYUTE" "Tokyo, Japan"
  "ZruaEnhtYLA" "Dhaka, Bangladesh"
  "MAj6y23vNuU" "Mumbai, India"
  "-TPJot7-HTs" "Buenos Aires, Argentina"
)

mkdir -p data/raw_clips

for ((i=0; i<${#VIDEOS[@]}; i+=2)); do
  VID="${VIDEOS[$i]}"
  NAME="${VIDEOS[$((i+1))]}"
  OUTFILE="data/raw_clips/${VID}.mp4"

  if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
    echo "Already have: $VID ($NAME) — $(du -h "$OUTFILE" | cut -f1)"
    continue
  fi

  echo "--- Downloading: $VID ($NAME) ---"

  # Try 360p first, fall back to worst available
  yt-dlp -f "best[height<=360]" \
    --max-filesize 300M \
    -o "data/raw_clips/%(id)s.%(ext)s" \
    "https://www.youtube.com/watch?v=${VID}" 2>/dev/null

  if [ $? -eq 0 ] && [ -s "$OUTFILE" ]; then
    echo "  OK: $VID ($(du -h "$OUTFILE" | cut -f1))"
  else
    echo "  Retrying with worst format..."
    rm -f "$OUTFILE" "${OUTFILE}.part" 2>/dev/null
    yt-dlp -f "worst" \
      -o "data/raw_clips/%(id)s.%(ext)s" \
      "https://www.youtube.com/watch?v=${VID}" 2>/dev/null

    if [ $? -eq 0 ] && [ -s "$OUTFILE" ]; then
      echo "  OK (worst): $VID ($(du -h "$OUTFILE" | cut -f1))"
    else
      echo "  FAILED: $VID"
      rm -f "$OUTFILE" "${OUTFILE}.part" 2>/dev/null
    fi
  fi
done

echo ""
echo "=== All done ==="
echo "Downloaded files:"
ls -lh data/raw_clips/ 2>/dev/null || echo "(empty)"

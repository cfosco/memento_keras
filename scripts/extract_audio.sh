#!/bin/bash
if [ "$1" == '' ] || [ "$2" == '' ] || [ "$3" == '' ]; then
    echo "Usage: $0 <input folder> <output folder> <file extension>";
    exit;
fi
for file in "$1"/*/*."$3"; do
	path="${file%/*}";
	filenameext="${file##*/}"
	filename="${filenameext%.*}"
    destination="$2${path##*/}";
	echo "$destination";
	echo "$filename";
	
    if [ ! -d "$destination" ]; then
      mkdir -p "$destination";
	fi
	ffmpeg -i "$file" -acodec mp3 "$destination/$filename.mp3" 

done

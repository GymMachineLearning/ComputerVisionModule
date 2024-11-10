#!/bin/bash

# Zdefiniuj zdalny folder, z którego chcesz pobrać pliki
REMOTE_DIR="inzynierka:Inżynierka/Nagrania/Squat"  # Zdalny folder na dysku 'inzynierka'

# Zdefiniuj lokalny folder, do którego chcesz pobrać pliki
LOCAL_DIR="/data/Videos"  # Zmienna lokalnego folderu, do którego będą pobrane pliki

# Sprawdź, czy lokalny folder istnieje, jeśli nie, utwórz go
if [ ! -d "$LOCAL_DIR" ]; then
  echo "Folder $LOCAL_DIR nie istnieje. Tworzę folder..."
  mkdir -p "$LOCAL_DIR"
fi

while true; do

	# Znajdź pierwszy plik MP4 w folderze
	first_video=$(find "$LOCAL_DIR" -type f -name "*.mp4" | head -n 1)

	# Sprawdź, czy znaleziono plik
	if [ -z "$first_video" ]; then
	  echo "Brak plików MP4 do przetworzenia."
	  exit 1
	fi

	echo "Znaleziono pierwszy plik wideo: $first_video"

	# Tutaj możesz dodać dowolną operację na pliku, np. analizę wideo
	# Na przykład uruchomienie jakiegoś procesu na pliku:
	# ./przetworz_wideo.sh "$first_video"


	# Zakładając, że $first_video zawiera pełną ścieżkę
	filename=$(basename "$first_video")
	folder=$(basename "$first_video" .mp4)
	# Wyświetlanie samej nazwy pliku
	echo "Nazwa pliku: $filename"


	LOCAL_OUTPUT="./demo/output/$folder/$filename"
	REMOTE_OUTPUT="inzynierka:Inżynierka/Nagrania/Procesowane"  # Zdalny folder na dysku 'inzynierka'


	python demo/movenet.py --video $first_video

	echo "LOCAL_OUTPUT : $LOCAL_OUTPUT"
	echo "REMOTE_OUTPUT : $REMOTE_OUTPUT"

	rclone copy "$LOCAL_OUTPUT" "$REMOTE_OUTPUT"  --progress --drive-shared-with-me

	# Po zakończeniu operacji usuń plik
	rm "$first_video"
	rm -r "./demo/output/$folder"
	# Informacja o zakończeniu
	echo "Przetwarzanie zakończone. Plik $first_video został usunięty."
done

echo "Przetwarzanie wszystkich plików zakończone."


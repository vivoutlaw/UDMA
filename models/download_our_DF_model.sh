FILE=$1

if [[  $FILE != "DeepFashion"]]; then
  echo "Available model pre-trained on DeepFashion (trainval set)"
  exit 1
fi

echo "Specified [$FILE]"

URL=https://cvhci.anthropomatik.kit.edu/~vsharma/UDMA/model/DeepFashion.tar.gz
TAR_FILE=./features/$FILE.tar.gz
TARGET_DIR=./features/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./features/
rm $TAR_FILE


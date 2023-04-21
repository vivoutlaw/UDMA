FILE=$1

if [[  $FILE != "DeepFashion" ]]; then
  echo "Available model pre-trained on DeepFashion (trainval set)"
  exit 1
fi

echo "Specified [$FILE]"

URL=https://cvhci.anthropomatik.kit.edu/~datasets-publisher/published_datasets/domain_adaption/UDMA/models/$FILE.tar.gz
TAR_FILE=./models/$FILE.tar.gz
TARGET_DIR=./models/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./models/
rm $TAR_FILE


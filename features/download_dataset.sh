FILE=$1

if [[ $FILE != "ClusteringFeats" &&  $FILE != "DeepFashion" &&  $FILE != "Street2Shop" ]]; then
  echo "Available datasets are ClusteringFeats: Trainset:DeepFashion and Trainset:Street2Shop; Testset for DeepFashion; and Testset for Street2Shop"
  exit 1
fi

echo "Specified [$FILE]"

URL=https://cvhci.anthropomatik.kit.edu/~vsharma/DeepFashion.tar.gz
TAR_FILE=./features/$FILE.tar.gz
TARGET_DIR=./features/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./features/
rm $TAR_FILE


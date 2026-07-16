#!/bin/bash


cd data/raw


for i in {100..124}
do
    echo "Downloading record $i..."
    wget -nc --quiet https://physionet.org/files/mitdb/1.0.0/${i}.dat
    wget -nc --quiet https://physionet.org/files/mitdb/1.0.0/${i}.hea
    wget -nc --quiet https://physionet.org/files/mitdb/1.0.0/${i}.atr
    
  
    if [ $? -eq 0 ]; then
        echo "Record $i downloaded successfully"
    else
        echo "Error downloading record $i"
    fi
done


echo -e "\nChecking downloaded files..."
count=0
for i in {100..124}
do
    if [ -f "${i}.dat" ] && [ -f "${i}.hea" ] && [ -f "${i}.atr" ]; then
        ((count++))
    fi
done

echo -e "\nDownload completed!"
echo "Successfully downloaded $count complete records"
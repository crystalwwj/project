#curl "https://datashare.is.ed.ac.uk/bitstream/handle/10283/853/wav_data.aa.tar.gz?sequence=6&isAllowed=y" -o wav_data.aa.tar.gz
curl "https://datashare.is.ed.ac.uk/bitstream/handle/10283/853/wav_data.ab.tar.gz?sequence=7&isAllowed=y" -o wav_data.ab.tar.gz
curl "https://datashare.is.ed.ac.uk/bitstream/handle/10283/853/wav_data.ac.tar.gz?sequence=8&isAllowed=y" -o wav_data.ac.tar.gz
cat wav_data.*.tar.gz | tar xzvf -
echo Files were successfully downloaded and uncompressed
#echo running main
#python main.py
#echo finished experiment

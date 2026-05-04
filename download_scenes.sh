for item in indoor_synthetic/kitchen.zip indoor_synthetic/bedroom.zip indoor_synthetic/livingroom.zip indoor_synthetic/bathroom.zip; do
  mkdir -p data/${item%.*}
  wget "https://kaldir.vc.cit.tum.de/intrinsix/${item}" -O "data/${item}"
  unzip "data/${item}" -d data/${item%.*}/..
  rm "data/${item}"
done
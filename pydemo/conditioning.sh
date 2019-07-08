replace "basisChoice=3" "basisChoice=1" -- testelliptic.py
python testelliptic.py > conditioning1.out
mv testelliptic.dump conditioning1.dump
replace "basisChoice=1" "basisChoice=2" -- testelliptic.py
python testelliptic.py > conditioning2.out
mv testelliptic.dump conditioning2.dump
replace "basisChoice=2" "basisChoice=3" -- testelliptic.py
python testelliptic.py > conditioning3.out
mv testelliptic.dump conditioning3.dump

grep "method:" conditioning1.out  > conditioning.out
echo >> conditioning
echo >> conditioning
grep "method:" conditioning2.out >> conditioning.out
echo >> conditioning
echo >> conditioning
grep "method:" conditioning3.out >> conditioning.out

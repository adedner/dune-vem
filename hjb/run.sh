grid=$1

minStab=0.1
maxStab=100
for p in "0" "-1" "1"
do
    python kappa.py --grid $grid --stab $minStab --startStab $maxStab --proj $p  --order 3 --no-linear  >  run3$p.out
done

for p in "0" "-1" "1"
do
    python kappa.py --grid $grid --stab $minStab --startStab $maxStab --proj $p  --order 4 --no-linear  >  run4$p.out
done

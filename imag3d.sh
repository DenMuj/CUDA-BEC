#!/bin/bash 
 
for ((i = 70; i < 71; i++)); do 
   for ((j = 1; j <= 30; j++)); do 
      mkdir $j 
      cd $j 
      cat ../input/imag3d-input-droplet-template | sed "s/CCCAS/70/g" | sed "s/CCCADD/132.607/g" > input 
 
       if [[ $i -gt 100 ]] || [[ $j -gt 1 ]] 
       then 
          sed -i "s/^.*SX = .*/   SX = $sx/" input 
          sed -i "s/^.*SY = .*/   SY = $sy/" input 
          sed -i "s/^.*SZ = .*/   SZ = $sz/" input 
       fi 
 
       ../imag3d-cuda -i input 
       
 
       read sx sy sz <<< $(tail -3 imag3d-rms.txt | head -1 | awk '{print $(NF-2), $(NF-1), $NF}') 
       cd .. 
 
   done 
done

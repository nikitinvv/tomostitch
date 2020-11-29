#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J rec
#SBATCH --mem 160G
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --tasks-per-node=32
#SBATCH -p v100

for f in 170
 do
  echo $f
  python ./stitchopencv.py /data/staff/tomograms/experiments/APS/2019-12/${f}_sand_gh_form.h5 /data/staff/tomograms/experiments/APS/2019-12_stitched/s${f}_sand_gh_form.h5
 done


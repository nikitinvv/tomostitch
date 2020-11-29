for f in $1
 do
  echo $f
  python ./stitch360.py --axis 120 /data/staff/tomograms/experiments/APS/2019-12/${f}_sand_gh_form.h5 /data/staff/tomograms/experiments/APS/2019-12_stitched/s${f}_sand_gh_form.h5
 done


#!/bin/bash

PROJECT_DIR=$1


cd
module load python/3.8
virtualenv ~/earth
source ~/earth/bin/activate
pip install --no-index --upgrade pip
pip install --no-index xarray==0.16.2 scipy==1.6.0 netCDF4==1.5.6
pip install --no-index h5netcdf==0.7.4 matplotlib==3.3.4
pip install --no-index pandas matplotlib seaborn
pip install --no-index torch horovod jupyterlab click opencv_python tensorboard torchvision
pip install python-dotenv

# create bash script for opening jupyter notebooks https://stackoverflow.com/a/4879146/9214620
cat << EOF >$VIRTUAL_ENV/bin/notebook.sh
#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter-lab --ip \$(hostname -f) --no-browser
EOF

chmod u+x $VIRTUAL_ENV/bin/notebook.sh

cd $PROJECT_DIR
pip install -e .
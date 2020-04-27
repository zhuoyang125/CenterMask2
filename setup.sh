pip install --upgrade pip
sudo apt-get install git
pip install cython pyyaml==5.1
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/zhuoyang125/CenterMask2.git
cd detectron2
python setup.py build develop
ls

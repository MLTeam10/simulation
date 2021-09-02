### Stop service with same port (Fabrizio's PC)
sudo docker stop bc

### Export path
export PYTHONPATH=$PYTHONPATH:/home/fabrizio/Workspace/CARLA_0.9.12/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg

### Launch Carla 
/home/fabrizio/Workspace/CARLA_0.9.12/CarlaUE4.sh -quality-level=Low

### Go to directory
cd /home/fabrizio/Workspace/CARLA_0.9.12/PythonAPI/test/

### Start simulation ('python' for CARLA 0.9.11, 'python3' for CARLA 0.9.12)
python3 test.py -n 20 -w 300 -k 82

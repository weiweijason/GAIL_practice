# Create directory structure
mkdir -p /root/.mujoco

# Download and extract MuJoCo 2.1.0
cd /root
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco

# Set environment variables
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210' >> ~/.bashrc
source ~/.bashrc

# Install additional dependencies
apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    xpra \
    xserver-xorg-dev \
    libglfw3-dev

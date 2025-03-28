# Step 1:

## install and setup Docker


###  Update your package list
`sudo apt-get update`

### Install required dependencies
`sudo apt-get install apt-transport-https ca-certificates curl software-properties-common`

###  Add Docker’s official GPG key
`curl -fsSL https://download.docker.com/linux/raspbian/gpg | sudo apt-key add -`

###  Set up the Docker stable repository
`echo "deb [arch=arm64] https://download.docker.com/linux/raspbian $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list`

###  Update apt and install Docker
`sudo apt-get update`
`sudo apt-get install docker-ce`

###  Verify Docker installation
`docker --version`

`sudo systemctl enable docker`
`sudo systemctl start docker`




# Step 2:

## install and setup the mongoDB service

`sudo docker pull mongo:4.0`

`sudo docker run -d --name project -p 27017:27017 -v ~/mongo_data:/data/db --restart unless-stopped mongo:4.0`



# Step 3:
## Create a new virtual environment and install requirements.txt

`python -m venv doorlock`
`source doorlock/bin/activate  `
`pip install -r requirements.txt`
`cd doorlock/files`

# Step 4:
## Register user

`source doorlock/bin/activate  `
`cd doorlock/files`
`python register.py`

Follow instructions in the console.

# Step 5: 
## Run the doorlock

`source doorlock/bin/activate  `
`cd doorlock/files`
`python mainstate.py`

# Usage:
Stand in front of the radar sensor. 
When green light is observed on the webcam, say passphrase. 
If voice authenticated, press s to capture face image. 
If face authenticated, door will be unlocked.




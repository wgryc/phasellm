# Tests

### Structure

#### Release Checklist

`release_checklist_code.py` contains manual tests for the release checklist.

#### E2E Tests

E2E style tests are used when network communication is required.

To run the E2E tests, run `python -m unittest discover tests/e2e`.

##### NOTE

You may want to disable testing local models due to resource constraints. If so, set the environment variable
`SKIP_LOCAL_MODELS` to `True`.

#### Unit Tests

Unit tests are for testing individual functions and when network communication is not required.

Can mock network communication if necessary.

To run the unit tests, run `python -m unittest discover tests/unit`.

#### Running tests on a VM

We use a Paperspace VM to run PhaseLLM tests.

##### VM Requirements

- GPU with >= 30GM VRAM
- 100GB disk (for model weights + packages + docker)

##### Connecting to VM

You need to create + add an SSH key to the VM.
1) Create a SSH key if you haven’t already.
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent 
2) Copy it into paperspace
https://console.paperspace.com/account/settings/ssh-keys 

##### Initial VM Setup (Already Done)

Install Docker
https://docs.docker.com/engine/install/ubuntu/

Get base python image.
```
docker pull python:3
```

Create a code directory in the VM.
```
mkdir code
```

##### New Tester Setup

Create code folder for repos + forks. Currently, there is a ‘garett’ folder for holding my repo fork. You may want to 
add a ‘wojciech’ folder for the master repo.
```
cd code
mkdir myname
```

Clone repo into your folder. Make sure you clone using https, otherwise you need to add an SSH key to the VM to access 
the repo.
```
cd myname
git clone https://github.com/...
cd phasellm
```

Create a virtual environment in the cloned repository.
```
python -m venv .env
```

##### Test Specific Setup

Ensure you are on the branch you want to test.
```
cd code/myname/phasellm
git fetch –all
git checkout -- track origin/your-branch
```

Create a testing bash script if you don’t already have one. Example below.
```
#!/bin/bash
export ANTHROPIC_API_KEY=key
export COHERE_API_KEY=key
export HUGGING_FACE_API_KEY=key
export OPENAI_API_KEY=key

cd phasellm
. .env/bin/activate
pip install -r requirements.txt
nohup python -m unittest discover -s tests -v > ../test.log &
cd ..
```

Run the bootstrap_tests.sh
```
sudo /bin/sh bootstrap_tests.sh
```

The tests run as a background process, so once you see the nohup message, uou can escape the script and follow the 
output.
`ctrl + c (or command + c on mac)` 
```
tail -f test.log
```

##### Useful Commands
Monitor the process resources and find process IDs
```
top
```

Kill a process
```
sudo kill process_id
```

Check why process was killed
```
dmesg | less
```

Check Nvidia GPU usage
```
nvidia-smi
```

Shut down VM
```
sudo shutdown now
```

##### Gotchas

The Paperspace VM turns off on its own after 1 hour, so watch out for that.
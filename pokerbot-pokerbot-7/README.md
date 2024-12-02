[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kHV1rFnK)
# PokerBot Group Project (Weeks 4-8)

This repository contains the skeleton code for the PokerBot project agent (server client). In this group project you will develop an agent that plays a simplified version of poker (Kuhn poker) against fellow students. We first explain the setup and then the assignment.


## Setup

_Clone the Client and Server Repositories_

Accept the assignment invitation in GitHub Classroom. This creates a new personal remote with the skeleton code for the PokerBot project. Start VSCode, and in a terminal window, navigate to the root directory where you want to download the assignment repository. Then clone the repository, `git clone git@github.com:tue-5ARA0-<YYYY-QQ>/<YOUR TEAM REPOSITORY HANDLE>.git poker-server-client`, where you insert the correct year, quartile and team repository.

In order to play games with your agent on your own computer, you'll need to install the poker server as well. Type `git clone git@github.com:tue-5ARA0/poker-server-backend.git poker-server-backend`. This will clone the server repository. In order to get the server up and running locally, please consult the readme in the `poker-server-backend` repository.


_Setup the Virtual Environment_

Open a new terminal in VSCode and create a new virtual environment by typing `conda env create -f environment.yml` (on non-windows systems, users should use the corresponding environment file). This command will create a new `pokerbot39` virtual environment and install required packages. Activate the newly created environment by `conda activate pokerbot39`.


_Install TensorFlow_

TensorFlow must be installed using pip. With the `pokerbot39` environment active, install TensorFlow using
```bash
pip install -r requirements.txt
```
Again if you're using a non-windows system then use the corresponding environment file instead.


_Start a Local Game_

Follow the intructions in the `poker-server-backend` [readme](https://github.com/tue-5ARA0/poker-server-backend) to start a local server. In the terminal that runs your local server you will see test player tokens. These tokens represent the agent ids that the server expects to connect with. An agent can be connected to a token on the local server by opening a _new_ terminal and running

```bash
python main.py --token <token UUID here> --play "random" --local
``` 

You'll need to repeat this procedure to connect a second agent using the second token in the second terminal. The game will start automatically once both agents are connected. The server waits only a limited amount of time for both agents to connect.

You can also play a local game against a bot (with this setting you need only one terminal):

```bash
python main.py --token <token UUID here> --play "bot" --local
```

_Start an Online Game_

We also have a public (cloud) server running for you to play online games against bots or agents of fellow students. At the start of the project you will receive a unique secret token that identifies your agent to the server (keep this token secret, otherwise internet hackers will steal all of your virtual money, and you won't be able to play online games).

In order to play an online game you need to specify a `--global` flag for the script and wait for your opponent to connect as well:

```bash
python main.py --token <token UUID here> --global --play "random" # or --play "bot"
```

You may also omit the `--token` argument if you store your secret token in a `token_key.txt` file in the same folder as the `main.py` script. 

In case if you want to play against a specific team, you can create a private game with the `--create` argument:

```bash
python main.py --token <host agent token UUID here> --global --create
```

The server will then respond with a private game coordinator token that you can share with your opponent, e.g.

```bash
id: "de2c20f1-c6b9-4536-8cb0-c5c5ac816634"
```

Your opponent can use this token to connect to your private game, e.g. (on the opponent's side):

```bash
> python main.py --token <opponent agent token UUID here> --play de2c20f1-c6b9-4536-8cb0-c5c5ac816634
```


_Name Your Agent_

Think of a fierce name for your agent and specify it with the `--rename` flag:

```bash
> python main.py --rename "<fierce name here>" --global 
```

Don't forget the `--global` flag - you want the whole world to thrill before your strenght, right?
After this command the updated name will then appear in the leaderboard once you start playing online games.


_Build and Train an Image Recognition Model_

For easy image recognition model testing, `main.py` provides two extra arguments for building and training your image recogntiion model. 

Use the `--build-model` flag to build an image recognition model:
```
> python main.py --build-model
```

Use the `--train-model 'n_validation'` command to train image recognition model with the specified `n_validation`:
```
> python main.py --train-model 100
```

Note: in order for this commands to work properly, first, you need to implement the `build_model()` and `train_model()` functions from the `model.py` file.

_Full List of Available Options_

For a full list of available options/arguments, use

```bash
python main.py --help
```

## Assignment

You'll need to implement an agent strategy and a card image classifier that recognizes cards dealt as images. The project will be graded on three aspects: _Software Engineering_, _Data Management_, and _Project Management_. In the first place we care about clean, correct, well-tested and well-motivated code; the performance of your agent is secondary. All group members need to contribute to the code base.

Details on the grading criteria can be found in the Rubric that is available on Canvas, together with a list of critical questions that verify whether your group is on the right track.


_Assignment Details_

The current mockup agent plays a random game, and does not yet recognize dealt cards. Your assignment is to equip the poker agent with a card image classifier and a betting strategy for games with _three_ cards, as well as games with _four_ cards. So you will need to write _two_ betting strategies.

For this, you'll need to implement several subroutines that are properly tested and documented. Some guidance is provided, but you'll need to be creative and implement any additional classes/modules/functionality yourself. You can choose your own machine learning toolbox (we suggest TensorFlow with Keras) and are free to modify/create files as you see fit. 

While it is not forbidden to inspect server communication code, we highly recommend that you don't modify `main.py`, `client\events.py` and `client\controller.py` in order to prevent connection errors. When playing with the online server, a stable internet connection is required (you lose immediately if the connection is severed).

For grading, we need insight in your development process. Make sure to motivate and document your process and key decisions. We will also try to run your agent ourselves. Therefore, ensure that the agent runs on a ``clean'' machine, and that your agent, dataset and model can be reproduced.

You are also required to record a video where you play a game with your bot (hand-in via Canvas). You can also use this video to further explain your approach and design choices.

We highly encourage you to play some online games before you finish the assignment (this might reveal some weak points in your agent implementation). If you want to join the (optional) tournament at the end, then prior online participation is mandatory.
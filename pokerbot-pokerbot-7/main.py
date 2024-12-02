from model import load_model, build_model, train_model
from agent import PokerAgent
from client.controller import Controller

import argparse

parser = argparse.ArgumentParser(description = 'Poker client CLI, with no arguments provided plays a local random game with token from '
                                               '"token_key.txt" file')

parser.add_argument('--build-model', help = 'Build image recongition model and exit', action = 'store_true', default = False)
parser.add_argument('--train-model', help = 'Train image recongition model with the specified `n_validation` and `write_to_file = True` and exit', type = int, default = -1, metavar = 'n_validation')
parser.add_argument('--token', help = 'Player\'s token', default = None)
parser.add_argument('--play', help = 'Existing game id, \'random\' for random match with real player, or \'bot\' to play against bot', default = 'random')
parser.add_argument('--cards', help = 'Number of cards used in a game', choices=[ '3', '4' ], default = '3', type = str)
parser.add_argument('--create', action = 'store_true', help = 'Create a new game', default = False)
parser.add_argument('--local', dest = 'server_local', action = 'store_true', help = 'Connect to a local server', default = False)
parser.add_argument('--global', dest = 'server_global', action = 'store_true', help = 'Connect to a default global server', default = False)
parser.add_argument('--server', help = 'Connect to a particular server')
parser.add_argument('--rename', help = 'Rename player', type = str)


def __main__():
    args = parser.parse_args()

    if args.build_model is True:
        print("Building model")
        build_model()
        print("Building model has been completed")
        return 
    
    if args.train_model > 0:
        print("Training model")
        train_model(load_model(), args.train_model, write_to_file = True)
        print("Training model has been completed")
        return

    token = args.token

    if token is None:
        try:
            with open("token_key.txt", "r") as f:
                token = f.read(36)
        except FileNotFoundError:
            print('Token has not been specified. Either create a `token_key.txt` file or use `--token` CLI argument.')
            return
        except Exception:
            print('Error reading token from `token_key.txt` file. Ensure that token has a valid UUID structure and has not extra spaces before and after the token.')
            return

    server_address = 'localhost:50051'
    if args.server_local is True:
        server_address = 'localhost:50051'
    elif args.server is not None:
        server_address = args['server']
    elif args.server_global is True and args.server is None:
        try:
            with open("server_address.txt", "r") as f:
                server_address = f.readline()
        except FileNotFoundError:
            server_address = '51.159.25.188:50051'
        except Exception:
            print('Error fetching global server address from `server_address.txt` file. Ensure that server address file contains only one line, which represents a valid URL.')
            return

    client = Controller(token, server_address)

    if args.rename:
        print(client.rename(args.rename))
    elif args.create:
        print(client.create(args.cards))
    else:
        client.play(args.play, args.cards, lambda: PokerAgent())

if __name__ == '__main__':
    __main__()

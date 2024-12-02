import random
from model import load_model, identify
from client.state import ClientGameRoundState, ClientGameState

class PokerAgent(object):
    def __init__(self):
        self.model = load_model()
        if self.model is None:
            raise ValueError("Failed to load the model. Ensure the model file exists and is correctly loaded.")
        
        # Initialize tracking for opponent's behaviors and game state variables
        self.opponent_tendencies = {'bet': 0, 'check': 0, 'fold': 0, 'call': 0, 'total': 0}
        self.round_count = 0
        self.total_bank = 0
        self.is_4_card_game = False
        self.current_card = None

    def make_action(self, state: ClientGameState, round: ClientGameRoundState) -> str:
        available_actions = round.get_available_actions()
        card = self.current_card
        opponent_moves = round.get_moves_history()

        # Handle unidentified card scenario
        if not card:
            card = '?'

        print(f"Available actions: {available_actions}, Card: {card}")

        # Dynamically detect if it's a 4-card game
        if card == 'A':
            self.is_4_card_game = True

        # Use different strategies for 3-card and 4-card games
        if self.is_4_card_game:
            action = self._make_4_card_strategy(card, available_actions, opponent_moves)
        else:
            action = self._make_3_card_strategy(card, available_actions, opponent_moves)

        # Ensure the action is valid
        if action not in available_actions:
            print(f"Invalid action: {action}. Choosing a valid action.")
            action = self._fallback_action(available_actions, card)  # Improved fallback mechanism

        return action

    def _make_3_card_strategy(self, card, available_actions, opponent_moves):
        # Strategy logic for 3-card poker based on the identified card
        if card == 'J':
            if 'FOLD' in available_actions:
                if self._opponent_is_tight(opponent_moves) and random.random() < 0.2:
                    return 'BET'
                return 'FOLD'
            return 'CHECK'
        elif card == 'Q':
            if 'BET' in available_actions:
                if self._opponent_is_aggressive(opponent_moves):
                    return 'CHECK'
                return 'BET'
            return 'CHECK'
        elif card == 'K':
            return 'BET' if 'BET' in available_actions else 'CHECK'
        else:
            return random.choice(available_actions)

    def _make_4_card_strategy(self, card, available_actions, opponent_moves):
        # Adjusted strategy for games involving an Ace card
        if card == 'J':
            if 'FOLD' in available_actions:
                if self._opponent_is_tight(opponent_moves) and random.random() < 0.2:
                    return 'BET'
                return 'FOLD'
            return 'CHECK'
        elif card == 'Q':
            if 'BET' in available_actions:
                return 'BET' if not self._opponent_is_aggressive(opponent_moves) else 'CHECK'
            return 'CHECK'
        elif card == 'K':
            return 'BET' if 'BET' in available_actions else 'CALL' if 'CALL' in available_actions else 'CHECK'
        elif card == 'A':
            if 'BET' in available_actions:
                return 'BET'
            elif 'CALL' in available_actions:
                return 'CALL'
            elif 'CHECK' in available_actions:
                return 'CHECK'
        else:
            return random.choice(available_actions)

    def _fallback_action(self, available_actions, card):
        # Fallback action for any unexpected situations
        if card == 'A' or card == 'K':
            return 'CALL' if 'CALL' in available_actions else 'CHECK'
        elif card == 'J':
            return 'FOLD' if 'FOLD' in available_actions else 'CHECK'
        else:
            return random.choice(available_actions)

    def _opponent_is_tight(self, opponent_moves):
        # Check if opponent tends to fold or check often
        if not opponent_moves:
            return False
        total_moves = self.opponent_tendencies['total']
        fold_ratio = self.opponent_tendencies['fold'] / total_moves if total_moves > 0 else 0
        return fold_ratio > 0.6

    def _opponent_is_aggressive(self, opponent_moves):
        # Check if opponent tends to bet frequently
        if not opponent_moves:
            return False
        total_moves = self.opponent_tendencies['total']
        bet_ratio = self.opponent_tendencies['bet'] / total_moves if total_moves > 0 else 0
        return bet_ratio > 0.5

    def _opponent_is_extremely_aggressive(self, opponent_moves):
        # Extremely aggressive detection threshold
        if not opponent_moves:
            return False
        total_moves = self.opponent_tendencies['total']
        bet_ratio = self.opponent_tendencies['bet'] / total_moves if total_moves > 0 else 0
        return bet_ratio > 0.8

    def on_image(self, image):
        # Image recognition to identify the card from the given image
        if self.model is None:
            raise ValueError("Model not loaded properly. Cannot perform card recognition.")
        
        recognized_card = identify(image, self.model)
        self.current_card = recognized_card
        print(f'Identified card: {recognized_card}')

    def on_error(self, error):
        # Handle any errors during game play
        print(f'Error: {error}')

    def on_game_start(self):
        # Setup initial game state at the start
        print("Game has started.")
        self.round_count = 0
        self.opponent_tendencies = {'bet': 0, 'check': 0, 'fold': 0, 'call': 0, 'total': 0}
        self.total_bank = 0
        self.is_4_card_game = False

    def on_new_round_request(self, state: ClientGameState):
        # Increment round count and reset card data
        self.round_count += 1
        print(f"Round {self.round_count} starting...")
        self.current_card = None

    def on_round_end(self, state: ClientGameState, round: ClientGameRoundState):
        # Log the results of the round and update opponent tendencies
        opponent_moves = round.get_moves_history()
        if opponent_moves:
            last_opponent_move = opponent_moves[-1]
            if last_opponent_move == 'BET':
                self.opponent_tendencies['bet'] += 1
            elif last_opponent_move == 'CHECK':
                self.opponent_tendencies['check'] += 1
            elif last_opponent_move == 'FOLD':
                self.opponent_tendencies['fold'] += 1
            elif last_opponent_move == 'CALL':
                self.opponent_tendencies['call'] += 1
            self.opponent_tendencies['total'] += 1

        self.total_bank = state.get_player_bank()

        print(f'----- Round {round.get_round_id()} results -----')
        print(f'  Your card       : {self.current_card}')
        print(f'  Your turn order : {round.get_turn_order()}')
        print(f'  Moves history   : {round.get_moves_history()}')
        print(f'  Your outcome    : {round.get_outcome()}')
        print(f'  Current bank    : {self.total_bank}')
        print(f'  Show-down       : {round.get_cards()}')

    def on_game_end(self, state: ClientGameState, result: str):
        # Announce the final results of the game
        print(f'----- Game results -----')
        print(f'  Outcome:    {result}')
        print(f'  Final bank: {self.total_bank}')

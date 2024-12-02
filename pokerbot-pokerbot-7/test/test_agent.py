import pytest
from unittest.mock import Mock
from agent import PokerAgent
from client.state import ClientGameState, ClientGameRoundState

class TestPokerAgent:
    @pytest.fixture
    def agent(self):
        return PokerAgent()

    def test_make_action_with_3_cards(self, agent):
        # Mocking the game and round state
        mock_game_state = Mock(spec=ClientGameState)
        mock_round_state = Mock(spec=ClientGameRoundState)

        # Simulate a 3-card game with the card 'Q' and available actions
        mock_round_state.get_card.return_value = 'Q'
        mock_round_state.get_available_actions.return_value = ['CHECK', 'BET']
        mock_round_state.get_moves_history.return_value = ['CHECK', 'BET']

        # Call the make_action function
        action = agent.make_action(mock_game_state, mock_round_state)

        # Check if the action is valid (either 'CHECK' or 'BET')
        assert action in ['CHECK', 'BET'], f"Invalid action: {action}"

    def test_make_action_with_4_cards(self, agent):
        # Mocking the game and round state
        mock_game_state = Mock(spec=ClientGameState)
        mock_round_state = Mock(spec=ClientGameRoundState)

        # Simulate a 4-card game with the card 'A' and available actions
        agent.is_4_card_game = True  # Force a 4-card game
        mock_round_state.get_card.return_value = 'A'
        mock_round_state.get_available_actions.return_value = ['CHECK', 'BET']
        mock_round_state.get_moves_history.return_value = ['CHECK', 'FOLD']

        # Call the make_action function
        action = agent.make_action(mock_game_state, mock_round_state)

        # Check if the action is valid (either 'CHECK' or 'BET')
        assert action in ['CHECK', 'BET'], f"Invalid action: {action}"

    def test_bluffing_decision(self, agent):
        # Mocking the opponent's move history
        mock_opponent_moves = ['FOLD', 'CHECK']

        # Test bluffing decision logic
        bluff_chance = 0.3
        action = agent._make_bluffing_decision(mock_opponent_moves, bluff_chance)

        # Since bluff_chance is low, it should mostly return 'CHECK'
        assert action in ['CHECK', 'BET'], f"Invalid action: {action}"

    def test_opponent_is_tight(self, agent):
        # Simulate opponent folding frequently
        agent.opponent_tendencies['fold'] = 5
        agent.opponent_tendencies['total'] = 8

        # Check if the opponent is considered 'tight'
        assert agent._opponent_is_tight(['FOLD', 'CHECK']) == True

    def test_opponent_is_aggressive(self, agent):
        # Simulate opponent betting frequently
        agent.opponent_tendencies['bet'] = 6
        agent.opponent_tendencies['total'] = 8

        # Check if the opponent is considered 'aggressive'
        assert agent._opponent_is_aggressive(['BET', 'BET']) == True

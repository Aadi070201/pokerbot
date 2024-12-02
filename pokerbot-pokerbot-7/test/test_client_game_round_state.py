import pytest
from client.state import ClientGameRoundState

@pytest.fixture
def round_state():
    return ClientGameRoundState("coordinator_123", 1)

def test_set_and_get_card(round_state):
    round_state.set_card('K')
    assert round_state.get_card() == 'K'

def test_set_and_get_turn_order(round_state):
    round_state.set_turn_order(1)
    assert round_state.get_turn_order() == 1

def test_add_and_get_moves_history(round_state):
    round_state.add_move_history('BET')
    assert round_state.get_moves_history() == ['BET']

def test_set_and_get_available_actions(round_state):
    round_state.set_available_actions(['BET', 'CHECK'])
    assert round_state.get_available_actions() == ['BET', 'CHECK']

def test_is_ended(round_state):
    round_state.set_outcome(50)
    assert round_state.is_ended() is True

def test_get_outcome(round_state):
    round_state.set_outcome(100)
    assert round_state.get_outcome() == 100

def test_set_and_get_cards(round_state):
    round_state.set_cards('KQ')
    assert round_state.get_cards() == 'KQ'

import pytest
from client.state import ClientGameState, ClientGameRoundState

@pytest.fixture
def game_state():
    return ClientGameState("coordinator_123", "player_token_456", 1000)

def test_get_coordinator_id(game_state):
    assert game_state.get_coordinator_id() == "coordinator_123"

def test_get_player_token(game_state):
    assert game_state.get_player_token() == "player_token_456"

def test_get_player_bank(game_state):
    assert game_state.get_player_bank() == 1000

def test_update_bank(game_state):
    game_state.update_bank(500)
    assert game_state.get_player_bank() == 1500

def test_start_new_round(game_state):
    # Start a new round
    game_state.start_new_round()
    
    # Get the newly created round (without filtering)
    all_rounds = game_state._rounds  # Access the raw _rounds attribute directly
    assert len(all_rounds) == 1  # We expect one round to exist
    assert isinstance(all_rounds[0], ClientGameRoundState)
    assert all_rounds[0].get_round_id() == 1  # First round, so ID should be 1

def test_get_last_round_state(game_state):
    # Start two new rounds
    game_state.start_new_round()
    game_state.start_new_round()
    
    # Access the last round
    last_round = game_state.get_last_round_state()
    assert isinstance(last_round, ClientGameRoundState)
    assert last_round.get_round_id() == 2

def test_get_rounds_after_moves(game_state):
    # Start a new round and simulate moves
    game_state.start_new_round()
    last_round = game_state.get_last_round_state()
    
    # Add moves to the round
    last_round.add_move_history('CHECK')
    
    # Now, get_rounds() should return the round since it has a move
    rounds = game_state.get_rounds()
    assert len(rounds) == 1
    assert isinstance(rounds[0], ClientGameRoundState)
    assert rounds[0].get_round_id() == 1

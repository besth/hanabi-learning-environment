import numpy as np

class ObservationAdapter(object):

  def __init__(self, environment, run_experiment_module):
    ### used for debug:
    self.debug_obs = None # placeholder for observation
    self.debug_other_obs = None
    # ^^ set these with agent.obs.debug_obs

    ### testing pretrained mdp with full knowledge plus some error
    self.debug_with_nearly_perfect_knowledge = False
    # [0-1] add bits from card knowledge with this chance
    # 0 is same as qmdp, 1 is same as debug_full_obs
    self.ERROR_RATE = 0.25

    game = environment.game
    ## load/calculate params from game for low level belief
    self.players = game.num_players()
    self.handsize = game.hand_size()

    self.colors = game.num_colors()
    self.ranks = game.num_ranks()
    self.infotokens = game.max_information_tokens()
    self.lifetokens = game.max_life_tokens()

    self.maxdeck = 0 # should be 50
    for card_i in range(self.colors):
        for rank_i in range(self.ranks):
            self.maxdeck += game.num_cards(card_i, rank_i)

    self.cardbits = self.colors * self.ranks # should be 25

    self.hands_bits = ((self.players-1) * self.handsize * self.cardbits + self.players)
    self.deck_bits = self.hands_bits + (
                      self.maxdeck - self.players * self.handsize )
    self.fireworks_bits = self.deck_bits + (
                      self.colors * self.ranks )
    self.info_tokens_bits = self.fireworks_bits + (
                      self.infotokens )
    self.life_tokens_bits = self.info_tokens_bits + (
                      self.lifetokens )
    self.board_bits = self.life_tokens_bits
    self.discard_bits = self.board_bits + self.maxdeck
    self.action_bits = self.discard_bits + (self.players +
                                        4 +
                                        self.players +
                                        self.colors +
                                        self.ranks +
                                        self.handsize +
                                        self.handsize +
                                        self.cardbits
                                        + 2)
    self.knowledge_bits = (self.action_bits +
                           self.players * self.handsize *
                            (self.cardbits + self.colors + self.ranks))
    assert self.knowledge_bits == environment.vectorized_observation_shape()[0]

    # in case we need back compatibility with old full observation vector
    REDUNDANT = False
    if (REDUNDANT):
      self.num_knowledge_bits = (self.players * self.handsize *
                                (self.cardbits + self.colors + self.ranks))
    else:
      self.num_knowledge_bits = (self.players * self.handsize *
                                (self.cardbits))

    self.low_rank_end = 3
    self.mid_rank_end = 9
    self.cards_per_color = 10 # = 3 + 2 + 2 + 2 + 1

    self.card_totals = np.zeros((self.colors, self.ranks))
    self.card_totals[:,0] = 3
    self.card_totals[:,1:4] = 2
    self.card_totals[:,-1] = 1

    obs_size = (environment.vectorized_observation_shape()[0]
                #no hands, no actions
                - (self.hands_bits + (self.action_bits - self.discard_bits)
                   #no reveal hist section in knowledge
                   + (self.players * self.handsize * (self.colors + self.ranks))
                  )
               )

    self.obs_stacker = run_experiment_module.ObservationStacker(history_size=1,
                         observation_size = obs_size,
                         num_players = environment.players)

  ##-----------------------
  # access regular (partially observed) observation vector components
  ##-----------------------
  def hand_section(self, obs):
    '''
    hands portion of normal hanabi observation vector
    INcluding bits telling you if player has less than a full hand
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to hands ("by reference")
    '''
    return obs[:self.hands_bits]

  def hands(self, obs):
    '''
    ACTUAL hands portion of normal hanabi observation vector
    EXcluding bits telling you if player has less than a full hand
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to hands ("by reference")
    '''
    return obs[:self.hands_bits-self.players]

  def deck(self, obs):
    '''
    board's deck portion of normal hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to deck ("by reference")
    '''
    return obs[self.hands_bits:self.deck_bits]

  def fireworks(self, obs):
    '''
    board's fireworks portion of normal hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to fireworks ("by reference")
    '''
    return obs[self.deck_bits:self.fireworks_bits]

  def info_tokens(self, obs):
    '''
    board's info tokens portion of normal hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to info tokens ("by reference")
    '''
    return obs[self.fireworks_bits:self.info_tokens_bits]

  def life_tokens(self, obs):
    '''
    board's life tokens portion of normal hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to life tokens ("by reference")
    '''
    return obs[self.info_tokens_bits:self.life_tokens_bits]

  def board(self, obs):
    '''
    board portion of normal hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to board ("by reference")
    '''
    return obs[self.hands_bits:self.board_bits]

  def discard(self, obs):
    '''
    discard portion of normal hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to discard ("by reference")
    '''
    return obs[board_bits:discard_bits]

  def action(self, obs):
    '''
    action portion of normal hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to action ("by reference")
    '''
    return obs[discard_bits:action_bits]

  def knowledge(self, obs):
    '''
    knowledge portion of normal hanabi
    includes a small record of which rank/cards were hinted
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to knowledge ("by reference")
    '''
    return obs[self.action_bits:]

  def card_knowledge(self, obs):
    '''
    CARD HINT knowledge portion of normal hanabi
    EXcludes a small record of which rank/cards were hinted
    obs: a 1d np array containing vectorized observation
    returns: a 1d array slice of obs corresponding to knowledge ("by reference")
             of shape: (num_players, handsize, colors, ranks)
    '''
    knowledge = self.knowledge(obs)
    knowledge = knowledge.reshape(self.players,
                                  self.handsize,
                                  self.cardbits +
                                    self.colors +
                                    self.ranks)
    knowledge = knowledge[:,:,:self.cardbits]
    return knowledge

  ##-----------------------
  # access mdp ("fully observed") observation vector components
  ##-----------------------
  def mdp_hands(self, obs):
    '''
    hands portion of fully observed hanabi
    obs: a 1d array or list containing vectorized observation
    returns: a 1d array slice of obs corresponding to hands ("by reference")
    '''
    return obs[-self.num_knowledge_bits:]

  ##-----------------------
  # reshaping
  ##-----------------------
  def reshape_hands(self, hand):
    '''
    hand: a 1d array or list containing vectorized observation hand
      hand should be an array of (total) length
      num_players x handsize x colors x ranks
    num_players: default number of players is the players in the game - 1
      (same as the observation hands)
    returns: a multi dim array slice of hand ("by reference")
    '''
    res = hand.reshape(-1, self.handsize, self.colors, self.ranks)
    return res

  ##-----------------------
  # conversion from partial obs to full obs sections
  ##-----------------------
  #def extract_knowledge(self, knowledge_obs, current_player=True):
  #  '''
  #  knowledge_obs needs to be just the knowledge part of the obs vector
  #  return (by reference) a slice of obs corresponding to knowledge
  #  '''
  #  knowledge = knowledge_obs.reshape(self.players,
  #                                    self.handsize,
  #                                    self.cardbits +
  #                                      self.colors +
  #                                      self.ranks)
  #  # discard other player knowledge, and reveal history
  #  # (see hanabi_lib/canonical_encoders.cc)
  #  # what's left is our cards
  #  if (current_player):
  #    knowledge = self.reshape_hand(knowledge[0,:,:self.cardbits])
  #  else:
  #    knowledge = self.reshape_hand(knowledge[1:,:,:self.cardbits],
  #                                  num_players = self.players-1)
  #  return knowledge

  ##-----------------------
  # observation vector functions
  ##-----------------------
  def full_obs_vector(self, observation):
    '''
    returns the fully observable vector components with:
    - Board
      - Deck
      - Fireworks
      - Info tokens
      - Life tokens
    - Discards
    - Hands (or if not fully observed, card knowledge)
      - (without bits indicating which hints were given)
    '''
    knowledge = np.array(observation[self.action_bits:])
    knowledge = knowledge.reshape(self.players,
                                  self.handsize,
                                  self.cardbits + self.colors + self.ranks)
    no_reveal_hist = knowledge[:,:,:self.cardbits].flatten()
    knowledge_len = no_reveal_hist.shape[0]
    mdp_observation_vector = np.concatenate((
                       observation[self.hands_bits:self.discard_bits],
                       no_reveal_hist))

    return mdp_observation_vector

  def nearly_perfect_knowledge(self, knowledge, other_obs, error_rate):
      '''
      other_obs: other player_observation
      '''
      nearly_perfect = np.zeros_like(knowledge)
      # add error bits
      nearly_perfect_mask = (knowledge >= 1)
      error_bits = (np.random.rand(np.sum(nearly_perfect_mask)) < error_rate)
      nearly_perfect[nearly_perfect_mask] = error_bits
      # fill in true hand (using other player's observation)
      other_observation = other_obs['vectorized']
      true_hand = np.array(other_observation[:self.hands_bits-2]).reshape(
                           self.players-1, self.handsize, self.colors, self.ranks)
      nearly_perfect |= true_hand[0,...]
      return nearly_perfect

  ##-----------------------
  # card counting
  ##-----------------------
  def individual_card_counts(self, obs):
    # TODO: unit test this
    '''
    from parse_observation:
    obs = current_player_observation['vectorized']
    '''
    hand    = obs[  :self.hands_bits - self.players]
    board   = obs[self.hands_bits  :self.board_bits]
    discard = obs[self.board_bits  :self.discard_bits]

    ## hands
    hands = np.array(hand).reshape(self.players-1, self.handsize,
                                    self.colors, self.ranks)
    cards_in_hands = np.sum(hands, axis = (0,1))

    ## board
    deck      = board[:self.maxdeck - self.players * self.handsize]
    fireworks = board[len(deck):len(deck) + self.colors * self.ranks]

    cards_remaining = sum(deck)
    cards_played = np.array(fireworks).reshape(self.colors,self.ranks)

    ## discard
    # each color in discard looks like this:
    # lll      h
    # 1100011101
    discarded = np.array(discard).reshape(self.colors,self.cards_per_color)
    cards_discarded = np.zeros((self.colors,self.ranks))
    # total low (3 of each)
    cards_discarded[:,0] = np.sum(discarded[:,:self.low_rank_end], axis=1)
    # total mid (2 of each)
    cards_discarded[:,1:4] = (  discarded[:,self.low_rank_end:
                                            self.mid_rank_end:2] +
                                discarded[:,self.low_rank_end+1:
                                            self.mid_rank_end+1:2])
    # total high (1 of each)
    cards_discarded[:,-1] = discarded[:,-1]

    cards_gone = cards_played + cards_discarded + cards_in_hands

    card_counts = (self.card_totals - cards_gone).astype(float)
    #prob = card_counts / (self.maxdeck - np.sum(cards_gone))
    #return prob
    return card_counts

  def card_counts(self, obs, other_obs=None):
    '''
    from parse_observation:
    obs = current_player_observation['vectorized']
    '''
    counts = self.individual_card_counts(obs)
    #knowledge = self.extract_knowledge(obs[self.action_bits:])
    knowledge = self.reshape_hands(self.card_knowledge(obs))[0]
    # knowledge is a bitmask representing potential cards

    if (self.debug_with_nearly_perfect_knowledge and
        self.debug_other_obs is not None):
      knowledge = self.nearly_perfect_knowledge(self.reshape_hands(knowledge),
                            self.debug_other_obs, self.ERROR_RATE)

    all_counts = counts * knowledge
    #if self.debug_obs['deck_size'] < 2:
    #  pdb.set_trace()
    return all_counts

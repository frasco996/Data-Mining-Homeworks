import random

def shuffle_deck():
    """Returns a shuffled standard deck of cards."""
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    deck = [{'rank': rank, 'suit': suit} for rank in ranks for suit in suits]
    random.shuffle(deck)
    return deck


##The first two cards include at least one ace
def check_event_a(deck):
    """Checks if the first two cards include at least one ace."""
    first_two_cards = deck[:2]
    return any(card['rank'] == 'A' for card in first_two_cards)

def calculate_probability_event_a(num_simulations):
    """Calculates the probability of the first two cards including at least one ace."""
    favorable_outcomes = 0

    for _ in range(num_simulations):
        deck = shuffle_deck()
        if check_event_a(deck):
            favorable_outcomes += 1

    probability_event_a = favorable_outcomes / num_simulations
    return probability_event_a

##The first five cards include at least one ace
def check_event_b(deck):

    first_five_cards = deck[:5]
    return any(card['rank'] == 'A' for card in first_five_cards)

def calculate_probability_event_b(num_simulations):

    favorable_outcomes = 0

    for _ in range(num_simulations):
        deck = shuffle_deck()
        if check_event_b(deck):
            favorable_outcomes += 1

    probability_event_b = favorable_outcomes / num_simulations
    return probability_event_b

##The first two cards are a pair of the same rank
def check_event_c(deck):

    first_two_cards = deck[:2]

    return first_two_cards[0]['rank'] == first_two_cards[1]['rank']

def calculate_probability_event_c(num_simulations):

    favorable_outcomes = 0

    for _ in range(num_simulations):
        deck = shuffle_deck()
        if check_event_c(deck):
            favorable_outcomes += 1

    probability_event_c = favorable_outcomes / num_simulations
    return probability_event_c

##The first five cards are all diamonds
def check_event_d(deck):

    first_five_cards = deck[:5]
    return all(card['suit'] == 'Diamonds' for card in first_five_cards)

def calculate_probability_event_d(num_simulations):

    favorable_outcomes = 0

    for _ in range(num_simulations):
        deck = shuffle_deck()
        if check_event_d(deck):
            favorable_outcomes += 1

    probability_event_d = favorable_outcomes / num_simulations
    return probability_event_d

##The first five cards form a full house
def check_event_e(deck):
    l1=0
    c=0
    d=0
    l2=0
    first_five_cards = deck[:5]
    for card in first_five_cards:
        if l1 == 0 or l1 == card['rank']:
            l1 = card['rank']
            c+=1
        elif l2 == 0 or l2 == card['rank']:
            l2 = card['rank']
            d+=1
    if (d == 3 and c == 2) or (d == 2 and c == 3):
        return True
    else:
        return False


def calculate_probability_event_e(num_simulations):

    favorable_outcomes = 0

    for _ in range(num_simulations):
        deck = shuffle_deck()
        if check_event_e(deck):
            favorable_outcomes += 1

    probability_event_e = favorable_outcomes / num_simulations
    return probability_event_e

# Number of simulations to estimate the probability
num_simulations = 100000
print("Wait a few seconds...")
# Calculate the probability of event (a) using random sampling
probability_event_a = calculate_probability_event_a(num_simulations)

probability_event_b = calculate_probability_event_b(num_simulations)

probability_event_c = calculate_probability_event_c(num_simulations)

probability_event_d = calculate_probability_event_d(num_simulations)

probability_event_e = calculate_probability_event_e(num_simulations)



print(f"Estimated Probability of The first two cards include at least one ace: {probability_event_a:.6f}")
print(f"Estimated Probability of The first five cards include at least one ace: {probability_event_b:.6f}")
print(f"Estimated Probability of The first two cards are a pair of the same rank : {probability_event_c:.6f}")
print(f"Estimated Probability of The first five cards are all diamonds : {probability_event_d:.6f}")
print(f"Estimated Probability of The first five cards form a full house  : {probability_event_e:.6f}")
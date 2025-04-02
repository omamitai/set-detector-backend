from itertools import combinations

def is_set(cards):
    """
    Determine if a group of cards forms a valid set.
    For each feature (Count, Color, Fill, Shape), the values must be either all the same or all different.

    Args:
        cards (list of dict): List of card feature dictionaries.

    Returns:
        bool: True if the cards form a valid set, False otherwise.
    """
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        values = {card[feature] for card in cards}
        if len(values) not in [1, 3]:
            return False
    return True

def find_sets(card_df):
    """
    Find all valid sets from the card DataFrame by checking every combination of three cards.

    Args:
        card_df (pandas.DataFrame): DataFrame containing card features.

    Returns:
        list: List of dictionaries with set details.
    """
    sets_found = []
    card_combinations = combinations(card_df.iterrows(), 3)
    for combo in card_combinations:
        cards = [entry[1] for entry in combo]
        if is_set(cards):
            set_info = {
                'set_indices': [entry[0] for entry in combo],
                'cards': [{feature: card[feature] for feature in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']} for card in cards]
            }
            sets_found.append(set_info)
    return sets_found

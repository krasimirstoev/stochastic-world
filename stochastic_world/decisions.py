class WeightedRandomDecision:
    """Random decisions whose weights are modified by needs, memory and place."""

    def choose_action(self, person, rng, location):
        memory = person.aggregate_memory()
        weights = {
            "move": 6.0, "work": 24.0, "scavenge": 11.0, "buy_supplies": 9.0,
            "rest": 13.0, "heal": 4.0, "repair": 4.0, "help": 8.0,
            "steal": 4.0, "attack": 1.0, "idle": 16.0,
        }
        if person.food <= 3:
            weights["scavenge"] *= 2.6; weights["buy_supplies"] *= 3.0; weights["steal"] *= 2.0; weights["move"] *= 1.5
        if person.medicine == 0 and person.health < 75:
            weights["buy_supplies"] *= 2.8; weights["move"] *= 1.6
        if person.energy <= 30:
            weights["rest"] *= 4.0; weights["work"] *= .35; weights["move"] *= .4; weights["attack"] *= .5
        if person.health < 70:
            weights["heal"] *= 4.0 if person.medicine else .1; weights["attack"] *= .5
        if person.shelter < 45:
            weights["repair"] *= 4.0
        if person.money < 4:
            weights["work"] *= 1.8; weights["buy_supplies"] *= .35; weights["move"] *= 1.3
        if person.employer_id is None:
            weights["work"] *= 1.65
        if location.kind == "industrial":
            weights["work"] *= 1.25
        if location.kind == "outskirts":
            weights["scavenge"] *= 1.5

        if memory["positive_ties"]:
            weights["help"] *= 1 + min(2.0, memory["positive_ties"] / 4)
        if memory["hostile_ties"]:
            weights["attack"] *= 1 + memory["max_conflict"] / 12
            weights["steal"] *= 1 + memory["max_conflict"] / 60
        if memory["mean_affinity"] > 10:
            weights["help"] *= 1.4; weights["attack"] *= .65
        elif memory["mean_affinity"] < -10:
            weights["help"] *= .6; weights["attack"] *= 1.5

        return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]

import unittest

from stochastic_world.person import Person


class MemoryTests(unittest.TestCase):
    def test_witnessed_attack_harms_reputation_without_direct_victim_memory(self):
        witness = Person(1, "Witness")
        attacker = Person(2, "Attacker")
        witness.observe(attacker, day=3, action="attack", magnitude=1.5)
        memory = witness.memory_of(attacker)
        self.assertEqual(memory.observed_attack, 1)
        self.assertLess(memory.trust, 0)
        self.assertGreater(memory.grievance, 0)
        self.assertEqual(memory.attacks_suffered, 0)

    def test_witnessed_help_improves_reputation(self):
        witness = Person(1, "Witness")
        helper = Person(2, "Helper")
        witness.observe(helper, day=2, action="help", magnitude=1)
        self.assertGreater(witness.memory_of(helper).trust, 0)


if __name__ == "__main__":
    unittest.main()

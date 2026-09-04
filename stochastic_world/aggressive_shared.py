"""Built-in shared-memory transport for the opt-in aggressive agent engine.

The hot planner input, local-state output and fixed-width action intents live in
POSIX shared memory. Worker queues therefore carry only compact pid lists and
small control metadata instead of thousands of Python result tuples.
"""

from multiprocessing import shared_memory
import struct


_LOCATION_KINDS = ("residential", "market", "industrial", "clinic", "outskirts", "logistics", "new_venture")
_LOCATION_TO_CODE = {name: index for index, name in enumerate(_LOCATION_KINDS)}

_ACTIONS = ("", "move", "work", "buy_supplies", "help", "steal", "attack")
_ACTION_TO_CODE = {name: index for index, name in enumerate(_ACTIONS) if name}
_RESOURCES = (None, "food", "medicine", "money")
_RESOURCE_TO_CODE = {name: index for index, name in enumerate(_RESOURCES)}

# snapshot excluding pid (the record index is the pid):
# location, food, medicine, energy, health, shelter, money, has_employer,
# location_kind, positive_ties, hostile_ties, max_conflict, mean_affinity,
# scavenge_food_max, medicine_chance, is_working_age
_INPUT = struct.Struct("<iiiiiidBBiiddidB")
# food, medicine, energy, health, shelter, money
_STATE = struct.Struct("<iiiiid")
# kind, action, target, resource, value0..value4, witness_count
# Kinds: 0 none, 1 generic shared, 2 social, 3 prepared move,
# 4 prepared work, 5 prepared purchase.
_INTENT = struct.Struct("<BBiBdddddB")


class SharedAgentBuffers:
    """Shared fixed-size planner buffers owned by the main process."""

    def __init__(self, population, actions_per_day, max_witnesses, *, descriptor=None):
        if descriptor is None:
            self.population = max(1, int(population))
            self.actions_per_day = max(1, int(actions_per_day))
            self.max_witnesses = max(0, int(max_witnesses))
            self.intent_record_size = _INTENT.size + 4 * self.max_witnesses
            self._input = shared_memory.SharedMemory(create=True, size=self.population * _INPUT.size)
            self._state = shared_memory.SharedMemory(create=True, size=self.population * _STATE.size)
            self._intent = shared_memory.SharedMemory(
                create=True,
                size=self.population * self.actions_per_day * self.intent_record_size,
            )
            self._owner = True
        else:
            self.population = int(descriptor["population"])
            self.actions_per_day = int(descriptor["actions_per_day"])
            self.max_witnesses = int(descriptor["max_witnesses"])
            self.intent_record_size = int(descriptor["intent_record_size"])
            self._input = shared_memory.SharedMemory(name=descriptor["input_name"])
            self._state = shared_memory.SharedMemory(name=descriptor["state_name"])
            self._intent = shared_memory.SharedMemory(name=descriptor["intent_name"])
            self._owner = False

    @classmethod
    def attach(cls, descriptor):
        return cls(1, 1, 0, descriptor=descriptor)

    @property
    def descriptor(self):
        return {
            "population": self.population,
            "actions_per_day": self.actions_per_day,
            "max_witnesses": self.max_witnesses,
            "intent_record_size": self.intent_record_size,
            "input_name": self._input.name,
            "state_name": self._state.name,
            "intent_name": self._intent.name,
        }

    @property
    def allocated_bytes(self):
        return self._input.size + self._state.size + self._intent.size

    def write_snapshot(self, snapshot):
        (
            pid, location_id, food, medicine, energy, health, shelter, money,
            has_employer, location_kind, positive_ties, hostile_ties,
            max_conflict, mean_affinity, scavenge_food_max, medicine_chance,
            is_working_age,
        ) = snapshot
        _INPUT.pack_into(
            self._input.buf,
            int(pid) * _INPUT.size,
            int(location_id), int(food), int(medicine), int(energy), int(health), int(shelter), float(money),
            int(bool(has_employer)), _LOCATION_TO_CODE.get(location_kind, 0),
            int(positive_ties), int(hostile_ties), float(max_conflict), float(mean_affinity),
            int(scavenge_food_max), float(medicine_chance), int(bool(is_working_age)),
        )

    def read_snapshot(self, pid):
        values = _INPUT.unpack_from(self._input.buf, int(pid) * _INPUT.size)
        (
            location_id, food, medicine, energy, health, shelter, money,
            has_employer, location_kind_code, positive_ties, hostile_ties,
            max_conflict, mean_affinity, scavenge_food_max, medicine_chance,
            is_working_age,
        ) = values
        location_kind = _LOCATION_KINDS[location_kind_code] if location_kind_code < len(_LOCATION_KINDS) else "residential"
        return (
            int(pid), location_id, food, medicine, energy, health, shelter, money,
            bool(has_employer), location_kind, positive_ties, hostile_ties,
            max_conflict, mean_affinity, scavenge_food_max, medicine_chance,
            bool(is_working_age),
        )

    def write_state(self, pid, final_state):
        food, medicine, energy, health, shelter, money = final_state
        _STATE.pack_into(
            self._state.buf,
            int(pid) * _STATE.size,
            int(food), int(medicine), int(energy), int(health), int(shelter), float(money),
        )

    def read_state(self, pid):
        return _STATE.unpack_from(self._state.buf, int(pid) * _STATE.size)

    def _intent_offset(self, pid, round_index):
        slot = int(pid) * self.actions_per_day + int(round_index)
        return slot * self.intent_record_size

    def _pack_intent(self, offset, kind, action_code=0, target=-1, resource_code=0, values=(), witness_count=0):
        padded = tuple(float(v) for v in values[:5]) + (0.0,) * max(0, 5 - len(values))
        _INTENT.pack_into(
            self._intent.buf,
            offset,
            int(kind), int(action_code), int(target), int(resource_code),
            padded[0], padded[1], padded[2], padded[3], padded[4], int(witness_count),
        )

    def write_intent(self, pid, round_index, plan):
        offset = self._intent_offset(pid, round_index)
        if plan is None:
            self._pack_intent(offset, 0)
            return

        kind = plan[0]
        if kind == "shared":
            action = plan[1]
            self._pack_intent(offset, 1, _ACTION_TO_CODE.get(action, 0))
            return
        if kind == "move_prepared":
            _, destination_id, energy_cost = plan
            self._pack_intent(offset, 3, _ACTION_TO_CODE["move"], destination_id, values=(energy_cost,))
            return
        if kind == "work_prepared":
            (
                _, employer_id, gross, energy_cost, output_good,
                produced, service_revenue, career_delta,
            ) = plan
            self._pack_intent(
                offset,
                4,
                _ACTION_TO_CODE["work"],
                employer_id,
                _RESOURCE_TO_CODE.get(output_good, 0),
                (gross, energy_cost, produced, service_revenue, career_delta),
            )
            return
        if kind == "buy_prepared":
            _, resource, requested = plan
            self._pack_intent(
                offset,
                5,
                _ACTION_TO_CODE["buy_supplies"],
                -1,
                _RESOURCE_TO_CODE.get(resource, 0),
                (requested,),
            )
            return

        _, action, target_id, payload, witness_ids = plan
        resource_code = 0
        amount = 0.0
        extra = 0.0
        if action in ("help", "steal"):
            if payload:
                resource, proposed = payload
                resource_code = _RESOURCE_TO_CODE.get(resource, 0)
                amount = float(proposed or 0)
        elif action == "attack" and payload:
            damage, energy_cost = payload
            amount = float(damage)
            extra = float(energy_cost)
        witnesses = tuple(witness_ids or ())[: self.max_witnesses]
        self._pack_intent(
            offset,
            2,
            _ACTION_TO_CODE.get(action, 0),
            -1 if target_id is None else int(target_id),
            resource_code,
            (amount, extra),
            len(witnesses),
        )
        witness_offset = offset + _INTENT.size
        for index, witness_id in enumerate(witnesses):
            struct.pack_into("<i", self._intent.buf, witness_offset + index * 4, int(witness_id))

    def read_intent(self, pid, round_index):
        offset = self._intent_offset(pid, round_index)
        (
            kind, action_code, target_id, resource_code,
            value0, value1, value2, value3, value4, witness_count,
        ) = _INTENT.unpack_from(self._intent.buf, offset)
        if kind == 0 or action_code <= 0 or action_code >= len(_ACTIONS):
            return None
        action = _ACTIONS[action_code]
        if kind == 1:
            return ("shared", action)
        if kind == 3:
            return ("move_prepared", int(target_id), int(value0))
        if kind == 4:
            resource = _RESOURCES[resource_code] if resource_code < len(_RESOURCES) else None
            return (
                "work_prepared", int(target_id), int(value0), int(value1), resource,
                float(value2), float(value3), float(value4),
            )
        if kind == 5:
            resource = _RESOURCES[resource_code] if resource_code < len(_RESOURCES) else None
            return ("buy_prepared", resource, int(value0))

        witnesses = []
        witness_offset = offset + _INTENT.size
        for index in range(min(int(witness_count), self.max_witnesses)):
            (witness_id,) = struct.unpack_from("<i", self._intent.buf, witness_offset + index * 4)
            witnesses.append(witness_id)
        target = None if target_id < 0 else target_id
        if action in ("help", "steal"):
            resource = _RESOURCES[resource_code] if resource_code < len(_RESOURCES) else None
            payload = (resource, value0)
        else:
            payload = (int(value0), int(value1))
        return ("social", action, target, payload, tuple(witnesses))

    def write_result(self, pid, final_state, intents):
        self.write_state(pid, final_state)
        for round_index in range(self.actions_per_day):
            plan = intents[round_index] if round_index < len(intents) else None
            self.write_intent(pid, round_index, plan)

    def close(self, *, unlink=False):
        segments = (self._input, self._state, self._intent)
        for segment in segments:
            segment.close()
        if unlink and self._owner:
            for segment in segments:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass

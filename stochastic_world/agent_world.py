import multiprocessing as mp
import os

from .agent_coarse import AgentCoarsePool
from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .labor_market import BUSINESS_INTERVAL_DAYS
from .life_world import LifeWorld
from .multiprocessing_engine import _DeterministicStream, _seed_for
from .politics import ELECTION_INTERVAL_DAYS
from .population_index import permutation_ids
from .professions import MOBILITY_INTERVAL_DAYS


_PHASE_SOCIAL = 0x50C1A1
_SOCIAL_ACTIONS = {"help", "steal", "attack"}


def _sample_candidates(pool, actor_id, limit, stream):
    usable = [row for row in pool if row[0] != actor_id]
    if not usable or limit <= 0:
        return []
    want = min(int(limit), len(usable))
    if want == len(usable):
        return usable
    result = []
    seen = set()
    while len(result) < want:
        index = stream.randint(0, len(usable) - 1)
        if index in seen:
            continue
        seen.add(index)
        result.append(usable[index])
    return result


def _weighted_pick(candidates, memories, mode, stream):
    if not candidates:
        return None
    weighted = []
    total = 0.0
    for target in candidates:
        target_id, food, medicine, money, _health = target
        memory = memories.get(target_id)
        if memory is None:
            affinity = 0.0
            conflict = 0.0
            familiarity = 0
        else:
            trust, grievance, familiarity = memory
            affinity = max(-100.0, min(100.0, trust - grievance))
            conflict = max(0.0, grievance - min(0.0, trust))
        if mode == "help":
            value = 1.0 if memory is None else 1.0 + max(0.0, affinity) / 10.0 + familiarity / 20.0
        elif mode == "attack":
            value = 1.0 if memory is None else 1.0 + conflict / 5.0
        else:
            wealth = food + medicine * 2 + max(0.0, money) / 4.0
            value = 1.0 + wealth / 25.0 if memory is None else 1.0 + conflict / 18.0 + max(0.0, -affinity) / 30.0 + wealth / 25.0
        weight = max(0.1, value)
        weighted.append((target, weight))
        total += weight
    needle = stream.random() * total
    upto = 0.0
    for target, weight in weighted:
        upto += weight
        if needle <= upto:
            return target
    return weighted[-1][0]


def _plan_social_row(row, pools, master_seed, day, round_index, encounter_sample, max_witnesses, visibility):
    pid, action, location_id, actor_food, actor_medicine, memories_tuple = row
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_SOCIAL, round_index))
    pool = pools.get(location_id, ())
    candidates = _sample_candidates(pool, pid, encounter_sample, stream)
    target = _weighted_pick(candidates, dict(memories_tuple), action, stream)
    if target is None:
        return (pid, action, None, None, ())
    target_id, target_food, target_medicine, target_money, target_health = target
    if action == "help":
        if target_health < 70 and actor_medicine > 0:
            payload = ("medicine", 1)
        elif actor_food > 2:
            payload = ("food", stream.randint(1, min(2, actor_food - 1)))
        else:
            payload = (None, 0)
    elif action == "steal":
        resource = None
        amount = 0
        if stream.random() < 0.45:
            options = []
            if target_food > 0:
                options.extend(("food",) * 4)
            if target_money > 0:
                options.extend(("money",) * 2)
            if target_medicine > 0:
                options.append("medicine")
            if options:
                resource = options[stream.randint(0, len(options) - 1)]
                if resource == "food":
                    amount = min(target_food, stream.randint(1, 3))
                elif resource == "money":
                    amount = min(target_money, stream.randint(1, 5))
                else:
                    amount = 1
        payload = (resource, amount)
    else:
        payload = (stream.randint(5, 20), stream.randint(4, 9))
    witness_ids = ()
    if max_witnesses and stream.random() <= visibility:
        witness_pool = [candidate for candidate in pool if candidate[0] not in (pid, target_id)]
        witnesses = _sample_candidates(witness_pool, -1, max_witnesses, stream)
        witness_ids = tuple(item[0] for item in witnesses)
    return (pid, action, target_id, payload, witness_ids)


def _social_worker(worker_id, input_queue, result_queue, master_seed):
    while True:
        task = input_queue.get()
        if task is None:
            return
        day, round_index, rows, pools, encounter_sample, max_witnesses, visibility = task
        results = [_plan_social_row(row, pools, master_seed, day, round_index, encounter_sample, max_witnesses, visibility) for row in rows]
        result_queue.put((worker_id, results))


class SocialIntentPool:
    def __init__(self, master_seed, workers=0):
        cpu_count = os.cpu_count() or 1
        requested = max(0, int(workers))
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
        self.enabled = self.worker_count >= 2
        self.started = False
        self._ctx = None
        self._queues = []
        self._result_queue = None
        self._processes = []

    def _ensure_started(self):
        if not self.enabled or self.started:
            return
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self._ctx = mp.get_context(method)
        self._result_queue = self._ctx.Queue()
        for worker_id in range(self.worker_count):
            queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(target=_social_worker, args=(worker_id, queue, self._result_queue, self.master_seed), name=f"stochastic-social-{worker_id}", daemon=True)
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
        self.started = True

    def plan(self, day, round_index, rows, pools, encounter_sample, max_witnesses, visibility):
        if not rows:
            return []
        if not self.enabled:
            return [_plan_social_row(row, pools, self.master_seed, day, round_index, encounter_sample, max_witnesses, visibility) for row in rows]
        self._ensure_started()
        chunk_size = max(1, (len(rows) + self.worker_count - 1) // self.worker_count)
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        for worker_id, chunk in enumerate(chunks):
            self._queues[worker_id].put((day, round_index, chunk, pools, encounter_sample, max_witnesses, visibility))
        results = []
        for _ in chunks:
            _worker_id, planned = self._result_queue.get()
            results.extend(planned)
        return results

    def close(self):
        if not self.started:
            return
        for queue in self._queues:
            queue.put(None)
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for queue in self._queues:
            queue.close()
        if self._result_queue is not None:
            self._result_queue.close()
        self._queues.clear()
        self._processes.clear()
        self.started = False


class ParallelAgentWorld(LifeWorld):
    def __init__(self, *args, agent_workers=0, agent_worker_min_active=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_mode = "agent"
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.district_pool = AgentCoarsePool(seed, workers=agent_workers, min_active=agent_worker_min_active)
        self.social_pool = SocialIntentPool(seed, workers=agent_workers)

    def remember_interaction(self, actor, target, action, magnitude=1.0):
        if self.store.event_mode == "full":
            return super().remember_interaction(actor, target, action, magnitude)
        actor.remember(target, self.current_day, action, "actor", magnitude)
        target.remember(actor, self.current_day, action, "target", magnitude)
        self.spread_reputation(actor, target, action, magnitude)

    def spread_reputation(self, actor, target, action, magnitude):
        if self.store.event_mode == "full":
            return super().spread_reputation(actor, target, action, magnitude)
        if not self.max_witnesses or self.rng.random() > self.visibility:
            return
        witnesses = self.population_index.sample_people(actor.location_id, self.rng, self.max_witnesses, exclude=(actor.id, target.id))
        for witness in witnesses:
            witness.observe(actor, self.current_day, action, magnitude)
            self.total_observations += 1

    def _action_snapshot(self, person):
        memory = person.aggregate_memory()
        location = self.location_of(person)
        return (person.id, person.location_id, person.food, person.medicine, person.energy, person.health, person.shelter, person.money, person.employer_id is not None, location.kind, memory["positive_ties"], memory["hostile_ties"], memory["max_conflict"], memory["mean_affinity"], location.scavenge_food_max, location.medicine_chance, person.is_working_age)

    def _execute_shared_intent(self, person, action):
        if not person.alive or self.current_day < person.detained_until_day or not person.is_adult:
            return
        if not person.is_working_age:
            handler = {"buy_supplies": self.buy_supplies, "help": self.help}.get(action, self.rest)
            handler(person)
            return
        handler = {"move": self.move, "work": self.work, "buy_supplies": self.buy_supplies, "help": self.help, "steal": self.steal, "attack": self.attack}.get(action)
        if handler:
            handler(person)

    def _apply_safe_result(self, person, result):
        _pid, action, _safe, food, medicine, energy, health, shelter, money, event_data = result
        if not person.alive or self.current_day < person.detained_until_day:
            return
        person.food = food; person.medicine = medicine; person.energy = energy; person.health = health; person.shelter = shelter; person.money = money
        if event_data is None:
            return
        if action == "rest":
            energy_gain, health_gain = event_data
            self.store.event(self.current_day, self.next_sequence(), "rest", actor=person, location_id=person.location_id, energy_gain=energy_gain, health_gain=health_gain)
        elif action == "heal":
            (gain,) = event_data
            self.store.event(self.current_day, self.next_sequence(), "heal", actor=person, location_id=person.location_id, health_gain=gain)
        elif action == "repair":
            (gain,) = event_data
            self.store.event(self.current_day, self.next_sequence(), "repair", actor=person, location_id=person.location_id, shelter_gain=gain)
        elif action == "scavenge":
            food_found, medicine_found, cost = event_data
            self.store.event(self.current_day, self.next_sequence(), "scavenge", actor=person, location_id=person.location_id, food_found=food_found, medicine_found=medicine_found, energy_cost=cost)

    def _social_candidate_pools(self):
        pools = {}
        for location in self.locations:
            rows = []
            for pid in self.population_index.ids(location.id):
                person = self.people[pid]
                if person.alive:
                    rows.append((pid, person.food, person.medicine, person.money, person.health))
            pools[location.id] = tuple(rows)
        return pools

    def _social_row(self, person, action):
        memories = tuple((other_id, (memory.trust, memory.grievance, memory.familiarity)) for other_id, memory in person.memories.items())
        return (person.id, action, person.location_id, person.food, person.medicine, memories)

    def _apply_prepared_social(self, person, prepared):
        _pid, action, target_id, payload, witness_ids = prepared
        if target_id is None or not person.alive or self.current_day < person.detained_until_day or target_id >= len(self.people):
            return
        target = self.people[target_id]
        if not target.alive or target.location_id != person.location_id:
            return
        if action == "help":
            resource, proposed = payload; amount = 0
            if resource == "medicine" and proposed and person.medicine > 0:
                amount = 1; person.medicine -= 1; target.medicine += 1
            elif resource == "food" and proposed and person.food > 1:
                amount = min(int(proposed), max(0, person.food - 1)); person.food -= amount; target.food += amount
            self.store.event(self.current_day, self.next_sequence(), "help", actor=person, target=target, success=int(bool(amount)), location_id=person.location_id, resource=resource, amount=amount)
            if not amount:
                return
            self.total_helps += 1; magnitude = float(amount)
        elif action == "steal":
            resource, proposed = payload; amount = 0
            if resource == "food" and proposed:
                amount = min(target.food, int(proposed)); target.food -= amount; person.food += amount
            elif resource == "money" and proposed:
                amount = min(target.money, float(proposed)); target.money -= amount; person.money += amount
            elif resource == "medicine" and proposed and target.medicine > 0:
                amount = 1; target.medicine -= 1; person.medicine += 1
            if amount:
                self.total_thefts += 1; target.crime_suffered += 1; self.daily_crimes[person.location_id] += 1
            self.store.event(self.current_day, self.next_sequence(), "steal", actor=person, target=target, success=int(bool(amount)), location_id=person.location_id, resource=resource, amount=amount)
            if not amount:
                return
            magnitude = max(1.0, float(amount) / 2.0)
        else:
            damage, energy_cost = payload
            person.energy = max(0, person.energy - int(energy_cost)); target.health -= int(damage); self.total_attacks += 1; target.crime_suffered += 1; self.daily_crimes[person.location_id] += 1
            self.store.event(self.current_day, self.next_sequence(), "attack", actor=person, target=target, success=1, location_id=person.location_id, damage=damage, killed=target.health <= 0)
            magnitude = float(damage) / 10.0
        person.remember(target, self.current_day, action, "actor", magnitude)
        target.remember(person, self.current_day, action, "target", magnitude)
        for witness_id in witness_ids:
            if witness_id < len(self.people):
                witness = self.people[witness_id]
                if witness.alive and witness.location_id == person.location_id and witness.id not in (person.id, target.id):
                    witness.observe(person, self.current_day, action, magnitude); self.total_observations += 1
        if action in ("steal", "attack"):
            self.police_response(action, person, target, magnitude)
        if action == "attack" and target.health <= 0:
            self.kill(target, "violence")

    def _run_parallel_actions(self, day):
        eligible_order = [pid for pid in permutation_ids(len(self.people), self.rng) if self.people[pid].alive and self.people[pid].is_adult and day >= self.people[pid].detained_until_day]
        if not eligible_order or self.actions_per_day <= 0:
            return
        for round_index in range(self.actions_per_day):
            snapshots = [self._action_snapshot(self.people[pid]) for pid in eligible_order if self.people[pid].alive and day >= self.people[pid].detained_until_day]
            if not snapshots:
                break
            planned = {row[0]: row for row in self.district_pool.plan_round(day, round_index, snapshots)}
            prepared_social = {}
            if self.store.event_mode == "compact":
                social_rows = []
                for pid in eligible_order:
                    result = planned.get(pid); person = self.people[pid]
                    if result is not None and not result[2] and result[1] in _SOCIAL_ACTIONS and person.alive and day >= person.detained_until_day and person.is_working_age:
                        social_rows.append(self._social_row(person, result[1]))
                if social_rows:
                    social_results = self.social_pool.plan(day, round_index, social_rows, self._social_candidate_pools(), self.encounter_sample, self.max_witnesses, self.visibility)
                    prepared_social = {row[0]: row for row in social_results}
            for pid in eligible_order:
                person = self.people[pid]; result = planned.get(pid)
                if result is None or not person.alive or day < person.detained_until_day:
                    continue
                if result[2]:
                    self._apply_safe_result(person, result)
                elif pid in prepared_social:
                    self._apply_prepared_social(person, prepared_social[pid])
                else:
                    self._execute_shared_intent(person, result[1])

    def _end_snapshot(self, person, rates):
        location = self.location_of(person)
        return (person.id, person.food, person.energy, person.shelter, person.health, person.money, person.unemployment_days, person.employer_id is not None, person.is_working_age, person.is_dependent, person.is_adult, location.shelter_decay_bonus, rates.get(person.location_id, 0.0))

    def _apply_end_delta(self, person, delta):
        _pid, food, energy, shelter, health, unemployment_days, lifetime_unemployment_increment, ideology_shift, damage, causes = delta
        if not person.alive:
            return
        person.days_in_class += 1; person.food = food; person.energy = energy; person.shelter = shelter; person.health = health; person.unemployment_days = unemployment_days; person.lifetime_unemployment_days += lifetime_unemployment_increment
        if ideology_shift:
            person.shift_ideology(ideology_shift)
        if damage:
            self.store.event(self.current_day, self.next_sequence(), "daily_harm", actor=person, location_id=person.location_id, damage=damage, causes=list(causes))
        if person.health <= 0:
            self.kill(person, "+".join(causes) if causes else "injury")

    def _run_parallel_end_of_day(self):
        for location in self.locations:
            self.crime_history[location.id].append(self.daily_crimes.get(location.id, 0))
        rates = self.crime_rates(); alive_ids = [p.id for p in self.people if p.alive]
        for pid in alive_ids:
            self.demographics.support_dependent(self.people[pid])
        snapshots = [self._end_snapshot(self.people[pid], rates) for pid in alive_ids if self.people[pid].alive]
        deltas = {row[0]: row for row in self.district_pool.plan_end_of_day(self.current_day, snapshots)}
        for pid in alive_ids:
            person = self.people[pid]; delta = deltas.get(pid)
            if delta is not None and person.alive:
                self._apply_end_delta(person, delta)
        self.daily_crimes.clear()

    def close_parallel(self):
        self.district_pool.close(); self.social_pool.close()

    def run_day(self, day):
        if not self.district_pool.should_parallelize(self.alive_count):
            return super().run_day(day)
        self.current_day = day
        if day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0:
            self.run_election()
        self._run_parallel_actions(day)
        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        if day % BUSINESS_INTERVAL_DAYS == 0:
            self.welfare_cycle(); self.business_cycle(); self.police.rebalance()
        self._run_parallel_end_of_day(); self.goods_market.reprice()
        if day % MOBILITY_INTERVAL_DAYS == 0:
            self.mobility_cycle()
        police_snapshot = self.police.end_day(); self.store.write_daily_stats(day, self); self.store.write_location_stats(day, self); self.store.write_political_stats(day, self); self.store.write_social_stats(day, self); self.store.write_labor_stats(day, self); self.store.write_market_stats(day, self); self.store.write_police_stats(day, police_snapshot); self.store.commit_day()
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            self.demographics.cycle(day); self.demographics.write_stats(day); self.store.commit_day()

"""Optional Numba kernels for the very-large aggressive engine.

The BSP / temporal engines deliberately keep Python as the orchestration layer.
This module compiles the two hottest scalar kernels (action weighting and EOD
arithmetic) and installs API-compatible wrappers into the already-loaded domain
modules.  If NumPy / Numba are unavailable, the same allocation-free Python
kernels are used instead, so aggressive mode remains functional.
"""

_MASK64 = (1 << 64) - 1
_INV_U64 = 1.0 / float(1 << 64)
_PHASE_ACTION = 0xA11CE
_PHASE_END_OF_DAY = 0xE0D

_ACTIONS = (
    "move", "work", "scavenge", "buy_supplies", "rest", "heal",
    "repair", "help", "steal", "attack", "idle",
)

try:
    import numpy as _np
    from numba import njit as _njit
except Exception:  # pragma: no cover - accelerator is optional
    _np = None
    _njit = None


JIT_ENABLED = _njit is not None and _np is not None


def _splitmix64_py(value):
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _seed_py(master_seed, day, pid, phase, round_index=0):
    value = int(master_seed) & _MASK64
    value ^= (int(day) * 0xD6E8FEB86659FD93) & _MASK64
    value ^= (int(pid) * 0xA5A3564E27F8862B) & _MASK64
    value ^= (int(phase) * 0x9E3779B185EBCA87) & _MASK64
    value ^= (int(round_index) * 0xC2B2AE3D27D4EB4F) & _MASK64
    return _splitmix64_py(value)


def _weighted_code_py(pid, food, medicine, energy, health, shelter, money,
                      has_employer, location_code, positive_ties, hostile_ties,
                      max_conflict, mean_affinity, master_seed, day, round_index):
    # Keep this order identical to multiprocessing_engine._weighted_action.
    w0 = 6.0
    w1 = 24.0
    w2 = 11.0
    w3 = 9.0
    w4 = 13.0
    w5 = 4.0
    w6 = 4.0
    w7 = 8.0
    w8 = 4.0
    w9 = 1.0
    w10 = 16.0

    if food <= 3:
        w2 *= 2.6; w3 *= 3.0; w8 *= 2.0; w0 *= 1.5
    if medicine == 0 and health < 75:
        w3 *= 2.8; w0 *= 1.6
    if energy <= 30:
        w4 *= 4.0; w1 *= 0.35; w0 *= 0.4; w9 *= 0.5
    if health < 70:
        w5 *= 4.0 if medicine else 0.1; w9 *= 0.5
    if shelter < 45:
        w6 *= 4.0
    if money < 4:
        w1 *= 1.8; w3 *= 0.35; w0 *= 1.3
    if not has_employer:
        w1 *= 1.65
    if location_code == 1:  # industrial
        w1 *= 1.25
    if location_code == 2:  # outskirts
        w2 *= 1.5
    if positive_ties:
        w7 *= 1.0 + min(2.0, positive_ties / 4.0)
    if hostile_ties:
        w9 *= 1.0 + max_conflict / 12.0
        w8 *= 1.0 + max_conflict / 60.0
    if mean_affinity > 10.0:
        w7 *= 1.4; w9 *= 0.65
    elif mean_affinity < -10.0:
        w7 *= 0.6; w9 *= 1.5

    total = w0+w1+w2+w3+w4+w5+w6+w7+w8+w9+w10
    seed = _seed_py(master_seed, day, pid, _PHASE_ACTION, round_index)
    rnd = _splitmix64_py(seed) * _INV_U64
    pick = rnd * total
    cursor = w0
    if pick < cursor: return 0
    cursor += w1
    if pick < cursor: return 1
    cursor += w2
    if pick < cursor: return 2
    cursor += w3
    if pick < cursor: return 3
    cursor += w4
    if pick < cursor: return 4
    cursor += w5
    if pick < cursor: return 5
    cursor += w6
    if pick < cursor: return 6
    cursor += w7
    if pick < cursor: return 7
    cursor += w8
    if pick < cursor: return 8
    cursor += w9
    if pick < cursor: return 9
    return 10


def _eod_py(pid, food, energy, shelter, health, money, unemployment_days,
            employed, working_age, dependent, adult, shelter_decay_bonus,
            local_crime_rate, master_seed, day):
    seed = _seed_py(master_seed, day, pid, _PHASE_END_OF_DAY, 0)

    new_unemployment = int(unemployment_days)
    lifetime_inc = 0
    ideology_shift = 0.0
    if working_age:
        if employed:
            new_unemployment = 0
        else:
            new_unemployment += 1
            lifetime_inc = 1
        if new_unemployment > 30:
            ideology_shift -= 0.0004
    else:
        new_unemployment = 0

    food = int(food) - 1
    energy = max(0, int(energy) - (2 if dependent else 3))
    seed = _splitmix64_py(seed)
    shelter_loss = seed % 3
    shelter = max(0, int(shelter) - int(shelter_loss) - int(shelter_decay_bonus))

    if adult:
        if money < 6 or food <= 2:
            ideology_shift -= 0.0015
        if local_crime_rate > 0.10:
            ideology_shift += min(0.0030, local_crime_rate * 0.012)

    damage = 0
    cause_mask = 0
    if food < 0:
        food = 0
        seed = _splitmix64_py(seed)
        damage += 4 + int(seed % 7)
        cause_mask |= 1
    if energy == 0:
        seed = _splitmix64_py(seed)
        damage += 2 + int(seed % 5)
        cause_mask |= 2
    if shelter <= 20:
        seed = _splitmix64_py(seed)
        if seed * _INV_U64 < 0.35:
            seed = _splitmix64_py(seed)
            damage += 2 + int(seed % 6)
            cause_mask |= 4
    health = int(health) - damage
    return food, energy, shelter, health, new_unemployment, lifetime_inc, ideology_shift, damage, cause_mask


if JIT_ENABLED:
    @_njit(cache=True, nogil=True)
    def _splitmix64_nb(value):
        value = _np.uint64(value)
        value = _np.uint64(value + _np.uint64(0x9E3779B97F4A7C15))
        value = _np.uint64((value ^ (value >> _np.uint64(30))) * _np.uint64(0xBF58476D1CE4E5B9))
        value = _np.uint64((value ^ (value >> _np.uint64(27))) * _np.uint64(0x94D049BB133111EB))
        return _np.uint64(value ^ (value >> _np.uint64(31)))

    @_njit(cache=True, nogil=True)
    def _seed_nb(master_seed, day, pid, phase, round_index):
        value = _np.uint64(master_seed)
        value ^= _np.uint64(day) * _np.uint64(0xD6E8FEB86659FD93)
        value ^= _np.uint64(pid) * _np.uint64(0xA5A3564E27F8862B)
        value ^= _np.uint64(phase) * _np.uint64(0x9E3779B185EBCA87)
        value ^= _np.uint64(round_index) * _np.uint64(0xC2B2AE3D27D4EB4F)
        return _splitmix64_nb(value)

    @_njit(cache=True, nogil=True)
    def _weighted_code_nb(pid, food, medicine, energy, health, shelter, money,
                          has_employer, location_code, positive_ties, hostile_ties,
                          max_conflict, mean_affinity, master_seed, day, round_index):
        w0=6.0; w1=24.0; w2=11.0; w3=9.0; w4=13.0; w5=4.0
        w6=4.0; w7=8.0; w8=4.0; w9=1.0; w10=16.0
        if food <= 3:
            w2*=2.6; w3*=3.0; w8*=2.0; w0*=1.5
        if medicine == 0 and health < 75:
            w3*=2.8; w0*=1.6
        if energy <= 30:
            w4*=4.0; w1*=0.35; w0*=0.4; w9*=0.5
        if health < 70:
            if medicine:
                w5*=4.0
            else:
                w5*=0.1
            w9*=0.5
        if shelter < 45: w6*=4.0
        if money < 4.0:
            w1*=1.8; w3*=0.35; w0*=1.3
        if not has_employer: w1*=1.65
        if location_code == 1: w1*=1.25
        if location_code == 2: w2*=1.5
        if positive_ties != 0:
            bonus = positive_ties / 4.0
            if bonus > 2.0: bonus = 2.0
            w7 *= 1.0 + bonus
        if hostile_ties != 0:
            w9 *= 1.0 + max_conflict / 12.0
            w8 *= 1.0 + max_conflict / 60.0
        if mean_affinity > 10.0:
            w7*=1.4; w9*=0.65
        elif mean_affinity < -10.0:
            w7*=0.6; w9*=1.5
        total=w0+w1+w2+w3+w4+w5+w6+w7+w8+w9+w10
        seed=_seed_nb(master_seed, day, pid, _PHASE_ACTION, round_index)
        rnd=float(_splitmix64_nb(seed)) * _INV_U64
        pick=rnd*total
        cursor=w0
        if pick<cursor: return 0
        cursor+=w1
        if pick<cursor: return 1
        cursor+=w2
        if pick<cursor: return 2
        cursor+=w3
        if pick<cursor: return 3
        cursor+=w4
        if pick<cursor: return 4
        cursor+=w5
        if pick<cursor: return 5
        cursor+=w6
        if pick<cursor: return 6
        cursor+=w7
        if pick<cursor: return 7
        cursor+=w8
        if pick<cursor: return 8
        cursor+=w9
        if pick<cursor: return 9
        return 10

    @_njit(cache=True, nogil=True)
    def _eod_nb(pid, food, energy, shelter, health, money, unemployment_days,
                employed, working_age, dependent, adult, shelter_decay_bonus,
                local_crime_rate, master_seed, day):
        seed=_seed_nb(master_seed, day, pid, _PHASE_END_OF_DAY, 0)
        new_unemployment=unemployment_days
        lifetime_inc=0
        ideology_shift=0.0
        if working_age:
            if employed:
                new_unemployment=0
            else:
                new_unemployment+=1; lifetime_inc=1
            if new_unemployment>30: ideology_shift-=0.0004
        else:
            new_unemployment=0
        food-=1
        energy=max(0, energy-(2 if dependent else 3))
        seed=_splitmix64_nb(seed)
        shelter=max(0, shelter-int(seed % _np.uint64(3))-shelter_decay_bonus)
        if adult:
            if money<6.0 or food<=2: ideology_shift-=0.0015
            if local_crime_rate>0.10:
                shift=local_crime_rate*0.012
                ideology_shift += 0.0030 if shift>0.0030 else shift
        damage=0; cause_mask=0
        if food<0:
            food=0; seed=_splitmix64_nb(seed); damage+=4+int(seed % _np.uint64(7)); cause_mask|=1
        if energy==0:
            seed=_splitmix64_nb(seed); damage+=2+int(seed % _np.uint64(5)); cause_mask|=2
        if shelter<=20:
            seed=_splitmix64_nb(seed)
            if float(seed)*_INV_U64<0.35:
                seed=_splitmix64_nb(seed); damage+=2+int(seed % _np.uint64(6)); cause_mask|=4
        health-=damage
        return food,energy,shelter,health,new_unemployment,lifetime_inc,ideology_shift,damage,cause_mask
else:
    _weighted_code_nb = None
    _eod_nb = None


def weighted_action(snapshot, master_seed, day, round_index):
    (pid, _district_id, food, medicine, energy, health, shelter, money,
     has_employer, location_kind, positive_ties, hostile_ties,
     max_conflict, mean_affinity) = snapshot
    location_code = 1 if location_kind == "industrial" else (2 if location_kind == "outskirts" else 0)
    fn = _weighted_code_nb if JIT_ENABLED else _weighted_code_py
    code = fn(
        int(pid), int(food), int(medicine), int(energy), int(health), int(shelter),
        float(money), bool(has_employer), int(location_code), int(positive_ties),
        int(hostile_ties), float(max_conflict), float(mean_affinity),
        int(master_seed) & _MASK64, int(day), int(round_index),
    )
    return int(pid), _ACTIONS[int(code)]


def end_of_day_delta(snapshot, master_seed, day):
    (pid, food, energy, shelter, health, money, unemployment_days,
     employed, working_age, dependent, adult, shelter_decay_bonus,
     local_crime_rate) = snapshot
    fn = _eod_nb if JIT_ENABLED else _eod_py
    values = fn(
        int(pid), int(food), int(energy), int(shelter), int(health), float(money),
        int(unemployment_days), bool(employed), bool(working_age), bool(dependent),
        bool(adult), int(shelter_decay_bonus), float(local_crime_rate),
        int(master_seed) & _MASK64, int(day),
    )
    food,energy,shelter,health,new_unemployment,lifetime_inc,ideology_shift,damage,mask = values
    causes=[]
    if int(mask)&1: causes.append("starvation")
    if int(mask)&2: causes.append("exhaustion")
    if int(mask)&4: causes.append("exposure")
    return (
        int(pid), int(food), int(energy), int(shelter), int(health),
        int(new_unemployment), int(lifetime_inc), float(ideology_shift),
        int(damage), tuple(causes),
    )


def warmup():
    """Compile hot kernels before fork so Linux workers inherit machine code."""
    if not JIT_ENABLED:
        return False
    weighted_action((0,0,5,1,80,90,70,10.0,True,"industrial",0,0,0.0,0.0), 1, 1, 0)
    end_of_day_delta((0,5,80,70,90,10.0,0,True,True,False,True,0,0.0), 1, 1)
    return True


def install():
    """Install API-compatible kernels into the loaded BSP modules."""
    from . import aggressive_domain
    from . import aggressive_temporal
    warmup()
    aggressive_domain._weighted_action = weighted_action
    aggressive_temporal._end_of_day_delta = end_of_day_delta
    return JIT_ENABLED

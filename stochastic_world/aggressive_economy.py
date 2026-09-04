"""Shared hot economy snapshot for the throughput-first aggressive engine."""

from multiprocessing import shared_memory
import struct

from .professions import PROFESSIONS


PROFESSION_NAMES = tuple(PROFESSIONS)
PROFESSION_TO_CODE = {name: index for index, name in enumerate(PROFESSION_NAMES)}
SOCIAL_CLASSES = ("working", "lower_middle", "middle", "upper_middle", "affluent")
SOCIAL_CLASS_TO_CODE = {name: index for index, name in enumerate(SOCIAL_CLASSES)}
LOCATION_KINDS = ("residential", "market", "industrial", "clinic", "outskirts", "logistics", "new_venture")
LOCATION_TO_CODE = {name: index for index, name in enumerate(LOCATION_KINDS)}
OUTPUT_CODES = {None: 0, "food": 1, "medicine": 2}
OUTPUT_NAMES = (None, "food", "medicine")

# employer_id, profession_code, social_class_code
_PERSON = struct.Struct("<iii")
# location, capacity, employees, base_wage, cash, productivity, output_code,
# employer_kind_code, output_per_shift, preferred_profession_mask, alive
_EMPLOYER = struct.Struct("<iiidddBBdQB")
# kind, food_stock, medicine_stock, vacancies, population
_LOCATION = struct.Struct("<Bddii")


class SharedEconomyState:
    """Small but frequently-read economy/location state shared by all workers."""

    def __init__(self, population_capacity, employer_capacity, locations, *, descriptor=None):
        if descriptor is None:
            self.population_capacity = max(1, int(population_capacity))
            self.employer_capacity = max(1, int(employer_capacity))
            self.location_count = max(1, len(locations))
            self.neighbors = tuple(tuple(int(x) for x in loc.neighbors) for loc in locations)
            self._persons = shared_memory.SharedMemory(create=True, size=self.population_capacity * _PERSON.size)
            self._employers = shared_memory.SharedMemory(create=True, size=self.employer_capacity * _EMPLOYER.size)
            self._locations = shared_memory.SharedMemory(create=True, size=self.location_count * _LOCATION.size)
            self._owner = True
            self._employers.buf[:] = b"\x00" * self._employers.size
        else:
            self.population_capacity = int(descriptor["population_capacity"])
            self.employer_capacity = int(descriptor["employer_capacity"])
            self.location_count = int(descriptor["location_count"])
            self.neighbors = tuple(tuple(int(x) for x in row) for row in descriptor["neighbors"])
            self._persons = shared_memory.SharedMemory(name=descriptor["persons_name"])
            self._employers = shared_memory.SharedMemory(name=descriptor["employers_name"])
            self._locations = shared_memory.SharedMemory(name=descriptor["locations_name"])
            self._owner = False

    @classmethod
    def attach(cls, descriptor):
        return cls(1, 1, (), descriptor=descriptor)

    @property
    def descriptor(self):
        return {
            "population_capacity": self.population_capacity,
            "employer_capacity": self.employer_capacity,
            "location_count": self.location_count,
            "neighbors": self.neighbors,
            "persons_name": self._persons.name,
            "employers_name": self._employers.name,
            "locations_name": self._locations.name,
        }

    @property
    def allocated_bytes(self):
        return self._persons.size + self._employers.size + self._locations.size

    def write_person(self, pid, person):
        _PERSON.pack_into(
            self._persons.buf,
            int(pid) * _PERSON.size,
            -1 if person.employer_id is None else int(person.employer_id),
            PROFESSION_TO_CODE.get(person.profession, 0),
            SOCIAL_CLASS_TO_CODE.get(person.social_class, 0),
        )

    def read_person(self, pid):
        return _PERSON.unpack_from(self._persons.buf, int(pid) * _PERSON.size)

    def sync_world(self, world):
        for employer in world.labor_market.employers:
            eid = int(employer.id)
            if eid < 0 or eid >= self.employer_capacity:
                continue
            preferred_mask = 0
            for name in employer.preferred_professions:
                code = PROFESSION_TO_CODE.get(name)
                if code is not None and code < 64:
                    preferred_mask |= 1 << code
            _EMPLOYER.pack_into(
                self._employers.buf,
                eid * _EMPLOYER.size,
                int(employer.location_id),
                int(employer.capacity),
                len(employer.employee_ids),
                float(employer.base_wage),
                float(employer.cash),
                float(employer.productivity),
                OUTPUT_CODES.get(employer.output_good, 0),
                LOCATION_TO_CODE.get(employer.kind, 0),
                float(employer.output_per_shift),
                int(preferred_mask),
                int(bool(employer.alive)),
            )

        for location in world.locations:
            lid = int(location.id)
            if lid < 0 or lid >= self.location_count:
                continue
            market = world.goods_market.state(lid)
            _LOCATION.pack_into(
                self._locations.buf,
                lid * _LOCATION.size,
                LOCATION_TO_CODE.get(location.kind, 0),
                float(market.stock("food")),
                float(market.stock("medicine")),
                int(world.labor_market.vacancies(lid)),
                int(world.population_index.population(lid)),
            )

    def read_employer(self, employer_id):
        eid = int(employer_id)
        if eid < 0 or eid >= self.employer_capacity:
            return None
        values = _EMPLOYER.unpack_from(self._employers.buf, eid * _EMPLOYER.size)
        if not values[-1]:
            return None
        (
            location_id, capacity, employees, base_wage, cash, productivity,
            output_code, kind_code, output_per_shift, preferred_mask, alive,
        ) = values
        # Keep the planner tuple compact. Negative output_per_shift is a service
        # marker for non-logistics employers with no physical output.
        if output_code == 0 and kind_code != LOCATION_TO_CODE["logistics"]:
            output_per_shift = -1.0
        return (
            location_id, capacity, employees, base_wage, cash, productivity,
            output_code, output_per_shift, preferred_mask, alive,
        )

    def read_location(self, location_id):
        lid = int(location_id)
        if lid < 0 or lid >= self.location_count:
            return None
        return _LOCATION.unpack_from(self._locations.buf, lid * _LOCATION.size)

    def close(self, *, unlink=False):
        segments = (self._persons, self._employers, self._locations)
        for segment in segments:
            segment.close()
        if unlink and self._owner:
            for segment in segments:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass

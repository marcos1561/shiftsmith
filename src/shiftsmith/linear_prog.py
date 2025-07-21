from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus, PULP_CBC_CMD
from collections import namedtuple
import copy
import logging

from .shifts import ShiftList, Shift
from .base import SchedulerBase, Sheets

# Define a logger for this module
logger = logging.getLogger(__name__)

#==
# Linear Programming
#==
FixPeopleHours = namedtuple('FixPeopleHours', ['shifts', 'capacity', 'days'])

class ScheduleProblems:
    def __init__(self):
        self.availability: dict = None

class LinearProgSched(SchedulerBase):
    def __init__(self, sheets: Sheets, 
        people_boost:dict[str, float]=None, desired_work_load: dict[str, float]=None,
        max_open_per_people=2, max_close_per_people=2, 
        max_load_per_day=None, max_load_per_week=None, min_load_per_week=None,
        default_shift_capacity=2,
        shifts_weights: dict = None,
    ):
        '''
        System to generate a schedule. After creating this object, one should call `self.generate()`
        to create the schedule.

        OBS: 
            A shift is a string with the following structure: "HH:MM-HH:MM".
            Example: "07:30-08:00" 

        Parameters
        ----------
        sheets:
            Dada from sheets (availability, preference, etc). See docs for `Sheets` for
            more info.

        people_boost:
            Map between people and its boost in the objective function. The name
            used should be the same as the one in the preference/availability sheets.
            If the boost is greater/less than 1, for a give person, this person will tend to gain more/less
            work hours. Example boosting work hours for Marcos:
            
            >>> shed = Scheduler(people_boost={"Marcos": 1.5})

        desired_work_load:
            Map between people and its desired total work load in the week in hours.
        
        max_open_per_people:
            Maximum number of times a given person open the cafe.
        
        max_close_per_people:
            Maximum number of times a given person close the cafe.
        
        max_load_per_day:
            Maximum number of hours a given person works per day. If None there is no limit.

        max_load_per_week, min_load_per_week:
            Maximum/Minimum number of hours a person should work in a week. If `None`, there is no limit.
        
        default_shift_capacity:
            Default number of people per shift. This constraint is override for shifts
            specified by `add_fix_people_shifts()`
        '''
        super().__init__(sheets)


        # self.open_shift = Shift(open_shift)
        # self.close_shift = Shift(close_shift)
        self.open_shift = self.sheets.shifts.shifts[0]
        self.close_shift = self.sheets.shifts.shifts[-1]

        # self.work_shifts = ShiftList.from_start_end(
        #     start_shift=self.open_shift,
        #     end_shift=self.close_shift,
        # )
        self.work_shifts = self.sheets.shifts

        self.max_open_per_people = max_open_per_people
        self.max_close_per_people = max_close_per_people
        self.max_load_per_day = max_load_per_day
        self.max_load_per_week = max_load_per_week
        self.min_load_per_week = min_load_per_week
        self.default_shift_capacity = default_shift_capacity

        if desired_work_load is None:
            desired_work_load = {}
        self.work_load = desired_work_load
        
        if people_boost is None:
            people_boost = {}
        self.people_boost = people_boost

        self.fix_people_hours: list[FixPeopleHours] = []

        self.preference_work_load: dict = None
        self.preference = None
        self.availability = None

        self.x = None
        self.problems = ScheduleProblems()

    def add_fix_people_shifts(self, shifts: ShiftList, capacity: int, days="all"):
        "Add shifts where there should be `capacity` people at the cafe."
        if isinstance(shifts, list):
            shifts = ShiftList(shifts)

        filtered_shifts = []
        for t in shifts:
            if t in self.work_shifts:
                filtered_shifts.append(t)
        shifts = ShiftList(filtered_shifts)

        week_days = self.sheets.week_days

        if days == "all":
            days = week_days
        else:
            for d in days:
                if d not in week_days:
                    raise ValueError(f"O dia '{d}' em `days` não está na lista de dias: {week_days}.")

        self.fix_people_hours.append(
            FixPeopleHours(shifts=shifts, capacity=capacity, days=days)
        )

    def generate(self):
        '''
        Generates schedule trying to maximize peoples preference and respecting
        peoples availability. After running this method, one can save the
        schedule calling `save()`.
        '''
        logger.info("Gerando a escala..")

        preference = copy.deepcopy(self.sheets.pref)
        availability = copy.deepcopy(self.sheets.avail)

        week_days = self.sheets.week_days
        work_shifts = self.sheets.shifts.shifts_str
        people = self.sheets.people

        # If a person does not have a preference, is assumed 
        # that his preference is the same as his availability.
        for p in people:
            if p not in preference:
                preference[p] = availability[p]

        # Creating decision variables x[person][day][hour]
        x = {
            p: {
                d: {h: LpVariable(f"x_{p}_{d}_{h}", cat="Binary") for h in work_shifts}
                for d in week_days
            }
            for p in people
        }

        for p in people:
            if p not in preference.keys():
                logger.warning(f"A pessoa {p} não preencheu a preferência.")

        # Creating the maximization problem (prioritizing preferences)
        prob = LpProblem("Weekly_Schedule_Generation", LpMaximize)

        preference_work_load = {}
        for p in preference: 
            preference_work_load[p] = 0
            for day_shifts in preference[p].values(): 
                preference_work_load[p] += len(day_shifts)
        self.preference_work_load = preference_work_load
            
        # preference_work_load["Floriano"] = 1000

        # Objective function: Maximize allocations in preferred hours
        prob += lpSum(
            x[p][d][h] * (1/preference_work_load[p] * self.people_boost.get(p, 1)) for p in preference for d in week_days for h in preference[p][d] if h in x[p][d]
        ), "Maximize Preferred Hours"

        # Constraint: Each shift needs 2 or 3 people
        shifts_with_fix_people = {d: set() for d in week_days}
        for fix_people_hours in self.fix_people_hours:
            capacity = fix_people_hours.capacity
            shifts = fix_people_hours.shifts.shifts_str
            days = fix_people_hours.days

            for d in days:
                shifts_with_fix_people[d].update(shifts)
                for h in shifts:
                    prob += lpSum(x[p][d][h] for p in people) == capacity, f"Shift_{d}_{h}_{capacity}_People"

        # Constraint: Default shift needs 2 people 
        for d in week_days:
            for h in work_shifts:
                if h in shifts_with_fix_people[d]:
                    continue
                prob += lpSum(x[p][d][h] for p in people) == self.default_shift_capacity, f"Shift_{d}_{h}_max_default_people"

        # Constraint: Ensure that each person is only scheduled when available
        for p in people:
            for d in week_days:
                for h in work_shifts:
                    if h not in availability[p][d]: 
                        prob += x[p][d][h] == 0, f"Unavailability_{p}_{d}_{h}"

        # for p in people:
        #     prob += lpSum(x[p][d][h] for d in week_days for h in work_shifts) >= 2 * 2, f"{p}_minimal_work"

        # Constraint: Ensure that each person works close to the desired weekly workload
        delta_h = 1
        for p in people:
            if p not in self.work_load:
                continue
            
            prob += (
                lpSum(x[p][d][h] for d in week_days for h in work_shifts) >= self.work_load[p] - 2*delta_h,
                f"Load_Min_{p}"
            )
            prob += (
                lpSum(x[p][d][h] for d in week_days for h in work_shifts) <= self.work_load[p] + 2*delta_h,
                f"Load_Max_{p}"
            )


        # Constraint: Distribute who opens and closes the day evenly throughout the week
        for p in people:
            prob += lpSum(
                x[p][d][str(self.open_shift)] for d in week_days if str(self.open_shift) in availability[p].get(d, set())
            ) <= self.max_open_per_people, f"Balanced_Weekly_Opening_{p}"
            
            prob += lpSum(
                x[p][d][str(self.close_shift)] for d in week_days if str(self.close_shift) in availability[p].get(d, set())
            ) <= self.max_close_per_people, f"Balanced_Weekly_Closing_{p}"


        # Constraint: Ensure everyone doesn't exceed the maximum work load per day
        if self.max_load_per_day is not None:
            for p in people:
                for d in week_days:
                    prob += lpSum(x[p][d][h] for h in work_shifts) <= self.max_load_per_day*2, f"Max_Load_{p}_{d}"
        
        # Constraint: Ensure everyone doesn't exceed the maximum/minimum work load per week
        if self.min_load_per_week is not None:
            for p in people:
                prob += lpSum(x[p][d][h] for d in week_days for h in work_shifts) >= self.min_load_per_week*2, f"Min_Load_Week_{p}"
        
        if self.max_load_per_week is not None:
            for p in people:
                prob += lpSum(x[p][d][h] for d in week_days for h in work_shifts) <= self.max_load_per_week*2, f"Max_Load_Week_{p}"

        prob.solve(PULP_CBC_CMD(msg=False))

        logger.info("Schedule generated!")
        logger.info(f"Solution Status: {LpStatus[prob.status]}\n")

        # Generate schedule
        schedule = {d: {h: [] for h in work_shifts} for d in week_days}
        for p in people:
            for d in week_days:
                for h in work_shifts:
                    if x[p][d][h].value() == 1:
                        schedule[d][h].append(p)
        
        self.problems.availability = self.check_availability(schedule, availability)

        self.schedule = schedule
        self.x = x

    def check_availability(self, schedule: dict, availability: dict):
        "Returns locations where availability was violated."
        problems = {}
        
        for d, day_sched in schedule.items():
            for t, people in day_sched.items():
                for p in people:
                    if t not in availability[p][d]:
                        if p not in problems:
                            problems[p] = []
                        problems[p].append((d, t))

        # problems = {}
        # for people, days in variables.items():
        #     days: dict
        #     for day, shifts in days.items():
        #         shifts: dict
        #         for shift, var in shifts.items() :
        #             was_summoned = var.value() == 1
        #             is_available = shift in availability[people].get(day, set()) 
        #             if was_summoned and not is_available:
        #                 if people not in problems:
        #                     problems[people] = []

        #                 problems[people].append((day, shift))

        return problems                    

    def show(self):
        "Show schedule generated."
        for d in self.sheets.week_days:
            print(f"\nSchedule for {d.capitalize()}:")
            for h in self.sheets.shifts.shifts_str:
                print(f"{h}: {', '.join(self.schedule[d].get(h, []))}")

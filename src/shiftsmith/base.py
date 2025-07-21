import pandas as pd
import logging
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from collections import namedtuple
import os

from . import readers
from .shifts import ShiftList

# Define a logger for this module
logger = logging.getLogger(__name__)

Mappers = namedtuple('Mappers', [
    'shift_to_id', 'id_to_shift', 
    'week_day_to_id', 'id_to_week_day', 
    'person_to_id', 'id_to_person'
])

class Sheets:
    def __init__(self, 
        pref_path: readers.FileInfo, avail_path: readers.FileInfo, target_work_load_path: readers.FileInfo, shift_capacity_path: readers.FileInfo):
        '''
        Sheets with data to generate a schedule. Each path can be a string
        representing the file path or a `FileInfo` object when additional
        information, such as a sheet name, is needed.

        OBS: Preference and availability are adjusted so that availability becomes a superset
        of preference, and people with no preference have their preference set to match
        their availability.

        Parameters
        ----------
        pref_path:
            People preference file path.

        avail_path:
            People availability file path.
        
        target_work_load_path:
            People desired work load file path.
        
        shift_capacity_path:
            Target number of people per shift file path.
        '''
        self.pref, week_days1, shifts1 = readers.ScheduleReader.read(pref_path, True)
        self.avail, week_days2, shifts2 = readers.ScheduleReader.read(avail_path, True)
        self.target_work_load = readers.TargetWorkLoadReader.read(target_work_load_path)
        self.shift_capacity = readers.ShiftCapacityReader.read(shift_capacity_path)

        if week_days1 != week_days2:
            raise Exception((
                "Week days is not the same!\n"
                f"pref week days: {week_days1}"
                f"avail week days: {week_days2}"
            ))
        if shifts1 != shifts2:
            raise Exception((
                "Shifts is not the same!\n"
                f"pref shifts: {shifts1}"
                f"avail shifts: {shifts2}"
            ))

        self.adjust_pref_avail()

        self.week_days = week_days1
        self.shifts = ShiftList(shifts1)
        self.people = list(self.avail.keys())
        self.mappers = self.create_mappers()
        
        # If a person does not have a preference, is assumed 
        # that his preference is the same as his availability.
        for p in self.people:
            if p not in self.pref:
                self.pref[p] = self.avail[p]

    def adjust_pref_avail(self):
        for p, day_shift in self.pref.items():
            if p not in self.avail.keys():
                self.avail[p] = self.pref[p]
                continue

            for day, shifts in day_shift.items():
                for t in shifts:
                    if t not in self.avail[p][day]:
                        self.avail[p][day].add(t)
        
    def create_mappers(self):
        week_days, shifts, people = self.week_days, self.shifts, self.people

        shift_str_list = [str(t) for t in shifts.shifts]
        shift_to_id = dict(zip(shift_str_list, range(len(shifts))))
        id_to_shift = {v: k for k, v in shift_to_id.items()}

        week_day_to_id = dict(zip(week_days, range(len(week_days))))
        id_to_week_day = {v: k for k, v in week_day_to_id.items()}

        person_to_id = dict(zip(people, range(len(people))))
        id_to_person = {v: k for k, v in person_to_id.items()}

        return Mappers(
            shift_to_id=shift_to_id, id_to_shift=id_to_shift,
            week_day_to_id=week_day_to_id, id_to_week_day=id_to_week_day,
            person_to_id=person_to_id, id_to_person=id_to_person
        )

    def save_pref(self, path):
        self.save_sheet_shifts(self.pref, path)
    
    def save_avail(self, path):
        self.save_sheet_shifts(self.avail, path)

    def save_sheet_shifts(self, sheet: dict, path):
        data = np.full((len(self.shifts), len(self.week_days)), "", dtype=object)

        for person, week_shifts in sheet.items():
            for day, day_shifts in week_shifts.items():
                for shift in day_shifts:
                    col_id = self.week_days.index(day)
                    row_id = self.shifts.shifts_str.index(shift)
                    data[row_id, col_id] += f"{person}, " 

        df = pd.DataFrame(data, columns=self.week_days, index=self.shifts.shifts_str)
        df.to_csv(path)
        
class SchedulerBase(ABC):
    def __init__(self, sheets: Sheets, shifts_weights: dict = None):
        self.sheets = sheets
        self.schedule: dict = None
        
        if shifts_weights is None:
            shifts_weights = {}
        self.shifts_weights = shifts_weights

    @abstractmethod
    def generate(self):
        pass
    
    def add_missing_people(self, missing_name="MISSING"):
        for day in self.sheets.week_days:
            for shift in self.sheets.shifts:
                current_num = len(self.schedule[day][str(shift)])
                target_num = self.sheets.shift_capacity.loc[str(shift), day]
                missing_num = max(0, target_num - current_num)
                for _ in range(missing_num):
                    self.schedule[day][str(shift)].append(missing_name)

    def save(self, path: Path, start_shift_col_name="Start", end_shift_col_name="End"):
        "Save schedule generated at `path` as a .csv"
        week_days = self.sheets.week_days
        shifts = self.sheets.shifts

        data = []
        for h in shifts.shifts_str:
            row = h.split("-")
            for d in week_days:
                row.append(", ".join(self.schedule[d][h]))
            data.append(row)

        df = pd.DataFrame(data, columns=[start_shift_col_name, end_shift_col_name] + week_days)
        df.to_csv(path, index=False) 

        absolute_path = os.path.abspath(path)
        logger.info(f"Schedule saved in: {absolute_path}")

    def save_work_load(self, path, target_work_load=None):
        "Calculates total work load per person and save it at `path` as a .csv"
        people = list(self.sheets.avail.keys())
        people_work_num = dict(zip(people, np.zeros(len(people), dtype=float)))
        
        for d, day_sched in self.schedule.items():
            for t, shift_people in day_sched.items():
                for p in shift_people:
                    people_work_num[p] += self.shifts_weights.get((d, str(t)), 1)
        
        total_work = []
        for p in people:
            p_work = people_work_num[p]
            if p not in target_work_load:
                w_rel = None
            else:
                w_rel = p_work / target_work_load[p]
            
            total_work.append([p, p_work/2, target_work_load[p]/2, w_rel])

        df = pd.DataFrame(total_work, columns=["People", "Given Workload (Hours)", "Requested Workload (Hours)", "Relative Workload"])
        df.to_csv(path, index=False)

        logger.info(f"Workload saved in: {os.path.abspath(path)}")

    def save_pref_avail(self, path):
        self.sheets.save_avail(Path(path) / "availability_output.csv")
        self.sheets.save_pref(Path(path) / "preference_output.csv")
        logger.info(f"Avail. and Pref. saved in folder: {os.path.abspath(path)}")
    
    @staticmethod
    def load_schedule(path, missing_name="MISSING"):
        df = pd.read_csv(path)
        schedule = {}
        start_col_name = df.columns[0]
        end_col_name = df.columns[1]
        for day in df.columns[2:]:
            schedule[day] = {}
            for id, people in df[day].items():
                if pd.isna(people):
                    schedule[day][shift] = []
                    continue
                shift = f"{df.loc[id, start_col_name]}-{df.loc[id, end_col_name]}"
                people_list = []
                for p in people.split(","):
                    if p.strip() == missing_name:
                        continue
                    people_list.append(p.strip())

                schedule[day][shift] = people_list
        return schedule
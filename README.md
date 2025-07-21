# Shiftsmith

`Shiftsmith` uses statistical mechanics to generate shift schedules (a problem also known as workforce scheduling or rostering). Shift scheduling is a very complex problem with practical significance, such as the [nurse rostering problem](https://en.wikipedia.org/wiki/Nurse_scheduling_problem).

`Shiftsmith` uses a metaheuristic known as Simulated Annealing (SA) [[1]](#ref1) to generate shift schedules. This technique has already been successfully applied to real-world problems, such as the nurse rostering problem [[2]](#ref2). SA is a powerful method and can be used in numerous problems. For example, see its application to the famous NP-hard [travelling salesman problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) in [[3]](#ref3).

`Shiftsmith` is intended to generate a shift schedule for a week, given workers' availabilities and preferences. Each weekday is divided into equal shifts. Here is what `Shiftsmith` attempts to do:
- Respect workers' availabilities.
- Assigns workers to their preferences.
- Fills every shift. Shifts capacities (how many workers should be in the shift) can be customized by the user.
- Prioritizes heavy shifts, even if it means violating soft constraints (heavy shifts are specified by the user).
- Ensures every worker receives a lunch break (lunch break shifts are specified by the user).
- Distributes target workloads equally to each worker, calculated from total work time and individual availability, in such a way that the sum of all workers' workload is the total available work time and a worker workload is never greater than his total availability. Shifts can have different weights when calculating total work time.
- Assign compact shifts to each worker.

`Shiftsmith` has a general system to load data that is in the form of tables (availability, preference, shifts capacities, etc). Currently, it supports .ods and .csv files, but the system can easily be expanded to support other file formats.

# Installation
Install the package with the following command
```bash
pip install "git+https://github.com/marcos1561/shiftsmith.git/#egg=shiftsmith"
```
To update the package, run
```bash
pip install --upgrade "git+https://github.com/marcos1561/shiftsmith.git/#egg=shiftsmith"
```

# Quick Start
Here is an example generating a schedule. To see how data should be represent, go to [here](#how-data-should-be-represented).
```python
from shiftsmith.annealing import AnnealingSched, SystemParams, TempScalingLaw, TempConst
from shiftsmith import ShiftList, Sheets
from shiftsmith.readers import OdsInfo

# Parameters used by the annealing algorithm
params = SystemParams(
    k_pref=1,
    k_disp=100,
    k_border=8,
    k_fix_people_gt=100,
    k_fix_people_sm=3,
    k_fix_people_sm_peak=2*10,
    k_work_load=0.6,
    k_overflow_work_load=0.6,
    k_lunch=11,
    k_continuos_lunch=11,
    k_no_people=1,
    temp_strat=[
        TempScalingLaw(
            num_steps=200,
            t1=16,
            exponent=0.6,
        ),
        TempConst(
            num_steps=5,
            t=0.001,
        ),
    ],
)

# Scheduler instantiation, specifying peak shifts, lunch shifts and shift weights.
sched = AnnealingSched(
    params=params,
    init_state="random",
    sheets=Sheets(
        # Loading .csv data
        pref_path="path-to-preference.csv",
        avail_path="path-to-availability.csv",
        target_work_load_path="path-to-target_workload.csv",
        
        # Loading a specific sheet in a .ods file
        shift_capacity_path=OdsInfo(
            path="path-to-shift_capacity.ods", 
            sheet_name="sheet_name",
        ),
    ),
    peak_shifts=ShiftList([
        "07:30-08:00", "08:00-08:30", "10:00-10:30", 
        "11:30-12:00", "13:00-13:30", "13:00-13:30", 
        "15:00-15:30",
    ]),
    lunch_shifts=ShiftList.from_start_end("11:30-12:00", "13:00-13:30"),
    shifts_weights={
        ("Monday", "07:30-08:00"): 2, 
        ("Tuesday", "07:30-08:00"): 2, 
        ("Wednesday", "07:30-08:00"): 2,
        ("Thursday", "07:30-08:00"): 2, 
        ("Friday", "07:30-08:00"): 2,
        ("Friday", "16:30-17:00"): 4,
    },
)

# Calculates the schedule, saves any problems (if any), resulting workload, 
# availability and preference, as well as the final schedule, all under the
# "results" folder.

from pathlib import Path
root_path = Path("results")

sched.generate()
sched.show_problems(root_path / "problems")
sched.save_work_load(root_path / "work_load.csv")
sched.save_pref_avail(root_path)
sched.add_missing_people()
sched.save(root_path / "schedule.csv")
```
After running this code with the data in the following sections, the result is this schedule (which gives lunch to everyone and does not violate availability):

| Start | End | Monday | Tuesday | Wednesday | Thursday | Friday |
| --- | --- | --- | --- | --- | --- | --- |
| 07:30 | 08:00 | Oscar, Grace, Dave | Grace, Dave, Judy | Grace, Alice, Frank | Judy, Frank, Bob | Grace, Alice, Mallory |
| 08:00 | 08:30 | Oscar, Grace, Dave | Grace, Dave, Judy | Grace, Alice, Carol | Judy, Frank, Bob | Grace, Alice, Mallory |
| 08:30 | 09:00 | Oscar, Grace, Dave | Dave, Judy, MISSING | Alice, Ivan, Carol | Judy, Bob, Carol | Alice, Frank, Mallory |
| 09:00 | 09:30 | Oscar, Grace | Dave, Judy | Alice, Ivan, Carol | Bob, Carol | Alice, Frank |
| 09:30 | 10:00 | Oscar, Grace | Oscar, Dave | Alice, Ivan, Carol | Bob, Carol | Alice, Frank |
| 10:00 | 10:30 | Oscar, Grace, MISSING | Oscar, Dave, Judy | Alice, Ivan, Carol | Frank, Bob, Carol | Grace, Alice, Frank |
| 10:30 | 11:00 | Oscar, Grace, MISSING | Oscar, Alice, MISSING | Alice, Ivan, Carol | Frank, Mallory, Carol | Grace, Alice, Frank |
| 11:00 | 11:30 | Oscar, MISSING | Oscar, Alice | Ivan, Carol | Mallory, Carol | Grace, Frank |
| 11:30 | 12:00 | Ivan, MISSING | Alice, Ivan | Alice, Ivan | Mallory, Carol | Grace, Frank |
| 12:00 | 12:30 | MISSING, MISSING | Alice, MISSING | Ivan, MISSING | Mallory, MISSING | Grace, Frank |
| 12:30 | 13:00 | Grace, Frank | Grace, Mallory | Grace, Frank | Alice, Bob | Alice, Mallory |
| 13:00 | 13:30 | Grace, Frank, Ivan | Grace, Mallory, Ivan | Grace, Alice, Frank | Grace, Alice, Bob | Alice, Mallory, Ivan |
| 13:30 | 14:00 | Frank, Ivan | Judy, Ivan | Alice, Heidi | Bob, Mallory | Mallory, Ivan |
| 14:00 | 14:30 | Frank, Ivan | Judy, Ivan | Alice, Heidi | Bob, Mallory | Mallory, Ivan |
| 14:30 | 15:00 | Frank, Ivan | Judy, Ivan | Bob, Heidi | Bob, Mallory | Mallory, Ivan |
| 15:00 | 15:30 | Grace, Frank, Ivan | Judy, Bob, Ivan | Frank, Bob, Heidi | Grace, Bob, Mallory | Mallory, Ivan, MISSING |
| 15:30 | 16:00 | Frank, Ivan | Ivan, MISSING | Bob, Heidi | Bob, Mallory | Mallory, Ivan |
| 16:00 | 16:30 | Frank, Ivan | Ivan, MISSING | Bob, Heidi | Bob, Mallory | Mallory, Eve |
| 16:30 | 17:00 |  |  |  |  | Mallory, Eve |

# How data should be represented?
## Availability and Preference
Availability and preference should be a table with the following columns:

| Shift Start Time | Shift End Time | Monday         | Tuesday | Wednesday | Thursday | Friday |
| ---------------- | -------------- | -------------- | ------- | --------- | -------- | ------ |

- Shit times should be in the format HH:MM, such as 07:00.
- Names in cells should be separated by commas. A cell with names can be empty.
- The actual column names are irrelevant to the code.
- There can be any number of columns after the shift times columns.

Here is the availability used in the example above (with fake names):

| Start | End | Monday | Tuesday | Wednesday | Thursday | Friday |
| --- | --- | --- | --- | --- | --- | --- |
| 07:30 | 08:00 | Marcos, Cristiano, Henrique Hub,  | Grace, Dave, Judy, Alice | Grace, Alice, Frank, Ivan | Grace, Judy, Alice, Frank, Bob | Grace, Alice, Frank, Mallory, Ivan |
| 08:00 | 08:30 | Marcos, Cristiano, Henrique Hub,  | Grace, Dave, Judy, Alice | Grace, Alice, Frank, Ivan, Carol | Grace, Judy, Alice, Frank, Bob, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 08:30 | 09:00 | Marcos, Cristiano, Henrique Hub,  | Dave, Judy, Alice, Bob | Grace, Alice, Frank, Ivan, Carol | Judy, Alice, Frank, Bob, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 09:00 | 09:30 | Marcos, Cristiano,  | Dave, Judy, Alice, Bob | Grace, Alice, Frank, Ivan, Carol | Judy, Alice, Frank, Bob, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 09:30 | 10:00 | Marcos, Cristiano,  | Oscar, Dave, Judy, Alice, Bob, Ivan | Grace, Alice, Frank, Ivan, Carol | Alice, Frank, Bob, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 10:00 | 10:30 | Marcos, Cristiano,  | Oscar, Dave, Judy, Alice, Bob, Ivan | Grace, Alice, Frank, Ivan, Carol | Alice, Frank, Bob, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 10:30 | 11:00 | Marcos, Cristiano,  | Oscar, Grace, Alice, Ivan | Grace, Alice, Frank, Ivan, Carol | Grace, Alice, Frank, Mallory, Ivan, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 11:00 | 11:30 | Marcos,  | Oscar, Alice, Ivan | Alice, Ivan, Carol | Alice, Frank, Mallory, Ivan, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 11:30 | 12:00 | Vinícius Baêta,  | Alice, Ivan | Alice, Ivan | Alice, Mallory, Ivan, Carol | Grace, Alice, Frank, Mallory, Ivan |
| 12:00 | 12:30 | Cristiano, Eduarda B, Vinícius Baêta,  | Grace, Judy, Alice, Bob, Mallory, Ivan | Grace, Alice, Ivan | Grace, Alice, Bob, Mallory, Ivan | Grace, Alice, Frank, Mallory, Ivan |
| 12:30 | 13:00 | Cristiano, Eduarda B, Vinícius Baêta,  | Grace, Alice, Mallory, Ivan | Grace, Alice, Frank, Ivan | Grace, Alice, Bob, Mallory, Ivan | Grace, Alice, Mallory, Ivan |
| 13:00 | 13:30 | Cristiano, Eduarda B, Vinícius Baêta,  | Grace, Alice, Frank, Mallory, Ivan | Grace, Alice, Frank | Grace, Alice, Bob, Mallory, Ivan | Alice, Frank, Mallory, Ivan |
| 13:30 | 14:00 | Cristiano, Eduarda B, Vinícius Baêta,  | Grace, Judy, Alice, Bob, Mallory, Ivan | Grace, Alice, Frank, Heidi | Grace, Alice, Bob, Mallory, Ivan | Alice, Frank, Mallory, Ivan |
| 14:00 | 14:30 | Cristiano, Eduarda B, Vinícius Baêta,  | Grace, Judy, Alice, Bob, Mallory, Ivan | Grace, Alice, Frank, Heidi | Grace, Alice, Bob, Mallory, Ivan | Alice, Mallory, Ivan |
| 14:30 | 15:00 | Cristiano, Eduarda B, Vinícius Baêta,  | Grace, Judy, Alice, Bob, Mallory, Ivan | Grace, Alice, Frank, Bob, Heidi | Grace, Alice, Bob, Mallory, Ivan | Alice, Mallory, Ivan |
| 15:00 | 15:30 | Cristiano, Eduarda B, Vinícius Baêta,  | Grace, Judy, Bob, Ivan | Grace, Frank, Bob, Heidi | Grace, Bob, Mallory, Ivan | Mallory, Ivan |
| 15:30 | 16:00 | Eduarda B, Vinícius Baêta,  | Grace, Ivan | Frank, Bob, Heidi | Grace, Bob, Mallory, Ivan | Mallory, Ivan |
| 16:00 | 16:30 | Eduarda B, Vinícius Baêta,  | Ivan | Frank, Bob, Heidi | Grace, Bob, Mallory, Ivan | Grace, Mallory, Eve, Ivan |
| 16:30 | 17:00 |  |  |  |  | Grace, Mallory, Eve, Ivan |

## Shift Capacity
Shift capacity is a table in the same format as the availability and preference, the only difference being that its cells should have integer numbers (how many workers should be in this specific shift). Here is an example:

| Start | End | Segunda | Terça | Quarta | Quinta | Sexta |
| --- | --- | --- | --- | --- | --- | --- |
| 07:30 | 08:00 | 3 | 3 | 3 | 3 | 3 |
| 08:00 | 08:30 | 3 | 3 | 3 | 3 | 3 |
| 08:30 | 09:00 | 3 | 3 | 3 | 3 | 3 |
| 09:00 | 09:30 | 2 | 2 | 3 | 2 | 2 |
| 09:30 | 10:00 | 2 | 2 | 3 | 2 | 2 |
| 10:00 | 10:30 | 3 | 3 | 3 | 3 | 3 |
| 10:30 | 11:00 | 3 | 3 | 3 | 3 | 3 |
| 11:00 | 11:30 | 2 | 2 | 2 | 2 | 2 |
| 11:30 | 12:00 | 2 | 2 | 2 | 2 | 2 |
| 12:00 | 12:30 | 2 | 2 | 2 | 2 | 2 |
| 12:30 | 13:00 | 2 | 2 | 2 | 2 | 2 |
| 13:00 | 13:30 | 3 | 3 | 3 | 3 | 3 |
| 13:30 | 14:00 | 2 | 2 | 2 | 2 | 2 |
| 14:00 | 14:30 | 2 | 2 | 2 | 2 | 2 |
| 14:30 | 15:00 | 2 | 2 | 2 | 2 | 2 |
| 15:00 | 15:30 | 3 | 3 | 3 | 3 | 3 |
| 15:30 | 16:00 | 2 | 2 | 2 | 2 | 2 |
| 16:00 | 16:30 | 2 | 2 | 2 | 2 | 2 |
| 16:30 | 17:00 | 0 | 0 | 0 | 0 | 2 |

# References
1. <a id="ref1"></a> KIRKPATRICK, S.; GELATT, C. D.; VECCHI, M. P. Optimization by Simulated Annealing. Science, v. 220, n. 4598, p. 671–680, 13 maio 1983. 
2. <a id="ref2"></a> CESCHIA, Sara et al. Solving a real-world nurse rostering problem by Simulated Annealing. Operations Research for Health Care, v. 36, p. 100379, 1 mar. 2023. 
3. <a id="ref3"></a> DA SILVA, Roberto; FILHO, Eliseu Venites; ALVES, Alexandre. A thorough study of the performance of simulated annealing in the traveling salesman problem under correlated and long tailed spatial scenarios. Physica A: Statistical Mechanics and its Applications, v. 577, p. 126067, 1 set. 2021. 

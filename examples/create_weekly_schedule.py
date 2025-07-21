from shiftsmith import ShiftList, Sheets
from shiftsmith.annealing import AnnealingSched, SystemParams, TempScalingLaw

# Parameters used by the annealing algorithm
params = SystemParams(
    k_pref=1,
    k_disp=100,
    k_border=10,
    k_fix_people_gt=100,
    k_fix_people_sm=3,
    k_fix_people_sm_peak=2*10,
    k_work_load=0.6,
    k_overflow_work_load=0.6,
    k_lunch=11,
    k_continuos_lunch=11,
    k_no_people=1,
    temp_strat=TempScalingLaw(
        num_steps=200,
        t1=10,
        exponent=0.6,
    )
)

# Scheduler instantiation, specifying peak shifts, lunch shifts and shift weights.
sched = AnnealingSched(
    params=params,
    init_state="random",
    sheets=Sheets(
        pref_path="path-to-preference.csv",
        avail_path="path-to-availability.csv",
        shift_capacity_path="path-to-shift_capacity.ods",
    ),
    peak_shifts=ShiftList([
        "07:30-08:00", "08:00-08:30", "10:00-10:30", 
        "11:30-12:00", "13:00-13:30", "13:00-13:30", 
        "15:00-15:30",
    ]),
    lunch_shifts=ShiftList.from_start_end(
        "11:30-12:00", "13:00-13:30",
    ),
    shifts_weights={
        ("Segunda", "07:30-08:00"): 2, 
        ("Terça", "07:30-08:00"): 2, 
        ("Quarta", "07:30-08:00"): 2,
        ("Quinta", "07:30-08:00"): 2, 
        ("Sexta", "07:30-08:00"): 2,
        ("Sexta", "16:30-17:00"): 3,
    },
)

sched.generate()
sched.show_problems()
sched.save_work_load("work_load.csv")
sched.save("schedule.csv")
sched.show_energy()
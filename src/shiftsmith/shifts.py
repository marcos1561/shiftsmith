from datetime import datetime, timedelta
from datetime import time as TimeType

class Shift:
    def __init__(self, shift: str, duration: timedelta=None):
        '''
        A work shift initializing at `init_hour`.

        Parameter
        ---------
        shift:
            str:
                Shift in the format "HH:MM-HH:MM"
            
            datetime.Time
                start shift time, its duration is specified
                in `duration`.

        duration:
            Shift duration, only applicable if shift is
            datetime.Time.
        '''
        if isinstance(shift, str):
            i, f = shift.split("-")

            self.init = datetime.strptime(i, "%H:%M").time()
            self.end = datetime.strptime(f, "%H:%M").time()
        elif isinstance(shift, TimeType):
            self.init = shift
            self.end = (datetime.combine(datetime.today(), self.init) + duration).time()
        elif isinstance(shift, Shift):
            self.init = shift.init
            self.end = shift.end
            self.duration = shift.duration
            return
        else:
            raise ValueError(f"`shift` should be str, time or Shift, instead it is {type(shift)}.")
        
        self.duration = datetime.combine(datetime.today(), self.end) - datetime.combine(datetime.today(), self.init)

    def __str__(self):
        init = f"{self.init.hour:02}:{self.init.minute:02}"
        end = f"{self.end.hour:02}:{self.end.minute:02}"
        return f"{init}-{end}"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Shift):
            return self.init == other.init and self.end == other.end
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def next(self):
        return Shift(self.end, duration=self.duration)

    def __lt__(self, other):
        if isinstance(other, Shift):
            return self.init < other.init
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Shift):
            return self.init > other.init
        return NotImplemented


class ShiftList:
    def __init__(self, shifts: list[Shift]):
        "List of shifts."
        self.shifts = [Shift(t) for t in shifts]
        self.shifts_str = [t.__str__() for t in self.shifts]

    @classmethod
    def from_start_end(Cls, start_shift: Shift, end_shift: Shift):
        "Creates list of all shifts between `start_shift` and `end_shift`."
        start_shift = Shift(start_shift)
        end_shift = Shift(end_shift)

        shifts = [start_shift]
        current_shift = start_shift
        while current_shift != end_shift:
            current_shift = current_shift.next()
            shifts.append(current_shift)

        return Cls(shifts)
    
    def __add__(self, other):
        if isinstance(other, ShiftList):
            combined_shifts = self.shifts + other.shifts
            return ShiftList(combined_shifts)
        else:
            raise ValueError(f"Cannot add ShiftList with {type(other)}")

    def __contains__(self, item):
        item = Shift(item)
        return item in self.shifts

    def __len__(self):
        return len(self.shifts)
    
    def __iter__(self):
        return iter(self.shifts)

if __name__ == "__main__":
    a = ShiftList.from_start_end("07:30-08:00", "12:00-12:30")
    b = ShiftList.from_start_end("14:00-14:30", "15:30-16:00")
    c = a + b
    print(c.shifts_str)
    # print(a.shifts[0].duration.)
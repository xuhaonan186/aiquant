import pandas as pd


def get_calendars():
    cal = pd.read_hdf(r"K:\cta\fut_repo\calendar.h5", 'data')
    return cal


class XhnScheduler(object):
    def __init__(self, start_date, end_date, ntrain_days, rolling_days):
        self.start_date = start_date
        self.end_date = end_date
        self.ntrain_days = ntrain_days
        self.rolling_days = rolling_days

    def get_scheduler(self):
        cal = get_calendars()
        cal = cal[(cal[0] >= self.start_date) & (cal[0] <= self.end_date)]
        cal = cal.reset_index(drop=True)
        step = (cal.shape[0]-self.ntrain_days-self.rolling_days)//self.rolling_days
        if step<0:
            raise Exception('date error')
        train_scheduler = []
        index = self.ntrain_days
        for i in range(step+1):
            index = index+i*self.rolling_days
            train_sdt, train_edt = cal.iloc[index-self.ntrain_days, 0], cal.iloc[index-1, 0]
            test_sdt, test_edt = cal.iloc[index, 0], cal.iloc[index+self.rolling_days-1, 0]
            train_scheduler.append([train_sdt, train_edt, test_sdt, test_edt])
        train_scheduler[-1][-1] = cal.iloc[-1][0]
        return train_scheduler

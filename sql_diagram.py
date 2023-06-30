import clickhouse_connect
from datetime import datetime, timedelta
import math

def diagram_days_sql():

    #Поключение к clickhouse
    client = clickhouse_connect.get_client(host='pheerses.space', port=8123, username='practice', password='secretKey_lhv323as5vc_d23k32mk')
    #Получение данных
    countid_groupby_time = client.query('select count(case_id), dt from (select case_id, min(date(start_time)) as dt from main_table '
                                        'group by case_id ORDER BY dt) group by dt')

    start = countid_groupby_time.result_rows[0][1]
    end = countid_groupby_time.result_rows[-1][1]
    all_days = (end-start).days
    list_days = [] #[№дня, №дня, дата начала, дата конца, к-во уникальных Id]
    my_start = 0 #первый день для каждого периода
    step = round(int(all_days)/10) #длина промежутка в днях
    my_date = start
    now_date = 0#реально существующий день: порядковый номер

    #Добавление данных [№дня, №дня, дата начала, дата конца, к-во уникальных Id] в список
    while my_start < int(all_days):
        total_count = 0
        if my_start+step <= int(all_days):
            step_end = my_start+step
        else:
            step_end = all_days
        #Считаем количество уникальных Id для конкретного периода
        for i in range(now_date, len(countid_groupby_time.result_rows)):
            if countid_groupby_time.result_rows[i][1] > my_date + timedelta(days=step):
                now_date = i
                break
            total_count = total_count + countid_groupby_time.result_rows[i][0]

        list_days.append([my_start, step_end, my_date, my_date + timedelta(days=step), total_count])
        my_start = my_start + step + 1
        my_date = my_date + timedelta(days=step+1)

    # для получения часов/минут перевести my_start, step_end
    return list_days


def diagram_months_sql():

    #Поключение к clickhouse
    client = clickhouse_connect.get_client(host='pheerses.space', port=8123, username='practice', password='secretKey_lhv323as5vc_d23k32mk')
    #Получение данных
    countid_groupby_time = client.query('select count(case_id), dt from (select case_id, min(date(start_time)) as dt from main_table '
                                        'group by case_id ORDER BY dt) group by dt')

    list_month = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [0,8], [0,9], [0,10], [0,11], [0,12]]  # [к-во уникальных Id, №месяца]

    # Добавление данных [к-во уникальных Id, № месяца] в список
    # Считаем количество уникальных Id для конкретного месяца
    for i in countid_groupby_time.result_rows:
        list_month[i[1].month-1][0] = list_month[i[1].month-1][0] + i[0]

    return list_month


if __name__ == '__main__':

    print(diagram_days_sql())
    print()
    print(diagram_months_sql())


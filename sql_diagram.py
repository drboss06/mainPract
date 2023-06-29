import clickhouse_connect
from datetime import datetime, timedelta
import math

def diagram_sql():

    #Получение и обработка БД
    #Поключение к clickhouse
    client = clickhouse_connect.get_client(host='pheerses.space', port=8123, username='practice', password='secretKey_lhv323as5vc_d23k32mk')

    start = client.query('SELECT top 1 start_time FROM main_table where start_time = (SELECT MIN(start_time) FROM main_table)')
    end = client.query('SELECT top 1 start_time FROM main_table where start_time = (SELECT MAX(start_time) FROM main_table)')
    countid_groupby_time = client.query('SELECT count(case_id), start_time FROM main_table GROUP BY start_time')
    print(countid_groupby_time.result_rows)
    start = start.result_rows[0][0]
    end = end.result_rows[0][0]
    all_days = (end-start).days
    list_days = [] # [№дня, №дня, дата начала, дата конца, к-во уникальных Id]
    my_start = 0
    step = round(int(all_days)/10)
    my_data = datetime.date(start)

    while my_start < int(all_days):
        if my_start+step <= int(all_days):
            step_end = my_start+step
        else:
            step_end = all_days
        list_days.append([my_start, step_end, my_data, my_data + timedelta(days=step), 'уникальная хня'])
        my_start = my_start + step + 1
        my_data = my_data + timedelta(days=step+1)


    print(list_days)





if __name__ == '__main__':
    diagram_sql()
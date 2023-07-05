import clickhouse_connect
from datetime import timedelta
import logging

logging.basicConfig(level=20, filename="column_diagram_log.log",
                            format="%(asctime)s %(levelname)s %(message)s")

class Calc_diagrams():

    def calc_diagram_days(self, data):

        # Получение данных
        data = data

        list_days = []  # [к-во уникальных Id, №дня, №дня]
        start_day = 0
        for i in range(0, len(data.result_rows)-1):
            list_days.append([data.result_rows[i][0], start_day, data.result_rows[i][1]])
            start_day = data.result_rows[i][1]

        logging.info(f"call calc_diagram_days: {list_days}")
        # для получения часов/минут перевести my_start, step_end
        return list_days

    def calc_diagram_months(self, data):

        # Получение данных
        data = data

        list_month = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11],
                      [0, 12]]  # [к-во уникальных Id, №месяца]

        # Добавление данных [к-во уникальных Id, № месяца] в список
        # Считаем количество уникальных Id для конкретного месяца
        for i in data.result_rows:
            list_month[i[1].month - 1][0] = list_month[i[1].month - 1][0] + i[0]

        logging.info(f"call calc_diagram_months: {list_month}")
        return list_month



if __name__ == '__main__':

    client = clickhouse_connect.get_client(host='pheerses.space',
                                           port=8123,
                                           username='practice',
                                           password='secretKey_lhv323as5vc_d23k32mk')

    data_months = client.query('select count(case_id), dt from (select case_id, min(date(start_time)) as dt '
                                         'from main_table group by case_id ORDER BY dt) group by dt')
    data_days = client.query('select count(case_id), cast(round(max(diff)) as smallint), floor(diff/(select max(diff)/10 from '
                             '(select case_id, date_diff(second, min(start_time), max(end_time))/(60*60*24) as diff '
                             'from main_table group by case_id))) as a from (select case_id, '
                             'date_diff(second, min(start_time), max(end_time))/(60*60*24) as diff '
                             'from main_table group by case_id order by diff) group by a order by a')

    diagrams = Calc_diagrams()
    print(diagrams.calc_diagram_days(data_days))
    print()
    print(diagrams.calc_diagram_months(data_months))

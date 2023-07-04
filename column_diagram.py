import clickhouse_connect
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO, filename="column_diagram_log.log",
                            format="%(asctime)s %(levelname)s %(message)s")

class DB_connection_groupby():

    #подключение к clickhouse и запрос - разные методы
    '''
    def __clickhouse_connection(self, host, port, user, password):
        connection = clickhouse_connect.get_client(host=host, port=port, username=user, password=password)
        return connection


    def db_connection_groupby(self):

        client = self.__clickhouse_connection('pheerses.space', 8123, 'practice', 'secretKey_lhv323as5vc_d23k32mk')

        groupby_query = client.query('select count(case_id), dt from (select case_id, min(date(start_time)) as dt '
                               'from main_table group by case_id ORDER BY dt) group by dt')

        return groupby_query

    '''

    # подключение к clickhouse и запрос в одном методе
    def db_connection_groupby(self):

        try:
            client = clickhouse_connect.get_client(host='pheerses.space',
                                                   port=8123,
                                                   username='practice',
                                                   password='secretKey_lhv323as5vc_d23k32mk')
            logging.info("clickhouse connection successful")
        except:
            logging.error("clickhouse connection failed")

        try:
            groupby_query = client.query('select count(case_id), dt from (select case_id, min(date(start_time)) as dt '
                                         'from main_table group by case_id ORDER BY dt) group by dt')
            logging.info("query successful")
        except:
            logging.error("query failed")

        return groupby_query


class Calc_diagrams():

    def calc_diagram_days(self, data):

        # Получение данных
        data = data

        start = data.result_rows[0][1]
        end = data.result_rows[-1][1]
        all_days = (end - start).days
        list_days = []  # [№дня, №дня, дата начала, дата конца, к-во уникальных Id]
        my_start = 0  # первый день для каждого периода
        step = round(int(all_days) / 10)  # длина промежутка в днях
        my_date = start
        now_date = 0  # реально существующий день: порядковый номер

        # Добавление данных [№дня, №дня, дата начала, дата конца, к-во уникальных Id] в список
        while my_start < int(all_days):
            total_count = 0
            if my_start + step <= int(all_days):
                step_end = my_start + step
            else:
                step_end = all_days
            # Считаем количество уникальных Id для конкретного периода
            for i in range(now_date, len(data.result_rows)):
                if data.result_rows[i][1] > my_date + timedelta(days=step):
                    now_date = i
                    break
                total_count = total_count + data.result_rows[i][0]

            list_days.append([my_start, step_end, my_date, my_date + timedelta(days=step), total_count])
            my_start = my_start + step + 1
            my_date = my_date + timedelta(days=step + 1)

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

    data = DB_connection_groupby().db_connection_groupby()
    diagrams = Calc_diagrams()
    print(diagrams.calc_diagram_days(data))
    print(diagrams.calc_diagram_months(data))


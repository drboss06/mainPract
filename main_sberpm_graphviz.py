#Обработка данных
from sberpm import DataHolder
from sberpm.metrics import ActivityMetric, TransitionMetric, IdMetric, TraceMetric, UserMetric

#Graphviz
from sberpm.visual import GraphvizPainter

#Различные варинаты майнеров
from sberpm.miners import HeuMiner, SimpleMiner,CausalMiner,AlphaMiner,AlphaPlusMiner

import clickhouse_connect
import pandas as pd
import graphviz as gz


def  initializating_pm():

    #Получение и обработка БД
    #Поключение к clickhouse
    client = clickhouse_connect.get_client(host='pheerses.space', port=8123, username='practice', password='secretKey_lhv323as5vc_d23k32mk')

    # Получение данных и преобразование в DataFrame
    #input_activity =
    lst_data = []
    list_id = client.query('SELECT case_id, activity, start_time, end_time FROM main_table')
    print(1)
    for i in list_id.result_rows:
        lst_data.append(i)
    lst_name = list_id.column_names
    print(2)
    df = pd.DataFrame(lst_data, columns=lst_name)
    print(3)

    data_holder = DataHolder(data=df,
                         id_column='case_id',
                         activity_column='activity',
                         start_timestamp_column='start_time',
                         end_timestamp_column='end_time',
                         time_format='%Y-%m-%d %I:%M:%S')

    data_holder.check_or_calc_duration()
    data_holder.data.head()

    activity_metric = ActivityMetric(data_holder, time_unit='d')
    count_metric = activity_metric.count().to_dict()

    transition_metric = TransitionMetric(data_holder, time_unit='d')
    transition_metric.apply().head()
    edges_count_metric = transition_metric.count().to_dict()


#Область объявления майнеров

    #Обявление списка на отрисовку
    miner_graphs = []

    #Hei miner - эвристический майнер, который удаляет наиболее редкие связи в зависимости от задаваемого порога (threshold)
    heu_miner = HeuMiner(data_holder, threshold=0.8)
    heu_miner.apply()
    miner_graphs.append(heu_miner.graph)

    #Simple Miner - отрисовывает все ребра, найденные в логе (без какой-либо фильтрации)
    simple_miner = SimpleMiner(data_holder)
    simple_miner.apply()
    miner_graphs.append(simple_miner.graph)

    #Casual Miner -
    casual_miner = CausalMiner(data_holder)
    casual_miner.apply()
    miner_graphs.append(casual_miner.graph)

    #Alpha Miner
    alpha_miner = AlphaMiner(data_holder)
    alpha_miner.apply()
    miner_graphs.append(alpha_miner.graph)

    #AlphaPlus Miner
    alphaplus_miner = AlphaPlusMiner(data_holder)
    alphaplus_miner.apply()
    miner_graphs.append(alphaplus_miner.graph)


# Модуль отрисовки
    for (index, elem) in enumerate(miner_graphs):

        try:
            elem.add_node_metric('count', count_metric)
            elem.add_edge_metric('count', edges_count_metric)
        except:
            print('ошибочка')
        
        painter = GraphvizPainter()
        painter.apply(elem, node_style_metric='count', edge_style_metric='count')

        custom_graph(elem.nodes, elem.edges, 'graph' + str(index), format='svg')


def custom_graph(nodes,edges,file,format='svg'):
    ps = gz.Digraph(file, node_attr={'shape': 'plaintext', 'color': '#2d137d', 'fontcolor': '#2d137d',
                                              'fontsize': '12.0', 'size': '2', 'image':'1.png'},
                                              edge_attr={'color': '#2d137d', 'fontcolor': '#2d137d', 'fontsize': '9.0'})
    for g_node in nodes:
        metric = nodes.get(g_node).metrics.get('count')
        if g_node == 'startevent':
            ps.node(g_node, image='', label='')
        elif g_node == 'endevent':
            ps.node(g_node, image='', label='')
        else:
            ps.node(g_node, label=r'' + g_node + '\n' + str(metric) + '', )

    for g_edge in edges:
        metric = edges.get(g_edge).metrics.get('count')
        if metric == None:
            ps.edge(g_edge[0], g_edge[1])
        else:
            ps.edge(g_edge[0], g_edge[1], label=str(metric))

    ps.format = 'svg'
    ps.render()


if __name__ == '__main__':
    initializating_pm()

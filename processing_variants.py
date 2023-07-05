import clickhouse_connect
import logging
import array as arr
import networkx as nx
from networkx.algorithms import isomorphism

class Connect:
    def __init__(self, client=None, graphs=None, top_variables=None, other_variables=None):
        self.client = None
        self.graphs = None
        self.top_variables = None
        self.other_variables = None

    def apply(self,client,graphs):
        self.client = client
        self.graphs = graphs
        self.top_variables = []
        self.other_variables = []
        result = self.graphs.result_rows

        for item in result:
            if item[1] > 1:
                self.top_variables.append(item)
            else:
                self.other_variables.append(item)

    def combine_variables(self):
        result_top = self.top_variables
        combine_top = []
        result_other = self.other_variables
        for (index,res) in enumerate(result_top):
            a = (str(index), res)
            print(a)
            g1 = nx.Graph()
            for (intex, i) in enumerate(res[2]):
                try:
                    g1.add_node(i)
                    g1.add_edge(res[intex], res[intex + 1])
                except:
                    k = 0
            for res2 in result_other:
                #combine_top.append(a)
                #common_paths = [p for p in res[2] if p in res2[2] and p in res[2][res[2].index(p):]]
                #similarity = (len(common_paths)/1.5) / len(res[2])
                #if similarity >= 0.8:
                g2 = nx.Graph()
                for (inter, i) in enumerate(res2[2]):
                    try:
                        g2.add_node(i)
                        g2.add_edge(res2[inter], res2[inter + 1])
                    except:
                        k=0
                GM = isomorphism.GraphMatcher(g1, g2)
                if GM.is_isomorphic():
                    b = (str(index), res2)
                    print(b)
                    combine_top.append(b)
                    result_other.remove(res2)
        return(combine_top)




if __name__ == '__main__':
    #Поключение к clickhouse
    client = clickhouse_connect.get_client(host='pheerses.space', port=8123, username='practice', password='secretKey_lhv323as5vc_d23k32mk')
    #Получение данных
    graphs = client.query('select count(case_id), count(case_id)*100/(select count(distinct case_id) from main_table), a from'
                          '(select case_id, arrayMap((x)->x[1], arraySort((x)->toDateTime(x[2]), groupArray([activity,'
                          'toString(start_time)]))) as a  from main_table group by case_id) group by a order by count(case_id) desc')
    connect = Connect()
    connect.apply(client, graphs)
    connect.combine_variables()
    #print(connect.top_variables)
    print()
    print(connect.other_variables)